"""The agent's tools. All business logic lives here: URL guarding, caching, chunking,
evidence verification, Pydantic validation with a bounded repair loop, question commits.

Every write is an idempotent upsert so a replayed tool call after a crash is harmless.
Transient scrape failures raise `TransientScrapeError` (retried by middleware); permanent
ones are reported in the tool result and never retried.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain.tools import tool
from langgraph.types import interrupt
from pydantic import ValidationError

from profile_builder.config import (
    DISCOVERY_FILENAME,
    MAX_EVIDENCE_EXCERPT_CHARS,
    MAX_FINALIZE_REFUSALS,
    MAX_PAGE_CHARS,
    MAX_REPAIR_ATTEMPTS,
    MAX_URLS_PER_SCRAPE_CALL,
    MIN_EXCERPT_CHARS,
    PAGE_HEADINGS_TO_MODEL,
    PAGE_LEAD_CHARS,
    PAGES_DIRNAME,
    SCRAPE_ATTEMPTS_PER_PAGE,
    SCRAPE_INTER_REQUEST_DELAY_S,
    THIN_DRAFT_SECTION_FIELDS,
    Settings,
)
from profile_builder.logging_setup import event, get_logger, redact_text, set_run_context
from profile_builder.retrieval.chunking import text_hash
from profile_builder.retrieval.index import HybridIndex, tokenize
from profile_builder.schema import (
    FEATURE_LIST_PATH,
    STRING_LIST_PATHS,
    STRING_PATHS,
    CompanyBrain,
    FieldPathError,
    get_by_path,
    parse_field_path,
    set_by_path,
    strip_unknown_keys,
)
from profile_builder.security import (
    atomic_write_text,
    detect_injection,
    excerpt_in_page,
    sanitize_answer,
    url_cache_name,
)
from profile_builder.state.run_store import PageRecord, RunStore
from profile_builder.web.discovery import discover
from profile_builder.web.robots import RobotsChecker
from profile_builder.web.scraper import (
    PermanentScrapeError,
    ScrapedPage,
    Scraper,
    TransientScrapeError,
)
from profile_builder.web.url_guard import URLGuardError, normalize_url, same_site, validate_url
from profile_builder.workflow.export import evidence_paths, write_outputs
from profile_builder.workflow.gaps import grounding_report, prioritize_for_interview

log = get_logger("tools")

SKIP_WORDS = frozenset({"skip", "s", "pass", "next"})
UNKNOWN_WORDS = frozenset(
    {
        "idk",
        "i don't know",
        "i dont know",
        "dont know",
        "don't know",
        "unknown",
        "not sure",
        "no idea",
        "?",
    }
)
QUESTION_KINDS = ("product_selection", "gap", "conflict", "brand")
_NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*(?:\(?\d{1,2}[.)]|[a-d][.)]|[-•*])\s+\S")
MAX_QUESTION_MARKS = 2  # a question plus one clarifying sub-question is fine; a list is not
FINALIZE_GUARD_SECTIONS = frozenset({"product", "customer", "content_evidence"})


def is_transient_code(code: str | None) -> bool:
    return bool(code) and (code in TRANSIENT_CODES or code.startswith("HTTP_5"))


OBSERVATION_PATHS = frozenset({"brand.voice_and_tone", "brand.writing_style"})
STEM_CHARS = 5
MIN_CONTENT_TOKEN_CHARS = 4
TRANSIENT_CODES = frozenset(
    {"RATE_LIMITED", "TIMEOUT", "SERVER_ERROR", "NETWORK", "TRANSIENT", "HTTP_429", "HTTP_408"}
)
STAGES = ("discover", "scrape", "research", "draft", "interview", "finalize", "done")


class FatalProfileError(RuntimeError):
    """Model output stayed invalid after the allowed repair attempt (or another unrecoverable
    profile error). The runner persists state and reports an actionable error."""


@dataclass
class ToolContext:
    settings: Settings
    store: RunStore
    scraper: Scraper
    index: HybridIndex
    robots: RobotsChecker
    run_dir: Path
    run_id: str
    start_url: str
    product_focus: str | None = None
    check_dns: bool = True
    _page_text_cache: dict[str, str] = field(default_factory=dict)

    # ---- helpers -------------------------------------------------------------------
    @property
    def pages_dir(self) -> Path:
        return self.run_dir / PAGES_DIRNAME

    def set_stage(self, stage: str) -> None:
        current = self.store.get_run().stage
        if current != stage:
            self.store.set_stage(stage)
            set_run_context(stage=stage)
            event(log, "stage_changed", f"stage → {stage}", kind=stage)

    def page_budget_remaining(self) -> int:
        """Unique pages scraped (incl. duplicates/empties/permanent failures) vs. the cap."""
        return max(0, self.settings.max_pages - self.store.unique_pages_scraped())

    def page_text(self, url: str) -> str | None:
        url = normalize_url(url)
        if url in self._page_text_cache:
            return self._page_text_cache[url]
        rec = self.store.get_page(url)
        if not rec or rec.status != "fetched" or not rec.cache_file:
            return None
        path = self.pages_dir / rec.cache_file
        if not path.exists():
            return None
        page = ScrapedPage.from_json(path.read_text(encoding="utf-8"))
        self._page_text_cache[url] = page.markdown
        return page.markdown

    def cached_page(self, url: str) -> ScrapedPage | None:
        path = self.pages_dir / url_cache_name(normalize_url(url))
        if not path.exists():
            return None
        try:
            page = ScrapedPage.from_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        page.cached = True
        return page

    def warn(self, code: str, message: str, **details: Any) -> None:
        message = redact_text(message)
        self.store.add_warning(code, message, details or None)
        event(log, "warning", message, level=logging.WARNING, code=code)


def is_multi_question(text: str) -> bool:
    """True when a single ask_user call tries to bundle several questions."""
    numbered = len(_NUMBERED_ITEM_RE.findall(text or ""))
    return numbered >= 2 or (text or "").count("?") > MAX_QUESTION_MARKS


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=None, default=str)


def _format_validation_error(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors()[:12]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        out.append(f"{loc}: {err.get('msg')}")
    return out


# --------------------------------------------------------------------------------------
# Page processing (shared by discover + scrape)
# --------------------------------------------------------------------------------------


def process_page(ctx: ToolContext, page: ScrapedPage, requested_url: str) -> dict[str, Any]:
    """Validate, cache, index and record a fetched page. Returns the compact summary the
    model sees. Idempotent: re-processing the same page overwrites the same rows."""
    url = normalize_url(requested_url)
    final_url = page.final_url or url
    try:
        final_norm = validate_url(final_url, check_dns=ctx.check_dns)
    except URLGuardError as exc:
        ctx.store.upsert_page(
            PageRecord(
                url,
                final_url,
                "failed",
                page.http_status,
                page.title,
                None,
                0,
                None,
                "REDIRECT_REJECTED",
                str(exc),
                True,
                time.time(),
            )
        )
        ctx.warn("REDIRECT_REJECTED", f"{url} redirected to a disallowed destination: {exc}")
        return {"url": url, "status": "rejected", "error": f"redirect target rejected: {exc}"}
    if not same_site(final_norm, ctx.start_url):
        ctx.store.upsert_page(
            PageRecord(
                url,
                final_norm,
                "failed",
                page.http_status,
                page.title,
                None,
                0,
                None,
                "REDIRECT_OFFSITE",
                "redirected off-site",
                True,
                time.time(),
            )
        )
        ctx.warn("REDIRECT_OFFSITE", f"{url} redirected off-site to {final_norm}; skipped")
        return {"url": url, "status": "rejected", "error": "redirected off-site"}

    markdown = (page.markdown or "").strip()
    truncated = False
    if len(markdown) > MAX_PAGE_CHARS:
        markdown = markdown[:MAX_PAGE_CHARS]
        truncated = True
    if len(markdown) < 80:
        ctx.store.upsert_page(
            PageRecord(
                url,
                final_norm,
                "skipped",
                page.http_status,
                page.title,
                None,
                len(markdown),
                None,
                "EMPTY_CONTENT",
                "page had no usable text",
                True,
                time.time(),
            )
        )
        ctx.warn("EMPTY_CONTENT", f"{url} returned no usable text; skipped")
        return {"url": url, "status": "skipped", "error": "empty content"}

    content_hash = text_hash(markdown)
    duplicate_of = ctx.store.content_hash_exists(content_hash, other_than=url)
    if duplicate_of:
        ctx.store.upsert_page(
            PageRecord(
                url,
                final_norm,
                "skipped",
                page.http_status,
                page.title,
                content_hash,
                len(markdown),
                None,
                "DUPLICATE_CONTENT",
                f"same content as {duplicate_of}",
                True,
                time.time(),
            )
        )
        ctx.warn("DUPLICATE_CONTENT", f"{url} has the same content as {duplicate_of}; skipped")
        return {"url": url, "status": "duplicate", "duplicate_of": duplicate_of}

    hits = detect_injection(markdown)
    if hits:
        ctx.warn(
            "INJECTION_SUSPECTED",
            f"{url} contains instruction-like text; treated as data only",
            patterns=hits[:3],
        )

    cache_name = url_cache_name(url)
    stored = ScrapedPage(
        url=url,
        final_url=final_norm,
        title=page.title,
        description=page.description,
        markdown=markdown,
        http_status=page.http_status,
        links=[],
    )
    atomic_write_text(ctx.pages_dir / cache_name, stored.to_json())
    ctx._page_text_cache[url] = markdown

    kept, dropped = ctx.index.index_page(url, markdown, title=page.title)
    ctx.store.upsert_page(
        PageRecord(
            url,
            final_norm,
            "fetched",
            page.http_status,
            page.title,
            content_hash,
            len(markdown),
            cache_name,
            None,
            None,
            True,
            time.time(),
        )
    )
    if truncated:
        ctx.warn(
            "PAGE_TRUNCATED", f"{url} was longer than {MAX_PAGE_CHARS} chars and was truncated"
        )
    event(
        log,
        "page_fetch",
        f"indexed {url} ({len(markdown)} chars, {kept} chunks, {dropped} repeated blocks dropped)",
        url=url,
        status="cached" if page.cached else "fetched",
        count=kept,
    )
    headings = ctx.index.page_outline(url, PAGE_HEADINGS_TO_MODEL)
    lead = " ".join(markdown[: PAGE_LEAD_CHARS * 2].split())[:PAGE_LEAD_CHARS]
    return {
        "url": url,
        "status": "cached" if page.cached else "fetched",
        "title": page.title,
        "chars": len(markdown),
        "chunks": kept,
        "headings": headings,
        "lead": lead,
    }


# --------------------------------------------------------------------------------------
# Tool factory
# --------------------------------------------------------------------------------------


def make_tools(ctx: ToolContext) -> list[Any]:
    settings = ctx.settings
    store = ctx.store

    # ---- discovery -----------------------------------------------------------------
    @tool
    def discover_pages(start_url: str) -> str:
        """Discover candidate pages of the company website (sitemap/map first, homepage links
        as fallback). Returns up to a few dozen same-site candidates with a heuristic score and
        the remaining page budget. Pick pages that explain the company: product/platform,
        features, solutions, customers/case studies, about, pricing."""
        ctx.set_stage("discover")
        try:
            url = validate_url(start_url, check_dns=ctx.check_dns)
        except URLGuardError as exc:
            return _json({"ok": False, "error": f"start URL rejected: {exc}"})
        if not same_site(url, ctx.start_url):
            return _json({"ok": False, "error": "discover_pages only works on the run's website"})
        if not ctx.robots.allowed(url):
            ctx.warn("ROBOTS_DISALLOWED", f"robots.txt disallows fetching {url}")
            return _json(
                {"ok": False, "error": "robots.txt disallows the homepage; no pages can be fetched"}
            )
        cached_home = ctx.cached_page(url)
        result = discover(
            url,
            ctx.scraper,
            timeout_ms=settings.scrape_timeout_ms,
            check_dns=ctx.check_dns,
            homepage=cached_home,
        )
        if cached_home is None and result.homepage is not None:
            store.increment_counter("scrape_attempts")
        for note in result.notes:
            ctx.warn("DISCOVERY_NOTE", note)
        if result.homepage is not None and ctx.page_budget_remaining() > 0:
            process_page(ctx, result.homepage, url)
        event(
            log,
            "pages_discovered",
            f"{len(result.candidates)} candidates from {result.source} ({result.dropped} dropped)",
            count=len(result.candidates),
            kind=result.source,
        )
        atomic_write_text(
            ctx.run_dir / DISCOVERY_FILENAME,
            _json(
                [
                    {"url": c.url, "title": c.title, "description": c.description}
                    for c in result.candidates
                ]
            ),
        )
        return _json(
            {
                "ok": True,
                "source": result.source,
                "candidates": [
                    {"url": c.url, "title": c.title, "description": c.description, "score": c.score}
                    for c in result.candidates
                ],
                "already_fetched": sorted(store.fetched_urls()),
                "page_budget_remaining": ctx.page_budget_remaining(),
                "notes": result.notes,
            }
        )

    # ---- scraping ------------------------------------------------------------------
    @tool
    def scrape_pages(urls: list[str]) -> str:
        """Fetch and index up to the remaining page budget of same-site URLs (pass several at
        once). Already-fetched or duplicate pages are skipped for free. Returns, per URL, the
        status, title, size, headings and a short lead so you can decide what to search for.
        Afterwards use `search_pages` / `read_page` instead of asking for raw text."""
        ctx.set_stage("scrape")
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        live_fetches = 0
        requested = list(urls or [])
        if len(requested) > MAX_URLS_PER_SCRAPE_CALL:
            results.append(
                {
                    "note": f"only the first {MAX_URLS_PER_SCRAPE_CALL} of {len(requested)} URLs were considered"
                }
            )
            requested = requested[:MAX_URLS_PER_SCRAPE_CALL]
        max_attempts = settings.max_pages * SCRAPE_ATTEMPTS_PER_PAGE
        for raw in requested:
            try:
                url = validate_url(raw, check_dns=ctx.check_dns)
            except URLGuardError as exc:
                ctx.warn("URL_REJECTED", f"{raw} rejected: {exc}")
                results.append({"url": raw, "status": "rejected", "error": str(exc)})
                continue
            if url in seen:
                continue
            seen.add(url)
            if not same_site(url, ctx.start_url):
                results.append(
                    {"url": url, "status": "rejected", "error": "not on the company's website"}
                )
                continue
            existing = store.get_page(url)
            if existing and existing.status == "fetched":
                results.append(
                    {
                        "url": url,
                        "status": "already_fetched",
                        "title": existing.title,
                        "chars": existing.char_count,
                        "headings": ctx.index.page_outline(url, PAGE_HEADINGS_TO_MODEL),
                    }
                )
                continue
            if existing and existing.status == "skipped":
                results.append({"url": url, "status": "skipped", "error": existing.error})
                continue
            if (
                existing
                and existing.status == "failed"
                and not is_transient_code(existing.error_code)
            ):
                results.append(
                    {"url": url, "status": "failed", "error": f"already failed: {existing.error}"}
                )
                continue
            if store.get_counter("scrape_attempts") >= max_attempts:
                if not store.has_warning("LIMIT_SCRAPE_ATTEMPTS_REACHED"):
                    ctx.warn(
                        "LIMIT_SCRAPE_ATTEMPTS_REACHED",
                        f"fetch attempt limit of {max_attempts} reached",
                    )
                results.append(
                    {"url": url, "status": "not_fetched", "error": "fetch attempt budget exhausted"}
                )
                continue
            if ctx.page_budget_remaining() <= 0:
                if not store.has_warning("LIMIT_PAGES_REACHED"):
                    ctx.warn(
                        "LIMIT_PAGES_REACHED",
                        f"page limit of {settings.max_pages} reached; further pages were not fetched",
                    )
                    event(
                        log,
                        "limit_reached",
                        "page limit reached",
                        kind="pages",
                        count=settings.max_pages,
                    )
                results.append(
                    {"url": url, "status": "not_fetched", "error": "page budget exhausted"}
                )
                continue
            if not ctx.robots.allowed(url):
                store.upsert_page(
                    PageRecord(
                        url,
                        None,
                        "skipped",
                        None,
                        None,
                        None,
                        0,
                        None,
                        "ROBOTS_DISALLOWED",
                        "robots.txt disallows",
                        True,
                        time.time(),
                    )
                )
                ctx.warn("ROBOTS_DISALLOWED", f"robots.txt disallows fetching {url}; skipped")
                results.append({"url": url, "status": "skipped", "error": "robots.txt disallows"})
                continue
            page = ctx.cached_page(url)
            if page is None:
                if live_fetches and SCRAPE_INTER_REQUEST_DELAY_S > 0:
                    time.sleep(SCRAPE_INTER_REQUEST_DELAY_S)
                live_fetches += 1
                store.increment_counter("scrape_attempts")
                try:
                    page = ctx.scraper.scrape(url, timeout_ms=settings.scrape_timeout_ms)
                except TransientScrapeError as exc:
                    # Record the attempt, then let the retry middleware retry the whole call.
                    store.upsert_page(
                        PageRecord(
                            url,
                            None,
                            "failed",
                            None,
                            None,
                            None,
                            0,
                            None,
                            exc.code,
                            redact_text(str(exc)),
                            True,
                            time.time(),
                        )
                    )
                    event(
                        log,
                        "page_fetch",
                        f"{url} transient failure: {exc}",
                        level=logging.WARNING,
                        url=url,
                        status="transient",
                        code=exc.code,
                    )
                    raise
                except PermanentScrapeError as exc:
                    err = redact_text(str(exc))
                    store.upsert_page(
                        PageRecord(
                            url,
                            None,
                            "failed",
                            exc.http_status,
                            None,
                            None,
                            0,
                            None,
                            exc.code,
                            err,
                            True,
                            time.time(),
                        )
                    )
                    ctx.warn(f"PAGE_SKIPPED_{exc.code}", f"{url} skipped: {err}")
                    results.append({"url": url, "status": "failed", "error": err})
                    continue
            results.append(process_page(ctx, page, url))
        return _json(
            {
                "ok": True,
                "results": results,
                "pages_fetched": len(store.fetched_urls()),
                "page_budget_remaining": ctx.page_budget_remaining(),
            }
        )

    # ---- retrieval -----------------------------------------------------------------
    @tool
    def search_pages(query: str, url: str | None = None) -> str:
        """Search the indexed pages for a topic (hybrid keyword + semantic). Returns the best
        matching passages with their source URL and heading. Excerpts are verbatim page text
        you can cite as evidence. Optionally restrict to one URL."""
        ctx.set_stage("research")
        target = None
        if url:
            try:
                target = validate_url(url, check_dns=False)
            except URLGuardError as exc:
                return _json({"ok": False, "error": str(exc)})
        hits = ctx.index.search(query, url=target)
        if not hits:
            return _json(
                {
                    "ok": True,
                    "query": query,
                    "results": [],
                    "note": "no indexed content matched; scrape pages first or broaden the query",
                }
            )
        return _json(
            {
                "ok": True,
                "query": query,
                "results": [
                    {"url": h.url, "heading": h.heading, "excerpt": h.excerpt} for h in hits
                ],
            }
        )

    @tool
    def read_page(url: str, section: str | None = None, offset: int = 0) -> str:
        """Read a window of an already-fetched page (optionally one heading section, or
        continue from `offset`). Prefer `search_pages`; use this when you need surrounding
        context for a specific page."""
        ctx.set_stage("research")
        try:
            target = validate_url(url, check_dns=False)
        except URLGuardError as exc:
            return _json({"ok": False, "error": str(exc)})
        if target not in store.fetched_urls():
            return _json(
                {
                    "ok": False,
                    "error": "page not fetched; call scrape_pages first",
                    "fetched": sorted(store.fetched_urls()),
                }
            )
        text, more, next_offset = ctx.index.read(
            target, section=section, offset=max(0, int(offset or 0))
        )
        return _json(
            {
                "ok": True,
                "url": target,
                "section": section,
                "text": text,
                "more_available": more,
                "next_offset": next_offset,
                "outline": ctx.index.page_outline(target, PAGE_HEADINGS_TO_MODEL),
            }
        )

    # ---- interview -----------------------------------------------------------------
    @tool
    def ask_user(question: str, why_unclear: str, field_paths: list[str], kind: str = "gap") -> str:
        """Ask the user ONE focused question in the terminal and wait for the answer. Explain in
        `why_unclear` what the website left ambiguous. `field_paths` lists the contract fields the
        answer resolves (e.g. ["customer.target_customer"]). `kind` is one of product_selection,
        gap, conflict, brand. Returns the answer, or SKIPPED / UNKNOWN when the user cannot
        answer. Must be the only tool call in the step."""
        kind = kind if kind in QUESTION_KINDS else "gap"
        try:
            paths = [str(p).strip() for p in (field_paths or []) if str(p).strip()]
            for p in paths:
                parse_field_path(p)
        except FieldPathError as exc:
            return _json({"ok": False, "error": f"invalid field path: {exc}"})
        if not paths and kind != "product_selection":
            return _json(
                {"ok": False, "error": "field_paths must name at least one contract field"}
            )
        if is_multi_question(question):
            return _json(
                {
                    "ok": False,
                    "error": (
                        "ask ONE focused question per call (this text bundles several numbered or "
                        "question-marked items). Call ask_user again with the single most important "
                        "question; follow up separately if still useful."
                    ),
                }
            )
        normalized_q = " ".join((question or "").split()).lower()
        qid = hashlib.sha1(
            f"{ctx.run_id}|{kind}|{','.join(sorted(paths))}|{normalized_q}".encode(),
            usedforsecurity=False,
        ).hexdigest()[:12]

        existing = store.get_question(qid)
        if existing and existing["status"] != "pending":
            # Replay after a crash: the answer is already committed; never re-ask.
            return _json(
                {
                    "ok": True,
                    "qid": qid,
                    "status": existing["status"],
                    "answer": existing["answer"],
                    "questions_remaining": max(0, settings.max_questions - store.questions_asked()),
                    "note": "answer already recorded",
                }
            )
        if store.questions_asked() >= settings.max_questions:
            if not store.has_warning("LIMIT_QUESTIONS_REACHED"):
                ctx.warn(
                    "LIMIT_QUESTIONS_REACHED", f"question limit of {settings.max_questions} reached"
                )
                event(
                    log,
                    "limit_reached",
                    "question limit reached",
                    kind="questions",
                    count=settings.max_questions,
                )
            return _json(
                {
                    "ok": False,
                    "error": "question limit reached; finalize the profile with the evidence you have",
                }
            )

        if kind == "product_selection" and ctx.product_focus:
            store.upsert_question(qid, kind, question, why_unclear, paths)
            store.answer_question(qid, "answered", ctx.product_focus)
            return _json(
                {
                    "ok": True,
                    "qid": qid,
                    "status": "answered",
                    "answer": ctx.product_focus,
                    "note": "product focus was preset on the command line",
                }
            )

        record = store.upsert_question(qid, kind, question, why_unclear, paths)
        ctx.set_stage("interview")
        event(
            log,
            "interrupt_raised",
            f"asking user (q{record['ordinal']}): {question[:120]}",
            qid=qid,
            count=record["ordinal"],
        )
        # --- interrupt: everything above is idempotent and re-runs safely on resume ---
        raw = interrupt(
            {
                "qid": qid,
                "question": question,
                "why_unclear": why_unclear,
                "field_paths": paths,
                "kind": kind,
                "number": record["ordinal"],
                "max": settings.max_questions,
            }
        )
        answer = sanitize_answer(str(raw if raw is not None else ""))
        lowered = answer.lower().strip(" .!")
        if not answer or lowered in SKIP_WORDS:
            status, stored_answer = "skipped", None
        elif lowered in UNKNOWN_WORDS:
            status, stored_answer = "unknown", None
        else:
            status, stored_answer = "answered", answer
        store.answer_question(qid, status, stored_answer)
        event(log, "answer_committed", f"q{record['ordinal']} {status}", qid=qid, status=status)
        if kind == "product_selection" and status == "answered":
            ctx.product_focus = stored_answer
            store.set_product_focus(stored_answer or "")
        remaining = max(0, settings.max_questions - store.questions_asked())
        result_answer = stored_answer if status == "answered" else status.upper()
        return _json(
            {
                "ok": True,
                "qid": qid,
                "status": status,
                "answer": result_answer,
                "questions_remaining": remaining,
            }
        )

    # ---- drafting ------------------------------------------------------------------
    def _stems(text: str) -> set[str]:
        """Content words reduced to a 5-char stem so inflections (key/keys, release/releases,
        protect/protection) still count as overlap."""
        return {t[:STEM_CHARS] for t in tokenize(text) if len(t) >= MIN_CONTENT_TOKEN_CHARS}

    def _value_text(value: Any) -> str:
        if isinstance(value, dict):
            return " ".join(str(v) for v in value.values())
        if isinstance(value, list):
            return " ".join(_value_text(v) for v in value)
        return str(value)

    def _supports(field_path: str, value: Any, text: str) -> bool:
        """An excerpt/answer supports a value if they share a content-word stem. Observed brand
        patterns (tone, writing style) are exempt: their evidence is an illustrative passage."""
        if field_path.split("[")[0] in OBSERVATION_PATHS:
            return True
        vt = _stems(_value_text(value))
        return not vt or bool(vt & _stems(text))

    def _verify_website_evidence(
        profile: dict[str, Any], items: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        fetched = store.fetched_urls()
        for item in items or []:
            path = str(item.get("field_path", "")).strip()
            src = str(item.get("source_url", "")).strip()
            excerpt = str(item.get("excerpt", "")).strip()
            try:
                parse_field_path(path)
            except FieldPathError as exc:
                rejected.append({"field_path": path, "reason": str(exc)})
                continue
            value = get_by_path(profile, path)
            if value in ("", [], None):
                rejected.append({"field_path": path, "reason": "field is empty in the profile"})
                continue
            try:
                src_norm = normalize_url(src) if src else ""
            except ValueError:
                src_norm = ""
            if src_norm not in fetched:
                rejected.append(
                    {"field_path": path, "reason": "source_url is not a page fetched in this run"}
                )
                continue
            if len(excerpt) < MIN_EXCERPT_CHARS:
                rejected.append(
                    {
                        "field_path": path,
                        "reason": f"excerpt too short to ground a claim (min {MIN_EXCERPT_CHARS} chars)",
                    }
                )
                continue
            if len(excerpt) > MAX_EVIDENCE_EXCERPT_CHARS:
                rejected.append(
                    {
                        "field_path": path,
                        "reason": f"excerpt too long (max {MAX_EVIDENCE_EXCERPT_CHARS} chars); quote only the relevant passage",
                    }
                )
                continue
            page_text = ctx.page_text(src_norm) or ""
            if not excerpt_in_page(excerpt, page_text):
                rejected.append(
                    {"field_path": path, "reason": "excerpt is not a verbatim quote from that page"}
                )
                continue
            if not _supports(path, value, excerpt):
                rejected.append(
                    {
                        "field_path": path,
                        "reason": "excerpt does not mention the field value; quote the passage the value comes from",
                    }
                )
                continue
            accepted.append({"field_path": path, "source_url": src_norm, "excerpt": excerpt})
        return accepted, rejected

    def _repair_or_fail(errors: list[str], what: str) -> str:
        attempts = store.increment_counter("repair_attempts")
        event(
            log,
            "repair_attempt",
            f"{what} invalid (attempt {attempts}): " + " ; ".join(errors[:4])[:600],
            level=logging.WARNING,
            attempt=attempts,
            count=len(errors),
        )
        if attempts > MAX_REPAIR_ATTEMPTS:
            ctx.warn(
                "INVALID_MODEL_OUTPUT",
                f"{what} was still invalid after {MAX_REPAIR_ATTEMPTS} repair attempt(s)",
            )
            raise FatalProfileError(f"{what} failed validation twice: {'; '.join(errors[:5])}")
        return _json(
            {
                "ok": False,
                "errors": errors,
                "repair_attempts_remaining": MAX_REPAIR_ATTEMPTS - attempts + 1,
                "hint": "fix the listed fields and call again with the full corrected input",
            }
        )

    def _thin_draft_advice(profile: dict[str, Any], *, strict: bool = False) -> str | None:
        """Push back when the draft leaves sections empty although pages are indexed: the
        interview must not be used for facts the website already establishes."""
        from profile_builder.workflow.gaps import section_coverage

        if not store.fetched_urls():
            return None
        threshold = 0 if strict else THIN_DRAFT_SECTION_FIELDS
        sections = FINALIZE_GUARD_SECTIONS if strict else None
        thin = [
            sec
            for sec, (filled, _total) in section_coverage(profile).items()
            if filled <= threshold and sec != "company" and (sections is None or sec in sections)
        ]
        if not thin:
            return None
        hints = {
            "product": '"what the platform does", "key features and capabilities", "how it works", "why choose / differentiators"',
            "customer": '"who it is for", "industries and use cases", "challenges / problems solved", "outcomes and benefits", "compared to / alternatives"',
            "content_evidence": '"case study results", "customers like", "awards, research, certifications", "demo, benchmarks, proof"',
            "brand": '"tone of voice and recurring phrases" (read 2-3 page leads with read_page)',
        }
        lines = [
            f"- {sec}: search_pages for {hints.get(sec, 'the relevant topics')}" for sec in thin
        ]
        return (
            f"The draft leaves {len(thin)} section(s) nearly empty although {len(store.fetched_urls())} pages are indexed. "
            "Do NOT ask the user about facts the website can answer. First gather evidence, then fill these via apply_profile_updates:\n"
            + "\n".join(lines)
        )

    def _gap_payload(profile: dict[str, Any]) -> dict[str, Any]:
        asked = {p for q in store.list_questions() for p in q["field_paths"]}
        grounding = grounding_report(profile, evidence_paths(store))
        gaps = prioritize_for_interview(
            profile,
            open_conflicts=store.list_conflicts("open"),
            asked_paths=asked,
            ungrounded=grounding["ungrounded"],
        )
        advice = _thin_draft_advice(profile)
        return {
            **({"advice": advice} if advice else {}),
            "grounding": {
                "grounded": grounding["grounded"],
                "populated": grounding["populated"],
                "ungrounded": grounding["ungrounded"][:10],
            },
            "gaps": gaps,
            "questions_remaining": max(0, settings.max_questions - store.questions_asked()),
        }

    @tool
    def save_profile_draft(profile: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
        """Save the full draft profile (exact company_brain shape) plus evidence. Each evidence
        item is {"field_path": "...", "source_url": "...", "excerpt": "<verbatim text from that
        page>"}; a field path may point at a whole list ("customer.buyers") or an item
        ("customer.buyers[0]"). Unverifiable evidence is rejected and listed back. Returns the
        validation result, grounding stats and the prioritized gaps to interview about."""
        ctx.set_stage("draft")
        data = copy.deepcopy(profile) if isinstance(profile, dict) else {}
        data, unknown = strip_unknown_keys(data)
        if unknown:
            ctx.warn(
                "UNKNOWN_KEYS_IGNORED",
                f"draft contained keys outside the contract; ignored: {', '.join(unknown[:8])}",
            )
        try:
            if isinstance(data.get("company"), dict) and not data["company"].get("website_url"):
                data["company"]["website_url"] = ctx.start_url
            brain = CompanyBrain.model_validate(data)
        except ValidationError as exc:
            return _repair_or_fail(_format_validation_error(exc), "profile draft")
        clean = brain.model_dump(mode="json")
        # Merge with the previous draft: a later draft may never erase a field that an earlier
        # one filled (the model sometimes re-sends a thinner profile after the interview).
        kept_from_previous: list[str] = []
        protected_by_user: list[str] = []
        previous = store.latest_draft()
        if previous is not None:
            _, prev = previous
            user_bases = store.interview_bases()
            for base in (*STRING_PATHS, *STRING_LIST_PATHS, FEATURE_LIST_PATH):
                new_val, old_val = get_by_path(clean, base), get_by_path(prev, base)
                if base in user_bases and new_val != old_val:
                    # An explicit user answer always wins over a later website-based draft.
                    set_by_path(clean, base, copy.deepcopy(old_val))
                    protected_by_user.append(base)
                elif new_val in ("", [], None) and old_val not in ("", [], None):
                    set_by_path(clean, base, copy.deepcopy(old_val))
                    kept_from_previous.append(base)
            clean = CompanyBrain.model_validate(clean).model_dump(mode="json")
        restated = {
            base
            for base in (*STRING_PATHS, *STRING_LIST_PATHS, FEATURE_LIST_PATH)
            if base not in kept_from_previous and base not in protected_by_user
        }
        accepted, rejected = _verify_website_evidence(clean, evidence)
        store.replace_website_evidence(accepted, only_fields=restated)
        for r in rejected:
            ctx.warn("EVIDENCE_REJECTED", f"{r['field_path']}: {r['reason']}")
        store.set_counter("repair_attempts", 0)
        version = store.save_draft(clean, "draft")
        payload = _gap_payload(clean)
        event(
            log,
            "draft_saved",
            f"draft v{version} saved ({len(accepted)} evidence rows, {len(rejected)} rejected, {len(payload['gaps'])} gaps)",
            version=version,
            count=len(payload["gaps"]),
        )
        return _json(
            {
                "ok": True,
                "version": version,
                "evidence_accepted": len(accepted),
                "evidence_rejected": rejected,
                "kept_from_previous_draft": kept_from_previous,
                "protected_by_user_answers": protected_by_user,
                "ignored_unknown_keys": unknown,
                **payload,
                "next": "ask focused questions for the top gaps/conflicts, or finalize_profile if none are worth asking",
            }
        )

    @tool
    def apply_profile_updates(updates: list[dict[str, Any]]) -> str:
        """Apply targeted changes to the saved draft. Each update is {"field_path": "...",
        "value": <string, list of strings, or feature object>, "evidence": {"kind": "interview",
        "question_id": "<qid>"} or {"kind": "website", "source_url": "...", "excerpt": "<verbatim>"}}.
        Use list index "[n]" to replace an item or "[len]" to append. A user answer that
        contradicts the website supersedes it (the original evidence is kept and a warning is
        recorded). Updates without valid evidence are rejected."""
        ctx.set_stage("draft")
        latest = store.latest_draft()
        if latest is None:
            return _json({"ok": False, "error": "no draft yet; call save_profile_draft first"})
        _, profile = latest
        data = copy.deepcopy(profile)
        applied: list[str] = []
        rejected: list[dict[str, Any]] = []
        pending_evidence: list[dict[str, Any]] = []
        for upd in updates or []:
            path = str(upd.get("field_path", "")).strip()
            ev = upd.get("evidence") or {}
            try:
                parse_field_path(path)
            except FieldPathError as exc:
                rejected.append({"field_path": path, "reason": str(exc)})
                continue
            kind = str(ev.get("kind", "")).lower()
            if kind == "interview":
                q = store.get_question(str(ev.get("question_id", "")))
                if not q or q["status"] != "answered":
                    rejected.append(
                        {
                            "field_path": path,
                            "reason": "evidence must reference an answered question id",
                        }
                    )
                    continue
                covered = {fp.split("[")[0] for fp in q["field_paths"]}
                if path.split("[")[0] not in covered and q["kind"] != "product_selection":
                    rejected.append(
                        {
                            "field_path": path,
                            "reason": f"question {q['qid']} did not cover this field (it covered {sorted(covered)})",
                        }
                    )
                    continue
                if not _supports(path, upd.get("value"), q["answer"] or ""):
                    rejected.append(
                        {"field_path": path, "reason": "value does not reflect the user's answer"}
                    )
                    continue
            elif kind == "website":
                if path.split("[")[0] in store.interview_bases():
                    rejected.append(
                        {
                            "field_path": path,
                            "reason": "this field was set by a user answer; website evidence "
                            "cannot override it (ask the user instead)",
                        }
                    )
                    continue
            else:
                rejected.append(
                    {"field_path": path, "reason": "evidence.kind must be 'interview' or 'website'"}
                )
                continue
            old = get_by_path(data, path)
            try:
                set_by_path(data, path, upd.get("value"))
            except FieldPathError as exc:
                rejected.append({"field_path": path, "reason": str(exc)})
                continue
            if kind == "website":
                acc, rej = _verify_website_evidence(
                    data,
                    [
                        {
                            "field_path": path,
                            "source_url": ev.get("source_url"),
                            "excerpt": ev.get("excerpt"),
                        }
                    ],
                )
                if rej:
                    set_by_path(data, path, old)
                    rejected.append(rej[0])
                    continue
                pending_evidence.append(
                    {"kind": "website", "old": old, "value": upd.get("value"), **acc[0]}
                )
            else:
                pending_evidence.append(
                    {
                        "kind": "interview",
                        "field_path": path,
                        "question_id": q["qid"],
                        "answer": q["answer"],
                        "old": old,
                        "value": upd.get("value"),
                    }
                )
            applied.append(path)
        if not applied:
            return _json({"ok": False, "applied": [], "rejected": rejected, **_gap_payload(data)})
        try:
            brain = CompanyBrain.model_validate(data)
        except ValidationError as exc:
            return _repair_or_fail(_format_validation_error(exc), "profile update")
        clean = brain.model_dump(mode="json")
        for ev in pending_evidence:
            old, value = ev.get("old"), ev.get("value")
            changed = old not in ("", [], None) and old != value
            stale = store.supersede_evidence_tree(ev["field_path"]) if changed else {}
            if ev["kind"] == "website":
                store.add_evidence(
                    ev["field_path"], "website", source_url=ev["source_url"], excerpt=ev["excerpt"]
                )
            else:
                if stale.get("website"):
                    ctx.warn(
                        "USER_CORRECTION_SUPERSEDES_SITE",
                        f"{ev['field_path']}: user answer replaced the website claim (original evidence preserved)",
                    )
                store.add_evidence(
                    ev["field_path"],
                    "interview",
                    question_id=ev["question_id"],
                    answer=ev["answer"],
                )
                for c in store.list_conflicts("open"):
                    if c["field_path"] == ev["field_path"]:
                        store.resolve_conflict(
                            c["field_path"], f"user answer (q {ev['question_id']}): {ev['answer']}"
                        )
        store.set_counter("repair_attempts", 0)
        version = store.save_draft(clean, "update")
        payload = _gap_payload(clean)
        event(
            log,
            "draft_saved",
            f"draft v{version} updated ({len(applied)} fields)",
            version=version,
            count=len(applied),
        )
        return _json(
            {"ok": True, "version": version, "applied": applied, "rejected": rejected, **payload}
        )

    @tool
    def note_conflict(field_path: str, claims: list[dict[str, Any]], summary: str) -> str:
        """Record that two sources disagree about a field, e.g. the homepage targets startups
        while the customer page shows only banks. `claims` is a list of {"source_url", "excerpt"
        (verbatim), "claim"}. Then ask the user a targeted question if it matters; unresolved
        conflicts are omitted from the final profile and reported as warnings."""
        try:
            parse_field_path(field_path)
        except FieldPathError as exc:
            return _json({"ok": False, "error": str(exc)})
        fetched = store.fetched_urls()
        verified: list[dict[str, Any]] = []
        for c in claims or []:
            try:
                src = normalize_url(str(c.get("source_url", "")))
            except ValueError:
                src = ""
            text = ctx.page_text(src) if src in fetched else None
            if text is None or not excerpt_in_page(str(c.get("excerpt", "")), text):
                continue
            verified.append(
                {
                    "source_url": src,
                    "excerpt": str(c.get("excerpt", "")),
                    "claim": str(c.get("claim", "")),
                }
            )
        if len(verified) < 2:
            return _json(
                {
                    "ok": False,
                    "error": "a conflict needs at least two verifiable claims from fetched pages",
                }
            )
        store.upsert_conflict(field_path.strip(), verified, summary)
        ctx.warn("CONFLICT_RECORDED", f"{field_path}: {summary}")
        event(log, "conflict_noted", f"conflict on {field_path}", code=field_path)
        return _json(
            {
                "ok": True,
                "field_path": field_path,
                "claims": len(verified),
                "next": "ask the user a targeted question with kind='conflict' if this affects the profile",
            }
        )

    # ---- finalize ------------------------------------------------------------------
    @tool
    def finalize_profile() -> str:
        """Validate the latest draft, omit disputed unresolved claims, write company_brain.json
        plus evidence.json and report.md, and return the run summary. Call once at the end."""
        latest = store.latest_draft()
        if latest is not None and store.get_counter("finalize_refusals") < MAX_FINALIZE_REFUSALS:
            advice = _thin_draft_advice(latest[1], strict=True)
            ungrounded = ungrounded_paths(store, latest[1])
            if advice or ungrounded:
                store.increment_counter("finalize_refusals")
                return _json(
                    {
                        "ok": False,
                        "error": "profile not ready to export",
                        **({"advice": advice} if advice else {}),
                        **(
                            {
                                "ungrounded": ungrounded[:20],
                                "ungrounded_hint": (
                                    "these populated fields have no accepted evidence and will be "
                                    "OMITTED from the export unless you cite a verbatim page excerpt "
                                    "(apply_profile_updates with kind=website) or an answered question. "
                                    "Call finalize_profile again when done."
                                ),
                            }
                            if ungrounded
                            else {}
                        ),
                    }
                )
        ctx.set_stage("finalize")
        result = finalize(ctx)
        ctx.set_stage("done")
        return _json(result)

    return [
        discover_pages,
        scrape_pages,
        search_pages,
        read_page,
        ask_user,
        save_profile_draft,
        apply_profile_updates,
        note_conflict,
        finalize_profile,
    ]


# --------------------------------------------------------------------------------------
# Finalization (also used by the runner for best-effort partial exports)
# --------------------------------------------------------------------------------------

# Reaching the page/question caps is normal bounded behavior; only being cut off before the
# agent could finish (model-call cap, budget, forced stop) makes a profile "partial". The
# runner passes forced_partial for the model-call cap; the budget is re-checked live so a
# resumed run with a raised budget can complete.


def ungrounded_paths(store: RunStore, profile: dict[str, Any]) -> list[str]:
    return list(grounding_report(profile, evidence_paths(store))["ungrounded"])


def _omit_path(store: RunStore, data: dict[str, Any], path: str) -> None:
    """Remove the value at `path` from the profile dict and keep evidence/conflict paths
    consistent (list items are deleted and later indexes renumbered)."""
    base, index, sub = parse_field_path(path)
    current = get_by_path(data, path)
    if current in ("", [], None):
        return
    if index is not None and (sub is None or sub == "name"):
        lst = get_by_path(data, base)
        if isinstance(lst, list) and index < len(lst):
            del lst[index]
            store.shift_list_paths(base, index)
    elif sub is not None:
        set_by_path(data, path, "")
    elif isinstance(current, list):
        set_by_path(data, path, [])
    else:
        set_by_path(data, path, "")


def _omit_many(store: RunStore, data: dict[str, Any], paths: list[str]) -> None:
    # Delete list items from the highest index down so earlier indexes stay valid.
    def sort_key(p: str) -> tuple[str, int]:
        base, index, _sub = parse_field_path(p)
        return (base, -(index if index is not None else -1))

    for path in sorted(set(paths), key=sort_key):
        _omit_path(store, data, path)


def finalize(ctx: ToolContext, *, forced_partial: bool = False) -> dict[str, Any]:
    store = ctx.store
    latest = store.latest_draft()
    if latest is None:
        raise FatalProfileError("no valid draft exists; nothing to export")
    _, profile = latest
    data = copy.deepcopy(profile)

    # 1. Unresolved conflicts: omit the disputed value once and mark the conflict so a later
    #    re-export does not delete a different item that moved into the same index.
    disputed = [c["field_path"] for c in store.list_conflicts("open")]
    for path in disputed:
        if get_by_path(data, path) in ("", [], None):
            store.set_conflict_status(path, "omitted", "value was already empty at export")
            continue
        ctx.warn(
            "CONFLICT_UNRESOLVED_OMITTED",
            f"{path}: conflicting evidence was not resolved; value omitted",
        )
        store.set_conflict_status(path, "omitted", "omitted from the export (unresolved)")
    _omit_many(store, data, disputed)

    # 2. Grounding: a populated field without accepted evidence is not exported.
    ungrounded = ungrounded_paths(store, data)
    if ungrounded:
        ctx.warn(
            "UNGROUNDED_OMITTED",
            f"{len(ungrounded)} populated field(s) had no accepted evidence and were omitted: "
            + ", ".join(ungrounded[:12]),
        )
        _omit_many(store, data, ungrounded)

    feats = data.get("product", {}).get("features_and_capabilities", [])
    data["product"]["features_and_capabilities"] = [
        f
        for f in feats
        if any(f.get(k) for k in ("name", "description", "how_it_works", "customer_benefit"))
    ]
    try:
        brain = CompanyBrain.model_validate(data)
    except ValidationError as exc:
        raise FatalProfileError(
            "final profile failed validation: " + "; ".join(_format_validation_error(exc))
        ) from exc
    clean = brain.model_dump(mode="json")
    budget = ctx.settings.budget_usd
    over_budget = budget is not None and store.total_cost() >= budget
    status = (
        "partial" if (forced_partial or over_budget or not store.fetched_urls()) else "complete"
    )
    output_path = write_outputs(ctx.run_dir, store, clean, status)
    store.save_draft(clean, "finalize")
    store.set_status(status, str(output_path))
    grounding = grounding_report(clean, evidence_paths(store))
    pending = store.pending_question()
    summary = {
        "ok": True,
        "status": status,
        "output_path": str(output_path),
        "grounded_fields": f"{grounding['grounded']}/{grounding['populated']}",
        "ungrounded": grounding["ungrounded"][:10],
        "gaps_remaining": [
            g["field_path"]
            for g in prioritize_for_interview(
                clean, open_conflicts=[], asked_paths=set(), ungrounded=[]
            )
        ][:10],
        "warnings": len(store.list_warnings()),
        "questions_asked": store.questions_asked(),
        "pending_question": pending["question"] if pending else None,
    }
    event(
        log,
        "run_finished",
        f"profile exported ({status}) → {output_path}",
        status=status,
        cost_usd=round(store.total_cost(), 6),
    )
    return summary
