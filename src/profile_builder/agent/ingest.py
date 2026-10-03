"""Page ingestion shared by `discover_pages` and `scrape_pages`: validate a fetched page
(redirect guard, size, duplicates, injection), cache it, index it and record the outcome;
plus the scrape-loop helpers that decide whether a URL needs a live fetch and perform it.

Every write is an idempotent upsert so a replayed tool call after a crash is harmless.
Transient scrape failures are recorded and surfaced to the caller (the retry middleware
retries the whole call); permanent ones are reported once and never retried.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from profile_builder.agent.context import ToolContext
from profile_builder.config import (
    MAX_PAGE_CHARS,
    MIN_PAGE_CHARS,
    PAGE_HEADINGS_TO_MODEL,
    PAGE_LEAD_CHARS,
    PAGE_LEAD_SLICE_FACTOR,
    SCRAPE_INTER_REQUEST_DELAY_S,
)
from profile_builder.logging_setup import event, get_logger, redact_text
from profile_builder.retrieval.chunking import text_hash
from profile_builder.security import atomic_write_text, detect_injection, url_cache_name
from profile_builder.state.run_store import PageRecord
from profile_builder.web.scraper import (
    PermanentScrapeError,
    ScrapedPage,
    TransientScrapeError,
    is_transient_code,
)
from profile_builder.web.url_guard import URLGuardError, normalize_url, same_site, validate_url

log = get_logger("tools")


def record_unusable_page(
    ctx: ToolContext,
    url: str,
    *,
    status: str,
    error_code: str,
    error: str,
    final_url: str | None = None,
    http_status: int | None = None,
    title: str | None = None,
    content_hash: str | None = None,
    char_count: int = 0,
) -> None:
    """Persist a page that yielded no indexed text (skipped, rejected or failed)."""
    ctx.store.upsert_page(
        PageRecord(
            url=url,
            final_url=final_url,
            status=status,
            http_status=http_status,
            title=title,
            content_hash=content_hash,
            char_count=char_count,
            cache_file=None,
            error_code=error_code,
            error=error,
        )
    )


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
        record_unusable_page(
            ctx,
            url,
            status="failed",
            error_code="REDIRECT_REJECTED",
            error=str(exc),
            final_url=final_url,
            http_status=page.http_status,
            title=page.title,
        )
        ctx.warn("REDIRECT_REJECTED", f"{url} redirected to a disallowed destination: {exc}")
        return {"url": url, "status": "rejected", "error": f"redirect target rejected: {exc}"}
    if not same_site(final_norm, ctx.start_url):
        record_unusable_page(
            ctx,
            url,
            status="failed",
            error_code="REDIRECT_OFFSITE",
            error="redirected off-site",
            final_url=final_norm,
            http_status=page.http_status,
            title=page.title,
        )
        ctx.warn("REDIRECT_OFFSITE", f"{url} redirected off-site to {final_norm}; skipped")
        return {"url": url, "status": "rejected", "error": "redirected off-site"}

    markdown = (page.markdown or "").strip()
    truncated = False
    if len(markdown) > MAX_PAGE_CHARS:
        markdown = markdown[:MAX_PAGE_CHARS]
        truncated = True
    if len(markdown) < MIN_PAGE_CHARS:
        record_unusable_page(
            ctx,
            url,
            status="skipped",
            error_code="EMPTY_CONTENT",
            error="page had no usable text",
            final_url=final_norm,
            http_status=page.http_status,
            title=page.title,
            char_count=len(markdown),
        )
        ctx.warn("EMPTY_CONTENT", f"{url} returned no usable text; skipped")
        return {"url": url, "status": "skipped", "error": "empty content"}

    content_hash = text_hash(markdown)
    duplicate_of = ctx.store.content_hash_exists(content_hash, other_than=url)
    if duplicate_of:
        record_unusable_page(
            ctx,
            url,
            status="skipped",
            error_code="DUPLICATE_CONTENT",
            error=f"same content as {duplicate_of}",
            final_url=final_norm,
            http_status=page.http_status,
            title=page.title,
            content_hash=content_hash,
            char_count=len(markdown),
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
    # Links are kept so a replayed/second discovery can reuse the cached homepage's links.
    stored = ScrapedPage(
        url=url,
        final_url=final_norm,
        title=page.title,
        description=page.description,
        markdown=markdown,
        http_status=page.http_status,
        links=[str(link) for link in (page.links or [])],
    )
    # Order matters: cache file, then index rows, then the "fetched" page row. A crash in
    # between leaves a cached-but-unrecorded page that discover/scrape replay re-processes.
    atomic_write_text(ctx.pages_dir / cache_name, stored.to_json())
    ctx._page_text_cache[url] = markdown

    kept, dropped = ctx.index.index_page(url, markdown, title=page.title)
    if ctx.index.last_embedding_error:
        ctx.warn(
            "EMBEDDINGS_UNAVAILABLE",
            f"embedding provider failed for {url}; indexed with keyword search only: "
            f"{ctx.index.last_embedding_error}",
        )
    ctx.store.upsert_page(
        PageRecord(
            url=url,
            final_url=final_norm,
            status="fetched",
            http_status=page.http_status,
            title=page.title,
            content_hash=content_hash,
            char_count=len(markdown),
            cache_file=cache_name,
            error_code=None,
            error=None,
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
    lead = " ".join(markdown[: PAGE_LEAD_CHARS * PAGE_LEAD_SLICE_FACTOR].split())[:PAGE_LEAD_CHARS]
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
# Scrape-loop helpers
# --------------------------------------------------------------------------------------


def classify_existing_page(
    ctx: ToolContext, url: str, existing: PageRecord | None
) -> dict[str, Any] | None:
    """The result row for a URL whose earlier outcome makes a new fetch pointless (already
    fetched, skipped, or permanently failed); None when it should be (re)fetched."""
    if existing is None:
        return None
    if existing.status == "fetched":
        return {
            "url": url,
            "status": "already_fetched",
            "title": existing.title,
            "chars": existing.char_count,
            "headings": ctx.index.page_outline(url, PAGE_HEADINGS_TO_MODEL),
        }
    if existing.status == "skipped":
        return {"url": url, "status": "skipped", "error": existing.error}
    if existing.status == "failed" and not is_transient_code(existing.error_code):
        return {
            "url": url,
            "status": "failed",
            "error": f"already failed: {existing.error}",
            "error_code": existing.error_code,
        }
    return None


@dataclass
class FetchOutcome:
    page: ScrapedPage | None = None
    result: dict[str, Any] | None = None  # the row to report when the fetch failed
    transient: TransientScrapeError | None = None
    permanent: PermanentScrapeError | None = None


def fetch_live(ctx: ToolContext, url: str, *, pace: bool, with_links: bool = False) -> FetchOutcome:
    """One counted live fetch. Failures are recorded on the page row and returned as a
    result row; a transient failure is also handed back so the caller can re-raise it once
    the rest of the batch is done (the retry middleware then retries the call). Discovery
    asks for the page's links too (`with_links`) so they can seed the candidate list."""
    if pace and SCRAPE_INTER_REQUEST_DELAY_S > 0:
        time.sleep(SCRAPE_INTER_REQUEST_DELAY_S)
    # Counted before the call: the attempt cap bounds spend, so a crash mid-fetch still costs
    # one attempt (unlike the page budget, which only counts pages that returned something).
    ctx.store.increment_counter("scrape_attempts")
    try:
        page = ctx.scraper.scrape(
            url, timeout_ms=ctx.settings.scrape_timeout_ms, with_links=with_links
        )
    except TransientScrapeError as exc:
        err = redact_text(str(exc))
        record_unusable_page(ctx, url, status="failed", error_code=exc.code, error=err)
        event(
            log,
            "page_fetch",
            f"{url} transient failure: {exc}",
            level=logging.WARNING,
            url=url,
            status="transient",
            code=exc.code,
        )
        return FetchOutcome(
            result={
                "url": url,
                "status": "transient_failure",
                "error": err,
                "error_code": exc.code,
            },
            transient=exc,
        )
    except PermanentScrapeError as exc:
        err = redact_text(str(exc))
        record_unusable_page(
            ctx, url, status="failed", error_code=exc.code, error=err, http_status=exc.http_status
        )
        ctx.warn(f"PAGE_SKIPPED_{exc.code}", f"{url} skipped: {err}")
        return FetchOutcome(
            result={"url": url, "status": "failed", "error": err, "error_code": exc.code},
            permanent=exc,
        )
    return FetchOutcome(page=page)
