"""Web tools: `discover_pages`, `scrape_pages`, `search_pages`, `read_page`. Page handling
lives in `agent.ingest`; these functions own the model-facing signatures and the budget
checks."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool

from profile_builder.agent.context import ToolContext
from profile_builder.agent.ingest import (
    classify_existing_page,
    fetch_live,
    process_page,
    record_unusable_page,
)
from profile_builder.agent.results import fail, ok, to_json
from profile_builder.config import (
    DISCOVERY_FILENAME,
    MAX_URLS_PER_SCRAPE_CALL,
    PAGE_HEADINGS_TO_MODEL,
    SCRAPE_ATTEMPTS_PER_PAGE,
)
from profile_builder.logging_setup import event, get_logger, redact_text
from profile_builder.security import atomic_write_text
from profile_builder.web.discovery import discover
from profile_builder.web.scraper import (
    PermanentScrapeError,
    TransientScrapeError,
    is_transient_code,
)
from profile_builder.web.url_guard import URLGuardError, normalize_url, same_site, validate_url

log = get_logger("tools")


def make_web_tools(ctx: ToolContext) -> list[Any]:
    settings = ctx.settings
    store = ctx.store

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
            return fail(f"start URL rejected: {exc}")
        if not same_site(url, ctx.start_url):
            return fail("discover_pages only works on the run's website")
        notes: list[str] = []
        home = normalize_url(ctx.start_url)
        if url != home:
            # Discovery only ever fetches the homepage; any other page goes through
            # scrape_pages and its budget checks.
            notes.append(
                f"discovery always starts from the run's homepage {home}; {url} was not fetched"
            )
            url = home
        if not ctx.robots.allowed(url):
            ctx.warn("ROBOTS_DISALLOWED", f"robots.txt disallows fetching {url}")
            return fail("robots.txt disallows the homepage; no pages can be fetched")
        max_attempts = settings.max_pages * SCRAPE_ATTEMPTS_PER_PAGE
        # The homepage goes through the same accounting as scrape_pages BEFORE the site map
        # is consulted: one counted live fetch, processed (cached, indexed, recorded) at once.
        # A transient map failure retried by the middleware then finds it in the cache and
        # never re-scrapes it; without budget, discovery runs from the site map alone.
        homepage = ctx.cached_page(url)
        homepage_error: PermanentScrapeError | None = None
        known = store.get_page(url)
        if homepage is not None:
            if known is None or known.status != "fetched":
                process_page(ctx, homepage, url)  # crash replay: cached but not yet recorded
        elif (
            known is not None
            and known.status == "failed"
            and not is_transient_code(known.error_code)
        ):
            # Permanent homepage failure (404/403/402...) recorded by an earlier call.
            homepage_error = PermanentScrapeError(
                known.error or "homepage unavailable",
                code=known.error_code or "PERMANENT",
                http_status=known.http_status,
            )
        elif store.get_counter("scrape_attempts") >= max_attempts:
            ctx.warn_once(
                "LIMIT_SCRAPE_ATTEMPTS_REACHED", f"fetch attempt limit of {max_attempts} reached"
            )
            notes.append(
                "homepage not fetched (fetch attempt budget exhausted); using the site map only"
            )
        elif ctx.page_budget_remaining() <= 0:
            notes.append("homepage not fetched (page budget exhausted); using the site map only")
        else:
            fetched = fetch_live(ctx, url, pace=False, with_links=True)
            if fetched.transient is not None:
                raise fetched.transient  # the attempt is counted; the middleware retries
            if fetched.page is None:
                homepage_error = fetched.permanent  # recorded + warned by fetch_live
            else:
                homepage = fetched.page
                process_page(ctx, homepage, url)
        try:
            result = discover(
                url,
                ctx.scraper,
                timeout_ms=settings.scrape_timeout_ms,
                check_dns=ctx.check_dns,
                homepage=homepage,
                homepage_error=homepage_error,
            )
        except TransientScrapeError as exc:
            # Only the site map can fail here; the homepage is already cached and counted,
            # so a retry costs nothing. After MAX_RETRIES transient map failures in this run
            # the map is given up and the homepage links are used (with no homepage there is
            # nothing to fall back to, so the error keeps propagating).
            failures = store.increment_counter("map_transient_failures")
            if homepage is None or failures <= settings.max_retries:
                raise
            notes.append(
                f"site map unavailable after {failures} attempts ({exc.code}); using homepage links"
            )
            result = discover(
                url,
                ctx.scraper,
                timeout_ms=settings.scrape_timeout_ms,
                check_dns=ctx.check_dns,
                homepage=homepage,
                homepage_error=homepage_error,
                use_map=False,
            )
        notes.extend(result.notes)
        for note in notes:
            ctx.warn("DISCOVERY_NOTE", note)
        if homepage_error is not None and not result.candidates:
            err = redact_text(str(homepage_error))
            return to_json(
                {
                    "ok": False,
                    "error_code": "HOMEPAGE_UNAVAILABLE",
                    "error": (
                        f"the homepage could not be fetched ({err}) and no other page was "
                        "discovered. Check the start URL (scheme, www vs. non-www, trailing "
                        "path) or start from another page of the same site."
                    ),
                    "notes": notes,
                }
            )
        event(
            log,
            "pages_discovered",
            f"{len(result.candidates)} candidates from {result.source} ({result.dropped} dropped)",
            count=len(result.candidates),
            kind=result.source,
        )
        atomic_write_text(
            ctx.run_dir / DISCOVERY_FILENAME,
            to_json(
                [
                    {"url": c.url, "title": c.title, "description": c.description}
                    for c in result.candidates
                ]
            ),
        )
        return ok(
            source=result.source,
            candidates=[
                {"url": c.url, "title": c.title, "description": c.description, "score": c.score}
                for c in result.candidates
            ],
            already_fetched=sorted(store.fetched_urls()),
            page_budget_remaining=ctx.page_budget_remaining(),
            notes=notes,
        )

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
        transient_failure: TransientScrapeError | None = None
        requested = [str(u) for u in (urls or [])]
        if len(requested) > MAX_URLS_PER_SCRAPE_CALL:
            results.append(
                {
                    "note": f"only the first {MAX_URLS_PER_SCRAPE_CALL} of {len(requested)} URLs were considered",
                    "error_code": "TOO_MANY_URLS",
                }
            )
            requested = requested[:MAX_URLS_PER_SCRAPE_CALL]
        max_attempts = settings.max_pages * SCRAPE_ATTEMPTS_PER_PAGE
        for raw in requested:
            try:
                url = validate_url(raw, check_dns=ctx.check_dns)
            except URLGuardError as exc:
                ctx.warn("URL_REJECTED", f"{raw} rejected: {exc}")
                results.append(
                    {
                        "url": raw,
                        "status": "rejected",
                        "error": str(exc),
                        "error_code": "URL_REJECTED",
                    }
                )
                continue
            if url in seen:
                continue
            seen.add(url)
            if not same_site(url, ctx.start_url):
                results.append(
                    {
                        "url": url,
                        "status": "rejected",
                        "error": "not on the company's website",
                        "error_code": "OFFSITE",
                    }
                )
                continue
            known = classify_existing_page(ctx, url, store.get_page(url))
            if known is not None:
                results.append(known)
                continue
            if store.get_counter("scrape_attempts") >= max_attempts:
                ctx.warn_once(
                    "LIMIT_SCRAPE_ATTEMPTS_REACHED",
                    f"fetch attempt limit of {max_attempts} reached",
                )
                results.append(
                    {
                        "url": url,
                        "status": "not_fetched",
                        "error": "fetch attempt budget exhausted",
                        "error_code": "FETCH_BUDGET_EXHAUSTED",
                    }
                )
                continue
            if ctx.page_budget_remaining() <= 0:
                if ctx.warn_once(
                    "LIMIT_PAGES_REACHED",
                    f"page limit of {settings.max_pages} reached; further pages were not fetched",
                ):
                    event(
                        log,
                        "limit_reached",
                        "page limit reached",
                        kind="pages",
                        count=settings.max_pages,
                    )
                results.append(
                    {
                        "url": url,
                        "status": "not_fetched",
                        "error": "page budget exhausted",
                        "error_code": "PAGE_BUDGET_EXHAUSTED",
                    }
                )
                continue
            if not ctx.robots.allowed(url):
                record_unusable_page(
                    ctx,
                    url,
                    status="skipped",
                    error_code="ROBOTS_DISALLOWED",
                    error="robots.txt disallows",
                )
                ctx.warn("ROBOTS_DISALLOWED", f"robots.txt disallows fetching {url}; skipped")
                results.append({"url": url, "status": "skipped", "error": "robots.txt disallows"})
                continue
            page = ctx.cached_page(url)
            if page is None:
                fetched = fetch_live(ctx, url, pace=live_fetches > 0)
                live_fetches += 1
                if fetched.page is None:
                    # Carry on with the other URLs; a transient error is re-raised once at
                    # the end so the retry middleware retries the call (already-fetched
                    # pages are then free) without starving the batch.
                    results.append(fetched.result or {})
                    transient_failure = transient_failure or fetched.transient
                    continue
                page = fetched.page
            results.append(process_page(ctx, page, url))
        if transient_failure is not None:
            raise transient_failure
        return ok(
            results=results,
            pages_fetched=len(store.fetched_urls()),
            page_budget_remaining=ctx.page_budget_remaining(),
        )

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
                return fail(str(exc))
        hits = ctx.index.search(query, url=target)
        if not hits:
            return ok(
                query=query,
                results=[],
                note="no indexed content matched; scrape pages first or broaden the query",
            )
        return ok(
            query=query,
            results=[{"url": h.url, "heading": h.heading, "excerpt": h.excerpt} for h in hits],
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
            return fail(str(exc))
        if target not in store.fetched_urls():
            return fail(
                "page not fetched; call scrape_pages first", fetched=sorted(store.fetched_urls())
            )
        text, more, next_offset = ctx.index.read(
            target, section=section, offset=max(0, int(offset or 0))
        )
        return ok(
            url=target,
            section=section,
            text=text,
            more_available=more,
            next_offset=next_offset,
            outline=ctx.index.page_outline(target, PAGE_HEADINGS_TO_MODEL),
        )

    return [discover_pages, scrape_pages, search_pages, read_page]
