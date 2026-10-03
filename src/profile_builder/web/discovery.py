"""Page discovery: sitemap/map first, homepage links as fallback, then deterministic
filtering and heuristic scoring. The agent chooses from the scored candidates. The homepage
itself is fetched (and counted) by `discover_pages`; this module never fetches."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from urllib.parse import urlsplit

from profile_builder.config import (
    DEFAULT_EXCLUDE_URL_PATTERNS,
    DISCOVERY_MAP_LIMIT,
    MAX_CANDIDATES_TO_MODEL,
    MAX_RAW_CANDIDATES,
    PAGE_SCORE_KEYWORDS,
)
from profile_builder.logging_setup import get_logger
from profile_builder.web.scraper import LinkCandidate, PermanentScrapeError, ScrapedPage, Scraper
from profile_builder.web.url_guard import (
    URLGuardError,
    _default_resolver,
    normalize_url,
    same_site,
    validate_url,
)

log = get_logger("discovery")

_EXCLUDE_RES = [re.compile(p, re.IGNORECASE) for p in DEFAULT_EXCLUDE_URL_PATTERNS]


@lru_cache(maxsize=256)
def cached_resolver(host: str) -> list[str]:
    return _default_resolver(host)


@dataclass
class ScoredCandidate:
    url: str
    title: str = ""
    description: str = ""
    score: int = 0


@dataclass
class DiscoveryResult:
    candidates: list[ScoredCandidate]
    source: str  # "map" | "homepage_links" | "none"
    homepage: ScrapedPage | None = None
    total_seen: int = 0
    dropped: int = 0
    notes: list[str] = field(default_factory=list)
    homepage_error: PermanentScrapeError | None = None  # 404/403/402...: recorded, not fatal


def _safe_normalize(url: str) -> str | None:
    try:
        return normalize_url(url)
    except URLGuardError:
        return None


def score_url(url: str, title: str = "", description: str = "", start_url: str = "") -> int:
    parts = urlsplit(url)
    path = parts.path.lower()
    text = f"{path} {title} {description}".lower()
    score = 0
    for kw, weight in PAGE_SCORE_KEYWORDS.items():
        if kw in text:
            score += weight
    depth = len([p for p in path.split("/") if p])
    score += max(0, 3 - depth)  # shallow pages explain the company more often
    if parts.query:
        score -= 2
    if start_url and normalize_url(url) == normalize_url(start_url):
        score += 10
    return score


def is_excluded(url: str) -> bool:
    return any(r.search(url) for r in _EXCLUDE_RES)


def filter_and_score(
    raw: list[LinkCandidate], start_url: str, *, check_dns: bool = True
) -> tuple[list[ScoredCandidate], int]:
    seen: set[str] = set()
    out: list[ScoredCandidate] = []
    dropped = max(0, len(raw) - MAX_RAW_CANDIDATES)
    for cand in raw[:MAX_RAW_CANDIDATES]:
        # Cheap offline checks first: hostile links must not trigger DNS lookups or crashes.
        try:
            url = normalize_url(cand.url)
        except URLGuardError:
            dropped += 1
            continue
        if not same_site(url, start_url) or is_excluded(url) or url in seen:
            dropped += 1
            continue
        try:
            url = validate_url(url, resolver=cached_resolver, check_dns=check_dns)
        except URLGuardError:
            dropped += 1
            continue
        seen.add(url)
        out.append(
            ScoredCandidate(
                url=url,
                title=(cand.title or "")[:120],
                description=(cand.description or "")[:200],
                score=score_url(url, cand.title or "", cand.description or "", start_url),
            )
        )
    out.sort(key=lambda c: c.score, reverse=True)
    return out, dropped


def discover(
    start_url: str,
    scraper: Scraper,
    *,
    timeout_ms: int,
    check_dns: bool = True,
    max_candidates: int = MAX_CANDIDATES_TO_MODEL,
    homepage: ScrapedPage | None = None,
    homepage_error: PermanentScrapeError | None = None,
    use_map: bool = True,
) -> DiscoveryResult:
    """Discovery never fetches a page itself: the caller hands in the homepage (fresh or
    cached, already counted against the budgets) or the permanent error that made it
    unavailable, and `discover` consults the site map. Transient map errors propagate (the
    retry middleware retries the tool call, which then serves the homepage from the cache);
    permanent map failures fall back to the homepage links. With `use_map=False` the map is
    not consulted at all and the candidates come from the homepage links alone."""
    notes: list[str] = []
    raw: list[LinkCandidate] = []
    source = "none"
    if use_map:
        try:
            raw = list(scraper.map(start_url, limit=DISCOVERY_MAP_LIMIT, timeout_ms=timeout_ms))
            if raw:
                source = "map"
        except PermanentScrapeError as exc:
            notes.append(f"site map unavailable ({exc.code}); using homepage links")
    known = {_safe_normalize(c.url) for c in raw if c.url} - {None}
    homepage_links = [
        LinkCandidate(url=u)
        for u in ((homepage.links if homepage is not None else None) or [])
        if u and _safe_normalize(u) not in known | {None}
    ]
    if homepage_links:
        raw.extend(homepage_links)
        source = "map+homepage_links" if source == "map" else "homepage_links"
    if not raw:
        notes.append(
            "no site map and the homepage exposed no links; only the start URL is available"
            if homepage_error is None
            else "no site map and the homepage could not be fetched; no pages are available"
        )
    # The homepage is a candidate only when it could be fetched; otherwise it is also removed
    # from the map's list so the model is not offered a page known to be unavailable.
    if homepage_error is None:
        raw.insert(0, LinkCandidate(url=start_url, title="Homepage"))
    candidates, dropped = filter_and_score(raw, start_url, check_dns=check_dns)
    if homepage_error is not None:
        start_norm = _safe_normalize(start_url)
        candidates = [c for c in candidates if c.url != start_norm]
    return DiscoveryResult(
        candidates=candidates[:max_candidates],
        source=source,
        homepage=homepage,
        total_seen=len(raw),
        dropped=dropped,
        notes=notes,
        homepage_error=homepage_error,
    )
