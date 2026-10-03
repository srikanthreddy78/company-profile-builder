"""Scraper abstraction: Firecrawl in production, saved fixtures offline.

Errors are classified so that only *transient* failures reach the retry middleware;
permanent ones (404, 402, blocked, bad request) are reported once and never retried.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

from profile_builder.config import USER_AGENT
from profile_builder.logging_setup import get_logger
from profile_builder.security import url_cache_name
from profile_builder.web.url_guard import normalize_url

log = get_logger("scraper")


class TransientScrapeError(Exception):
    """Retryable: rate limit, timeout, 5xx, network blip."""

    def __init__(self, message: str, *, code: str = "TRANSIENT") -> None:
        super().__init__(message)
        self.code = code


class PermanentScrapeError(Exception):
    """Not retryable: 4xx (except 408/429), blocked, unsupported, invalid credentials."""

    def __init__(
        self, message: str, *, code: str = "PERMANENT", http_status: int | None = None
    ) -> None:
        super().__init__(message)
        self.code = code
        self.http_status = http_status


@dataclass
class ScrapedPage:
    url: str
    final_url: str
    title: str = ""
    description: str = ""
    markdown: str = ""
    http_status: int | None = None
    links: list[str] = field(default_factory=list)
    cached: bool = False

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("cached", None)
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> ScrapedPage:
        d = json.loads(text)
        d.pop("cached", None)
        return cls(**d)


@dataclass
class LinkCandidate:
    url: str
    title: str = ""
    description: str = ""


class Scraper(Protocol):
    def scrape(self, url: str, *, timeout_ms: int, with_links: bool = False) -> ScrapedPage: ...

    def map(self, url: str, *, limit: int, timeout_ms: int) -> list[LinkCandidate]: ...


# --------------------------------------------------------------------------------------
# Firecrawl
# --------------------------------------------------------------------------------------


def is_transient(exc: BaseException) -> bool:
    """Retry predicate shared with ToolRetryMiddleware."""
    return isinstance(exc, TransientScrapeError)


def classify_firecrawl_error(exc: BaseException) -> TransientScrapeError | PermanentScrapeError:
    from firecrawl.v2.utils import error_handler as eh

    if isinstance(exc, (TransientScrapeError, PermanentScrapeError)):
        return exc
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return TransientScrapeError(f"network error: {exc}", code="NETWORK")
    if isinstance(exc, eh.RateLimitError):
        return TransientScrapeError("rate limited (429)", code="RATE_LIMITED")
    if isinstance(exc, eh.RequestTimeoutError):
        return TransientScrapeError("request timed out (408)", code="TIMEOUT")
    if isinstance(exc, eh.InternalServerError):
        return TransientScrapeError("server error (500)", code="SERVER_ERROR")
    if isinstance(exc, eh.UnauthorizedError):
        return PermanentScrapeError(
            "invalid Firecrawl credentials (401)", code="UNAUTHORIZED", http_status=401
        )
    if isinstance(exc, eh.PaymentRequiredError):
        return PermanentScrapeError(
            "Firecrawl credits exhausted (402)", code="PAYMENT_REQUIRED", http_status=402
        )
    if isinstance(exc, (eh.WebsiteNotSupportedError, eh.ProviderTermsRequiredError)):
        return PermanentScrapeError(
            "website blocked or unsupported (403)", code="BLOCKED", http_status=403
        )
    if isinstance(exc, eh.BadRequestError):
        return PermanentScrapeError(
            f"bad request: {_short(exc)}", code="BAD_REQUEST", http_status=400
        )
    if isinstance(exc, eh.FirecrawlError):
        status = getattr(exc, "status_code", None)
        if status is not None and int(status) >= 500:
            return TransientScrapeError(f"server error ({status})", code="SERVER_ERROR")
        return PermanentScrapeError(
            f"firecrawl error: {_short(exc)}", code="FIRECRAWL_ERROR", http_status=status
        )
    return PermanentScrapeError(
        f"unexpected scrape error: {type(exc).__name__}: {_short(exc)}", code="UNEXPECTED"
    )


def _short(exc: BaseException, limit: int = 200) -> str:
    text = str(exc).replace("\n", " ")
    return text[:limit]


class FirecrawlScraper:
    def __init__(self, api_key: str) -> None:
        from firecrawl import Firecrawl

        # The SDK's own retry loop is disabled so the middleware owns retries/backoff.
        self._client = Firecrawl(api_key=api_key, max_retries=0)

    def scrape(self, url: str, *, timeout_ms: int, with_links: bool = False) -> ScrapedPage:
        formats: list[Any] = ["markdown", "links"] if with_links else ["markdown"]
        try:
            doc = self._client.scrape(
                url,
                formats=formats,
                only_main_content=True,
                timeout=timeout_ms,
                headers={"User-Agent": USER_AGENT},
            )
        except Exception as exc:
            raise classify_firecrawl_error(exc) from exc
        meta = getattr(doc, "metadata", None)
        status = getattr(meta, "status_code", None) if meta else None
        if status is not None and int(status) >= 400:
            if int(status) in (408, 429) or int(status) >= 500:
                raise TransientScrapeError(f"target returned HTTP {status}", code=f"HTTP_{status}")
            raise PermanentScrapeError(
                f"target returned HTTP {status}", code=f"HTTP_{status}", http_status=int(status)
            )
        final_url = (
            (getattr(meta, "source_url", None) or getattr(meta, "url", None) or url)
            if meta
            else url
        )
        return ScrapedPage(
            url=url,
            final_url=final_url,
            title=(getattr(meta, "title", "") or "") if meta else "",
            description=(getattr(meta, "description", "") or "") if meta else "",
            markdown=getattr(doc, "markdown", "") or "",
            http_status=int(status) if status is not None else None,
            links=list(getattr(doc, "links", None) or []),
        )

    def map(self, url: str, *, limit: int, timeout_ms: int) -> list[LinkCandidate]:
        try:
            data = self._client.map(url, limit=limit, sitemap="include", timeout=timeout_ms)
        except Exception as exc:
            raise classify_firecrawl_error(exc) from exc
        out = []
        for link in getattr(data, "links", None) or []:
            out.append(
                LinkCandidate(
                    url=getattr(link, "url", "") or "",
                    title=getattr(link, "title", "") or "",
                    description=getattr(link, "description", "") or "",
                )
            )
        return out


# --------------------------------------------------------------------------------------
# Fixtures (offline / tests)
# --------------------------------------------------------------------------------------


class FixtureScraper:
    """Serves pages from a directory: ``map.json`` + ``pages/<sha1(url)>.json``.

    ``failures`` maps a normalized URL to a list of exceptions raised on successive calls
    before the page is served (used to simulate 429/timeouts in tests).
    """

    def __init__(
        self, fixture_dir: Path, failures: dict[str, list[BaseException]] | None = None
    ) -> None:
        self.dir = Path(fixture_dir)
        self.pages_dir = self.dir / "pages"
        self._failures = {normalize_url(k): list(v) for k, v in (failures or {}).items()}
        self.calls: list[tuple[str, str]] = []

    def _maybe_fail(self, url: str) -> None:
        queue = self._failures.get(normalize_url(url))
        if queue:
            raise queue.pop(0)

    def scrape(self, url: str, *, timeout_ms: int, with_links: bool = False) -> ScrapedPage:
        self.calls.append(("scrape", url))
        self._maybe_fail(url)
        path = self.pages_dir / url_cache_name(normalize_url(url))
        if not path.exists():
            raise PermanentScrapeError(
                "fixture page not found (404)", code="HTTP_404", http_status=404
            )
        page = ScrapedPage.from_json(path.read_text(encoding="utf-8"))
        if not with_links:
            page.links = []
        return page

    def map(self, url: str, *, limit: int, timeout_ms: int) -> list[LinkCandidate]:
        self.calls.append(("map", url))
        self._maybe_fail(url)
        path = self.dir / "map.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            LinkCandidate(**{k: v for k, v in d.items() if k in ("url", "title", "description")})
            for d in data
        ][:limit]

    @staticmethod
    def write_page(fixture_dir: Path, page: ScrapedPage) -> Path:
        pages = Path(fixture_dir) / "pages"
        pages.mkdir(parents=True, exist_ok=True)
        path = pages / url_cache_name(normalize_url(page.url))
        path.write_text(page.to_json(), encoding="utf-8")
        return path

    @staticmethod
    def write_map(fixture_dir: Path, candidates: list[LinkCandidate]) -> Path:
        Path(fixture_dir).mkdir(parents=True, exist_ok=True)
        path = Path(fixture_dir) / "map.json"
        path.write_text(json.dumps([asdict(c) for c in candidates], indent=2), encoding="utf-8")
        return path
