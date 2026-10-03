"""robots.txt checks (cached per host, fetched through the URL guard with timeouts, a size cap
and bounded, re-validated redirects). 5xx answers are treated as "disallow all" (RFC 9309)."""

from __future__ import annotations

import logging
import urllib.robotparser
from urllib.parse import urljoin, urlsplit, urlunsplit

import requests

from profile_builder.config import ROBOTS_MAX_BYTES, ROBOTS_MAX_REDIRECTS, USER_AGENT
from profile_builder.logging_setup import get_logger
from profile_builder.web.url_guard import URLGuardError, same_site, validate_url

log = get_logger("robots")

ROBOTS_TIMEOUT_S = 10
_DISALLOW_ALL = "DISALLOW_ALL"


class RobotsChecker:
    def __init__(self, *, enabled: bool = True, session: requests.Session | None = None) -> None:
        self.enabled = enabled
        self._session = session or requests.Session()
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | str | None] = {}

    @staticmethod
    def _robots_url(url: str) -> str:
        p = urlsplit(url)
        return urlunsplit((p.scheme, p.netloc, "/robots.txt", "", ""))

    def _fetch(self, robots_url: str) -> tuple[int | None, str]:
        """GET robots.txt following at most ROBOTS_MAX_REDIRECTS same-site, guard-validated hops."""
        current = robots_url
        for _ in range(ROBOTS_MAX_REDIRECTS + 1):
            validate_url(current)
            resp = self._session.get(
                current,
                timeout=ROBOTS_TIMEOUT_S,
                headers={"User-Agent": USER_AGENT},
                allow_redirects=False,
                stream=True,
            )
            try:
                if 300 <= resp.status_code < 400 and resp.headers.get("location"):
                    nxt = urljoin(current, resp.headers["location"])
                    if not same_site(nxt, robots_url):
                        return resp.status_code, ""
                    current = nxt
                    continue
                body = resp.raw.read(ROBOTS_MAX_BYTES + 1, decode_content=True)
                if len(body) > ROBOTS_MAX_BYTES:
                    log.debug(
                        "robots.txt at %s exceeds %d bytes; ignoring", current, ROBOTS_MAX_BYTES
                    )
                    return resp.status_code, ""
                return resp.status_code, body.decode("utf-8", errors="replace")
            finally:
                resp.close()
        return None, ""

    def _load(self, url: str) -> urllib.robotparser.RobotFileParser | str | None:
        robots_url = self._robots_url(url)
        key = urlsplit(robots_url).netloc
        if key in self._parsers:
            return self._parsers[key]
        parser: urllib.robotparser.RobotFileParser | str | None = None
        try:
            status, text = self._fetch(robots_url)
            if status is not None and status >= 500:
                parser = _DISALLOW_ALL  # RFC 9309: unreachable robots.txt → assume disallow
            elif status == 200 and text:
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(text.splitlines())
            # 4xx or redirect loop → no restrictions
        except (requests.RequestException, URLGuardError, ValueError) as exc:
            log.debug("robots.txt unavailable for %s: %s", key, exc)
        self._parsers[key] = parser
        return parser

    def allowed(self, url: str) -> bool:
        if not self.enabled:
            return True
        parser = self._load(url)
        if parser is None:
            return True
        if parser == _DISALLOW_ALL:
            return False
        try:
            # RobotFileParser falls back to the "*" group itself when our agent has no entry.
            return parser.can_fetch(USER_AGENT, url)  # type: ignore[union-attr]
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("robots parse error for %s: %s", url, exc)
            return True


logging.getLogger("urllib3").setLevel(logging.WARNING)
