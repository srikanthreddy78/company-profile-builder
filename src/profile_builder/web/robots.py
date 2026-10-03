"""robots.txt checks (cached per host, fetched through the URL guard with a timeout)."""

from __future__ import annotations

import logging
import urllib.robotparser
from urllib.parse import urlsplit, urlunsplit

import requests

from profile_builder.config import USER_AGENT
from profile_builder.logging_setup import get_logger
from profile_builder.web.url_guard import URLGuardError, validate_url

log = get_logger("robots")

ROBOTS_TIMEOUT_S = 10


class RobotsChecker:
    def __init__(self, *, enabled: bool = True, session: requests.Session | None = None) -> None:
        self.enabled = enabled
        self._session = session or requests.Session()
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def _robots_url(self, url: str) -> str:
        p = urlsplit(url)
        return urlunsplit((p.scheme, p.netloc, "/robots.txt", "", ""))

    def _load(self, url: str) -> urllib.robotparser.RobotFileParser | None:
        robots_url = self._robots_url(url)
        key = urlsplit(robots_url).netloc
        if key in self._parsers:
            return self._parsers[key]
        parser: urllib.robotparser.RobotFileParser | None = None
        try:
            validate_url(robots_url)
            resp = self._session.get(
                robots_url,
                timeout=ROBOTS_TIMEOUT_S,
                headers={"User-Agent": USER_AGENT},
                allow_redirects=False,
            )
            if resp.status_code == 200 and resp.text:
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(resp.text.splitlines())
            # 4xx/5xx or redirect → treat as "no robots restrictions" (standard behavior for 404)
        except (requests.RequestException, URLGuardError) as exc:
            log.debug("robots.txt unavailable for %s: %s", key, exc)
        self._parsers[key] = parser
        return parser

    def allowed(self, url: str) -> bool:
        if not self.enabled:
            return True
        parser = self._load(url)
        if parser is None:
            return True
        try:
            return parser.can_fetch(USER_AGENT, url) or parser.can_fetch("*", url)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("robots parse error for %s: %s", url, exc, exc_info=False)
            return True


logging.getLogger("urllib3").setLevel(logging.WARNING)
