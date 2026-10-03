"""Per-run context shared by every tool: settings, the durable store, scraper, index and
robots checker, plus the small helpers (stage changes, page text cache, warnings) that
keep tool bodies short."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from profile_builder.config import PAGES_DIRNAME, Settings
from profile_builder.logging_setup import event, get_logger, redact_text, set_run_context
from profile_builder.retrieval.index import HybridIndex
from profile_builder.security import url_cache_name
from profile_builder.state.run_store import RunStore
from profile_builder.web.robots import RobotsChecker
from profile_builder.web.scraper import ScrapedPage, Scraper
from profile_builder.web.url_guard import normalize_url

log = get_logger("tools")


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

    def warn_once(self, code: str, message: str, **details: Any) -> bool:
        """Record a warning unless one with this code already exists (limits are reported
        once per run). Returns True when the warning was recorded now."""
        if self.store.has_warning(code):
            return False
        self.warn(code, message, **details)
        return True
