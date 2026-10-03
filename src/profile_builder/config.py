"""Configuration: the single place where every tunable lives.

User-facing settings come from CLI flags, environment variables (prefix ``PROFILE_BUILDER_``)
or a ``.env`` file, in that order of precedence. Internal constants are module-level values
below; the rest of the codebase imports from here and never hardcodes numbers.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from profile_builder.models import DEFAULT_EMBEDDING_MODEL, MODEL_TIERS, Tier

# --------------------------------------------------------------------------------------
# Internal constants (tunable in code, intentionally not exposed as flags)
# --------------------------------------------------------------------------------------

APP_NAME = "CompanyProfileBuilder"
APP_VERSION = "0.1.0"
USER_AGENT = f"{APP_NAME}/{APP_VERSION} (+https://github.com/srikanth/company-profile-builder)"

# Derived caps -------------------------------------------------------------------------
MODEL_CALL_BASE = 20  # model calls for discovery/selection/drafting overhead
MODEL_CALLS_PER_PAGE = 3
MODEL_CALLS_PER_QUESTION = 4
PAGES_PER_SCRAPE_CALL = 3  # scrape_pages accepts several URLs per call

# Retry/backoff (used by ToolRetry + ModelRetry middleware) ----------------------------
RETRY_INITIAL_DELAY_S = 2.0
RETRY_MAX_DELAY_S = 30.0
RETRY_BACKOFF_FACTOR = 3.0  # 2s, 6s: Firecrawl's free tier rate-limits bursts of ~10 requests
SCRAPE_INTER_REQUEST_DELAY_S = 1.5  # polite pacing between live fetches inside one tool call
MODEL_REQUEST_TIMEOUT_S = 180  # a full structured draft from a reasoning model can take >60s

# Discovery ----------------------------------------------------------------------------
FIRECRAWL_API_URL = "https://api.firecrawl.dev"  # pinned: never taken from the environment
DISCOVERY_MAP_LIMIT = 200
MAX_RAW_CANDIDATES = 500  # cap on raw links considered before DNS checks
MAX_DISCOVER_CALLS = 3
MAX_URLS_PER_SCRAPE_CALL = 10
SCRAPE_ATTEMPTS_PER_PAGE = 2  # cap on fetch attempts (incl. failures) = MAX_PAGES * this
ROBOTS_MAX_BYTES = 512 * 1024
ROBOTS_MAX_REDIRECTS = 5
MAX_CANDIDATES_TO_MODEL = 30
MAX_URL_LENGTH = 2048
DEFAULT_EXCLUDE_URL_PATTERNS: tuple[str, ...] = (
    r"\.(png|jpe?g|gif|svg|webp|ico|pdf|zip|gz|mp4|mp3|css|js|xml|json|txt)(\?|$)",
    r"/(privacy|terms|legal|cookie|gdpr|careers?|jobs?|login|signin|signup|register|cart|checkout)(/|$)",
    r"/(tag|tags|category|categories|author|page)/",
    r"/(wp-admin|wp-json|feed|rss|sitemap)(/|$)",
    r"[?&](page|p|utm_[a-z]+|ref|fbclid|gclid)=",
)
PAGE_SCORE_KEYWORDS: dict[str, int] = {
    "product": 5,
    "platform": 5,
    "solution": 4,
    "feature": 4,
    "capabilit": 4,
    "customer": 4,
    "case-stud": 4,
    "case_stud": 4,
    "casestud": 4,
    "success": 3,
    "story": 3,
    "about": 3,
    "company": 3,
    "pricing": 4,
    "plans": 3,
    "why": 3,
    "compare": 3,
    "vs": 2,
    "use-case": 4,
    "usecase": 4,
    "industr": 3,
    "overview": 3,
    "how-it-works": 4,
    "security": 2,
    "integration": 2,
    "faq": 2,
    "resources": 1,
    "blog": -2,
    "news": -2,
    "press": -2,
    "event": -3,
    "webinar": -3,
    "podcast": -3,
}

# Content limits -----------------------------------------------------------------------
MAX_PAGE_CHARS = 60_000
CHUNK_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 60
CHARS_PER_TOKEN = 4  # cheap estimate; exactness is not needed for chunk sizing
SEARCH_K = 6
MAX_EXCERPT_CHARS = 500
MIN_EXCERPT_CHARS = 25
THIN_DRAFT_SECTION_FIELDS = 1  # a section with <= this many filled fields counts as thin  # shorter "quotes" cannot ground a claim
MAX_EVIDENCE_EXCERPT_CHARS = 2 * MAX_EXCERPT_CHARS
MAX_READ_CHARS = 4000
PAGE_LEAD_CHARS = 300
PAGE_HEADINGS_TO_MODEL = 10
RRF_K = 60  # reciprocal-rank-fusion constant
MAX_ANSWER_CHARS = 2000

# Interview / validation ---------------------------------------------------------------
MAX_REPAIR_ATTEMPTS = 1
MAX_FINALIZE_NUDGES = 1
MAX_FINALIZE_REFUSALS = 1  # finalize_profile pushes back once if sections are empty

# Logging ------------------------------------------------------------------------------
EVENTS_FILENAME = "events.jsonl"
RUN_LOG_FILENAME = "run.log"
CHECKPOINT_DB = "checkpoints.sqlite"
RUN_DB = "run.sqlite"
PAGES_DIRNAME = "pages"
DISCOVERY_FILENAME = "discovery.json"
OUTPUT_FILENAME = "company_brain.json"
EVIDENCE_FILENAME = "evidence.json"
REPORT_FILENAME = "report.md"

RUN_ID_PREFIX = "pb"
RUN_ID_PATTERN = r"^pb-\d{8}-[a-z0-9]{6}$"


# --------------------------------------------------------------------------------------
# User-facing settings
# --------------------------------------------------------------------------------------


class Settings(BaseSettings):
    """Runtime settings. Precedence: CLI flag > env var > .env > default."""

    model_config = SettingsConfigDict(
        env_prefix="PROFILE_BUILDER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider credentials (no prefix; standard names)
    openai_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("OPENAI_API_KEY")
    )
    firecrawl_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("FIRECRAWL_API_KEY")
    )

    # Limits
    max_pages: int = Field(default=10, ge=1, le=100)
    max_questions: int = Field(default=5, ge=0, le=50)

    # Model
    model: str | None = None
    tier: Tier = "fast"
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    use_embeddings: bool = True

    # Cost / network
    budget_usd: float | None = Field(default=None, gt=0)
    scrape_timeout_ms: int = Field(default=30_000, ge=1_000)
    max_retries: int = Field(default=2, ge=0, le=10)

    # Paths / logging
    runs_dir: Path = Path(".runs")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ---- derived -------------------------------------------------------------------
    @property
    def model_id(self) -> str:
        return self.model or MODEL_TIERS[self.tier]

    @property
    def max_model_calls(self) -> int:
        return (
            MODEL_CALL_BASE
            + MODEL_CALLS_PER_PAGE * self.max_pages
            + MODEL_CALLS_PER_QUESTION * self.max_questions
        )

    @property
    def max_scrape_calls(self) -> int:
        return math.ceil(self.max_pages / PAGES_PER_SCRAPE_CALL) + 1

    # ---- persistence -------------------------------------------------------------
    SNAPSHOT_FIELDS: ClassVar[tuple[str, ...]] = (
        "max_pages",
        "max_questions",
        "model",
        "tier",
        "embedding_model",
        "use_embeddings",
        "budget_usd",
        "scrape_timeout_ms",
        "max_retries",
        "log_level",
    )

    def snapshot(self) -> dict[str, Any]:
        """Non-secret settings, saved with the run so `resume` keeps the same limits."""
        data = {k: getattr(self, k) for k in self.SNAPSHOT_FIELDS}
        data["model_id"] = self.model_id
        data["max_model_calls"] = self.max_model_calls
        data["max_scrape_calls"] = self.max_scrape_calls
        return data

    def with_overrides(self, **overrides: Any) -> Settings:
        """Return a validated copy with the given non-None overrides applied (CLI flags)."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        if not clean:
            return self
        return type(self)(_env_file=None, **{**self.model_dump(), **clean})

    @classmethod
    def from_snapshot(cls, snapshot: dict[str, Any], **overrides: Any) -> Settings:
        """Rebuild settings for `resume`: secrets from the environment, limits from the run."""
        base = cls()
        fields = {k: snapshot[k] for k in cls.SNAPSHOT_FIELDS if k in snapshot}
        merged = cls(_env_file=None, **{**base.model_dump(), **fields})
        return merged.with_overrides(**overrides)

    def has_openai(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value())

    def has_firecrawl(self) -> bool:
        return bool(self.firecrawl_api_key and self.firecrawl_api_key.get_secret_value())
