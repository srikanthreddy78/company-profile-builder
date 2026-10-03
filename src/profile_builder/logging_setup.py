"""Logging system: structured events.jsonl + human-readable run.log + Rich console.

Every record automatically carries ``run_id`` and ``stage`` (via contextvars) and passes
through a secret-redaction filter before reaching any sink.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler

from profile_builder.config import EVENTS_FILENAME, RUN_LOG_FILENAME

LOGGER_NAME = "profile_builder"

# Process-wide (not contextvars): tools execute in the tool node's worker threads, and a
# stage set there must be visible to model-call telemetry logged from the main thread.
_CONTEXT: dict[str, str] = {"run_id": "-", "stage": "init"}

EVENT_FIELDS = (
    "event",
    "attempt",
    "duration_ms",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "tool",
    "url",
    "code",
    "qid",
    "status",
    "count",
    "version",
    "kind",
    "model",
)

_KNOWN_SECRETS: list[str] = []  # set by attach_run_sinks; used by redact_text()

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-*]{8,}"),
    re.compile(r"fc-[A-Za-z0-9_\-*]{8,}"),
    re.compile(r"lsv2_[A-Za-z0-9_\-*]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"(?i)(api[_-]?key|authorization)\s*[:=]\s*['\"]?[A-Za-z0-9._\-]{8,}"),
]


def redact_text(text: str) -> str:
    """Mask known secrets and key-like tokens in arbitrary text (for values that bypass
    logging, e.g. warnings stored in the run database or printed to the console)."""
    for secret in _KNOWN_SECRETS:
        text = text.replace(secret, "***REDACTED***")
    for pat in _SECRET_PATTERNS:
        text = pat.sub("***REDACTED***", text)
    return text


def set_run_context(run_id: str | None = None, stage: str | None = None) -> None:
    if run_id is not None:
        _CONTEXT["run_id"] = run_id
    if stage is not None:
        _CONTEXT["stage"] = stage


def current_stage() -> str:
    return _CONTEXT["stage"]


class ContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _CONTEXT["run_id"]
        record.stage = _CONTEXT["stage"]
        return True


class RedactSecretsFilter(logging.Filter):
    """Mask API keys and bearer tokens anywhere in the record (message, args, extras)."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._secrets = [s for s in secrets if s and len(s) >= 8]

    def redact(self, text: str) -> str:
        for s in self._secrets:
            text = text.replace(s, "***REDACTED***")
        return redact_text(text)

    def filter(self, record: logging.LogRecord) -> bool:
        with contextlib.suppress(Exception):  # never break logging
            record.msg = self.redact(str(record.getMessage()))
            record.args = ()
        if record.exc_info and record.exc_info[1] is not None:
            exc = record.exc_info[1]
            with contextlib.suppress(Exception):
                exc.args = tuple(self.redact(str(a)) for a in exc.args)
        for field in EVENT_FIELDS:
            val = getattr(record, field, None)
            if isinstance(val, str):
                setattr(record, field, self.redact(val))
        return True


def _exc_text(record: logging.LogRecord) -> str:
    exc_type, exc = record.exc_info[0], record.exc_info[1]  # type: ignore[index]
    return redact_text(f"{getattr(exc_type, '__name__', 'Error')}: {exc}")


_CONTROL_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _clean(text: str) -> str:
    return _CONTROL_RE.sub("", text)


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "run_id": getattr(record, "run_id", "-"),
            "stage": getattr(record, "stage", "-"),
            "logger": record.name,
            "message": _clean(record.getMessage()),
        }
        for field in EVENT_FIELDS:
            val = getattr(record, field, None)
            if val is not None:
                payload[field] = val
        if record.exc_info and record.exc_info[1] is not None:
            payload["exception"] = _exc_text(record)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} {record.levelname:<7} "
            f"[{getattr(record, 'run_id', '-')}:{getattr(record, 'stage', '-')}] "
            f"{_clean(record.getMessage()).replace(chr(10), ' | ')}"
        )
        extras = []
        for field in EVENT_FIELDS:
            val = getattr(record, field, None)
            if val is not None:
                extras.append(f"{field}={val}")
        if extras:
            base += "  (" + " ".join(extras) + ")"
        if record.exc_info and record.exc_info[1] is not None:
            base += f"\n  {_exc_text(record)}"
        return base


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def configure_console(level: str = "INFO", console: Console | None = None) -> None:
    """Console sink only (used before a run directory exists)."""
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(logging.DEBUG)
    root.propagate = False
    for h in list(root.handlers):
        if getattr(h, "_pb_console", False):
            root.removeHandler(h)
    handler = RichHandler(
        console=console, show_path=False, show_time=False, rich_tracebacks=False, markup=False
    )
    handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler._pb_console = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    _ensure_filters(root, ())


def attach_run_sinks(run_dir: Path, secrets: Iterable[str] = ()) -> None:
    """Add events.jsonl + run.log sinks for a run (idempotent per run_dir)."""
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(logging.DEBUG)  # sinks filter by their own level; never drop events here
    run_dir.mkdir(parents=True, exist_ok=True)
    for h in list(root.handlers):
        if getattr(h, "_pb_run_dir", None) == str(run_dir):
            return
        if getattr(h, "_pb_run_dir", None):
            root.removeHandler(h)
            h.close()
    for name in (EVENTS_FILENAME, RUN_LOG_FILENAME):  # create private before any handler opens
        path = run_dir / name
        if not path.exists():
            path.touch(mode=0o600)
    events = logging.FileHandler(run_dir / EVENTS_FILENAME, encoding="utf-8")
    events.setLevel(logging.DEBUG)
    events.setFormatter(JsonLinesFormatter())
    events._pb_run_dir = str(run_dir)  # type: ignore[attr-defined]
    text = logging.FileHandler(run_dir / RUN_LOG_FILENAME, encoding="utf-8")
    text.setLevel(logging.DEBUG)
    text.setFormatter(TextFormatter())
    text._pb_run_dir = str(run_dir)  # type: ignore[attr-defined]
    root.addHandler(events)
    root.addHandler(text)
    _KNOWN_SECRETS[:] = [s for s in secrets if s and len(s) >= 8]
    _ensure_filters(root, secrets)
    for noisy in ("openai", "httpx", "httpcore", "langchain", "langgraph"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for path in (run_dir / EVENTS_FILENAME, run_dir / RUN_LOG_FILENAME):
        try:
            path.chmod(0o600)
        except OSError:  # pragma: no cover
            pass


def _ensure_filters(logger: logging.Logger, secrets: Iterable[str]) -> None:
    secrets = list(secrets)
    for f in list(logger.filters):
        if isinstance(f, (ContextFilter, RedactSecretsFilter)):
            logger.removeFilter(f)
    logger.addFilter(ContextFilter())
    logger.addFilter(RedactSecretsFilter(secrets))
    # Filters on a logger only apply to records logged *through* it, not children; attach to
    # handlers too so child-logger records are also redacted.
    for h in logger.handlers:
        for f in list(h.filters):
            if isinstance(f, (ContextFilter, RedactSecretsFilter)):
                h.removeFilter(f)
        h.addFilter(ContextFilter())
        h.addFilter(RedactSecretsFilter(secrets))


def event(
    logger: logging.Logger, name: str, message: str, level: int = logging.INFO, **fields: Any
) -> None:
    """Log a structured event. Unknown fields are dropped to keep the schema stable."""
    extra = {"event": name}
    for k, v in fields.items():
        if k in EVENT_FIELDS and v is not None:
            extra[k] = v
    logger.log(level, message, extra=extra)


def read_events(
    run_dir: Path, tail: int | None = None, level: str | None = None
) -> list[dict[str, Any]]:
    path = run_dir / EVENTS_FILENAME
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if level:
        order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        min_idx = order.index(level.upper()) if level.upper() in order else 0
        rows = [r for r in rows if order.index(r.get("level", "INFO")) >= min_idx]
    if tail:
        rows = rows[-tail:]
    return rows
