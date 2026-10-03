"""Security helpers: run-id validation, path containment, atomic writes, terminal escaping,
evidence verification and injection heuristics."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

from rich.markup import escape as rich_escape

from profile_builder.config import MAX_ANSWER_CHARS, RUN_ID_PATTERN

_RUN_ID_RE = re.compile(RUN_ID_PATTERN)
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\](\([^)]*\))?")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_NOISE_RE = re.compile(r"[*_`#>|\[\]]")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*+•]|\d+[.)])\s+")
_WS_RE = re.compile(r"\s+")

INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore (all |any )?(previous|prior|above) (instructions|prompts?)",
        r"disregard (the )?(previous|prior|above|system) (instructions|prompt)",
        r"you are now (a|an) ",
        r"system prompt",
        r"as an ai (language )?model",
        r"do not (tell|reveal|mention) (the )?user",
        r"call the (tool|function) ",
        r"<\s*/?\s*(system|assistant|tool|user|developer)[-_ ]?(reminder|prompt|message)?\s*>",
        r"(developer|system) (message|instructions?)",
        r"(begin|end) (of )?(untrusted|trusted) (website )?content",
    )
]


class SecurityError(ValueError):
    pass


# --- run ids / paths -----------------------------------------------------------------


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id or ""):
        raise SecurityError(f"invalid run id {run_id!r}")
    return run_id


def safe_child(root: Path, *parts: str) -> Path:
    """Join `parts` under `root` and assert the result stays inside `root`."""
    root_resolved = root.resolve()
    candidate = root_resolved.joinpath(*parts).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise SecurityError(f"path escapes runs directory: {candidate}")
    return candidate


def url_cache_name(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8"), usedforsecurity=False).hexdigest() + ".json"


def atomic_write_text(path: Path, text: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# --- terminal safety -----------------------------------------------------------------


def strip_control(text: str) -> str:
    text = _ANSI_RE.sub("", text or "")
    return _CONTROL_RE.sub("", text)


def safe_text(text: str, max_len: int | None = None) -> str:
    """Untrusted text → safe for Rich printing (no markup, no control chars)."""
    cleaned = strip_control(text)
    if max_len is not None and len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 1] + "…"
    return rich_escape(cleaned)


def sanitize_answer(text: str) -> str:
    cleaned = strip_control(text or "").strip()
    return cleaned[:MAX_ANSWER_CHARS]


# --- evidence verification -----------------------------------------------------------


def normalize_for_match(text: str) -> str:
    """Markdown-insensitive normalization: images removed, links reduced to their text,
    brackets/emphasis/headings/list markers dropped, whitespace collapsed, case folded."""
    text = _MD_IMAGE_RE.sub(" ", text or "")
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _LIST_MARKER_RE.sub("", text)
    text = _MD_NOISE_RE.sub("", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return _WS_RE.sub(" ", text).strip().lower()


def excerpt_in_page(excerpt: str, page_text: str) -> bool:
    """True if `excerpt` is a verbatim (whitespace/markdown-normalized) substring of the page."""
    needle = normalize_for_match(excerpt)
    # Below 8 normalized chars nearly anything is a substring of something; this guards direct
    # callers such as note_conflict, the tools enforce the larger MIN_EXCERPT_CHARS themselves.
    if len(needle) < 8:
        return False
    return needle in normalize_for_match(page_text)


def detect_injection(text: str) -> list[str]:
    hits = []
    for pat in INJECTION_PATTERNS:
        if pat.search(text or ""):
            hits.append(pat.pattern)
    return hits
