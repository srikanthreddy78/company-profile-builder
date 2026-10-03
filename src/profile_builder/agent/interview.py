"""Interview helpers: question identity, answer classification and the result payload
`ask_user` returns. Pure functions; the tool owns the interrupt and the store writes."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from profile_builder.config import MAX_QUESTION_MARKS

SKIP_WORDS = frozenset({"skip", "s", "pass", "next"})
UNKNOWN_WORDS = frozenset(
    {
        "idk",
        "i don't know",
        "i dont know",
        "dont know",
        "don't know",
        "unknown",
        "not sure",
        "no idea",
        "?",
    }
)
QUESTION_KINDS = ("product_selection", "gap", "conflict", "brand")
_NUMBERED_ITEM_RE = re.compile(r"(?m)^\s*(?:\(?\d{1,2}[.)]|[a-d][.)]|[-•*])\s+\S")


def is_multi_question(text: str) -> bool:
    """True when a single ask_user call tries to bundle several questions."""
    numbered = len(_NUMBERED_ITEM_RE.findall(text or ""))
    return numbered >= 2 or (text or "").count("?") > MAX_QUESTION_MARKS


def question_id(run_id: str, kind: str, paths: list[str], question: str) -> str:
    """Stable id for a question: the same question re-asked after a crash maps to the
    same record, so an answer is never re-asked or counted twice."""
    normalized_q = " ".join((question or "").split()).lower()
    return hashlib.sha1(
        f"{run_id}|{kind}|{','.join(sorted(paths))}|{normalized_q}".encode(),
        usedforsecurity=False,
    ).hexdigest()[:12]


def classify_answer(answer: str) -> tuple[str, str | None]:
    """(status, stored_answer): skipped / unknown answers are recorded without text."""
    lowered = answer.lower().strip(" .!")
    if not answer or lowered in SKIP_WORDS:
        return "skipped", None
    if lowered in UNKNOWN_WORDS:
        return "unknown", None
    return "answered", answer


def answered_payload(
    qid: str, status: str, answer: str | None, remaining: int, *, note: str | None = None
) -> dict[str, Any]:
    """What the model sees for a committed question: the answer text, or SKIPPED/UNKNOWN."""
    payload: dict[str, Any] = {
        "ok": True,
        "qid": qid,
        "status": status,
        "answer": answer if status == "answered" else status.upper(),
        "questions_remaining": remaining,
    }
    if note:
        payload["note"] = note
    return payload
