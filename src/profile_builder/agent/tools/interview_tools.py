"""Interview tool: `ask_user`. Everything before the interrupt is idempotent so a resumed
run replays safely; the answer commit and classification live in `agent.interview`."""

from __future__ import annotations

from typing import Any

from langchain.tools import tool
from langgraph.types import interrupt

from profile_builder.agent.context import ToolContext
from profile_builder.agent.interview import (
    QUESTION_KINDS,
    answered_payload,
    classify_answer,
    is_multi_question,
    question_id,
)
from profile_builder.agent.results import fail, to_json
from profile_builder.logging_setup import event, get_logger
from profile_builder.schema import (
    FieldPathError,
    parse_field_path,
)
from profile_builder.security import sanitize_answer

log = get_logger("tools")


def make_interview_tools(ctx: ToolContext) -> list[Any]:
    settings = ctx.settings
    store = ctx.store

    def _remaining() -> int:
        return max(0, settings.max_questions - store.questions_asked())

    @tool
    def ask_user(question: str, why_unclear: str, field_paths: list[str], kind: str = "gap") -> str:
        """Ask the user ONE focused question in the terminal and wait for the answer. Explain in
        `why_unclear` what the website left ambiguous. `field_paths` lists the contract fields the
        answer resolves (e.g. ["customer.target_customer"]). `kind` is one of product_selection,
        gap, conflict, brand. Returns the answer, or SKIPPED / UNKNOWN when the user cannot
        answer. Must be the only tool call in the step."""
        kind = kind if kind in QUESTION_KINDS else "gap"
        try:
            paths = [str(p).strip() for p in (field_paths or []) if str(p).strip()]
            for p in paths:
                parse_field_path(p)
        except FieldPathError as exc:
            return fail(f"invalid field path: {exc}", code="BAD_PATH")
        if not paths and kind != "product_selection":
            return fail("field_paths must name at least one contract field", code="NO_FIELD_PATHS")
        if is_multi_question(question):
            return to_json(
                {
                    "ok": False,
                    "error_code": "MULTI_QUESTION",
                    "error": (
                        "ask ONE focused question per call (this text bundles several numbered or "
                        "question-marked items). Call ask_user again with the single most important "
                        "question; follow up separately if still useful."
                    ),
                }
            )
        qid = question_id(ctx.run_id, kind, paths, question)

        existing = store.get_question(qid)
        if existing and existing["status"] != "pending":
            # Replay after a crash: the answer is already committed; never re-ask.
            status = "answered" if existing["status"] == "preset" else existing["status"]
            return to_json(
                answered_payload(
                    qid, status, existing["answer"], _remaining(), note="answer already recorded"
                )
            )
        if kind == "product_selection" and ctx.product_focus:
            # Preset on the command line: recorded for traceability, never asked, not counted.
            store.upsert_question(qid, kind, question, why_unclear, paths)
            store.answer_question(qid, "preset", ctx.product_focus)
            return to_json(
                answered_payload(
                    qid,
                    "answered",
                    ctx.product_focus,
                    _remaining(),
                    note="product focus was preset on the command line (not counted as a question)",
                )
            )
        if store.questions_asked() >= settings.max_questions:
            if ctx.warn_once(
                "LIMIT_QUESTIONS_REACHED", f"question limit of {settings.max_questions} reached"
            ):
                event(
                    log,
                    "limit_reached",
                    "question limit reached",
                    kind="questions",
                    count=settings.max_questions,
                )
            return to_json(
                {
                    "ok": False,
                    "error_code": "QUESTION_LIMIT",
                    "error": "question limit reached; finalize the profile with the evidence you have",
                }
            )

        record = store.upsert_question(qid, kind, question, why_unclear, paths)
        ctx.set_stage("interview")
        event(
            log,
            "interrupt_raised",
            f"asking user (q{record['ordinal']}): {question[:120]}",
            qid=qid,
            count=record["ordinal"],
        )
        # --- interrupt: everything above is idempotent and re-runs safely on resume ---
        raw = interrupt(
            {
                "qid": qid,
                "question": question,
                "why_unclear": why_unclear,
                "field_paths": paths,
                "kind": kind,
                "number": record["ordinal"],
                "max": settings.max_questions,
            }
        )
        answer = sanitize_answer(str(raw if raw is not None else ""))
        status, stored_answer = classify_answer(answer)
        store.answer_question(qid, status, stored_answer)
        event(log, "answer_committed", f"q{record['ordinal']} {status}", qid=qid, status=status)
        if kind == "product_selection" and status == "answered":
            ctx.product_focus = stored_answer
            store.set_product_focus(stored_answer or "")
        return to_json(answered_payload(qid, status, stored_answer, _remaining()))

    return [ask_user]
