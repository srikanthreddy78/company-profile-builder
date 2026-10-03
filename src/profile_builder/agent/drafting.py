"""Draft lifecycle shared by `save_profile_draft` and `apply_profile_updates`: contract
validation with a bounded repair loop, merging with the previous draft, thin-draft advice,
the gap payload the model sees, and the commit tail (save + event)."""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from profile_builder.agent.context import FatalProfileError, ToolContext
from profile_builder.agent.prompts import render_thin_draft_advice
from profile_builder.agent.results import to_json
from profile_builder.config import (
    FINALIZE_GUARD_SECTIONS,
    MAX_REPAIR_ATTEMPTS,
    THIN_DRAFT_SECTION_FIELDS,
)
from profile_builder.logging_setup import event, get_logger
from profile_builder.schema import MERGEABLE_BASES, CompanyBrain, get_by_path, set_by_path
from profile_builder.workflow.export import evidence_paths
from profile_builder.workflow.gaps import (
    grounding_report,
    prioritize_for_interview,
    section_coverage,
)

log = get_logger("tools")

REPAIR_KEY_META = "repair_last_key"  # tool call + payload that last failed validation


def format_validation_error(exc: ValidationError) -> list[str]:
    out = []
    for err in exc.errors()[:12]:
        loc = ".".join(str(p) for p in err.get("loc", ()))
        out.append(f"{loc}: {err.get('msg')}")
    return out


def payload_key(tool_call_id: str, payload: Any) -> str:
    """Identity of one tool call: a crash replay re-runs the same call id with the same
    payload and must not count as a second repair attempt; a fresh call does."""
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    return f"{tool_call_id}:{digest}"


def validate_brain(data: dict[str, Any]) -> dict[str, Any]:
    """Validate against the contract and return the canonical JSON form (raises
    `ValidationError`)."""
    return CompanyBrain.model_validate(data).model_dump(mode="json")


def repair_or_fail(ctx: ToolContext, errors: list[str], what: str, *, call_key: str) -> str:
    store = ctx.store
    if store.get_meta(REPAIR_KEY_META) == call_key:
        attempts = store.get_counter("repair_attempts")  # crash replay of the same call
    else:
        attempts = store.increment_counter("repair_attempts")
        store.set_meta(REPAIR_KEY_META, call_key)
    event(
        log,
        "repair_attempt",
        f"{what} invalid (attempt {attempts}): " + " ; ".join(errors[:4])[:600],
        level=logging.WARNING,
        attempt=attempts,
        count=len(errors),
    )
    if attempts > MAX_REPAIR_ATTEMPTS:
        ctx.warn(
            "INVALID_MODEL_OUTPUT",
            f"{what} was still invalid after {MAX_REPAIR_ATTEMPTS} repair attempt(s)",
        )
        raise FatalProfileError(f"{what} failed validation twice: {'; '.join(errors[:5])}")
    return to_json(
        {
            "ok": False,
            "error_code": "INVALID_PROFILE",
            "errors": errors,
            "repair_attempts_remaining": MAX_REPAIR_ATTEMPTS - attempts + 1,
            "hint": "fix the listed fields and call again with the full corrected input",
        }
    )


def reset_repairs(ctx: ToolContext) -> None:
    ctx.store.set_counter("repair_attempts", 0)
    ctx.store.set_meta(REPAIR_KEY_META, "")


@dataclass
class MergeResult:
    profile: dict[str, Any]
    kept_from_previous: list[str] = field(default_factory=list)
    protected_by_user: list[str] = field(default_factory=list)
    changed: set[str] = field(default_factory=set)
    user_bases: set[str] = field(default_factory=set)


def merge_with_previous_draft(ctx: ToolContext, clean: dict[str, Any]) -> MergeResult:
    """A later draft may never erase a field that an earlier one filled (the model sometimes
    re-sends a thinner profile after the interview), and an explicit user answer always wins
    over a later website-based draft. `changed` lists the bases whose value differs from the
    previous draft (every base when there is none)."""
    result = MergeResult(profile=clean, user_bases=ctx.store.interview_bases())
    previous = ctx.store.latest_draft()
    if previous is None:
        result.changed = set(MERGEABLE_BASES)
        return result
    _, prev = previous
    for base in MERGEABLE_BASES:
        new_val, old_val = get_by_path(clean, base), get_by_path(prev, base)
        if base in result.user_bases and new_val != old_val:
            set_by_path(clean, base, copy.deepcopy(old_val))
            result.protected_by_user.append(base)
        elif new_val in ("", [], None) and old_val not in ("", [], None):
            set_by_path(clean, base, copy.deepcopy(old_val))
            result.kept_from_previous.append(base)
    result.profile = validate_brain(clean)
    result.changed = {
        base
        for base in MERGEABLE_BASES
        if base not in result.user_bases
        and get_by_path(result.profile, base) != get_by_path(prev, base)
    }
    return result


def thin_draft_advice(
    ctx: ToolContext, profile: dict[str, Any], *, strict: bool = False
) -> str | None:
    """Push back when the draft leaves sections empty although pages are indexed: the
    interview must not be used for facts the website already establishes."""
    fetched = ctx.store.fetched_urls()
    if not fetched:
        return None
    threshold = 0 if strict else THIN_DRAFT_SECTION_FIELDS
    sections = FINALIZE_GUARD_SECTIONS if strict else None
    thin = [
        sec
        for sec, (filled, _total) in section_coverage(profile).items()
        if filled <= threshold and sec != "company" and (sections is None or sec in sections)
    ]
    if not thin:
        return None
    return render_thin_draft_advice(thin, len(fetched))


def gap_payload(ctx: ToolContext, profile: dict[str, Any]) -> dict[str, Any]:
    store = ctx.store
    asked = {p for q in store.list_questions() for p in q["field_paths"]}
    grounding = grounding_report(profile, evidence_paths(store))
    gaps = prioritize_for_interview(
        profile,
        open_conflicts=store.list_conflicts("open"),
        asked_paths=asked,
        ungrounded=grounding["ungrounded"],
    )
    advice = thin_draft_advice(ctx, profile)
    return {
        **({"advice": advice} if advice else {}),
        "grounding": {
            "grounded": grounding["grounded"],
            "populated": grounding["populated"],
            "ungrounded": grounding["ungrounded"][:10],
        },
        "gaps": gaps,
        "questions_remaining": max(0, ctx.settings.max_questions - store.questions_asked()),
    }


def commit_draft(
    ctx: ToolContext,
    clean: dict[str, Any],
    source: str,
    *,
    describe: Callable[[int, dict[str, Any]], str],
    count: int | None = None,
) -> tuple[int, dict[str, Any]]:
    """Shared tail of a successful draft/update: the repair counter resets, the draft is
    saved, and the gap payload + `draft_saved` event are produced. `count` defaults to the
    number of gaps."""
    reset_repairs(ctx)
    version = ctx.store.save_draft(clean, source)
    payload = gap_payload(ctx, clean)
    event(
        log,
        "draft_saved",
        describe(version, payload),
        version=version,
        count=len(payload["gaps"]) if count is None else count,
    )
    return version, payload
