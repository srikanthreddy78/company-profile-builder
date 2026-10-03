"""Finalization: omit disputed/ungrounded values, validate, write the outputs and record
the result. Also used by the runner for best-effort partial exports.

Reaching the page/question caps is normal bounded behavior; only being cut off before the
agent could finish (model-call cap, budget, forced stop) makes a profile "partial". The
runner passes forced_partial for the model-call cap; the budget is re-checked live so a
resumed run with a raised budget can complete.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import ValidationError

from profile_builder.agent.context import FatalProfileError, ToolContext
from profile_builder.agent.drafting import format_validation_error, thin_draft_advice
from profile_builder.agent.evidence import ungrounded_paths
from profile_builder.agent.prompts import ungrounded_hint
from profile_builder.config import MAX_FINALIZE_REFUSALS
from profile_builder.logging_setup import event, get_logger
from profile_builder.schema import (
    FEATURE_FIELDS,
    FEATURE_LIST_PATH,
    CompanyBrain,
    FieldPathError,
    get_by_path,
    parse_field_path,
    set_by_path,
)
from profile_builder.state.run_store import RunStore, shift_path_set
from profile_builder.workflow.export import evidence_paths, write_outputs
from profile_builder.workflow.gaps import grounding_report, prioritize_for_interview

log = get_logger("tools")


def no_usable_pages_message(store: RunStore) -> tuple[str, dict[str, Any]]:
    """Warning text + details for a run in which no page could be fetched."""
    attempted = [{"url": p.url, "error_code": p.error_code or p.status} for p in store.list_pages()]
    detail = (
        "; ".join(f"{p['url']} ({p['error_code']})" for p in attempted[:10])
        or "no page was attempted"
    )
    return f"no usable pages were fetched: {detail}", {"pages": attempted}


def finalize_refusal(ctx: ToolContext, profile: dict[str, Any]) -> dict[str, Any] | None:
    """`finalize_profile` pushes back (at most MAX_FINALIZE_REFUSALS times) when guarded
    sections are empty or populated fields are ungrounded. Returns the NOT_READY payload,
    or None when the export may proceed."""
    store = ctx.store
    if store.get_counter("finalize_refusals") >= MAX_FINALIZE_REFUSALS:
        return None
    advice = thin_draft_advice(ctx, profile, strict=True)
    ungrounded = ungrounded_paths(store, profile)
    if not advice and not ungrounded:
        return None
    store.increment_counter("finalize_refusals")
    return {
        "ok": False,
        "error": "profile not ready to export",
        "error_code": "NOT_READY",
        **({"advice": advice} if advice else {}),
        **(
            {"ungrounded": ungrounded[:20], "ungrounded_hint": ungrounded_hint()}
            if ungrounded
            else {}
        ),
    }


def omit_many(
    store: RunStore | None, data: dict[str, Any], paths: list[str]
) -> list[tuple[str, int]]:
    """Remove the values at `paths` from the profile dict, deterministically:

    1. whole fields are blanked ("" / []);
    2. feature sub-fields are blanked, except when the whole item (or its name) is omitted
       too, in which case the item goes as a unit;
    3. list items are deleted highest index first, and feature objects that became empty are
       deleted the same way, so evidence/conflict paths can be renumbered consistently.

    Returns the (base, removed_index) deletions in the order applied. They are applied to
    `store` immediately when one is given; `finalize` passes None and applies them later in
    one transaction after validation and the output write succeeded.
    """
    shifts: list[tuple[str, int]] = []
    deletions: set[tuple[str, int]] = set()
    sub_blanks: list[tuple[str, int, str]] = []
    clears: list[str] = []
    for p in sorted(set(paths)):
        base, index, sub = parse_field_path(p)
        if index is None:
            clears.append(p)
        elif sub is None or sub == "name":
            deletions.add((base, index))
        else:
            sub_blanks.append((base, index, sub))
    for p in clears:
        current = get_by_path(data, p)
        if current in ("", [], None):
            continue
        set_by_path(data, p, [] if isinstance(current, list) else "")
    for base, index, sub in sub_blanks:
        if (base, index) in deletions:
            continue
        item = get_by_path(data, f"{base}[{index}]")
        if isinstance(item, dict):
            item[sub] = ""
    for i, feat in enumerate(get_by_path(data, FEATURE_LIST_PATH) or []):
        if isinstance(feat, dict) and not any(feat.get(k) for k in FEATURE_FIELDS):
            deletions.add((FEATURE_LIST_PATH, i))
    by_base: dict[str, set[int]] = {}
    for base, index in deletions:
        by_base.setdefault(base, set()).add(index)
    for base in sorted(by_base):
        lst = get_by_path(data, base)
        if not isinstance(lst, list):
            continue
        for index in sorted(by_base[base], reverse=True):
            if index >= len(lst):
                continue
            del lst[index]
            shifts.append((base, index))
            if store is not None:
                store.shift_list_paths(base, index)
    return shifts


def _omit_and_shift(
    data: dict[str, Any], paths: list[str], ev_paths: set[str]
) -> tuple[list[tuple[str, int]], set[str]]:
    """Omit `paths` on the in-memory copy and renumber the in-memory evidence paths."""
    shifts = omit_many(None, data, paths)
    for base, index in shifts:
        ev_paths = shift_path_set(ev_paths, base, index)
    return shifts, ev_paths


def finalize(
    ctx: ToolContext, *, forced_partial: bool = False, keep_status: bool = False
) -> dict[str, Any]:
    """Omit disputed/ungrounded values, validate, write the outputs and record the result.

    All omissions are computed on a copy with an in-memory view of the evidence paths, so a
    validation failure leaves the store untouched. The store mutations (warnings, conflict
    statuses, evidence re-indexing, finalize draft, status) and the output write then run in
    one transaction: a failed write rolls everything back and a re-export recomputes the
    same omissions from the same state. `keep_status` exports an unfinished run (paused /
    interrupted / running) without changing its status.
    """
    store = ctx.store
    latest = store.latest_draft()
    if latest is None:
        raise FatalProfileError("no valid draft exists; nothing to export")
    _, profile = latest
    data = copy.deepcopy(profile)
    ev_paths = evidence_paths(store)
    warnings: list[tuple[str, str, dict[str, Any] | None]] = []
    conflict_updates: list[tuple[str, str, str]] = []

    # 1. Unresolved conflicts: omit the disputed value once and mark the conflict so a later
    #    re-export does not delete a different item that moved into the same index.
    to_omit: list[str] = []
    for c in store.list_conflicts("open"):
        path = c["field_path"]
        try:
            current = get_by_path(data, path)
        except (FieldPathError, KeyError, TypeError):
            current = None
        if current in ("", [], None):
            conflict_updates.append((path, "omitted", "value was already empty at export"))
            continue
        warnings.append(
            (
                "CONFLICT_UNRESOLVED_OMITTED",
                f"{path}: conflicting evidence was not resolved; value omitted",
                None,
            )
        )
        conflict_updates.append((path, "omitted", "omitted from the export (unresolved)"))
        to_omit.append(path)
    shifts, ev_paths = _omit_and_shift(data, to_omit, ev_paths)

    # 2. Grounding: a populated field without accepted evidence is not exported.
    ungrounded = list(grounding_report(data, ev_paths)["ungrounded"])
    if ungrounded:
        warnings.append(
            (
                "UNGROUNDED_OMITTED",
                f"{len(ungrounded)} populated field(s) had no accepted evidence and were omitted: "
                + ", ".join(ungrounded[:12]),
                None,
            )
        )
        more, ev_paths = _omit_and_shift(data, ungrounded, ev_paths)
        shifts.extend(more)

    try:
        brain = CompanyBrain.model_validate(data)
    except ValidationError as exc:
        raise FatalProfileError(
            "final profile failed validation: " + "; ".join(format_validation_error(exc))
        ) from exc
    clean = brain.model_dump(mode="json")
    budget = ctx.settings.budget_usd
    over_budget = budget is not None and store.total_cost() >= budget
    no_pages = not store.fetched_urls()
    if no_pages and not store.has_warning("NO_USABLE_PAGES"):
        message, details = no_usable_pages_message(store)
        warnings.append(("NO_USABLE_PAGES", message, details))
    status = "partial" if (forced_partial or over_budget or no_pages) else "complete"

    # 3. Commit: store mutations and the output write succeed or fail together.
    with store.transaction():
        for code, message, details in warnings:
            ctx.warn(code, message, **(details or {}))
        for path, cstatus, resolution in conflict_updates:
            store.set_conflict_status(path, cstatus, resolution)
        for base, index in shifts:
            store.shift_list_paths(base, index)
        output_path = write_outputs(ctx.run_dir, store, clean, status)
        store.save_draft(clean, "finalize")
        if keep_status:
            store.set_output_path(str(output_path))
        else:
            store.set_status(status, str(output_path))
    grounding = grounding_report(clean, evidence_paths(store))
    pending = store.pending_question()
    summary = {
        "ok": True,
        "status": status,
        "output_path": str(output_path),
        "grounded_fields": f"{grounding['grounded']}/{grounding['populated']}",
        "ungrounded": grounding["ungrounded"][:10],
        "gaps_remaining": [
            g["field_path"]
            for g in prioritize_for_interview(
                clean, open_conflicts=[], asked_paths=set(), ungrounded=[]
            )
        ][:10],
        "warnings": len(store.list_warnings()),
        "questions_asked": store.questions_asked(),
        "pending_question": pending["question"] if pending else None,
    }
    event(
        log,
        "run_finished",
        f"profile exported ({status}) → {output_path}",
        status=status,
        cost_usd=round(store.total_cost(), 6),
    )
    return summary
