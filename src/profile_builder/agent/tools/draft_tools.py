"""Draft tools: `save_profile_draft`, `apply_profile_updates`, `note_conflict` and
`finalize_profile`. Validation, merge, evidence and export logic live in `agent.drafting`,
`agent.evidence` and `agent.finalize`."""

from __future__ import annotations

import copy
from typing import Annotated, Any

from langchain.tools import InjectedToolCallId, tool
from pydantic import ValidationError

from profile_builder.agent.context import ToolContext
from profile_builder.agent.drafting import (
    commit_draft,
    format_validation_error,
    gap_payload,
    merge_with_previous_draft,
    payload_key,
    repair_or_fail,
    validate_brain,
)
from profile_builder.agent.evidence import (
    descends,
    reject,
    supports,
    verify_website_evidence,
)
from profile_builder.agent.finalize import (
    finalize,
    finalize_refusal,
)
from profile_builder.agent.results import fail, ok, to_json
from profile_builder.logging_setup import event, get_logger
from profile_builder.schema import (
    CompanyBrain,
    FieldPathError,
    base_of,
    get_by_path,
    parse_field_path,
    set_by_path,
    strip_unknown_keys,
)
from profile_builder.security import excerpt_in_page
from profile_builder.web.url_guard import normalize_url

log = get_logger("tools")
NO_DRAFT_ERROR = "no draft yet; call save_profile_draft first"


def make_draft_tools(ctx: ToolContext) -> list[Any]:
    store = ctx.store

    @tool
    def save_profile_draft(
        profile: dict[str, Any],
        evidence: list[dict[str, Any]],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> str:
        """Save the full draft profile (exact company_brain shape) plus evidence. Each evidence
        item is {"field_path": "...", "source_url": "...", "excerpt": "<verbatim text from that
        page>"}; cite list items individually ("customer.buyers[0]"): an excerpt given for a
        whole list only grounds the items it mentions. Unverifiable evidence is rejected and
        listed back. Returns the validation result, grounding stats and the prioritized gaps
        to interview about."""
        ctx.set_stage("draft")
        data = copy.deepcopy(profile) if isinstance(profile, dict) else {}
        data, unknown = strip_unknown_keys(data)
        if unknown:
            ctx.warn(
                "UNKNOWN_KEYS_IGNORED",
                f"draft contained keys outside the contract; ignored: {', '.join(unknown[:8])}",
            )
        try:
            if isinstance(data.get("company"), dict) and not data["company"].get("website_url"):
                data["company"]["website_url"] = ctx.start_url
            clean = validate_brain(data)
        except ValidationError as exc:
            return repair_or_fail(
                ctx,
                format_validation_error(exc),
                "profile draft",
                call_key=payload_key(tool_call_id, profile),
            )
        merged = merge_with_previous_draft(ctx, clean)
        clean, user_bases = merged.profile, merged.user_bases
        accepted, rejected = verify_website_evidence(ctx, clean, evidence)
        # Website evidence never attaches to a field the user answered.
        protected_rows = [a for a in accepted if base_of(a["field_path"]) in user_bases]
        accepted = [a for a in accepted if base_of(a["field_path"]) not in user_bases]
        rejected.extend(
            reject(
                a["field_path"],
                "USER_ANSWER_PROTECTED",
                "this field was set by a user answer; website evidence cannot replace it",
            )
            for a in protected_rows
        )
        # Replace evidence only where the value changed or fresh evidence was supplied, so an
        # unchanged field never loses its grounding because the model omitted the evidence.
        replace_bases = (merged.changed | {base_of(a["field_path"]) for a in accepted}) - user_bases
        store.replace_website_evidence(accepted, only_fields=replace_bases)
        for r in rejected:
            ctx.warn(
                "EVIDENCE_REJECTED",
                f"{r['field_path']}: {r['reason']}",
                field_path=r["field_path"],
                reason_code=r.get("reason_code"),
            )
        version, payload = commit_draft(
            ctx,
            clean,
            "draft",
            describe=lambda v, p: (
                f"draft v{v} saved ({len(accepted)} evidence rows, {len(rejected)} rejected, "
                f"{len(p['gaps'])} gaps)"
            ),
        )
        return ok(
            version=version,
            evidence_accepted=len(accepted),
            evidence_rejected=rejected,
            kept_from_previous_draft=merged.kept_from_previous,
            protected_by_user_answers=merged.protected_by_user,
            ignored_unknown_keys=unknown,
            **payload,
            next="ask focused questions for the top gaps/conflicts, or finalize_profile if none are worth asking",
        )

    @tool
    def apply_profile_updates(
        updates: list[dict[str, Any]], tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> str:
        """Apply targeted changes to the saved draft. Each update is {"field_path": "...",
        "value": <string, list of strings, or feature object>, "evidence": {"kind": "interview",
        "question_id": "<qid>"} or {"kind": "website", "source_url": "...", "excerpt": "<verbatim>"}}.
        Use list index "[n]" to replace an item or "[len]" to append. A user answer that
        contradicts the website supersedes it (the original evidence is kept and a warning is
        recorded). Updates without valid evidence are rejected individually; the rest apply."""
        ctx.set_stage("draft")
        profile = store.latest_profile()
        if profile is None:
            return fail(NO_DRAFT_ERROR, code="NO_DRAFT")
        data = copy.deepcopy(profile)
        applied: list[str] = []
        rejected: list[dict[str, Any]] = []
        pending_evidence: list[dict[str, Any]] = []
        user_bases = store.interview_bases()
        for upd in updates if isinstance(updates, list) else []:
            if not isinstance(upd, dict):
                rejected.append(reject("", "BAD_PATH", "each update must be an object"))
                continue
            path = str(upd.get("field_path", "")).strip()
            ev = upd.get("evidence")
            try:
                parse_field_path(path)
            except FieldPathError as exc:
                rejected.append(reject(path, "BAD_PATH", str(exc)))
                continue
            if not isinstance(ev, dict):
                rejected.append(
                    reject(
                        path,
                        "BAD_EVIDENCE_KIND",
                        'evidence must be an object: {"kind": "interview", "question_id": ...} '
                        'or {"kind": "website", "source_url": ..., "excerpt": ...}',
                    )
                )
                continue
            kind = str(ev.get("kind", "")).lower()
            q: dict[str, Any] | None = None
            if kind == "interview":
                q = store.get_question(str(ev.get("question_id", "")))
                if not q or q["status"] not in ("answered", "preset"):
                    rejected.append(
                        reject(
                            path,
                            "QUESTION_NOT_ANSWERED",
                            "evidence must reference an answered question id",
                        )
                    )
                    continue
                covered = {base_of(fp) for fp in q["field_paths"]}
                if base_of(path) not in covered and q["kind"] != "product_selection":
                    rejected.append(
                        reject(
                            path,
                            "QUESTION_SCOPE",
                            f"question {q['qid']} did not cover this field (it covered {sorted(covered)})",
                        )
                    )
                    continue
                if not supports(path, upd.get("value"), q["answer"] or ""):
                    rejected.append(
                        reject(path, "ANSWER_MISMATCH", "value does not reflect the user's answer")
                    )
                    continue
            elif kind == "website":
                if base_of(path) in user_bases:
                    rejected.append(
                        reject(
                            path,
                            "USER_ANSWER_PROTECTED",
                            "this field was set by a user answer; website evidence cannot "
                            "override it (ask the user instead)",
                        )
                    )
                    continue
            else:
                rejected.append(
                    reject(
                        path, "BAD_EVIDENCE_KIND", "evidence.kind must be 'interview' or 'website'"
                    )
                )
                continue
            # Every update is applied on a snapshot: a rejected one (bad evidence, invalid
            # value) is rolled back completely, so an appended item never lingers as a
            # placeholder and never costs a repair attempt.
            snapshot = copy.deepcopy(data)
            old = get_by_path(data, path)
            try:
                set_by_path(data, path, upd.get("value"))
                CompanyBrain.model_validate(data)
            except FieldPathError as exc:
                data = snapshot
                rejected.append(reject(path, "BAD_PATH", str(exc)))
                continue
            except ValidationError as exc:
                data = snapshot
                rejected.append(
                    reject(
                        path,
                        "INVALID_VALUE",
                        "value is invalid: " + "; ".join(format_validation_error(exc)[:3]),
                    )
                )
                continue
            if kind == "website":
                acc, rej = verify_website_evidence(
                    ctx,
                    data,
                    [
                        {
                            "field_path": path,
                            "source_url": ev.get("source_url"),
                            "excerpt": ev.get("excerpt"),
                        }
                    ],
                )
                if not acc:
                    data = snapshot
                    rejected.append(rej[0])
                    continue
                rejected.extend(rej)  # list items the excerpt did not cover
                pending_evidence.append(
                    {
                        "kind": "website",
                        "field_path": path,
                        "old": old,
                        "value": upd.get("value"),
                        "rows": acc,
                    }
                )
            else:
                assert q is not None
                pending_evidence.append(
                    {
                        "kind": "interview",
                        "field_path": path,
                        "question_id": q["qid"],
                        "answer": q["answer"],
                        "old": old,
                        "value": upd.get("value"),
                    }
                )
            applied.append(path)
        if not applied:
            return to_json(
                {"ok": False, "applied": [], "rejected": rejected, **gap_payload(ctx, data)}
            )
        try:
            clean = validate_brain(data)
        except ValidationError as exc:  # pragma: no cover - every update was validated above
            return repair_or_fail(
                ctx,
                format_validation_error(exc),
                "profile update",
                call_key=payload_key(tool_call_id, updates),
            )
        for ev in pending_evidence:
            path = ev["field_path"]
            old, value = ev.get("old"), ev.get("value")
            changed = old not in ("", [], None) and old != value
            stale = store.supersede_evidence_tree(path) if changed else {}
            if ev["kind"] == "website":
                for row in ev["rows"]:
                    store.add_evidence(
                        row["field_path"],
                        "website",
                        source_url=row["source_url"],
                        excerpt=row["excerpt"],
                    )
            else:
                if stale.get("website"):
                    ctx.warn(
                        "USER_CORRECTION_SUPERSEDES_SITE",
                        f"{path}: user answer replaced the website claim (original evidence preserved)",
                    )
                # A whole-list answer grounds every item: record evidence per item so
                # grounding, omission and re-indexing work at item level.
                _base, index, _sub = parse_field_path(path)
                new_val = get_by_path(clean, path)
                rows = (
                    [f"{path}[{i}]" for i in range(len(new_val))]
                    if index is None and isinstance(new_val, list)
                    else [path]
                )
                for row_path in rows:
                    store.add_evidence(
                        row_path, "interview", question_id=ev["question_id"], answer=ev["answer"]
                    )
                for c in store.list_conflicts("open"):
                    if descends(c["field_path"], path):
                        store.resolve_conflict(
                            c["field_path"], f"user answer (q {ev['question_id']}): {ev['answer']}"
                        )
        version, payload = commit_draft(
            ctx,
            clean,
            "update",
            describe=lambda v, _p: f"draft v{v} updated ({len(applied)} fields)",
            count=len(applied),
        )
        return ok(version=version, applied=applied, rejected=rejected, **payload)

    @tool
    def note_conflict(field_path: str, claims: list[dict[str, Any]], summary: str) -> str:
        """Record that two sources disagree about a field, e.g. the homepage targets startups
        while the customer page shows only banks. `claims` is a list of {"source_url", "excerpt"
        (verbatim), "claim"}. Then ask the user a targeted question if it matters; unresolved
        conflicts are omitted from the final profile and reported as warnings."""
        try:
            parse_field_path(field_path)
        except FieldPathError as exc:
            return fail(str(exc), code="BAD_PATH")
        fetched = store.fetched_urls()
        verified: list[dict[str, Any]] = []
        for c in claims if isinstance(claims, list) else []:
            if not isinstance(c, dict):
                continue
            try:
                src = normalize_url(str(c.get("source_url", "")))
            except ValueError:
                src = ""
            text = ctx.page_text(src) if src in fetched else None
            if text is None or not excerpt_in_page(str(c.get("excerpt", "")), text):
                continue
            verified.append(
                {
                    "source_url": src,
                    "excerpt": str(c.get("excerpt", "")),
                    "claim": str(c.get("claim", "")),
                }
            )
        if len(verified) < 2:
            return fail(
                "a conflict needs at least two verifiable claims from fetched pages",
                code="INSUFFICIENT_CLAIMS",
            )
        store.upsert_conflict(field_path.strip(), verified, summary)
        ctx.warn("CONFLICT_RECORDED", f"{field_path}: {summary}")
        event(log, "conflict_noted", f"conflict on {field_path}", code=field_path)
        return ok(
            field_path=field_path,
            claims=len(verified),
            next="ask the user a targeted question with kind='conflict' if this affects the profile",
        )

    @tool
    def finalize_profile() -> str:
        """Validate the latest draft, omit disputed unresolved claims, write company_brain.json
        plus evidence.json and report.md, and return the run summary. Call once at the end."""
        profile = store.latest_profile()
        if profile is None:
            return fail(NO_DRAFT_ERROR, code="NO_DRAFT")
        refusal = finalize_refusal(ctx, profile)
        if refusal is not None:
            return to_json(refusal)
        ctx.set_stage("finalize")
        result = finalize(ctx)
        ctx.set_stage("done")
        return to_json(result)

    return [save_profile_draft, apply_profile_updates, note_conflict, finalize_profile]
