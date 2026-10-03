"""Write the run outputs: company_brain.json (contract only), evidence.json (everything
else: sources, questions, warnings, conflicts, usage) and report.md (human summary)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from profile_builder.config import (
    APP_VERSION,
    EVIDENCE_FILENAME,
    OUTPUT_FILENAME,
    REPORT_FILENAME,
)
from profile_builder.schema import CompanyBrain
from profile_builder.security import atomic_write_text
from profile_builder.state.run_store import RunStore
from profile_builder.workflow.gaps import empty_gaps, grounding_report, section_coverage


def evidence_paths(store: RunStore) -> set[str]:
    return {e["field_path"] for e in store.list_evidence(include_superseded=False)}


def build_evidence_document(
    store: RunStore, profile: dict[str, Any], status: str
) -> dict[str, Any]:
    run = store.get_run()
    grounding = grounding_report(profile, evidence_paths(store))
    return {
        "run_id": run.run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tool_version": APP_VERSION,
        "status": status,
        "start_url": run.start_url,
        "product_focus": run.product_focus,
        "settings": run.settings,
        "pages": [
            {
                "url": p.url,
                "final_url": p.final_url,
                "status": p.status,
                "http_status": p.http_status,
                "title": p.title,
                "chars": p.char_count,
                "error_code": p.error_code,
                "error": p.error,
            }
            for p in store.list_pages()
        ],
        "evidence": [
            {
                "field_path": e["field_path"],
                "kind": e["kind"],
                "source_url": e["source_url"],
                "excerpt": e["excerpt"],
                "question_id": e["question_id"],
                "answer": e["answer"],
                "superseded": bool(e["superseded"]),
            }
            for e in store.list_evidence()
        ],
        "questions": [
            {
                "qid": q["qid"],
                "n": q["ordinal"],
                "kind": q["kind"],
                "question": q["question"],
                "why_unclear": q["why_unclear"],
                "field_paths": q["field_paths"],
                "status": q["status"],
                "answer": q["answer"],
            }
            for q in store.list_questions()
        ],
        "conflicts": store.list_conflicts(),
        "warnings": [
            {"code": w["code"], "message": w["message"], "details": w["details"]}
            for w in store.list_warnings()
        ],
        "gaps": [g.to_dict() for g in empty_gaps(profile)],
        "grounding": grounding,
        "usage": store.usage_totals(),
    }


def build_report(store: RunStore, profile: dict[str, Any], status: str, output_path: Path) -> str:
    run = store.get_run()
    grounding = grounding_report(profile, evidence_paths(store))
    cov = section_coverage(profile)
    usage = store.usage_totals()
    lines = [
        f"# Company profile report — {profile['company'].get('name') or run.start_url}",
        "",
        f"- Run: `{run.run_id}`  ·  Status: **{status}**  ·  Output: `{output_path}`",
        f"- Website: {run.start_url}  ·  Product focus: {run.product_focus or '(not set)'}",
        f"- Model calls: {usage['model_calls']}  ·  Tokens in/out: {usage['input_tokens']}/{usage['output_tokens']}"
        f"  ·  Estimated cost: ${usage['cost_usd']:.4f}",
        "",
        "## Coverage",
        "",
        "| Section | Filled fields |",
        "|---|---|",
    ]
    for section, (filled, total) in cov.items():
        lines.append(f"| {section} | {filled}/{total} |")
    lines += [
        "",
        f"Grounding: {grounding['grounded']}/{grounding['populated']} populated fields have verified evidence.",
    ]
    if grounding["ungrounded"]:
        lines.append("Ungrounded fields: " + ", ".join(f"`{p}`" for p in grounding["ungrounded"]))
    lines += ["", "## Pages", ""]
    for p in store.list_pages():
        extra = f" — {p.error_code}: {p.error}" if p.error_code else ""
        lines.append(f"- [{p.status}] {p.url} ({p.char_count} chars){extra}")
    lines += ["", "## Interview", ""]
    questions = store.list_questions()
    if not questions:
        lines.append("No questions were asked.")
    for q in questions:
        lines.append(f"{q['ordinal']}. **{q['question']}**  ")
        lines.append(f"   _Why:_ {q['why_unclear'] or ''}  ")
        lines.append(f"   _Answer ({q['status']}):_ {q['answer'] or '—'}")
    gaps = empty_gaps(profile)
    lines += ["", "## Remaining gaps", ""]
    if not gaps:
        lines.append("None — every contract field is populated.")
    for g in gaps[:25]:
        lines.append(f"- `{g.field_path}` — {g.reason}")
    conflicts = store.list_conflicts()
    if conflicts:
        lines += ["", "## Conflicts", ""]
        for c in conflicts:
            lines.append(
                f"- `{c['field_path']}` [{c['status']}] {c.get('summary') or ''}"
                + (f" → {c['resolution']}" if c.get("resolution") else "")
            )
    warnings = store.list_warnings()
    lines += ["", "## Warnings", ""]
    if not warnings:
        lines.append("None.")
    for w in warnings:
        lines.append(f"- `{w['code']}` {w['message']}")
    return "\n".join(lines) + "\n"


def write_outputs(run_dir: Path, store: RunStore, profile: dict[str, Any], status: str) -> Path:
    """Validate once more and write all three artifacts atomically. Raises on invalid profile."""
    brain = CompanyBrain.model_validate(profile)
    output_path = run_dir / OUTPUT_FILENAME
    atomic_write_text(output_path, brain.to_json(), mode=0o644)
    evidence_doc = build_evidence_document(store, brain.model_dump(mode="json"), status)
    atomic_write_text(
        run_dir / EVIDENCE_FILENAME,
        json.dumps(evidence_doc, indent=2, ensure_ascii=False) + "\n",
        mode=0o644,
    )
    atomic_write_text(
        run_dir / REPORT_FILENAME,
        build_report(store, brain.model_dump(mode="json"), status, output_path),
        mode=0o644,
    )
    return output_path
