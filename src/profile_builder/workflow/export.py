"""Write the run outputs: company_brain.json (contract only), evidence.json (everything
else: sources, questions, warnings, conflicts, usage) and report.md (human summary)."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
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
from profile_builder.security import strip_control
from profile_builder.state.run_store import RunStore
from profile_builder.workflow.gaps import empty_gaps, grounding_report, section_coverage


def _md(text: object, limit: int = 400) -> str:
    """Untrusted (model- or web-derived) text → safe inline Markdown."""
    t = strip_control(str(text or "")).replace("\r", " ").replace("\n", " ")
    t = t.replace("<", "&lt;").replace(">", "&gt;").replace("`", "'").replace("|", "¦")
    t = t.replace("[", "［").replace("]", "］")
    return (t[: limit - 1] + "…") if len(t) > limit else t


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
        # True when no page could be fetched: every exported claim then rests on the
        # interview alone (the run is always labeled partial in that case).
        "interview_only": not store.fetched_urls(),
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
        f"# Company profile report — {_md(profile['company'].get('name') or run.start_url)}",
        "",
        f"- Run: `{run.run_id}`  ·  Status: **{status}**  ·  Output: `{output_path}`",
        f"- Website: {_md(run.start_url)}  ·  Product focus: {_md(run.product_focus or '(not set)')}",
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
        extra = f" — {_md(p.error_code)}: {_md(p.error)}" if p.error_code else ""
        lines.append(f"- [{p.status}] {_md(p.url)} ({p.char_count} chars){extra}")
    lines += ["", "## Interview", ""]
    questions = store.list_questions()
    if not questions:
        lines.append("No questions were asked.")
    for q in questions:
        lines.append(f"{q['ordinal']}. **{_md(q['question'])}**  ")
        lines.append(f"   _Why:_ {_md(q['why_unclear'] or '')}  ")
        lines.append(f"   _Answer ({q['status']}):_ {_md(q['answer'] or '—')}")
    gaps = empty_gaps(profile)
    lines += ["", "## Remaining gaps", ""]
    if not gaps:
        lines.append("None — every contract field is populated.")
    for g in gaps[:25]:
        lines.append(f"- `{g.field_path}` — {_md(g.reason)}")
    conflicts = store.list_conflicts()
    if conflicts:
        lines += ["", "## Conflicts", ""]
        for c in conflicts:
            lines.append(
                f"- `{c['field_path']}` [{c['status']}] {_md(c.get('summary') or '')}"
                + (f" → {_md(c['resolution'])}" if c.get("resolution") else "")
            )
    warnings = store.list_warnings()
    lines += ["", "## Warnings", ""]
    if not warnings:
        lines.append("None.")
    for w in warnings:
        lines.append(f"- `{_md(w['code'], 60)}` {_md(w['message'])}")
    return "\n".join(lines) + "\n"


def _unlink_quietly(path: Path | str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _stage(target: Path, text: str, mode: int) -> Path:
    """Write `text` to a temp file next to `target` (fsynced, with the final mode)."""
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=str(target.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, mode)
    except Exception:
        _unlink_quietly(tmp)
        raise
    return Path(tmp)


def write_all_or_nothing(artifacts: list[tuple[Path, str, int]]) -> None:
    """Publish several files as one unit: every artifact is staged to a temp file first,
    then renamed into place in sequence. If anything fails, the outputs already renamed are
    rolled back (the previous version is restored from a backup copy when one existed,
    otherwise the file is removed) and no temp or backup file is left behind."""
    staged: list[tuple[Path, Path]] = []
    published: list[tuple[Path, Path | None]] = []  # (target, backup of the previous version)
    try:
        for target, text, mode in artifacts:
            target.parent.mkdir(parents=True, exist_ok=True)
            staged.append((target, _stage(target, text, mode)))
        for target, tmp in staged:
            backup: Path | None = None
            if target.exists():
                fd, name = tempfile.mkstemp(prefix=".tmp-bak-", dir=str(target.parent))
                os.close(fd)
                backup = Path(name)
                shutil.copy2(target, backup)
            try:
                os.replace(tmp, target)
            except BaseException:
                # the target itself is untouched; only its backup copy needs to go
                if backup is not None:
                    _unlink_quietly(backup)
                raise
            published.append((target, backup))
    except BaseException:
        for target, backup in reversed(published):
            if backup is not None:
                os.replace(backup, target)
            else:
                _unlink_quietly(target)
        for _target, tmp in staged:
            _unlink_quietly(tmp)
        raise
    for _target, backup in published:
        if backup is not None:
            _unlink_quietly(backup)


def write_outputs(run_dir: Path, store: RunStore, profile: dict[str, Any], status: str) -> Path:
    """Validate once more and write all three artifacts as one unit: either every output
    is replaced or (on any failure) the previous set is left exactly as it was. Raises on an
    invalid profile."""
    brain = CompanyBrain.model_validate(profile)
    output_path = run_dir / OUTPUT_FILENAME
    dumped = brain.model_dump(mode="json")
    evidence_doc = build_evidence_document(store, dumped, status)
    # evidence.json and report.md contain interview answers → private; the profile is public-derived.
    write_all_or_nothing(
        [
            (output_path, brain.to_json(), 0o644),
            (
                run_dir / EVIDENCE_FILENAME,
                json.dumps(evidence_doc, indent=2, ensure_ascii=False) + "\n",
                0o600,
            ),
            (run_dir / REPORT_FILENAME, build_report(store, dumped, status, output_path), 0o600),
        ]
    )
    return output_path
