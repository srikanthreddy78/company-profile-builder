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
from profile_builder.logging_setup import event, get_logger
from profile_builder.schema import CompanyBrain
from profile_builder.security import SecurityError, atomic_write_text, safe_child, strip_control
from profile_builder.state.run_store import RunStore
from profile_builder.workflow.gaps import empty_gaps, grounding_report, section_coverage

log = get_logger("export")


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


PUBLISH_JOURNAL_FILENAME = ".publish-journal.json"


def _journal_paths(base: Path, entry: dict[str, Any]) -> tuple[Path, Path | None, Path | None]:
    """(target, backup, staged) of one journal entry, confined to the run directory."""
    base = base.resolve()
    target = safe_child(base, str(entry.get("target") or ""))
    if target == base:
        raise SecurityError("journal entry without a target")
    backup = safe_child(base, str(entry["backup"])) if entry.get("backup") else None
    staged = safe_child(base, str(entry["staged"])) if entry.get("staged") else None
    return target, backup, staged


def _undo_publish(base: Path, entries: list[dict[str, Any]]) -> list[str]:
    """Undo the renames listed in journal entries: every target is restored from its backup
    (or removed when it did not exist before) and staged temp files are deleted. Returns the
    names of the targets put back."""
    restored: list[str] = []
    for entry in reversed(entries):
        try:
            target, backup, staged = _journal_paths(base, entry)
        except SecurityError:
            continue
        if staged is not None:
            _unlink_quietly(staged)
        if backup is None:
            if target.exists():
                os.unlink(target)
                restored.append(target.name)
        elif backup.exists():
            os.replace(backup, target)
            restored.append(target.name)
    return restored


def write_all_or_nothing(artifacts: list[tuple[Path, str, int]], *, journal: Path) -> None:
    """Publish several files as one set: every artifact is staged to a temp file first, then
    renamed into place in sequence (each rename is atomic on its own). Before the first
    rename a publish journal is written next to the outputs listing every target with its
    backup copy (``null`` when the file did not exist); it is deleted only after the last
    rename, so the publish is committed exactly when the journal disappears.

    An ordinary exception is rolled back here (previous versions restored from the backups,
    new files removed) and leaves no journal, temp or backup file behind. A BaseException
    (Ctrl-C, SystemExit) is treated like a process kill: it propagates at once and the
    journal lets `recover_interrupted_publish` repair the set on the next command."""
    base = journal.parent
    staged: list[Path] = []
    entries: list[dict[str, Any]] = []
    try:
        for target, text, mode in artifacts:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = _stage(target, text, mode)
            staged.append(tmp)
            backup: Path | None = None
            if target.exists():
                fd, name = tempfile.mkstemp(prefix=".tmp-bak-", dir=str(target.parent))
                os.close(fd)
                backup = Path(name)
                shutil.copy2(target, backup)
            entries.append(
                {
                    "target": os.path.relpath(target, base),
                    "backup": os.path.relpath(backup, base) if backup is not None else None,
                    "staged": os.path.relpath(tmp, base),
                }
            )
        atomic_write_text(journal, json.dumps({"version": 1, "targets": entries}, indent=2) + "\n")
        renamed = 0
        for (target, _text, _mode), tmp in zip(artifacts, staged, strict=True):
            os.replace(tmp, target)
            renamed += 1
    except Exception:
        # Only the targets actually renamed are put back; the others are untouched, so
        # their backups and temp files are simply discarded.
        _undo_publish(base, entries[:renamed])
        for entry in entries[renamed:]:
            if entry["backup"]:
                _unlink_quietly(base / entry["backup"])
        for tmp in staged:
            _unlink_quietly(tmp)
        _unlink_quietly(journal)
        raise
    _unlink_quietly(journal)  # committed: from here on a kill leaves at most stray backups
    for entry in entries:
        if entry["backup"]:
            _unlink_quietly(base / entry["backup"])


def recover_interrupted_publish(run_dir: Path) -> bool:
    """Repair outputs left half-published by a process killed between renames: every target
    listed in the publish journal is restored from its backup (or removed when it did not
    exist before), then the journal is deleted. Idempotent and silent when there is no
    journal; returns True (and logs `publish_recovered`) when a journal was processed."""
    journal = run_dir / PUBLISH_JOURNAL_FILENAME
    if not journal.exists():
        return False
    entries: list[dict[str, Any]] = []
    try:
        data = json.loads(journal.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            entries = [e for e in (data.get("targets") or []) if isinstance(e, dict)]
    except (OSError, ValueError):
        log.warning("publish journal %s is unreadable; removing it", journal)
    restored = _undo_publish(run_dir, entries)
    _unlink_quietly(journal)
    event(
        log,
        "publish_recovered",
        "interrupted publish repaired from the journal: previous set restored "
        + (f"({', '.join(restored)})" if restored else "(nothing to undo)"),
        count=len(restored),
    )
    return True


def write_outputs(run_dir: Path, store: RunStore, profile: dict[str, Any], status: str) -> Path:
    """Validate once more and write all three artifacts as one set: either every output is
    replaced or (on any failure) the previous set is left exactly as it was; a set left
    half-published by an earlier kill is repaired first. Raises on an invalid profile."""
    recover_interrupted_publish(run_dir)
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
        ],
        journal=run_dir / PUBLISH_JOURNAL_FILENAME,
    )
    return output_path
