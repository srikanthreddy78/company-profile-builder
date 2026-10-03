"""Rich terminal rendering: interview question panels and the end-of-run summary."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from profile_builder.security import safe_text
from profile_builder.state.run_store import RunStore
from profile_builder.workflow.gaps import empty_gaps, grounding_report, section_coverage

STATUS_STYLE = {"complete": "bold green", "partial": "bold yellow", "failed": "bold red", "running": "cyan", "paused": "magenta"}


def render_question(console: Console, payload: dict[str, Any]) -> None:
    n = payload.get("number")
    total = payload.get("max")
    header = f"Question {n}/{total}" if n and total else "Question"
    body = Text()
    body.append(safe_text(payload.get("question", "")), style="bold")
    why = payload.get("why_unclear")
    if why:
        body.append("\n\n")
        body.append("Why this is unclear: ", style="dim")
        body.append(safe_text(why))
    fields = payload.get("field_paths") or []
    if fields:
        body.append("\n\n")
        body.append("Affects: ", style="dim")
        body.append(", ".join(safe_text(f) for f in fields), style="dim")
    body.append("\n\n")
    body.append("Answer, or type ", style="dim")
    body.append("skip", style="bold")
    body.append(" · ", style="dim")
    body.append("idk", style="bold")
    body.append(" · ", style="dim")
    body.append("exit", style="bold")
    body.append(" (save and resume later)", style="dim")
    console.print(Panel(body, title=header, border_style="magenta", expand=False))


def render_summary(console: Console, store: RunStore, profile: dict[str, Any] | None, status: str, output_path: str | None, resume_hint: str | None = None) -> None:
    run = store.get_run()
    usage = store.usage_totals()
    style = STATUS_STYLE.get(status, "white")
    head = Table.grid(padding=(0, 2))
    head.add_row("Run", f"[bold]{run.run_id}[/bold]")
    head.add_row("Website", safe_text(run.start_url))
    head.add_row("Product focus", safe_text(run.product_focus or "(not set)"))
    head.add_row("Status", f"[{style}]{status.upper()}[/{style}]")
    head.add_row("Pages", f"{len(store.list_pages('fetched'))} fetched · {len(store.list_pages('skipped'))} skipped · {len(store.list_pages('failed'))} failed")
    head.add_row("Questions", f"{store.questions_asked()} asked (max {run.settings.get('max_questions')})")
    head.add_row("Model", f"{run.settings.get('model_id')} · {usage['model_calls']} calls · {usage['input_tokens']:,} in / {usage['output_tokens']:,} out")
    head.add_row("Est. cost", f"${usage['cost_usd']:.4f}" + (f" (budget ${run.settings['budget_usd']:.2f})" if run.settings.get("budget_usd") else " (no budget cap)"))
    if output_path:
        head.add_row("Output", f"[bold]{output_path}[/bold]")
    console.print(Panel(head, title="Company Profile Builder", border_style=style.split()[-1]))

    if profile:
        cov = Table(title="Coverage", show_header=True, header_style="bold")
        cov.add_column("Section")
        cov.add_column("Filled", justify="right")
        for section, (filled, total) in section_coverage(profile).items():
            cov.add_row(section, f"{filled}/{total}")
        g = grounding_report(profile, {e["field_path"] for e in store.list_evidence(include_superseded=False)})
        cov.add_row("[dim]grounded fields[/dim]", f"{g['grounded']}/{g['populated']}")
        console.print(cov)
        gaps = empty_gaps(profile)
        if gaps:
            t = Table(title=f"Remaining gaps ({len(gaps)})", show_header=True, header_style="bold")
            t.add_column("Field")
            t.add_column("What is missing")
            for gap in gaps[:12]:
                t.add_row(safe_text(gap.field_path), safe_text(gap.reason))
            if len(gaps) > 12:
                t.add_row("…", f"{len(gaps) - 12} more in report.md")
            console.print(t)

    warnings = store.list_warnings()
    if warnings:
        t = Table(title=f"Warnings ({len(warnings)})", show_header=True, header_style="bold yellow")
        t.add_column("Code")
        t.add_column("Message")
        for w in warnings[:15]:
            t.add_row(safe_text(w["code"]), safe_text(w["message"], 140))
        if len(warnings) > 15:
            t.add_row("…", f"{len(warnings) - 15} more in evidence.json")
        console.print(t)
    if resume_hint:
        console.print(Panel(resume_hint, border_style="magenta", title="Resume"))
