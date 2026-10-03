"""Command-line interface: start | resume | status | list | inspect | export | schema | logs | doctor."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from profile_builder.config import APP_VERSION, RUN_DB, Settings
from profile_builder.logging_setup import configure_console, read_events
from profile_builder.models import MODEL_TIERS
from profile_builder.security import SecurityError, safe_child, safe_text, validate_run_id
from profile_builder.state.run_store import RunStore

app = typer.Typer(
    name="profile-builder",
    help="Build a grounded company_brain.json from a company website (Deep Agents + LangGraph + OpenAI + Firecrawl).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

TierOpt = Annotated[str | None, typer.Option("--tier", help="Model tier: fast | quality (overridden by --model)")]


def _settings(**overrides) -> Settings:
    base = Settings()
    tier = overrides.pop("tier", None)
    if tier is not None:
        if tier not in MODEL_TIERS:
            raise typer.BadParameter(f"--tier must be one of {', '.join(MODEL_TIERS)}")
        overrides["tier"] = tier
    if overrides.get("model") is not None:
        overrides["tier"] = overrides.get("tier") or base.tier
    return base.with_overrides(**overrides)


def _runner(settings: Settings, **kwargs):
    from profile_builder.workflow.runner import Runner

    return Runner(settings, console, **kwargs)


def _open_store(settings: Settings, run_id: str) -> RunStore:
    try:
        validate_run_id(run_id)
        run_dir = safe_child(settings.runs_dir, run_id)
    except SecurityError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not (run_dir / RUN_DB).exists():
        console.print(f"[red]Run {run_id} not found under {settings.runs_dir}[/red]")
        raise typer.Exit(code=2)
    return RunStore(run_dir)


@app.callback()
def _main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging on the console")] = False,
          quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Warnings and errors only")] = False) -> None:
    level = "DEBUG" if verbose else "WARNING" if quiet else "INFO"
    configure_console(level, console)


@app.command()
def start(
    url: Annotated[str, typer.Option("--url", help="Company website, e.g. https://www.fortanix.com/")],
    product: Annotated[str | None, typer.Option("--product", help="Product to focus on (skips the product question)")] = None,
    max_pages: Annotated[int | None, typer.Option("--max-pages", min=1, max=100, help="Max unique pages to scrape")] = None,
    max_questions: Annotated[int | None, typer.Option("--max-questions", min=0, max=50, help="Max interview questions incl. follow-ups")] = None,
    model: Annotated[str | None, typer.Option("--model", help="OpenAI model id (overrides --tier)")] = None,
    tier: TierOpt = None,
    budget_usd: Annotated[float | None, typer.Option("--budget-usd", min=0.01, help="Stop cleanly at this estimated spend (default: unlimited)")] = None,
    no_embeddings: Annotated[bool, typer.Option("--no-embeddings", help="Keyword-only retrieval (no embedding calls)")] = False,
    non_interactive: Annotated[bool, typer.Option("--non-interactive", help="Auto-skip every question (CI/demo)")] = False,
    fixtures: Annotated[Path | None, typer.Option("--fixtures", help="Serve pages from a fixture dir instead of Firecrawl")] = None,
    save_fixtures: Annotated[Path | None, typer.Option("--save-fixtures", help="After the run, save fetched pages as fixtures here")] = None,
    runs_dir: Annotated[Path | None, typer.Option("--runs-dir", help="Where run state is stored (default .runs)")] = None,
) -> None:
    """Start a new profile-building run."""
    settings = _settings(max_pages=max_pages, max_questions=max_questions, model=model, tier=tier,
                         budget_usd=budget_usd, use_embeddings=False if no_embeddings else None, runs_dir=runs_dir)
    runner = _runner(settings, non_interactive=non_interactive, fixtures_dir=fixtures, save_fixtures_dir=save_fixtures,
                     runs_dir_flag=str(runs_dir) if runs_dir else None)
    try:
        outcome = runner.start(url, product)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        console.print(f"[bold red]Error:[/bold red] {safe_text(str(exc))}")
        raise typer.Exit(code=2) from exc
    raise typer.Exit(code=outcome.exit_code)


@app.command()
def resume(
    run_id: Annotated[str, typer.Option("--run-id", help="Run id printed by `start`")],
    max_questions: Annotated[int | None, typer.Option("--max-questions", min=0, max=50, help="Allow more questions on resume")] = None,
    budget_usd: Annotated[float | None, typer.Option("--budget-usd", min=0.01, help="Raise the budget on resume")] = None,
    non_interactive: Annotated[bool, typer.Option("--non-interactive")] = False,
    fixtures: Annotated[Path | None, typer.Option("--fixtures")] = None,
    runs_dir: Annotated[Path | None, typer.Option("--runs-dir")] = None,
) -> None:
    """Resume a paused or interrupted run from its last checkpoint."""
    settings = _settings(runs_dir=runs_dir)
    runner = _runner(settings, non_interactive=non_interactive, fixtures_dir=fixtures, runs_dir_flag=str(runs_dir) if runs_dir else None)
    try:
        outcome = runner.resume(run_id, max_questions=max_questions, budget_usd=budget_usd)
    except (ValueError, RuntimeError, FileNotFoundError, SecurityError) as exc:
        console.print(f"[bold red]Error:[/bold red] {safe_text(str(exc))}")
        raise typer.Exit(code=2) from exc
    if outcome.message:
        console.print(safe_text(outcome.message))
    raise typer.Exit(code=outcome.exit_code)


@app.command()
def status(run_id: Annotated[str, typer.Option("--run-id")], runs_dir: Annotated[Path | None, typer.Option("--runs-dir")] = None) -> None:
    """Show stage, status, counters, pending question, warnings and cost for a run."""
    settings = _settings(runs_dir=runs_dir)
    store = _open_store(settings, run_id)
    run = store.get_run()
    usage = store.usage_totals()
    t = Table.grid(padding=(0, 2))
    t.add_row("Run", run.run_id)
    t.add_row("Website", safe_text(run.start_url))
    t.add_row("Product focus", safe_text(run.product_focus or "(not set)"))
    t.add_row("Status / stage", f"{run.status} / {run.stage}")
    t.add_row("Created", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run.created_at)))
    t.add_row("Pages", f"{len(store.list_pages('fetched'))} fetched, {len(store.list_pages('skipped'))} skipped, {len(store.list_pages('failed'))} failed (max {run.settings.get('max_pages')})")
    t.add_row("Questions", f"{store.questions_asked()} asked (max {run.settings.get('max_questions')})")
    t.add_row("Drafts", str(store.draft_count()))
    t.add_row("Model", f"{run.settings.get('model_id')} · {usage['model_calls']} calls · est. ${usage['cost_usd']:.4f}")
    t.add_row("Output", safe_text(run.output_path or "—"))
    console.print(Panel(t, title="Run status"))
    pending = store.pending_question()
    if pending:
        console.print(Panel(safe_text(pending["question"]), title=f"Pending question {pending['ordinal']}", border_style="magenta"))
    questions = store.list_questions()
    if questions:
        q = Table(title="Interview", show_header=True)
        for col in ("#", "Question", "Status", "Answer"):
            q.add_column(col)
        for item in questions:
            q.add_row(str(item["ordinal"]), safe_text(item["question"], 90), item["status"], safe_text(item["answer"] or "—", 60))
        console.print(q)
    warnings = store.list_warnings()
    if warnings:
        w = Table(title=f"Warnings ({len(warnings)})", show_header=True)
        w.add_column("Code")
        w.add_column("Message")
        for item in warnings:
            w.add_row(safe_text(item["code"]), safe_text(item["message"], 120))
        console.print(w)
    console.print("[dim]settings:[/dim] " + safe_text(json.dumps(run.settings)))


@app.command(name="list")
def list_runs(runs_dir: Annotated[Path | None, typer.Option("--runs-dir")] = None) -> None:
    """List runs in the runs directory."""
    settings = _settings(runs_dir=runs_dir)
    root = settings.runs_dir
    if not root.exists():
        console.print(f"No runs yet in {root}")
        return
    t = Table(title=f"Runs in {root}", show_header=True)
    for col in ("Run id", "Website", "Status", "Stage", "Pages", "Q", "Cost", "Created"):
        t.add_column(col)
    rows = []
    for d in sorted(root.glob("pb-*")):
        if not (d / RUN_DB).exists():
            continue
        try:
            store = RunStore(d)
            run = store.get_run()
        except (sqlite3.Error, LookupError):
            continue
        rows.append((run.created_at, run.run_id, run.start_url, run.status, run.stage, len(store.list_pages("fetched")), store.questions_asked(), store.total_cost()))
        store.close()
    for created, rid, url, st, stage, pages, q, cost in sorted(rows, reverse=True):
        t.add_row(rid, safe_text(url, 40), st, stage, str(pages), str(q), f"${cost:.4f}", time.strftime("%Y-%m-%d %H:%M", time.localtime(created)))
    console.print(t)


@app.command()
def inspect(
    run_id: Annotated[str, typer.Option("--run-id")],
    field: Annotated[str | None, typer.Option("--field", help="Only fields starting with this path, e.g. customer.")] = None,
    runs_dir: Annotated[Path | None, typer.Option("--runs-dir")] = None,
) -> None:
    """Show each populated field with its evidence (source URL + verbatim excerpt, or Q&A)."""
    from profile_builder.schema import ancestors, iter_leaf_paths

    settings = _settings(runs_dir=runs_dir)
    store = _open_store(settings, run_id)
    latest = store.latest_draft()
    if latest is None:
        console.print("No draft saved yet for this run.")
        raise typer.Exit(code=0)
    version, profile = latest
    evidence: dict[str, list[dict]] = {}
    for e in store.list_evidence():
        evidence.setdefault(e["field_path"], []).append(e)
    questions = {q["qid"]: q for q in store.list_questions()}
    console.print(f"[bold]Draft v{version}[/bold] — evidence per populated field" + (f" (filter: {safe_text(field)})" if field else ""))
    shown = 0
    for path, value in iter_leaf_paths(profile):
        if value in ("", [], None) or (field and not path.startswith(field)):
            continue
        rows = [r for a in ancestors(path) for r in evidence.get(a, [])]
        t = Table.grid(padding=(0, 1))
        t.add_row("[bold]value[/bold]", safe_text(json.dumps(value, ensure_ascii=False), 300))
        if not rows:
            t.add_row("[yellow]evidence[/yellow]", "[yellow]none (ungrounded)[/yellow]")
        for r in rows:
            tag = "[dim](superseded)[/dim] " if r["superseded"] else ""
            if r["kind"] == "website":
                t.add_row("website", f"{tag}{safe_text(r['source_url'] or '')}\n  “{safe_text(r['excerpt'] or '', 300)}”")
            else:
                q = questions.get(r["question_id"] or "", {})
                t.add_row("interview", f"{tag}Q: {safe_text(q.get('question', ''), 200)}\n  A: {safe_text(r['answer'] or '', 300)}")
        console.print(Panel(t, title=safe_text(path), title_align="left", border_style="blue" if rows else "yellow"))
        shown += 1
    if not shown:
        console.print("No populated fields matched.")


@app.command()
def export(run_id: Annotated[str, typer.Option("--run-id")], runs_dir: Annotated[Path | None, typer.Option("--runs-dir")] = None) -> None:
    """Re-export company_brain.json / evidence.json / report.md from the latest saved draft."""
    settings = _settings(runs_dir=runs_dir)
    _open_store(settings, run_id).close()
    runner = _runner(settings)
    outcome = runner.export(run_id)
    if outcome.message:
        console.print(f"[bold red]Error:[/bold red] {safe_text(outcome.message)}")
    raise typer.Exit(code=outcome.exit_code)


@app.command()
def schema(out: Annotated[Path | None, typer.Option("--out", help="Write the JSON Schema here (default: print)")] = None) -> None:
    """Print or write the company_brain JSON Schema generated from the Pydantic models."""
    from profile_builder.schema import json_schema, write_json_schema

    if out:
        path = write_json_schema(out)
        console.print(f"Schema written to {path}")
    else:
        console.print_json(json.dumps(json_schema()))


@app.command()
def logs(
    run_id: Annotated[str, typer.Option("--run-id")],
    tail: Annotated[int | None, typer.Option("--tail", min=1)] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Raw JSON lines")] = False,
    level: Annotated[str | None, typer.Option("--level", help="Minimum level: DEBUG|INFO|WARNING|ERROR")] = None,
    runs_dir: Annotated[Path | None, typer.Option("--runs-dir")] = None,
) -> None:
    """Show the structured event log of a run."""
    settings = _settings(runs_dir=runs_dir)
    store = _open_store(settings, run_id)
    rows = read_events(store.run_dir, tail=tail, level=level)
    store.close()
    if as_json:
        for r in rows:
            typer.echo(json.dumps(r, ensure_ascii=False))
        return
    t = Table(show_header=True, title=f"{len(rows)} events")
    for col in ("time", "lvl", "stage", "event", "message", "details"):
        t.add_column(col, overflow="fold")
    for r in rows:
        details = {k: v for k, v in r.items() if k not in {"ts", "level", "run_id", "stage", "logger", "message", "event"}}
        t.add_row(r.get("ts", "")[11:23], r.get("level", ""), r.get("stage", ""), safe_text(r.get("event", "") or ""), safe_text(r.get("message", ""), 100), safe_text(json.dumps(details), 80) if details else "")
    console.print(t)


@app.command()
def doctor(runs_dir: Annotated[Path | None, typer.Option("--runs-dir")] = None) -> None:
    """Check configuration: API keys present, providers reachable, runs dir writable."""
    settings = _settings(runs_dir=runs_dir)
    t = Table(title=f"profile-builder {APP_VERSION} — doctor", show_header=True)
    t.add_column("Check")
    t.add_column("Result")
    ok = True

    def row(name: str, good: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and good
        t.add_row(name, ("[green]OK[/green] " if good else "[red]FAIL[/red] ") + safe_text(detail))

    row("OPENAI_API_KEY", settings.has_openai(), "present" if settings.has_openai() else "missing (set it in .env)")
    row("FIRECRAWL_API_KEY", settings.has_firecrawl(), "present" if settings.has_firecrawl() else "missing (set it in .env, or use --fixtures)")
    row("Model", True, f"{settings.model_id} (tier {settings.tier}); quality tier = {MODEL_TIERS['quality']}")
    if settings.has_openai():
        try:
            from profile_builder.agent.builder import build_model

            build_model(settings).invoke("Reply with the single word OK.")
            row("OpenAI reachable", True, "model responded")
        except Exception as exc:
            row("OpenAI reachable", False, f"{type(exc).__name__}")
    if settings.has_firecrawl():
        try:
            from firecrawl import Firecrawl

            Firecrawl(api_key=settings.firecrawl_api_key.get_secret_value(), max_retries=0).get_credit_usage()  # type: ignore[union-attr]
            row("Firecrawl reachable", True, "credit endpoint responded")
        except Exception as exc:
            row("Firecrawl reachable", False, f"{type(exc).__name__}")
    try:
        settings.runs_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.runs_dir / ".write-probe"
        probe.write_text("ok")
        probe.unlink()
        row("Runs dir writable", True, str(settings.runs_dir.resolve()))
    except OSError as exc:
        row("Runs dir writable", False, str(exc))
    console.print(t)
    raise typer.Exit(code=0 if ok else 1)


def main() -> None:
    app()
