"""Workflow shell around the deep agent: start/resume, the CLI interview loop driven by
LangGraph interrupts, limit detection, best-effort partial export, and failure handling.

Only this module invokes the agent. It never injects new input while an interrupt is
pending (it resumes with `Command(resume=...)` or continues with `invoke(None)`), so the
pending `ask_user` call is never patched away.
"""

from __future__ import annotations

import logging
import secrets
import string
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from rich.console import Console

from profile_builder.agent.builder import (
    build_agent,
    build_checkpointer,
    build_embeddings,
    build_model,
)
from profile_builder.agent.prompts import render_initial_message
from profile_builder.agent.tools import FatalProfileError, ToolContext, finalize
from profile_builder.config import (
    DISCOVERY_FILENAME,
    MAX_FINALIZE_NUDGES,
    RUN_ID_PREFIX,
    Settings,
)
from profile_builder.logging_setup import attach_run_sinks, event, get_logger, set_run_context
from profile_builder.retrieval.index import Embeddings, HashEmbeddings, HybridIndex
from profile_builder.security import safe_child, sanitize_answer, validate_run_id
from profile_builder.state.run_store import RunStore
from profile_builder.web.robots import RobotsChecker
from profile_builder.web.scraper import (
    FirecrawlScraper,
    FixtureScraper,
    LinkCandidate,
    ScrapedPage,
    Scraper,
)
from profile_builder.web.url_guard import URLGuardError, validate_url
from profile_builder.workflow.summary import render_question, render_summary

log = get_logger("runner")

EXIT_WORDS = frozenset({"exit", "quit", "q", ":q", "stop"})
PAUSED = "paused"

ModelFactory = Callable[[Settings], BaseChatModel]
ScraperFactory = Callable[[Settings], Scraper]
EmbeddingsFactory = Callable[[Settings], Embeddings | None]
InputFn = Callable[[str], str]


@dataclass
class RunOutcome:
    run_id: str
    status: str  # complete | partial | failed | paused | interrupted
    output_path: str | None = None
    message: str | None = None
    resume_hint: str | None = None

    @property
    def exit_code(self) -> int:
        return 1 if self.status == "failed" else 0


def new_run_id() -> str:
    alphabet = string.ascii_lowercase + string.digits
    suffix = "".join(secrets.choice(alphabet) for _ in range(6))
    return f"{RUN_ID_PREFIX}-{time.strftime('%Y%m%d')}-{suffix}"


def default_scraper_factory(fixtures_dir: Path | None) -> ScraperFactory:
    def factory(settings: Settings) -> Scraper:
        if fixtures_dir is not None:
            return FixtureScraper(fixtures_dir)
        if not settings.has_firecrawl():
            raise RuntimeError("FIRECRAWL_API_KEY is not set (put it in .env or the environment), or use --fixtures DIR")
        return FirecrawlScraper(settings.firecrawl_api_key.get_secret_value())  # type: ignore[union-attr]

    return factory


def default_embeddings_factory(settings: Settings) -> Embeddings | None:
    if not settings.use_embeddings:
        return None
    emb = build_embeddings(settings)
    if emb is None:
        log.warning("embeddings requested but OPENAI_API_KEY missing; using keyword search only")
        return HashEmbeddings()
    return emb


class Runner:
    def __init__(
        self,
        settings: Settings,
        console: Console | None = None,
        *,
        model_factory: ModelFactory = build_model,
        scraper_factory: ScraperFactory | None = None,
        embeddings_factory: EmbeddingsFactory = default_embeddings_factory,
        input_fn: InputFn | None = None,
        non_interactive: bool = False,
        check_dns: bool = True,
        fixtures_dir: Path | None = None,
        save_fixtures_dir: Path | None = None,
        runs_dir_flag: str | None = None,
    ) -> None:
        self.settings = settings
        self.console = console or Console()
        self.model_factory = model_factory
        self.scraper_factory = scraper_factory or default_scraper_factory(fixtures_dir)
        self.embeddings_factory = embeddings_factory
        self.input_fn = input_fn or (lambda prompt: self.console.input(prompt))
        self.non_interactive = non_interactive
        self.check_dns = check_dns
        self.save_fixtures_dir = save_fixtures_dir
        self.runs_dir_flag = runs_dir_flag

    # ---- public API ----------------------------------------------------------------
    def start(self, url: str, product: str | None = None) -> RunOutcome:
        try:
            start_url = validate_url(url, check_dns=self.check_dns)
        except URLGuardError as exc:
            raise ValueError(f"URL rejected: {exc}") from exc
        run_id = new_run_id()
        run_dir = safe_child(self.settings.runs_dir, run_id)
        run_dir.mkdir(parents=True, exist_ok=False)
        self._attach_logging(run_id, run_dir)
        store = RunStore(run_dir)
        store.create_run(run_id, start_url, product, self.settings.snapshot())
        event(log, "run_started", f"run {run_id} started for {start_url}", kind="start")
        log.info("settings: %s", store.get_run().settings)
        ctx, agent, config = self._build(run_id, run_dir, store, start_url, product, self.settings)
        self.console.print(f"[bold]Run id:[/bold] {run_id}  [dim](resume later with: {self._resume_cmd(run_id)})[/dim]")
        initial = {
            "messages": [
                HumanMessage(
                    content=render_initial_message(
                        start_url=start_url, product_focus=product,
                        max_pages=self.settings.max_pages, max_questions=self.settings.max_questions,
                    )
                )
            ]
        }
        return self._run(ctx, agent, config, initial)

    def resume(self, run_id: str, **overrides: Any) -> RunOutcome:
        validate_run_id(run_id)
        run_dir = safe_child(self.settings.runs_dir, run_id)
        if not (run_dir / "run.sqlite").exists():
            raise FileNotFoundError(f"run {run_id} not found under {self.settings.runs_dir}")
        store = RunStore(run_dir)
        run = store.get_run()
        settings = Settings.from_snapshot(run.settings, **overrides)
        clean_overrides = {k: v for k, v in overrides.items() if v is not None}
        if clean_overrides:
            store.update_settings(settings.snapshot())
            store.add_warning("SETTINGS_OVERRIDDEN_ON_RESUME", f"limits changed on resume: {clean_overrides}")
        self.settings = settings
        self._attach_logging(run_id, run_dir)
        event(log, "run_started", f"run {run_id} resumed (status={run.status}, stage={run.stage})", kind="resume")
        if run.status in {"complete", "partial", "failed"}:
            latest = store.latest_draft()
            render_summary(self.console, store, latest[1] if latest else None, run.status, run.output_path)
            return RunOutcome(run_id, run.status, run.output_path, message=f"run already finished with status {run.status}; use `export` to re-export")
        ctx, agent, config = self._build(run_id, run_dir, store, run.start_url, run.product_focus, settings)
        snap = agent.get_state(config)
        if snap.interrupts:
            intr = snap.interrupts[0]
            answer = self._ask(intr.value)
            if answer is None:
                return self._paused(ctx)
            return self._run(ctx, agent, config, Command(resume={intr.id: answer}))
        if snap.next:
            return self._run(ctx, agent, config, None)
        if not snap.values:
            initial = {"messages": [HumanMessage(content=render_initial_message(start_url=run.start_url, product_focus=run.product_focus, max_pages=settings.max_pages, max_questions=settings.max_questions))]}
            return self._run(ctx, agent, config, initial)
        return self._finish(ctx, agent, config)

    def export(self, run_id: str) -> RunOutcome:
        """Re-export from the latest saved draft without running the agent."""
        validate_run_id(run_id)
        run_dir = safe_child(self.settings.runs_dir, run_id)
        store = RunStore(run_dir)
        run = store.get_run()
        self._attach_logging(run_id, run_dir)
        ctx = self._context(run_id, run_dir, store, run.start_url, run.product_focus, Settings.from_snapshot(run.settings), scraper=None)
        forced = run.status not in {"complete"}
        try:
            summary = finalize(ctx, forced_partial=forced)
        except FatalProfileError as exc:
            return RunOutcome(run_id, "failed", message=str(exc))
        latest = store.latest_draft()
        render_summary(self.console, store, latest[1] if latest else None, summary["status"], summary["output_path"])
        return RunOutcome(run_id, summary["status"], summary["output_path"])

    # ---- internals -----------------------------------------------------------------
    def _attach_logging(self, run_id: str, run_dir: Path) -> None:
        secrets_ = [
            s.get_secret_value()
            for s in (self.settings.openai_api_key, self.settings.firecrawl_api_key)
            if s is not None
        ]
        attach_run_sinks(run_dir, secrets_)
        set_run_context(run_id=run_id, stage="init")

    def _context(self, run_id: str, run_dir: Path, store: RunStore, start_url: str, product: str | None, settings: Settings, scraper: Scraper | None) -> ToolContext:
        embeddings = self.embeddings_factory(settings)
        index = HybridIndex(store, embeddings, embedding_model=settings.embedding_model, use_embeddings=settings.use_embeddings)
        return ToolContext(
            settings=settings, store=store, scraper=scraper,  # type: ignore[arg-type]
            index=index, robots=RobotsChecker(enabled=True), run_dir=run_dir, run_id=run_id,
            start_url=start_url, product_focus=product, check_dns=self.check_dns,
        )

    def _build(self, run_id: str, run_dir: Path, store: RunStore, start_url: str, product: str | None, settings: Settings):
        scraper = self.scraper_factory(settings)
        ctx = self._context(run_id, run_dir, store, start_url, product, settings, scraper)
        model = self.model_factory(settings)
        checkpointer = build_checkpointer(run_dir)
        agent = build_agent(ctx, model, checkpointer)
        config = {"configurable": {"thread_id": run_id}}
        return ctx, agent, config

    def _resume_cmd(self, run_id: str) -> str:
        extra = f" --runs-dir {self.runs_dir_flag}" if self.runs_dir_flag else ""
        return f"python -m profile_builder resume --run-id {run_id}{extra}"

    def _ask(self, payload: dict[str, Any]) -> str | None:
        """Show the question and read the answer. Returns None when the user wants to exit."""
        if self.non_interactive:
            log.info("non-interactive mode: auto-skipping question %s", payload.get("qid"))
            return "skip"
        render_question(self.console, payload)
        try:
            raw = self.input_fn("> ")
        except (EOFError, KeyboardInterrupt):
            self.console.print()
            return None
        answer = sanitize_answer(raw)
        if answer.lower() in EXIT_WORDS:
            return None
        return answer

    def _paused(self, ctx: ToolContext) -> RunOutcome:
        ctx.store.set_status(PAUSED)
        event(log, "run_finished", "run paused by user during interview", status=PAUSED)
        latest = ctx.store.latest_draft()
        hint = f"Progress is saved. Continue with:\n  {self._resume_cmd(ctx.run_id)}"
        render_summary(self.console, ctx.store, latest[1] if latest else None, PAUSED, None, resume_hint=hint)
        return RunOutcome(ctx.run_id, PAUSED, resume_hint=self._resume_cmd(ctx.run_id))

    def _run(self, ctx: ToolContext, agent: Any, config: dict[str, Any], initial: Any) -> RunOutcome:
        ctx.store.set_status("running")
        try:
            result = agent.invoke(initial, config, durability="sync")
            while True:
                interrupts = result.get("__interrupt__") or []
                if not interrupts:
                    break
                intr = interrupts[0]
                answer = self._ask(intr.value)
                if answer is None:
                    return self._paused(ctx)
                result = agent.invoke(Command(resume={intr.id: answer}), config, durability="sync")
            return self._finish(ctx, agent, config)
        except FatalProfileError as exc:
            return self._failed(ctx, str(exc))
        except KeyboardInterrupt:
            return self._paused(ctx)
        except Exception as exc:  # unexpected: keep the run resumable, report clearly
            log.error("run interrupted by error: %s: %s", type(exc).__name__, exc, exc_info=True)
            ctx.store.set_status("interrupted")
            ctx.store.add_warning("RUN_INTERRUPTED", f"{type(exc).__name__}: {str(exc)[:300]}")
            hint = f"The run stopped on an error and can be resumed:\n  {self._resume_cmd(ctx.run_id)}"
            latest = ctx.store.latest_draft()
            render_summary(self.console, ctx.store, latest[1] if latest else None, "interrupted", None, resume_hint=hint)
            return RunOutcome(ctx.run_id, "interrupted", message=f"{type(exc).__name__}: {exc}", resume_hint=self._resume_cmd(ctx.run_id))

    def _finish(self, ctx: ToolContext, agent: Any, config: dict[str, Any]) -> RunOutcome:
        store = ctx.store
        run = store.get_run()
        if run.status in {"complete", "partial"}:
            return self._done(ctx, run.status, run.output_path)
        snap = agent.get_state(config)
        values = snap.values or {}
        model_calls = int(values.get("thread_model_call_count", 0) or 0)
        limit_hit = False
        if model_calls >= self.settings.max_model_calls:
            store.add_warning("LIMIT_MODEL_CALLS_REACHED", f"model call limit of {self.settings.max_model_calls} reached")
            event(log, "limit_reached", "model call limit reached", level=logging.WARNING, kind="model_calls", count=model_calls)
            limit_hit = True
        if store.has_warning("BUDGET_EXCEEDED"):
            limit_hit = True
        if not limit_hit and store.get_counter("finalize_nudges") < MAX_FINALIZE_NUDGES:
            store.increment_counter("finalize_nudges")
            log.info("agent stopped without finalizing; nudging once")
            nudge = {"messages": [HumanMessage(content="You stopped before exporting. If a draft exists, call finalize_profile now; otherwise call save_profile_draft with what the evidence supports, then finalize_profile.")]}
            return self._run(ctx, agent, config, nudge)
        try:
            summary = finalize(ctx, forced_partial=True)
        except FatalProfileError as exc:
            return self._failed(ctx, f"stopped before a valid draft existed ({exc})")
        return self._done(ctx, summary["status"], summary["output_path"])

    def _done(self, ctx: ToolContext, status: str, output_path: str | None) -> RunOutcome:
        self._maybe_save_fixtures(ctx)
        latest = ctx.store.latest_draft()
        render_summary(self.console, ctx.store, latest[1] if latest else None, status, output_path)
        return RunOutcome(ctx.run_id, status, output_path)

    def _failed(self, ctx: ToolContext, message: str) -> RunOutcome:
        ctx.store.set_status("failed")
        ctx.store.add_warning("RUN_FAILED", message[:300])
        event(log, "run_finished", f"run failed: {message}", level=logging.ERROR, status="failed")
        latest = ctx.store.latest_draft()
        render_summary(self.console, ctx.store, latest[1] if latest else None, "failed", None)
        self.console.print(f"[bold red]Error:[/bold red] {message}")
        if latest:
            self.console.print(f"[dim]A valid earlier draft exists; export it with: python -m profile_builder export --run-id {ctx.run_id}[/dim]")
        return RunOutcome(ctx.run_id, "failed", message=message)

    def _maybe_save_fixtures(self, ctx: ToolContext) -> None:
        if not self.save_fixtures_dir:
            return
        out = Path(self.save_fixtures_dir)
        n = 0
        for rec in ctx.store.list_pages("fetched"):
            if not rec.cache_file:
                continue
            path = ctx.pages_dir / rec.cache_file
            if path.exists():
                FixtureScraper.write_page(out, ScrapedPage.from_json(path.read_text(encoding="utf-8")))
                n += 1
        disc = ctx.run_dir / DISCOVERY_FILENAME
        if disc.exists():
            import json

            FixtureScraper.write_map(out, [LinkCandidate(**d) for d in json.loads(disc.read_text(encoding="utf-8"))])
        log.info("saved %d fixture pages to %s", n, out)
