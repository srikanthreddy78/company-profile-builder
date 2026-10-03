# Architecture

One Deep Agent with nine tools, wrapped by a thin deterministic shell. Agent reasoning
decides *which* pages to read, *what* to search for, *what* is a gap worth asking about and
*how* to phrase the question. Plain code owns everything that must be exact: URL safety,
limits, caching, retries, validation, evidence, persistence.

```
CLI (typer + rich)
  └─ Runner (workflow/runner.py) ── start / resume / export, interview loop, limit detection
       └─ create_deep_agent(model, tools, middleware, checkpointer=SqliteSaver)
            ├─ tools (agent/tools.py)      discover_pages · scrape_pages · search_pages · read_page
            │                              ask_user (interrupt) · save_profile_draft
            │                              apply_profile_updates · note_conflict · finalize_profile
            ├─ middleware (agent/middleware.py)  see table below
            └─ state: LangGraph checkpoints (checkpoints.sqlite)  +  RunStore (run.sqlite)
                                                                   +  page cache (pages/*.json)
```

## Where each behavior lives

| Concern | Middleware | Tools | Workflow code (runner) |
|---|---|---|---|
| Retries with backoff (cap = `MAX_RETRIES`, default 2) | `ToolRetryMiddleware` on `scrape_pages`/`discover_pages` with `retry_on=is_transient`; `ModelRetryMiddleware` | classify Firecrawl errors: transient exceptions propagate, permanent ones become results + warnings (never retried) | — |
| Bounded execution | `ModelCallLimitMiddleware` (derived from pages+questions), `ToolCallLimitMiddleware` for `scrape_pages` and a backstop for `ask_user` | exact page budget and question cap enforced in code, with `LIMIT_*` warnings | detects the model-call cap / budget and exports the last valid draft as **partial** |
| Cost tracking + budget | `RunTelemetryMiddleware` records tokens/cost per attempt; `BudgetCapMiddleware.before_model` jumps to `end` (never raises, so nothing is retried or lost) | — | prints spend; adds `BUDGET_EXCEEDED` hint with a resume command |
| Serial tool calls | `SerialToolCallsMiddleware` sets `parallel_tool_calls=False`; `SoloAskUserGuardMiddleware` rejects a bundled `ask_user` | — | — |
| Untrusted content | `UntrustedContentMiddleware` frames web tool output | injection heuristics → `INJECTION_SUSPECTED` | — |
| Logging | telemetry logs every model/tool attempt (duration, tokens, cost, status) | stage changes, page fetches, limits, questions, drafts | run start/finish, resume, failures |
| Validation + repair | — | Pydantic `CompanyBrain`; one repair attempt (`MAX_REPAIR_ATTEMPTS`), then `FatalProfileError` | turns the fatal error into status `failed`, keeps state, never writes a profile |
| Grounding | — | evidence must quote a fetched page verbatim; field paths allow-listed; conflicts tracked and omitted if unresolved | `inspect` shows evidence per field |
| Interview | — | `ask_user` calls `langgraph.types.interrupt`, commits the answer idempotently | shows the question, reads input, resumes with `Command(resume={interrupt_id: answer})` |

Middleware order (first = outermost): `SerialToolCalls → [BudgetCap] → ModelCallLimit →
ToolCallLimit(ask_user) → ToolCallLimit(scrape_pages) → SoloAskUserGuard → ModelRetry →
ToolRetry → RunTelemetry → UntrustedContent`. Telemetry sits *inside* the retry middleware so
each attempt is logged and charged. Deep Agents' own core stack stays in place except the
filesystem middleware, which is replaced by a read-only instance (`ls`, `read_file`).

All custom middleware re-raises `GraphBubbleUp` (the base of `GraphInterrupt`) before any
generic handler, so interrupts are never swallowed. The built-in retry middleware does the
same.

## Context engineering (the token problem)

Ten pages can be 100k+ tokens. The model never sees raw pages:

1. **Discovery is pre-filtered in code.** Firecrawl `map` (sitemap) plus homepage links are
   normalized, de-duplicated, restricted to the same registrable domain, stripped of assets /
   legal / careers / pagination, and scored by path keywords. The model sees ≤ 40 candidates
   (`url`, `title`, `description`, `score`) and the remaining page budget.
2. **Scraping returns a digest**, not text: title, size, headings, a 300-character lead. The
   markdown goes to a per-run page cache and into the index.
3. **Heading-aware chunking** (~400 tokens, small overlap). Navigation-heavy paragraphs are
   dropped; paragraphs already seen on *other* pages (shared nav, footers, cookie banners) are
   removed before chunking; chunks whose text repeats elsewhere are skipped.
4. **Hybrid retrieval.** `search_pages` runs BM25 (rank-bm25) and cosine similarity over
   `text-embedding-3-small` vectors stored in SQLite, fused with reciprocal-rank fusion, and
   returns ≤ 6 verbatim excerpts (≤ 600 chars) with URL + heading. `read_page` returns a bounded
   window of one page/section when surrounding context is needed. `--no-embeddings` switches to
   BM25 only.
5. **Compact tool contracts.** Draft/update tools return validation results, grounding
   statistics and a prioritized gap list rather than echoing the profile.

A typical Fortanix run uses roughly 25-35 model calls at a few thousand input tokens each.

## Persistence and the replay boundary

Two stores, one directory per run (`.runs/<run_id>/`):

- `checkpoints.sqlite` — LangGraph `SqliteSaver`, `thread_id = run_id`, `durability="sync"`.
  Holds the agent's messages and control flow, including the pending interrupt.
- `run.sqlite` — the RunStore: run metadata, stage, status, settings snapshot, counters, pages,
  chunks + embeddings, paragraph hashes, evidence (field path → source URL + verbatim excerpt,
  or question + answer; superseded rows kept), questions (pending/answered/skipped/unknown),
  warnings, conflicts, draft versions, usage. Every write is an idempotent upsert keyed by a
  natural key (normalized URL, question id, field path).
- `pages/<sha1(url)>.json` — cached page text; `events.jsonl` / `run.log` — logs;
  `company_brain.json`, `evidence.json`, `report.md` — outputs.

**Replay boundary.** LangGraph re-executes the *whole tools step* from its start on resume, and
all tool calls in one model message form one step. Therefore:

- The model is forced to one tool call per step, and `ask_user` must be alone.
- `ask_user` does nothing non-idempotent before `interrupt()`; it upserts the pending question
  (so `status` can show it), and on resume commits the answer *before* the step checkpoints.
  If the process dies after the commit but before the checkpoint, the replayed call finds the
  committed answer and returns it without asking again.
- A crash between Firecrawl returning and the checkpoint may re-run `scrape_pages`; the page
  cache makes already-fetched URLs free and the store rows are overwritten identically.
- The runner never injects new input while an interrupt is pending; it resumes with
  `Command(resume=...)` or continues with `invoke(None, config)`. Otherwise Deep Agents'
  `PatchToolCallsMiddleware` would mark the pending `ask_user` as cancelled.
- Resume rebuilds the agent from scratch (new process), reads `get_state(config)`: a pending
  interrupt → show the question and resume; `next` non-empty → continue; otherwise finish.

Settings are snapshotted into the run at `start`; `resume` reuses them unless
`--max-questions`/`--budget-usd` are passed explicitly (recorded as a warning).

## Run lifecycle and statuses

`running → paused` (user typed `exit`) `→ running → complete | partial | failed`, plus
`interrupted` (unexpected error; resumable).

- **complete** — `finalize_profile` ran; the profile is schema-valid. Remaining gaps and
  skipped questions are reported, not hidden. Reaching the page or question cap is normal.
- **partial** — the run was cut short (model-call cap, budget, or the agent stopped twice
  without finalizing); the last valid draft is exported and labeled partial in `evidence.json`.
- **failed** — model output stayed invalid after the repair attempt, or no valid draft existed
  when the run was cut off. State is kept; nothing is written as a profile.

## Tradeoffs

- **One agent vs. a hand-built graph.** A single deep agent keeps the design small and lets the
  model decide page selection and questions. The cost is less deterministic control flow, which
  the limits, the tool-level checks and the runner's finish logic compensate for.
- **Tool-level interrupt vs. `HumanInTheLoopMiddleware`.** The tool-level interrupt keeps the
  commit of the answer inside business logic (the tool), at the price of needing serial tool
  calls. HITL middleware would avoid the index-matching concern but moves the commit into the
  runner.
- **Evidence verification is strict.** A paraphrased quote is rejected. This costs the model a
  correction round sometimes, but it makes "do not invent quotations" enforceable.
- **Hybrid retrieval adds an embedding call per page** (~$0.001). It noticeably helps
  paraphrased queries ("who buys this" vs. "decision makers"); BM25-only remains available.
- **Prices are a table in code.** Cost shown is an estimate; the budget cap is therefore
  approximate too.

## Time spent

Roughly 10 hours in total: ~1.5 h reading the brief and verifying the current Deep Agents /
LangGraph / Firecrawl APIs, ~5 h implementation, ~2 h tests and fixtures, ~1 h live run and
tuning, ~0.5 h docs. AI coding assistance (Claude Code) was used throughout; every module was
reviewed and the tricky parts (interrupt replay, middleware ordering, SSRF guard) were
checked against the library sources.
