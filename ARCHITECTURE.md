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

## Core scope vs. extras

The core of the brief is the pipeline above: take a company website, discover and read a
bounded number of pages (≤ 10), draft `company_brain.json` against the fixed contract, find
the gaps and conflicts the website leaves open, interview the user in the terminal one focused
question at a time (≤ 5), stop when no useful question remains, and export a profile in which
every claim is grounded in a verbatim page excerpt or a user answer, with evidence and a report
next to it, all of it bounded, logged and resumable (`start` / `resume` / `status`).

Everything else is an addition that made the core easier to run, debug and review and is not
required by the brief: the `inspect` evidence viewer, `logs`, `doctor`, `list`, `export`,
`schema`, the USD budget cap, model tiers and cost estimates, hybrid retrieval with embeddings,
the Docker image, the fixture mode (`--fixtures` / `--save-fixtures`) used by the offline tests,
and optional LangSmith tracing. Removing any of them leaves the core pipeline intact.

## Where each behavior lives

| Concern | Middleware | Tools | Workflow code (runner) |
|---|---|---|---|
| Retries with backoff (cap = `MAX_RETRIES`, default 2) | `ToolRetryMiddleware` on `scrape_pages`/`discover_pages` with `retry_on=is_transient`; `ModelRetryMiddleware` with `retry_on=is_retryable_model_error` (everything except an OpenAI auth / permission / bad-request / not-found error) and `on_failure="error"` | classify Firecrawl errors: transient exceptions propagate, permanent ones become results + warnings (never retried) | provider rejections → status `failed` with an actionable message; exhausted transient retries → `interrupted` (resumable) |
| Bounded execution | `ModelCallLimitMiddleware` (derived: `20 + 3·pages + 4·questions`), `ToolCallLimitMiddleware` for `scrape_pages` (`ceil(pages/3) + 1`), `discover_pages` (3) and a backstop for `ask_user` (`2·max_questions + 2`; the tool enforces the exact cap) | exact page budget, fetch-attempt budget (`2 × pages`, failures included) and question cap enforced in code, with `LIMIT_*` warnings | detects the model-call cap / budget and exports the last valid draft as **partial** with a resume hint |
| Cost tracking + budget | `RunTelemetryMiddleware` records tokens/cost per attempt; `BudgetCapMiddleware.before_model` jumps to `end` (never raises, so nothing is retried or lost) and records `BUDGET_EXCEEDED` | — | prints spend; prints `resume --run-id … --budget-usd <2 × budget>` |
| Serial tool calls | `SerialToolCallsMiddleware` sets `parallel_tool_calls=False`; `SoloAskUserGuardMiddleware` rejects a bundled `ask_user` | — | — |
| Untrusted content | `UntrustedContentMiddleware` frames web tool output | injection heuristics → `INJECTION_SUSPECTED` | — |
| Logging | telemetry logs every model/tool attempt (duration, tokens, cost, status) | stage changes, page fetches, limits, questions, drafts | run start/finish, resume, failures |
| Validation + repair | — | Pydantic `CompanyBrain`; one consecutive repair attempt (`MAX_REPAIR_ATTEMPTS`, counter resets after a valid save; a crash replay of the same tool call id + payload does not count twice), then `FatalProfileError` | turns the fatal error into status `failed`, keeps state, never writes a profile |
| Grounding | — | evidence must quote a fetched page verbatim (25–1000 chars) *and* mention the value it supports; **per list item**: an excerpt cited for a whole list is recorded as `base[i]` rows for exactly the items it mentions, the rest are reported back as `NO_OVERLAP`; interview evidence must come from a question that covered the field; user answers take precedence over later drafts and cannot be overridden by website evidence; field paths allow-listed; stale evidence superseded when a value changes; `finalize_profile` pushes back once listing ungrounded fields, then export **omits** any populated field without accepted evidence and any unresolved conflict (`UNGROUNDED_OMITTED` / `CONFLICT_UNRESOLVED_OMITTED`), re-indexing list evidence so re-export is idempotent | `inspect` shows evidence per field |
| Interview | — | `ask_user` calls `langgraph.types.interrupt`, commits the answer idempotently; with `--product` the product-selection question is recorded with status `preset` and not counted | shows the question, reads input, resumes with `Command(resume={interrupt_id: answer})` |

Middleware order (first = outermost): `SerialToolCalls → [BudgetCap] → ModelCallLimit →
ToolCallLimit(ask_user) → ToolCallLimit(scrape_pages) → ToolCallLimit(discover_pages) →
SoloAskUserGuard → ModelRetry → ToolRetry → RunTelemetry → UntrustedContent`. Telemetry sits
*inside* the retry middleware so each attempt is logged and charged. Deep Agents' own core stack
stays in place except the filesystem middleware, which is replaced by a read-only instance
(`ls`, `read_file`), and the default general-purpose subagent (`task` tool), which is disabled
through a harness profile so no tool can run outside this stack; the builder fails fast if
`task` or `execute` is ever exposed.

All custom middleware re-raises `GraphBubbleUp` (the base of `GraphInterrupt`) before any
generic handler, so interrupts are never swallowed. The built-in retry middleware does the
same.

## Context engineering (the token problem)

Ten pages can be 100k+ tokens. The model never sees raw pages:

1. **Discovery is pre-filtered in code.** Firecrawl `map` (sitemap) plus homepage links are
   normalized, de-duplicated, restricted to the same registrable domain, stripped of assets /
   legal / careers / pagination, and scored by path keywords. The model sees ≤ 30 candidates
   (`MAX_CANDIDATES_TO_MODEL`; `url`, `title`, `description`, `score`) and the remaining page
   budget.
2. **Scraping returns a digest**, not text: title, size, headings, a 300-character lead. The
   markdown goes to a per-run page cache and into the index.
3. **Heading-aware chunking** (~400 tokens, small overlap). Navigation-heavy paragraphs are
   dropped; paragraphs already seen on *other* pages (shared nav, footers, cookie banners) are
   removed before chunking; chunks whose text repeats elsewhere are skipped.
4. **Hybrid retrieval.** `search_pages` runs BM25 (rank-bm25) and cosine similarity over
   `text-embedding-3-small` vectors stored in SQLite, fused with reciprocal-rank fusion, and
   returns ≤ 6 verbatim excerpts (≤ 500 chars, `MAX_EXCERPT_CHARS`) with URL + heading.
   `read_page` returns a bounded window (≤ 4000 chars) of one page/section when surrounding
   context is needed. `--no-embeddings` switches to BM25 only; if the embedding call fails the
   page is indexed with keywords only and `EMBEDDINGS_UNAVAILABLE` is recorded.
5. **Compact tool contracts.** Draft/update tools return validation results, grounding
   statistics and a prioritized gap list rather than echoing the profile.

The committed Fortanix run (`examples/fortanix/run.log`) used 17 model calls in total
(1 discover, 1 scrape, 7 searches, 2 reads, 1 draft, 2 questions, 1 update, 1 finalize, plus the
closing summary) and 197k input / 4.5k output tokens — about 11.6k input tokens per call on
average, growing over the run as tool results accumulate. The default derived cap is 70 calls.

## Persistence and the replay boundary

Two stores, one directory per run (`.runs/<run_id>/`):

- `checkpoints.sqlite` — LangGraph `SqliteSaver`, `thread_id = run_id`, `durability="sync"`.
  Holds the agent's messages and control flow, including the pending interrupt.
- `run.sqlite` — the RunStore: run metadata, stage, status, settings snapshot, counters, pages,
  chunks + embeddings, paragraph hashes, evidence (field path → source URL + verbatim excerpt,
  or question + answer; superseded rows kept), questions (pending/answered/skipped/unknown/
  preset), warnings, conflicts, draft versions, usage. Pages, questions, conflicts and warnings
  are idempotent upserts keyed by a natural key (normalized URL, question id, field path,
  `(code, message)`); chunks and paragraphs are replaced per URL; evidence rows are
  de-duplicated on insert; drafts and usage are append-only; repair attempts are de-duplicated
  per tool call id + payload digest so a replayed call is not a second attempt.
- `pages/<sha1(url)>.json` — cached page text; `discovery.json` — the candidate list;
  `events.jsonl` / `run.log` — logs; `company_brain.json`, `evidence.json`, `report.md` — outputs.

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
- `resume` rebuilds the agent from scratch (a fresh Runner/agent instance over the same SQLite
  state) and reads `get_state(config)`. A pending interrupt always wins over the stored status:
  if its answer is already committed in the RunStore (crash between commit and checkpoint) the
  graph is replayed with the durable answer without asking; otherwise the question is shown and
  answered. With no pending interrupt: a finished run (`complete` / `partial` / `failed`) prints
  the summary and the message "run already finished" unless it is `partial` and
  `--budget-usd` / `--max-questions` was passed, in which case the run is set back to `running`
  and nudged to continue; `next` non-empty → continue; no state at all → start from the initial
  message; otherwise the runner's finish logic runs.

Settings are snapshotted into the run at `start`; `resume` reuses them unless
`--max-questions`/`--budget-usd` are passed explicitly (recorded as
`SETTINGS_OVERRIDDEN_ON_RESUME`).

**Finalize is atomic.** `finalize()` computes conflict and grounding omissions on a copy of the
draft with an in-memory view of the evidence paths, validates the result with Pydantic, and only
then opens one SQLite transaction in which the warnings, conflict statuses, evidence re-indexing
(list items renumbered after deletions), the output write and the status change are applied.
A validation failure leaves the store untouched; a failed write rolls everything back; a
re-export recomputes the same omissions from the same state.

## Run lifecycle and statuses

`running → paused` (user typed `exit`) `→ running → complete | partial | failed`, plus
`interrupted` (unexpected error; resumable).

- **complete** — `finalize_profile` ran; the profile is schema-valid and every exported claim
  has accepted evidence (ungrounded values were omitted and reported). Remaining gaps and
  skipped questions are reported, not hidden. Reaching the page or question cap is normal.
- **partial** — the run was cut short (model-call cap, budget, or the agent stopped twice
  without finalizing), or zero pages could be fetched (`NO_USABLE_PAGES`; the export is
  interview-only and `evidence.json` carries `"interview_only": true`). The last valid draft is
  exported and labeled partial in `evidence.json`. `_finish` prints and returns a resume hint:
  `resume --run-id … --budget-usd <2 × budget>` when the budget was hit, otherwise
  `--max-questions <current + 2>` for the model-call cap.
- **failed** — model output stayed invalid after the repair attempt, no valid draft existed
  when the run was cut off, no page could be fetched and no draft exists, or OpenAI rejected
  the request (bad key / no access / bad request / unknown model). State is kept; nothing is
  written as a profile.
- **interrupted** — any other exception, including exhausted transient model retries;
  `RUN_INTERRUPTED` is recorded and `resume` continues from the checkpoint.

`export` of a paused / running / interrupted run writes the three outputs labeled `partial`
(and records the output path) but keeps the run status, so `resume` still works afterwards.

## Warning codes and tool result codes

Warnings (in `evidence.json`, `report.md`, `status`), from `agent/tools.py`,
`workflow/runner.py` and `agent/middleware.py`:

| Code | Meaning |
|---|---|
| `UNGROUNDED_OMITTED` | Populated fields without accepted evidence were omitted at export (listed in the message). |
| `CONFLICT_UNRESOLVED_OMITTED` | An open conflict's value was omitted at export. |
| `CONFLICT_RECORDED` | `note_conflict` stored two verified, disagreeing claims. |
| `USER_CORRECTION_SUPERSEDES_SITE` | A user answer replaced a website claim; original evidence kept as superseded. |
| `EVIDENCE_REJECTED` | An evidence item failed verification; `details.field_path` and `details.reason_code` (below). |
| `UNKNOWN_KEYS_IGNORED` | A draft contained keys outside the contract; they were stripped. |
| `INVALID_MODEL_OUTPUT` | Model output stayed invalid after the repair attempt (run fails). |
| `NO_USABLE_PAGES` | No page was fetched; `details.pages` lists every attempted `url` with its `error_code`. |
| `EMBEDDINGS_UNAVAILABLE` | Embedding call failed for a page; indexed with keyword search only. |
| `LIMIT_PAGES_REACHED` / `LIMIT_QUESTIONS_REACHED` / `LIMIT_MODEL_CALLS_REACHED` / `LIMIT_SCRAPE_ATTEMPTS_REACHED` | A cap was hit (recorded once). |
| `BUDGET_EXCEEDED` | Estimated spend reached `--budget-usd`; `details.spent_usd`, `details.budget_usd`. |
| `SETTINGS_OVERRIDDEN_ON_RESUME` | `--max-questions` / `--budget-usd` changed the snapshot on resume. |
| `INJECTION_SUSPECTED` | A page contains instruction-like text; `details.patterns`. Treated as data. |
| `REDIRECT_REJECTED` / `REDIRECT_OFFSITE` | The final URL after redirects failed the guard / left the site. |
| `PAGE_SKIPPED_<code>` | A permanent scrape failure, e.g. `PAGE_SKIPPED_HTTP_404`, `_BLOCKED`, `_PAYMENT_REQUIRED`, `_UNAUTHORIZED`, `_BAD_REQUEST`, `_FIRECRAWL_ERROR`, `_UNEXPECTED`. |
| `PAGE_TRUNCATED` / `EMPTY_CONTENT` / `DUPLICATE_CONTENT` | Page longer than 60 000 chars (cut) / under 80 chars (skipped) / same content as another URL (skipped). |
| `URL_REJECTED` / `ROBOTS_DISALLOWED` / `DISCOVERY_NOTE` | A requested URL failed the guard / robots.txt forbids it / a note from discovery (map or homepage unavailable). |
| `RUN_INTERRUPTED` / `RUN_FAILED` | Terminal outcome recorded by the runner with the (redacted) reason. |

Tool result `error_code`s (returned to the model, not warnings):

| Tool | Codes |
|---|---|
| `discover_pages` | `HOMEPAGE_UNAVAILABLE` (homepage failed permanently and no other page was discovered) |
| `scrape_pages` | per call: `TOO_MANY_URLS` (> 10 URLs; only the first 10 are considered); per URL: `URL_REJECTED`, `OFFSITE`, `FETCH_BUDGET_EXHAUSTED`, `PAGE_BUDGET_EXHAUSTED`, or the scrape error code (`RATE_LIMITED`, `TIMEOUT`, `SERVER_ERROR`, `NETWORK`, `HTTP_<status>`, `UNAUTHORIZED`, `PAYMENT_REQUIRED`, `BLOCKED`, `BAD_REQUEST`, `FIRECRAWL_ERROR`, `UNEXPECTED`); per-URL `status` is one of `fetched`, `cached`, `already_fetched`, `skipped`, `duplicate`, `rejected`, `failed`, `transient_failure`, `not_fetched` |
| `ask_user` | `BAD_PATH`, `NO_FIELD_PATHS`, `MULTI_QUESTION`, `QUESTION_LIMIT` |
| `save_profile_draft` / `apply_profile_updates` | `INVALID_PROFILE` (with `repair_attempts_remaining`); `apply_profile_updates` also `NO_DRAFT` |
| `note_conflict` | `BAD_PATH`, `INSUFFICIENT_CLAIMS` |
| `finalize_profile` | `NO_DRAFT`, `NOT_READY` (thin core section or ungrounded fields; pushed back once) |

Evidence `reason_code`s (in `evidence_rejected` / `rejected` tool results and in
`EVIDENCE_REJECTED.details`): `BAD_PATH`, `FIELD_EMPTY`, `SOURCE_NOT_FETCHED`, `TOO_SHORT`
(< 25 chars), `TOO_LONG` (> 1000 chars), `NOT_VERBATIM`, `NO_OVERLAP` (excerpt does not mention
the value / list item), `USER_ANSWER_PROTECTED`, `BAD_EVIDENCE_KIND`, `QUESTION_NOT_ANSWERED`,
`QUESTION_SCOPE`, `ANSWER_MISMATCH`, `INVALID_VALUE`.

## Tradeoffs

- **One agent vs. a hand-built graph.** A single deep agent keeps the design small and lets the
  model decide page selection and questions. The cost is less deterministic control flow, which
  the limits, the tool-level checks and the runner's finish logic compensate for.
- **Tool-level interrupt vs. `HumanInTheLoopMiddleware`.** The tool-level interrupt keeps the
  commit of the answer inside business logic (the tool), at the price of needing serial tool
  calls. HITL middleware would avoid the index-matching concern but moves the commit into the
  runner.
- **Evidence verification is strict.** A paraphrased quote is rejected, and a list needs one
  excerpt per item. This costs the model a correction round sometimes, but it makes "do not
  invent quotations" enforceable.
- **Hybrid retrieval adds an embedding call per page** (~$0.001). It noticeably helps
  paraphrased queries ("who buys this" vs. "decision makers"); BM25-only remains available.
- **Prices are a table in code.** Cost shown is an estimate; the budget cap is therefore
  approximate too.

## What the live runs taught (and changed)

Nine live runs against fortanix.com shaped the final behavior:

- A 60 s model timeout cut off the first full draft from a reasoning model → 180 s.
- The agent interviewed before researching → the draft tool now reports thin sections with
  concrete search queries, and `finalize_profile` refuses once if a core section is empty.
- A later, thinner draft erased earlier work → drafts merge; an empty value never overwrites a
  filled one, and evidence is replaced per field rather than wholesale.
- Markdown link/image syntax made honest quotes fail verification → links are reduced to their
  text before chunking and the matcher ignores brackets, images and list markers.
- Feature-detail gaps produced weak questions ("write how it works") → priorities favour
  decisions and preferences; questions must be single (bundled lists are rejected).
- The model invented an extra key once and failed the run → unknown keys are stripped with a
  warning before validation.
- Firecrawl's free tier rate-limits bursts → longer backoff and 1.5 s pacing between fetches.
- A second external review found grounding/recovery gaps → export now omits unsupported
  claims, user answers are protected from later drafts, conflict omission is idempotent with
  evidence re-indexing, a crash between answer-commit and checkpoint no longer re-asks,
  partial runs resume after raising limits, 5xx failures are refetched later, the page budget
  counts unique scrapes (duplicates included) and discovery reuses the cached homepage,
  transient error text is redacted before storage, interrupted runs exit non-zero, short
  factual lines survive chunking, and model-call telemetry carries the real stage.
- A third review round closed the remaining loopholes → one excerpt no longer grounds a whole
  list (evidence is per item), finalize is one transaction, OpenAI auth/model errors fail fast
  with a clear message instead of being retried, "no usable pages" is diagnosed per URL and
  exports an interview-only partial when a draft exists, rejected `ask_user` calls and crash
  replays no longer consume the question or repair budget, `--product` is recorded without
  counting as a question, and `export` of an unfinished run keeps its status.

## Time spent

Roughly 18 hours in total: ~1.5 h reading the brief and verifying the current Deep Agents /
LangGraph / Firecrawl APIs against their sources, ~5 h implementation, ~2.5 h tests and fixtures,
~1 h independent security review and fixes, ~3 h live runs and tuning (nine runs, ~$0.60 of API
spend in total), ~4 h on the second and third review rounds (fixes, regression tests, and
reconciling the docs with the code), ~1 h initial docs. AI coding assistance (Claude Code) was
used throughout; every module was reviewed and the tricky parts (interrupt replay, middleware
ordering, SSRF guard, evidence verification) were checked against library sources and live
behaviour.
