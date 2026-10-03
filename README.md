# Company Profile Builder

A Python CLI that turns a company website into a grounded `company_brain.json`: it discovers
and reads the most relevant pages, drafts a profile against a fixed contract, finds the gaps
and conflicts the website leaves open, interviews you in the terminal (one focused question
at a time), and exports a validated profile with full evidence.

Built on the **Deep Agents SDK** (LangGraph runtime), **OpenAI** models and **Firecrawl**.
Every claim in the output is tied to a verbatim excerpt of a fetched page or to one of your
answers; the evidence lives next to the profile in `evidence.json`.

Abridged from [`examples/fortanix/transcript.txt`](examples/fortanix/transcript.txt) (quality
tier; `…` marks elided lines; the answer line is taken from `report.md` because the terminal
transcript does not echo typed input):

```
$ uv run python -m profile_builder start --url https://www.fortanix.com/ --product "Confidential Computing Platform" --tier quality
INFO     run pb-20261002-ulvek5 started for https://www.fortanix.com/
Run id: pb-20261002-ulvek5  (resume later with: python -m profile_builder resume --run-id pb-20261002-ulvek5)
INFO     stage → discover
INFO     indexed https://www.fortanix.com/ (28985 chars, 19 chunks, 0 repeated blocks dropped)
INFO     30 candidates from map+homepage_links (24 dropped)
INFO     stage → scrape
INFO     indexed https://www.fortanix.com/platform/confidential-computing (49642 chars, 33 chunks, 6 repeated blocks dropped)
INFO     indexed https://www.fortanix.com/platform/confidential-computing-manager (19445 chars, 7 chunks, 6 repeated blocks dropped)
…
INFO     stage → research
INFO     search_pages finished
…
INFO     stage → draft
WARNING  product.features_and_capabilities[3]: excerpt is not a verbatim quote from that page
INFO     draft v1 saved (60 evidence rows, 0 rejected, 3 gaps)
INFO     stage → interview
╭──────────────────────────────── Question 1/5 ────────────────────────────────╮
│ For the Confidential Computing Platform, who typically makes the buying      │
│ decision?                                                                    │
│                                                                              │
│ Why this is unclear: The website establishes industries and technical        │
│ capabilities, but it does not clearly say which role usually owns purchase   │
│ decisions for this product.                                                  │
│                                                                              │
│ Affects: customer.buyers                                                     │
│                                                                              │
│ Answer, or type skip · idk · exit (save and resume later)                    │
╰──────────────────────────────────────────────────────────────────────────────╯
> CISO, CIO, Head of Data Security and Compliance, VP of Cloud Infrastructure
…
INFO     profile exported (complete) → …/.runs/pb-20261002-ulvek5/company_brain.json
╭────────────────────────── Company Profile Builder ───────────────────────────╮
│ Run            pb-20261002-ulvek5                                            │
│ Website        https://www.fortanix.com/                                     │
│ Product focus  Confidential Computing Platform                               │
│ Status         COMPLETE                                                      │
│ Pages          10 fetched · 0 skipped · 0 failed                             │
│ Questions      2 asked (max 5)                                               │
│ Model          gpt-5.4 · 21 calls · 277,449 in / 7,521 out                   │
│ Est. cost      $0.1941 (no budget cap)                                       │
│ Output         /Users/srikanth/projects/company-profile-builder/.runs/pb-20… │
╰──────────────────────────────────────────────────────────────────────────────╯
```

The committed example artifacts were produced by an earlier code version (before the
per-list-item grounding rule and the `interview_only` field) and will be regenerated; the
numbers above match the files as committed.

## Quick start

Requirements: Python 3.12+, [`uv`](https://docs.astral.sh/uv/), an OpenAI API key and a
Firecrawl API key (Firecrawl's free tier is enough for a few runs).

```bash
git clone <this repo> && cd company-profile-builder
uv sync --extra dev                 # or: make install
cp .env.example .env                # then put OPENAI_API_KEY and FIRECRAWL_API_KEY in .env
uv run python -m profile_builder doctor
uv run python -m profile_builder start --url https://www.fortanix.com/ --product "Confidential Computing Platform"
```

During the interview type an answer, `skip`, `idk` (I don't know) or `exit` to save and leave.
Resume any time:

```bash
uv run python -m profile_builder resume --run-id pb-20261002-k3x9qa
```

Outputs land in `.runs/<run_id>/`: `company_brain.json` (the contract, nothing else),
`evidence.json` (sources, excerpts, questions, warnings, conflicts, usage), `report.md`
(coverage, gaps, pages, interview), `events.jsonl` and `run.log`. The JSON Schema of the
contract is committed as `schema/company_brain.schema.json`; it marks every property as
required at every level (unknown values are `""` / `[]`, never missing keys).

An example Fortanix run (quality tier) is committed under [`examples/fortanix/`](examples/fortanix/)
with its profile, evidence, report, logs and terminal transcript.

**Model tiers.** `fast` (`gpt-5-mini`, default) and `quality` (`gpt-5.4`). In live runs so far
the fast tier completed a run for roughly $0.03–0.08 with more variance between runs; the
quality tier cost about $0.15–0.25 (the committed example: $0.19) and grounded and phrased more
consistently. Use `--tier quality` for a profile you intend to keep.

## Commands

| Command | What it does |
|---|---|
| `start --url URL [--product NAME] [options]` | New run. `--product` presets the product focus: the "which product?" question is recorded with status `preset` and does not count toward the question cap. |
| `resume --run-id ID [--max-questions N] [--budget-usd X] [--non-interactive] [--fixtures DIR] [--runs-dir DIR]` | Continue from the last checkpoint; answered questions are never re-asked. A `partial` run continues only when `--max-questions` or `--budget-usd` is passed. |
| `status --run-id ID` | Stage, status, counters, pending question, interview, warnings, cost. |
| `inspect --run-id ID [--field PATH]` | Every populated field with its evidence (URL + verbatim excerpt, or Q&A). |
| `export --run-id ID` | Re-export the three output files from the latest saved draft. For a paused / running / interrupted run the outputs are labeled `partial` but the run status is kept so `resume` still works. |
| `logs --run-id ID [--tail N] [--level L] [--json]` | Structured event log of a run. |
| `list` | All runs in the runs directory. |
| `schema [--out PATH]` | JSON Schema generated from the Pydantic contract. |
| `doctor` | Keys present, providers reachable, runs dir writable (never prints keys). Exits 1 if any check fails. |

`start` options: `--max-pages N`, `--max-questions N`, `--model ID` / `--tier fast|quality`,
`--budget-usd X`, `--no-embeddings`, `--non-interactive` (auto-skip questions, for demos/CI),
`--fixtures DIR` (serve pages from saved fixtures instead of Firecrawl), `--save-fixtures DIR`,
`--runs-dir DIR`.

Global options go **before** the subcommand: `-v/--verbose` (debug logging on the console) and
`-q/--quiet` (warnings and errors only), e.g. `python -m profile_builder -v start --url …`.

## Configuration

Precedence: CLI flag > environment variable > `.env` > default. Every limit is one variable;
derived caps (model calls, scrape calls) follow it automatically.

| Variable | Flag | Default | Meaning |
|---|---|---|---|
| `PROFILE_BUILDER_MAX_PAGES` | `--max-pages` | 10 | Unique pages scraped per run |
| `PROFILE_BUILDER_MAX_QUESTIONS` | `--max-questions` | 5 | Interview questions incl. follow-ups |
| `PROFILE_BUILDER_TIER` / `PROFILE_BUILDER_MODEL` | `--tier` / `--model` | `fast` → `gpt-5-mini` | `quality` → `gpt-5.4`; `--model` takes any OpenAI id |
| `PROFILE_BUILDER_EMBEDDING_MODEL` | — | `text-embedding-3-small` | Embedding model for hybrid retrieval |
| `PROFILE_BUILDER_USE_EMBEDDINGS` | `--no-embeddings` | true | Hybrid BM25 + embeddings, or BM25 only |
| `PROFILE_BUILDER_BUDGET_USD` | `--budget-usd` | unlimited | Stop cleanly at this estimated spend; the run stays resumable with a higher cap |
| `PROFILE_BUILDER_SCRAPE_TIMEOUT_MS` | — | 30000 | Firecrawl timeout |
| `PROFILE_BUILDER_MAX_RETRIES` | — | 2 | Retries after the first attempt (model and scrape) |
| `PROFILE_BUILDER_RUNS_DIR` | `--runs-dir` | `.runs` | Where run state lives |
| `OPENAI_API_KEY`, `FIRECRAWL_API_KEY` | — | required | Provider credentials (never passed as flags) |

Optional LangSmith tracing works through the standard `LANGSMITH_*` variables.

### Exit codes

- `0` — run `complete` or `partial`, or paused by the user (`exit` / Ctrl-C during the interview).
- `1` — run `failed`: invalid model output after the single repair attempt, no usable pages and
  no draft, or OpenAI rejected the request (authentication / permission / bad-request / not-found
  errors are not retried and are reported as
  `OpenAI rejected the request (check OPENAI_API_KEY / model id)`). No profile is written.
- `2` — usage or configuration error: start URL rejected by the guard, missing API key, run id
  not found, invalid option. (`doctor` exits `1` when a check fails.)
- `3` — stopped on an unexpected error; state is kept and the run is resumable with
  `resume --run-id …` (exhausted transient model retries end here too).

### What happens at the limits

- **Page cap reached** — normal; the run finishes as `complete`. A single `LIMIT_PAGES_REACHED`
  warning is recorded. URLs that did not fit are returned to the agent in the `scrape_pages`
  result as `not_fetched` (`PAGE_BUDGET_EXHAUSTED`); they are not listed in the report.
- **Question cap reached** — normal; `ask_user` returns `QUESTION_LIMIT`, a
  `LIMIT_QUESTIONS_REACHED` warning is recorded and the run finishes as `complete`.
- **Budget reached** — the budget middleware stops before the next model call
  (`BUDGET_EXCEEDED`), the last valid draft is exported as `partial`, and the summary prints
  `resume --run-id … --budget-usd <2 × budget>`.
- **Model-call cap reached** (`20 + 3·pages + 4·questions`) — same as budget (`partial`,
  `LIMIT_MODEL_CALLS_REACHED`); the hint is `resume --run-id … --max-questions <current + 2>`,
  because more questions raise the derived cap.
- **Resuming a `partial` run** continues from the checkpoint only when `--budget-usd` or
  `--max-questions` is passed (recorded as `SETTINGS_OVERRIDDEN_ON_RESUME`); without an override
  `resume` prints the summary and says the run already finished. A pending interview question
  always takes precedence: `resume` asks it regardless of the stored status.
- **Invalid model output** — one repair attempt; if it fails again the run is `failed`, state
  is kept, and no profile is written. If an earlier valid draft exists the console shows how to
  `export` it.
- **No usable pages** — a `NO_USABLE_PAGES` warning lists every attempted URL with its
  `error_code`. With no draft the run is `failed`; with a draft (interview-only) it is exported
  as `partial` with `"interview_only": true` in `evidence.json`. Nothing is fabricated.
- **Unexpected error** — status `interrupted`, exit code 3, `RUN_INTERRUPTED` warning, resume hint.

## How it works

See [ARCHITECTURE.md](ARCHITECTURE.md) for the tools / middleware / workflow split,
persistence and the replay boundary, the context-engineering design and tradeoffs, and
[SECURITY.md](SECURITY.md) for the threat model.

In short: deterministic code discovers and pre-scores candidate pages; the agent picks ≤ 10,
pages are cached, chunked (heading-aware, nav/footer de-duplicated across pages) and indexed;
the agent gathers evidence through hybrid search rather than reading raw pages; drafts are
validated with Pydantic and every evidence excerpt must be a verbatim quote of a fetched page
that mentions the value it supports (list items are grounded one by one); gaps and conflicts
are prioritized in code and the agent asks one focused question at a time via a LangGraph
interrupt; answers are committed to SQLite before the graph advances and take precedence over
anything the website says. At export, a populated field without accepted evidence (a verbatim
page excerpt or a user answer) is omitted rather than shipped (`UNGROUNDED_OMITTED`), so every
claim in `company_brain.json` is traceable in `evidence.json`.

## Docker

```bash
docker build -t profile-builder .
docker run -it --user $(id -u):$(id -g) \
  -e OPENAI_API_KEY -e FIRECRAWL_API_KEY \
  -v "$PWD/.runs:/runs" \
  profile-builder start --url https://www.fortanix.com/ --product "Confidential Computing Platform"
```

`-it` is required for the interview (the agent waits for terminal input). Run state is written
to `/runs` inside the container (`PROFILE_BUILDER_RUNS_DIR=/runs`), so mount a host directory
there and pass `--user` so the files belong to you. Any subcommand works the same way, e.g.
`… profile-builder resume --run-id pb-…`. The image runs as a non-root user (UID 10001) by default.

## Development

```bash
make test       # offline test suite (scripted model + fixture site; no keys needed)
make lint       # ruff check + format check
make schema     # regenerate schema/company_brain.schema.json
make audit      # pip-audit (advisory: never fails the build)
make docker     # docker build -t profile-builder .
```

Tests cover: settings precedence and derived limits; the output contract and JSON Schema (every
property required); SSRF guard cases (private/loopback/link-local/IPv6/DNS-rebinding/userinfo/
ports/redirects, including a redirect whose final URL is a private IP); secret redaction in logs
and stored errors; chunking + hybrid search; a full scrape → interview → export run; a
conflicting fact resolved by a user correction (original evidence preserved); 429/timeout
followed by recovery with capped retries, 404 never retried, 5xx refetched later; the
fetch-attempt budget and the 10-URLs-per-call cap; invalid model output with one repair then a
clean failure (a crash replay of the same call does not count twice); OpenAI auth errors not
retried and reported; an unexpected error leaving the run `interrupted` and resumable; exit
during the interview and resume with a fresh Runner/agent instance over the same SQLite state
without re-asking; a list excerpt grounding only the items it mentions; ungrounded fields and
unresolved conflicts omitted at export with idempotent re-indexing; a failed output write rolling
the store back; export of a paused run keeping its status; no usable pages → failed or
interview-only partial; `--product` not counted as a question; the CLI commands and exit codes.

Run a fixture-backed session without Firecrawl credits (the LLM is still real):

```bash
uv run python -m profile_builder start --url https://www.fortanix.com/ --fixtures tests/fixtures/fortanix --product "Confidential Computing Platform"
```

## Project layout

```
src/profile_builder/
  cli.py                 typer commands            config.py        settings + internal constants
  models.py              model tiers + prices      schema.py        company_brain contract + field paths
  logging_setup.py       events.jsonl/run.log      security.py      run ids, paths, atomic writes, evidence checks
  web/url_guard.py       SSRF guard                web/robots.py    robots.txt
  web/scraper.py         Firecrawl + fixtures      web/discovery.py candidate discovery + scoring
  retrieval/chunking.py  heading-aware chunks      retrieval/index.py  BM25 + embeddings + RRF
  agent/tools/           the 9 tools (web, interview, draft groups)   agent/middleware.py middleware stack
  agent/context.py       ToolContext + fatal error  agent/ingest.py     page caching/chunking/indexing
  agent/evidence.py      evidence verification      agent/drafting.py   validate/merge/repair drafts
  agent/interview.py     question ids + answers     agent/finalize.py   omission + atomic export
  agent/prompts.py       system prompt             agent/builder.py  model/checkpointer/agent
  state/run_store.py     SQLite run state          workflow/runner.py start/resume/interview loop
  workflow/gaps.py       gap + grounding analysis  workflow/export.py / summary.py  outputs + terminal UI
schema/company_brain.schema.json   examples/fortanix/   tests/
```

License: MIT.
