# Company Profile Builder

A Python CLI that turns a company website into a grounded `company_brain.json`: it discovers
and reads the most relevant pages, drafts a profile against a fixed contract, finds the gaps
and conflicts the website leaves open, interviews you in the terminal (one focused question
at a time), and exports a validated profile with full evidence.

Built on the **Deep Agents SDK** (LangGraph runtime), **OpenAI** models and **Firecrawl**.
Every claim in the output is tied to a verbatim excerpt of a fetched page or to one of your
answers; the evidence lives next to the profile in `evidence.json`.

```
$ python -m profile_builder start --url https://www.fortanix.com/ --product "Confidential Computing Platform"
Run id: pb-20261002-5bsvmi  (resume later with: python -m profile_builder resume --run-id pb-20261002-5bsvmi)
INFO  stage → discover … 30 candidates from map+homepage_links (22 dropped)
INFO  stage → scrape   … indexed https://www.fortanix.com/platform/confidential-computing-manager (18 chunks)
INFO  stage → draft    … draft v1 saved (71 evidence rows, 2 rejected, 4 gaps)
╭─ Question 1/5 ───────────────────────────────────────────────────────────────╮
│ For the Confidential Computing Platform, who typically makes the buying      │
│ decision?                                                                    │
│ Why this is unclear: The website establishes industries and technical        │
│ capabilities, but it does not clearly say which role usually owns purchase   │
│ decisions for this product.                                                  │
│ Answer, or type skip · idk · exit (save and resume later)                    │
╰──────────────────────────────────────────────────────────────────────────────╯
> CISO, CIO, Head of Data Security and Compliance, VP of Cloud Infrastructure
…
╭─ Company Profile Builder ─────────────────────────────────────────────────────╮
│ Status     COMPLETE   Pages 10 fetched   Questions 2 asked (max 5)            │
│ Model      gpt-5.4 · 17 calls · 197,097 in / 4,503 out   Est. cost $0.1941    │
│ grounded fields 75/79                                                         │
│ Output     .runs/pb-20261002-5bsvmi/company_brain.json                        │
╰───────────────────────────────────────────────────────────────────────────────╯
```

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
(coverage, gaps, pages, interview), `events.jsonl` and `run.log`.

An example Fortanix run (quality tier) is committed under [`examples/fortanix/`](examples/fortanix/)
with its profile, evidence, report, logs and terminal transcript.

**Model tiers.** `fast` (`gpt-5-mini`, default) completes a run for roughly $0.03–0.08 but varies
more between runs; `quality` (`gpt-5.4`) costs about $0.15–0.25 and grounds and phrases more
consistently. Use `--tier quality` for a profile you intend to keep.

## Commands

| Command | What it does |
|---|---|
| `start --url URL [--product NAME] [options]` | New run. `--product` skips the "which product?" question. |
| `resume --run-id ID [--max-questions N] [--budget-usd X]` | Continue from the last checkpoint; answered questions are never re-asked. |
| `status --run-id ID` | Stage, status, counters, pending question, interview, warnings, cost. |
| `inspect --run-id ID [--field PATH]` | Every populated field with its evidence (URL + verbatim excerpt, or Q&A). |
| `export --run-id ID` | Re-export the three output files from the latest saved draft. |
| `logs --run-id ID [--tail N] [--level L] [--json]` | Structured event log of a run. |
| `list` | All runs in the runs directory. |
| `schema [--out PATH]` | JSON Schema generated from the Pydantic contract. |
| `doctor` | Keys present, providers reachable, runs dir writable (never prints keys). |

`start` options: `--max-pages N`, `--max-questions N`, `--model ID` / `--tier fast|quality`,
`--budget-usd X`, `--no-embeddings`, `--non-interactive` (auto-skip questions, for demos/CI),
`--fixtures DIR` (serve pages from saved fixtures instead of Firecrawl), `--save-fixtures DIR`,
`--runs-dir DIR`, `-v/--verbose`, `-q/--quiet`.

## Configuration

Precedence: CLI flag > environment variable > `.env` > default. Every limit is one variable;
derived caps (model calls, scrape calls) follow it automatically.

| Variable | Flag | Default | Meaning |
|---|---|---|---|
| `PROFILE_BUILDER_MAX_PAGES` | `--max-pages` | 10 | Unique pages scraped per run |
| `PROFILE_BUILDER_MAX_QUESTIONS` | `--max-questions` | 5 | Interview questions incl. follow-ups |
| `PROFILE_BUILDER_TIER` / `PROFILE_BUILDER_MODEL` | `--tier` / `--model` | `fast` → `gpt-5-mini` | `quality` → `gpt-5.4`; `--model` takes any OpenAI id |
| `PROFILE_BUILDER_BUDGET_USD` | `--budget-usd` | unlimited | Stop cleanly at this estimated spend; the run stays resumable with a higher cap |
| `PROFILE_BUILDER_USE_EMBEDDINGS` | `--no-embeddings` | true | Hybrid BM25 + `text-embedding-3-small`, or BM25 only |
| `PROFILE_BUILDER_SCRAPE_TIMEOUT_MS` | — | 30000 | Firecrawl timeout |
| `PROFILE_BUILDER_MAX_RETRIES` | — | 2 | Retries after the first attempt (model and scrape) |
| `PROFILE_BUILDER_RUNS_DIR` | `--runs-dir` | `.runs` | Where run state lives |
| `OPENAI_API_KEY`, `FIRECRAWL_API_KEY` | — | required | Provider credentials (never passed as flags) |

Optional LangSmith tracing works through the standard `LANGSMITH_*` variables.

### Exit codes

`0` complete, partial or paused by the user · `1` failed (no profile written) · `3` stopped on an
error and resumable (`resume --run-id …`).

### What happens at the limits

- **Page / question cap reached** — normal; the run finishes as `complete` and the report
  lists what was not fetched or asked.
- **Budget reached** — the agent stops before the next model call, the last valid draft is
  exported as `partial`, and the summary prints `resume --run-id … --budget-usd <higher>`.
  Resuming with a higher budget (or `--max-questions`) continues the same run from its checkpoint.
- **Model-call cap reached** — same as budget (`partial`).
- **Invalid model output** — one repair attempt; if it fails again the run is `failed`, state
  is kept, and no profile is written.
- **No usable pages** — the run fails with the reason; nothing is fabricated.

## How it works

See [ARCHITECTURE.md](ARCHITECTURE.md) for the tools / middleware / workflow split,
persistence and the replay boundary, the context-engineering design and tradeoffs, and
[SECURITY.md](SECURITY.md) for the threat model.

In short: deterministic code discovers and pre-scores candidate pages; the agent picks ≤ 10,
pages are cached, chunked (heading-aware, nav/footer de-duplicated across pages) and indexed;
the agent gathers evidence through hybrid search rather than reading raw pages; drafts are
validated with Pydantic and every evidence excerpt must be a verbatim quote of a fetched page;
gaps and conflicts are prioritized in code and the agent asks one focused question at a time
via a LangGraph interrupt; answers are committed to SQLite before the graph advances and take
precedence over anything the website says. At export, a populated field without accepted
evidence (a verbatim page excerpt or a user answer) is omitted rather than shipped, so every
claim in `company_brain.json` is traceable in `evidence.json`.

## Development

```bash
make test       # offline test suite (scripted model + fixture site; no keys needed)
make lint       # ruff check + format check
make schema     # regenerate schema/company_brain.schema.json
make audit      # pip-audit
```

Tests cover: settings precedence and derived limits; the output contract and JSON Schema;
SSRF guard cases (private/loopback/link-local/IPv6/DNS-rebinding/userinfo/ports/redirects);
secret redaction in logs; chunking + hybrid search; a full scrape → interview → export run;
a conflicting fact resolved by a user correction (original evidence preserved); 429/timeout
followed by recovery with capped retries and 404 never retried; invalid model output with
one repair then a clean failure; exit during the interview and resume in a new process
without re-asking; the CLI commands.

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
  agent/tools.py         the 9 tools               agent/middleware.py middleware stack
  agent/prompts.py       system prompt             agent/builder.py  model/checkpointer/agent
  state/run_store.py     SQLite run state          workflow/runner.py start/resume/interview loop
  workflow/gaps.py       gap + grounding analysis  workflow/export.py / summary.py  outputs + terminal UI
schema/company_brain.schema.json   examples/fortanix/   tests/
```

License: MIT.
