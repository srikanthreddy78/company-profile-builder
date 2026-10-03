# Example run: Fortanix — Confidential Computing Platform

A real, unedited run against https://www.fortanix.com/ (2026-10-02) produced with:

```bash
uv run python -m profile_builder start --url https://www.fortanix.com/ --product "Confidential Computing Platform" --tier quality
# … the run pauses at each interview question; this example answered them via
uv run python -m profile_builder resume --run-id <run-id>
```

| File | What it is |
|---|---|
| `company_brain.json` | The exported profile: exactly the contract keys, nothing else. |
| `evidence.json` | Everything outside the contract: pages fetched, every evidence row (field path → source URL + verbatim excerpt, or question + answer, with superseded rows kept), the interview, conflicts, warnings, gaps, grounding stats, settings snapshot and token usage. |
| `report.md` | Human-readable coverage, grounding, pages, interview transcript, remaining gaps and warnings. |
| `transcript.txt` | The terminal session (`start` plus each `resume`), including the question panels and the final summary. |
| `events.jsonl` / `run.log` | Structured and plain logs: stages, every model and tool attempt with duration/tokens/cost, retries, interrupts, committed answers, warnings. Secrets are redacted. |

Result: status **complete**, 10 pages fetched, 2 interview questions asked (the agent stopped
early because no useful question remained), 75 of 79 populated fields grounded with verified
excerpts, 17 model calls with `gpt-5.4`, estimated cost $0.19. Run ids and timestamps in the logs
belong to this run.

The `quality` tier was used for the committed example because it grounds and phrases more
consistently. The default `fast` tier (`gpt-5-mini`) completes the same run for roughly
$0.03–0.08 with more variance between runs (typically 40–55 of ~65 fields grounded, and
occasionally a run that fails cleanly after the single allowed repair attempt).

## How to read it

- The page cap (10) and question cap (5) applied; the agent stopped asking before the cap when
  no useful question remained, which the brief asks for.
- Every populated field with website evidence has a verbatim excerpt in `evidence.json`;
  fields the model filled without accepted evidence are listed under `grounding.ungrounded` and
  in the report. Rejected evidence (not verbatim, too short, or not mentioning the value) is
  recorded as `EVIDENCE_REJECTED` warnings rather than silently accepted.
- Where an interview answer replaced a website claim, the original website evidence is kept
  and marked `superseded`, and a `USER_CORRECTION_SUPERSEDES_SITE` warning is recorded.

## About the interview answers

Tavyn's design partner was not available for this run, so the candidate answered the
interview questions as a proxy using publicly reasonable knowledge of Fortanix (buyer/user
roles, alternatives, claims to avoid). They are plausible, not authoritative; a real run with the
partner would replace them. The questions themselves, and the way answers flow into the
profile, are what this example demonstrates.
