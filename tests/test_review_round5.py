"""Regression tests for the fifth review round: short-token values (acronyms) are verified
instead of auto-passing, discovery accounts for the homepage before the site map so a
transient `map` failure never re-scrapes it, and a process kill between output renames is
repaired from the publish journal."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from typer.testing import CliRunner

from profile_builder.agent.evidence import supports, unsupported_by_answer
from profile_builder.cli import app
from profile_builder.logging_setup import read_events
from profile_builder.security import url_cache_name
from profile_builder.state.run_store import RunStore
from profile_builder.web.scraper import FixtureScraper, TransientScrapeError
from profile_builder.workflow.export import (
    PUBLISH_JOURNAL_FILENAME,
    recover_interrupted_publish,
    write_all_or_nothing,
)
from tests.conftest import (
    DRAFT_EVIDENCE,
    DRAFT_PROFILE,
    SITE,
    finalize_steps,
    last_tool_result,
    make_runner,
    tool_call,
)

OUTPUTS = ("company_brain.json", "evidence.json", "report.md")


def _recorder(seen, next_step):
    def step(messages):
        seen.append(last_tool_result(messages))
        return next_step(messages) if callable(next_step) else next_step

    return step


def _prefix():
    return [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call(
            "scrape_pages", {"urls": [f"{SITE}/product", f"{SITE}/customers", f"{SITE}/about"]}
        ),
    ]


# ---------------------------------------------------------------------------------------
# 1. values made of short tokens are verified, not waved through
# ---------------------------------------------------------------------------------------


def test_R1_acronym_list_is_not_supported_by_an_unrelated_answer():
    assert unsupported_by_answer(
        "customer.buyers", ["CEO", "CTO", "CIO"], "Security teams only"
    ) == [
        "CEO",
        "CTO",
        "CIO",
    ]
    assert not supports("customer.buyers[0]", "CEO", "Security teams only")


def test_R1_short_token_in_answer_matches_short_value_exactly():
    assert supports("customer.buyers[0]", "CIO", "CISO, CIO")
    assert supports("customer.buyers[0]", "cio", "The CIO and the CFO decide")
    # a short value never matches on a prefix of a longer word, and vice versa
    assert not supports("customer.buyers[0]", "CIO", "CISO only")
    assert not supports("customer.buyers[0]", "CISO", "CIO only")
    # long-token values keep the stem rule
    assert supports("customer.users", ["Security teams"], "Security teams only")
    assert not supports("customer.users", ["Martian procurement officers"], "Security teams only")


def test_R1_value_without_tokens_cannot_be_verified():
    assert not supports("customer.buyers[0]", "...", "Security teams only")
    assert not supports("customer.buyers[0]", "—", "anything at all")
    assert unsupported_by_answer("customer.buyers", ["..."], "Security teams only") == ["..."]


def test_R1_observed_brand_patterns_keep_their_exemption():
    assert supports("brand.voice_and_tone", "...", "whatever passage")
    assert supports("brand.voice_and_tone[0]", "CIO", "no overlap here")
    assert supports("brand.writing_style", ["Terse"], "a long illustrative passage")


def test_R1_interview_acronyms_are_rejected_with_answer_mismatch(settings, acme_fixtures):
    q = {
        "question": "Who buys Acme Vault?",
        "why_unclear": "the site does not say",
        "field_paths": ["customer.buyers"],
        "kind": "gap",
    }
    seen: list[dict] = []

    def apply(messages):
        res = last_tool_result(messages)
        assert res["status"] == "answered"
        return tool_call(
            "apply_profile_updates",
            {
                "updates": [
                    {
                        "field_path": "customer.buyers",
                        "value": ["CEO", "CTO", "CIO"],
                        "evidence": {"kind": "interview", "question_id": res["qid"]},
                    }
                ]
            },
        )

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call("ask_user", q),
        apply,
        _recorder(seen, tool_call("finalize_profile", {})),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(
        settings, steps, scraper=FixtureScraper(acme_fixtures), answers=["Security teams only"]
    )
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    res = seen[0]
    assert res["ok"] is False and res["applied"] == []
    assert res["rejected"][0]["reason_code"] == "ANSWER_MISMATCH"
    assert res["rejected"][0]["unsupported"] == ["CEO", "CTO", "CIO"]
    brain = json.loads((settings.runs_dir / outcome.run_id / "company_brain.json").read_text())
    assert brain["customer"]["buyers"] == []


# ---------------------------------------------------------------------------------------
# 2. discovery accounts for the homepage before the site map
# ---------------------------------------------------------------------------------------


class MapFlakyScraper(FixtureScraper):
    """`map` raises a transient error the first `n` times; `scrape` behaves normally."""

    def __init__(self, fixture_dir: Path, n: int) -> None:
        super().__init__(fixture_dir)
        self.map_failures_left = n

    def map(self, url, *, limit, timeout_ms):
        if self.map_failures_left > 0:
            self.map_failures_left -= 1
            self.calls.append(("map", url))
            raise TransientScrapeError("map timed out", code="TIMEOUT")
        return super().map(url, limit=limit, timeout_ms=timeout_ms)


def _two_discover_steps(seen):
    return [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        _recorder(seen, tool_call("discover_pages", {"start_url": f"{SITE}/"})),
        _recorder(seen, AIMessage(content="done")),
        AIMessage(content="done"),
    ]


def test_R2_transient_map_failures_never_rescrape_the_homepage(settings, acme_fixtures):
    settings = settings.model_copy(update={"max_retries": 2})
    scraper = MapFlakyScraper(acme_fixtures, 10)
    seen: list[dict] = []
    runner, _, _ = make_runner(settings, _two_discover_steps(seen), scraper=scraper)
    outcome = runner.start(f"{SITE}/")
    run_dir = settings.runs_dir / outcome.run_id
    store = RunStore(run_dir)
    assert [c for c in scraper.calls if c[0] == "scrape"] == [("scrape", f"{SITE}/")]
    assert store.get_counter("scrape_attempts") == 1
    assert store.unique_pages_scraped() == 1 and f"{SITE}/" in store.fetched_urls()
    assert (run_dir / "pages" / url_cache_name(f"{SITE}/")).exists()
    # the second discover call succeeds from the cached homepage's links
    assert len(seen) == 2 and seen[-1]["ok"] is True, seen
    assert f"{SITE}/product" in {c["url"] for c in seen[-1]["candidates"]}
    assert seen[-1]["source"] == "homepage_links"
    assert store.has_warning("DISCOVERY_NOTE")
    # the map was retried a bounded number of times, never once per attempt forever
    assert 1 <= len([c for c in scraper.calls if c[0] == "map"]) <= 2 * (settings.max_retries + 1)


def test_R2_map_retry_reuses_the_cached_homepage(settings, acme_fixtures):
    settings = settings.model_copy(update={"max_retries": 2})
    scraper = MapFlakyScraper(acme_fixtures, 1)
    seen: list[dict] = []
    steps = [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        _recorder(seen, AIMessage(content="done")),
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=scraper)
    outcome = runner.start(f"{SITE}/")
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert [c for c in scraper.calls if c[0] == "scrape"] == [("scrape", f"{SITE}/")]
    assert [c for c in scraper.calls if c[0] == "map"] == [("map", f"{SITE}/")] * 2
    assert store.get_counter("scrape_attempts") == 1
    # the map lists every homepage link already, so the source is the map itself
    assert seen[0]["ok"] is True and seen[0]["source"] == "map"
    assert f"{SITE}/customers" in {c["url"] for c in seen[0]["candidates"]}


class ScrapeFlakyScraper(FixtureScraper):
    """`scrape` raises a transient error the first `n` times; `map` behaves normally
    (`FixtureScraper.failures` would make both fail, being keyed by URL)."""

    def __init__(self, fixture_dir: Path, n: int) -> None:
        super().__init__(fixture_dir)
        self.scrape_failures_left = n

    def scrape(self, url, *, timeout_ms, with_links=False):
        if self.scrape_failures_left > 0:
            self.scrape_failures_left -= 1
            self.calls.append(("scrape", url))
            raise TransientScrapeError("rate limited", code="RATE_LIMITED")
        return super().scrape(url, timeout_ms=timeout_ms, with_links=with_links)


def test_R2_homepage_fetch_counts_toward_the_attempt_cap(settings, acme_fixtures):
    scraper = ScrapeFlakyScraper(acme_fixtures, 10)
    settings = settings.model_copy(update={"max_retries": 2, "max_pages": 1})
    seen: list[dict] = []
    steps = [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        _recorder(seen, tool_call("discover_pages", {"start_url": f"{SITE}/"})),
        _recorder(seen, AIMessage(content="done")),
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=scraper)
    outcome = runner.start(f"{SITE}/")
    store = RunStore(settings.runs_dir / outcome.run_id)
    # 2 attempts allowed (max_pages × 2): the third and later attempts are refused
    assert len([c for c in scraper.calls if c[0] == "scrape"]) == 2
    assert store.get_counter("scrape_attempts") == 2
    assert store.has_warning("LIMIT_SCRAPE_ATTEMPTS_REACHED")
    assert seen[-1]["ok"] is True and seen[-1]["source"] == "map"


# ---------------------------------------------------------------------------------------
# 3. a process kill between renames is repaired from the publish journal
# ---------------------------------------------------------------------------------------


def _artifacts(run_dir: Path, tag: str) -> list[tuple[Path, str, int]]:
    return [
        (run_dir / "company_brain.json", f'{{"tag": "{tag}"}}\n', 0o644),
        (run_dir / "evidence.json", f'{{"evidence": "{tag}"}}\n', 0o600),
        (run_dir / "report.md", f"# report {tag}\n", 0o600),
    ]


def _kill_on_second_rename(monkeypatch):
    """The second output rename raises KeyboardInterrupt: the process 'dies' with
    company_brain.json replaced and the other two outputs still old (or missing)."""
    real = os.replace

    def crash(src, dst, *args, **kwargs):
        if Path(dst).name == "evidence.json":
            raise KeyboardInterrupt
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", crash)


def _tmp_files(run_dir: Path) -> list[str]:
    return sorted(p.name for p in run_dir.iterdir() if p.name.startswith(".tmp-"))


def test_R3_kill_between_renames_on_fresh_run_is_repaired(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        _kill_on_second_rename(mp)
        with pytest.raises(KeyboardInterrupt):
            write_all_or_nothing(
                _artifacts(run_dir, "v1"), journal=run_dir / PUBLISH_JOURNAL_FILENAME
            )
    journal = run_dir / PUBLISH_JOURNAL_FILENAME
    assert journal.exists()
    assert (run_dir / "company_brain.json").exists()
    assert not (run_dir / "evidence.json").exists()  # inconsistent set on disk
    assert recover_interrupted_publish(run_dir) is True
    assert [n for n in OUTPUTS if (run_dir / n).exists()] == []
    assert not journal.exists() and _tmp_files(run_dir) == []
    assert recover_interrupted_publish(run_dir) is False  # idempotent, silent


def test_R3_kill_between_renames_restores_previous_consistent_set(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_all_or_nothing(_artifacts(run_dir, "v1"), journal=run_dir / PUBLISH_JOURNAL_FILENAME)
    before = {n: (run_dir / n).read_bytes() for n in OUTPUTS}
    with pytest.MonkeyPatch.context() as mp:
        _kill_on_second_rename(mp)
        with pytest.raises(KeyboardInterrupt):
            write_all_or_nothing(
                _artifacts(run_dir, "v2"), journal=run_dir / PUBLISH_JOURNAL_FILENAME
            )
    journal = run_dir / PUBLISH_JOURNAL_FILENAME
    assert journal.exists()
    assert b"v2" in (run_dir / "company_brain.json").read_bytes()
    assert (run_dir / "evidence.json").read_bytes() == before["evidence.json"]  # half-published
    assert recover_interrupted_publish(run_dir) is True
    assert {n: (run_dir / n).read_bytes() for n in OUTPUTS} == before
    assert not journal.exists() and _tmp_files(run_dir) == []


def test_R3_normal_publish_leaves_no_journal_or_backups(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    for tag in ("v1", "v2"):
        write_all_or_nothing(_artifacts(run_dir, tag), journal=run_dir / PUBLISH_JOURNAL_FILENAME)
        assert sorted(p.name for p in run_dir.iterdir()) == sorted(OUTPUTS)
    assert b"v2" in (run_dir / "report.md").read_bytes()
    assert recover_interrupted_publish(run_dir) is False


def test_R3_ordinary_failure_still_rolls_back_in_process(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_all_or_nothing(_artifacts(run_dir, "v1"), journal=run_dir / PUBLISH_JOURNAL_FILENAME)
    before = {n: (run_dir / n).read_bytes() for n in OUTPUTS}
    real = os.replace

    def flaky(src, dst, *args, **kwargs):
        if Path(dst).name == "report.md":
            raise OSError("disk full")
        return real(src, dst, *args, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "replace", flaky)
        with pytest.raises(OSError):
            write_all_or_nothing(
                _artifacts(run_dir, "v2"), journal=run_dir / PUBLISH_JOURNAL_FILENAME
            )
    assert {n: (run_dir / n).read_bytes() for n in OUTPUTS} == before
    assert sorted(p.name for p in run_dir.iterdir()) == sorted(OUTPUTS)


def _complete_run(settings, acme_fixtures):
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    return runner, outcome


def test_R3_run_outputs_leave_no_journal_and_export_repairs_a_killed_publish(
    settings, acme_fixtures
):
    runner, outcome = _complete_run(settings, acme_fixtures)
    run_dir = settings.runs_dir / outcome.run_id
    assert not (run_dir / PUBLISH_JOURNAL_FILENAME).exists() and _tmp_files(run_dir) == []
    before = {n: (run_dir / n).read_bytes() for n in OUTPUTS}
    store = RunStore(run_dir)
    changed = json.loads(json.dumps(store.latest_profile()))
    changed["product"]["positioning"] = ""
    store.save_draft(changed, "draft")
    store.close()
    with pytest.MonkeyPatch.context() as mp:
        _kill_on_second_rename(mp)
        with pytest.raises(KeyboardInterrupt):
            runner.export(outcome.run_id)
    assert (run_dir / PUBLISH_JOURNAL_FILENAME).exists()
    assert (run_dir / "company_brain.json").read_bytes() != before["company_brain.json"]
    assert (run_dir / "evidence.json").read_bytes() == before["evidence.json"]
    assert RunStore(run_dir).get_run().status == "complete"  # the finalize transaction rolled back
    # the next export repairs the set first, then publishes the new one as a unit
    out = runner.export(outcome.run_id)
    assert out.status == "complete"
    assert not (run_dir / PUBLISH_JOURNAL_FILENAME).exists() and _tmp_files(run_dir) == []
    assert any(e.get("event") == "publish_recovered" for e in read_events(run_dir))
    brain = json.loads((run_dir / "company_brain.json").read_text())
    assert brain["product"]["positioning"] == ""
    assert json.loads((run_dir / "evidence.json").read_text())["run_id"] == outcome.run_id


def test_R3_resume_and_cli_status_repair_a_killed_publish(settings, acme_fixtures, monkeypatch):
    runner, outcome = _complete_run(settings, acme_fixtures)
    run_dir = settings.runs_dir / outcome.run_id
    before = {n: (run_dir / n).read_bytes() for n in OUTPUTS}
    with pytest.MonkeyPatch.context() as mp:
        _kill_on_second_rename(mp)
        with pytest.raises(KeyboardInterrupt):
            runner.export(outcome.run_id)
    assert (run_dir / PUBLISH_JOURNAL_FILENAME).exists()
    # `status` is read-only but repairs the set before reading it
    monkeypatch.setenv("PROFILE_BUILDER_RUNS_DIR", str(settings.runs_dir))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "")
    monkeypatch.chdir(settings.runs_dir.parent)
    result = CliRunner().invoke(app, ["status", "--run-id", outcome.run_id])
    assert result.exit_code == 0, result.output
    assert not (run_dir / PUBLISH_JOURNAL_FILENAME).exists()
    assert {n: (run_dir / n).read_bytes() for n in OUTPUTS} == before
    # a second kill, then `resume` of the finished run repairs it too
    with pytest.MonkeyPatch.context() as mp:
        _kill_on_second_rename(mp)
        with pytest.raises(KeyboardInterrupt):
            runner.export(outcome.run_id)
    assert (run_dir / PUBLISH_JOURNAL_FILENAME).exists()
    out = runner.resume(outcome.run_id)
    assert out.status == "complete"
    assert not (run_dir / PUBLISH_JOURNAL_FILENAME).exists()
    assert {n: (run_dir / n).read_bytes() for n in OUTPUTS} == before
