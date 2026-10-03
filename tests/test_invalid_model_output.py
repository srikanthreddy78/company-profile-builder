"""Malformed model output: one repair attempt, then a clean failure with saved state."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from profile_builder.state.run_store import RunStore
from profile_builder.web.scraper import FixtureScraper
from tests.conftest import DRAFT_EVIDENCE, DRAFT_PROFILE, SITE, make_runner, tool_call

BAD_PROFILE = {
    **DRAFT_PROFILE,
    "customer": {**DRAFT_PROFILE["customer"], "target_customer": "unknown"},
    "product": {**DRAFT_PROFILE["product"], "name": None},
}


def _prefix():
    return [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call(
            "scrape_pages", {"urls": [f"{SITE}/product", f"{SITE}/customers", f"{SITE}/about"]}
        ),
    ]


def test_invalid_then_repaired(settings, acme_fixtures):
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": BAD_PROFILE, "evidence": []}, "bad1"),
        tool_call(
            "save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}, "good"
        ),
        tool_call("finalize_profile", {}),
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    store = RunStore(settings.runs_dir / outcome.run_id)
    # the counter resets after a successful save (one *consecutive* repair is allowed);
    # the attempt itself is recorded in the event log
    assert store.get_counter("repair_attempts") == 0
    from profile_builder.logging_setup import read_events

    assert [
        e["attempt"] for e in read_events(store.run_dir) if e.get("event") == "repair_attempt"
    ] == [1]
    assert store.draft_count() == 2  # good draft + finalize snapshot; the bad one was never saved


def test_invalid_twice_fails_without_writing_profile(settings, acme_fixtures):
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": BAD_PROFILE, "evidence": []}, "bad1"),
        tool_call("save_profile_draft", {"profile": BAD_PROFILE, "evidence": []}, "bad2"),
        AIMessage(content="unreachable"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "failed"
    assert "validation" in (outcome.message or "")
    run_dir = settings.runs_dir / outcome.run_id
    assert not (run_dir / "company_brain.json").exists()
    store = RunStore(run_dir)
    assert store.get_run().status == "failed"
    assert store.has_warning("INVALID_MODEL_OUTPUT") and store.has_warning("RUN_FAILED")
    assert len(store.fetched_urls()) == 4  # scraped work is preserved for inspection


def test_wrong_argument_types_are_reported_to_model(settings, acme_fixtures):
    """Argument-level validation errors come back as error tool messages (no crash)."""
    steps = [
        *_prefix(),
        tool_call("scrape_pages", {"urls": "not-a-list"}, "badargs"),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call("finalize_profile", {}),
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
