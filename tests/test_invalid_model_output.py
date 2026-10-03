"""Malformed model output: one repair attempt, then a clean failure with saved state."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from profile_builder.state.run_store import RunStore
from profile_builder.web.scraper import FixtureScraper
from tests.conftest import (
    DRAFT_EVIDENCE,
    DRAFT_PROFILE,
    SITE,
    finalize_steps,
    make_runner,
    tool_call,
)

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
        *finalize_steps(),
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
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"


def test_later_thin_draft_cannot_erase_earlier_fields(settings, acme_fixtures):
    """A second save_profile_draft that omits sections keeps the earlier values + evidence."""
    thin = {
        **DRAFT_PROFILE,
        "content_evidence": {k: [] for k in DRAFT_PROFILE["content_evidence"]},
        "brand": {k: [] for k in DRAFT_PROFILE["brand"]},
    }
    steps = [
        *_prefix(),
        tool_call(
            "save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}, "full"
        ),
        tool_call("save_profile_draft", {"profile": thin, "evidence": []}, "thin"),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    store = RunStore(settings.runs_dir / outcome.run_id)
    import json

    brain = json.loads((store.run_dir / "company_brain.json").read_text())
    assert (
        brain["content_evidence"]["customer_stories"]
        == DRAFT_PROFILE["content_evidence"]["customer_stories"]
    )
    assert brain["brand"]["preferred_terms"] == ["confidential computing"]
    assert any(
        e["field_path"].startswith("content_evidence.customer_stories")
        for e in store.list_evidence(include_superseded=False)
    )


def test_finalize_pushes_back_once_on_empty_sections(settings, acme_fixtures):
    bare = {
        **DRAFT_PROFILE,
        "customer": {
            k: ([] if isinstance(v, list) else "") for k, v in DRAFT_PROFILE["customer"].items()
        },
    }
    seen = []

    def after_finalize_attempt(messages):
        from tests.conftest import last_tool_result

        res = last_tool_result(messages)
        seen.append(res)
        return tool_call("finalize_profile", {}, "second")

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": bare, "evidence": []}),
        tool_call("finalize_profile", {}, "first"),
        after_finalize_attempt,
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    assert seen and seen[0]["ok"] is False and "customer" in seen[0]["advice"]
