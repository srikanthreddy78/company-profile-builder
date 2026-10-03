"""Regression tests for the fourth review round: per-subfield feature grounding, per-item
interview answers, user-answer protection within one update batch, partial evidence
refresh on an unchanged list, discovery vs. the page budget, and all-or-nothing exports."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from profile_builder.state.run_store import RunStore
from profile_builder.web.discovery import discover
from profile_builder.web.scraper import FixtureScraper
from profile_builder.workflow.gaps import covering_paths, grounding_report
from tests.conftest import (
    CONFLICT_QUESTION,
    DRAFT_EVIDENCE,
    DRAFT_PROFILE,
    SITE,
    finalize_steps,
    last_tool_result,
    make_runner,
    tool_call,
)

FEATURES = "product.features_and_capabilities"
HOME_EXCERPT = "helps startups and enterprises protect sensitive data"
BANKS_EXCERPT = "used by Fortune 500 banks and healthcare networks"
TEAMS_EXCERPT = "Trusted by security teams at fast-moving startups and global enterprises alike."
RUNTIME_SHORT = (
    "Runtime encryption protects data while it is processed in memory. "
    "It uses Intel SGX and AMD SEV enclaves."
)


def _prefix():
    return [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call(
            "scrape_pages", {"urls": [f"{SITE}/product", f"{SITE}/customers", f"{SITE}/about"]}
        ),
    ]


def _brain(settings, run_id):
    return json.loads((settings.runs_dir / run_id / "company_brain.json").read_text())


def _recorder(seen, next_step):
    def step(messages):
        seen.append(last_tool_result(messages))
        return next_step(messages) if callable(next_step) else next_step

    return step


# ---------------------------------------------------------------------------------------
# 1a. a feature-level excerpt grounds only the subfields it supports
# ---------------------------------------------------------------------------------------


def test_R1a_feature_excerpt_grounds_only_supported_subfields(settings, acme_fixtures):
    prof = copy.deepcopy(DRAFT_PROFILE)
    feat = prof["product"]["features_and_capabilities"][0]
    feat["how_it_works"] = "Teleportation of workloads across galaxies"
    feat["customer_benefit"] = "1000% revenue increase guaranteed"
    ev = [e for e in DRAFT_EVIDENCE if e["field_path"] != f"{FEATURES}[0]"]
    ev.append(
        {"field_path": f"{FEATURES}[0]", "source_url": f"{SITE}/product", "excerpt": RUNTIME_SHORT}
    )
    seen: list[dict] = []
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": prof, "evidence": ev}),
        _recorder(seen, tool_call("finalize_profile", {})),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    draft = seen[0]
    assert {(r["field_path"], r["reason_code"]) for r in draft["evidence_rejected"]} >= {
        (f"{FEATURES}[0].how_it_works", "NO_OVERLAP"),
        (f"{FEATURES}[0].customer_benefit", "NO_OVERLAP"),
    }
    assert {f"{FEATURES}[0].how_it_works", f"{FEATURES}[0].customer_benefit"} <= set(
        draft["grounding"]["ungrounded"]
    )
    brain = _brain(settings, outcome.run_id)
    assert brain["product"]["features_and_capabilities"][0] == {
        "name": "Runtime encryption",
        "description": "Protects data while it is processed in memory.",
        "how_it_works": "",
        "customer_benefit": "",
    }
    assert len(brain["product"]["features_and_capabilities"]) == 2
    store = RunStore(settings.runs_dir / outcome.run_id)
    rows = {e["field_path"] for e in store.list_evidence() if e["field_path"].startswith(FEATURES)}
    assert {f"{FEATURES}[0].name", f"{FEATURES}[0].description"} <= rows
    assert not any(
        p.endswith("[0].how_it_works") or p.endswith("[0].customer_benefit") for p in rows
    )
    assert store.has_warning("UNGROUNDED_OMITTED")


def test_R1a_feature_with_unsupported_name_is_dropped(settings, acme_fixtures):
    prof = copy.deepcopy(DRAFT_PROFILE)
    prof["product"]["features_and_capabilities"].append(
        {
            "name": "Quantum teleport",
            "description": "Runtime encryption protects data while it is processed in memory.",
            "how_it_works": "",
            "customer_benefit": "",
        }
    )
    ev = [
        *DRAFT_EVIDENCE,
        {"field_path": f"{FEATURES}[2]", "source_url": f"{SITE}/product", "excerpt": RUNTIME_SHORT},
    ]
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": prof, "evidence": ev}),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    names = [
        f["name"] for f in _brain(settings, outcome.run_id)["product"]["features_and_capabilities"]
    ]
    assert names == ["Runtime encryption", "Policy-based key release"]


def test_R1a_item_row_no_longer_covers_feature_subfields():
    assert covering_paths(f"{FEATURES}[0].how_it_works") == [f"{FEATURES}[0].how_it_works"]
    profile = copy.deepcopy(DRAFT_PROFILE)
    report = grounding_report(profile, {f"{FEATURES}[0]"})
    assert {f"{FEATURES}[0].how_it_works", f"{FEATURES}[0].customer_benefit"} <= set(
        report["ungrounded"]
    )
    report = grounding_report(
        profile,
        {f"{FEATURES}[0].{s}" for s in ("name", "description", "how_it_works", "customer_benefit")},
    )
    assert not [p for p in report["ungrounded"] if p.startswith(f"{FEATURES}[0]")]


# ---------------------------------------------------------------------------------------
# 1b. an interview list value is checked item by item against the answer
# ---------------------------------------------------------------------------------------


def test_R1b_interview_list_items_are_checked_individually(settings, acme_fixtures):
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
                        "value": ["Security teams", "Martian procurement officers"],
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
    assert "Martian procurement officers" in res["rejected"][0]["reason"]
    assert res["rejected"][0].get("unsupported") == ["Martian procurement officers"]
    assert _brain(settings, outcome.run_id)["customer"]["buyers"] == []
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert not [e for e in store.list_evidence() if e["kind"] == "interview"]


# ---------------------------------------------------------------------------------------
# 2. a user answer applied earlier in the same batch protects the field
# ---------------------------------------------------------------------------------------


def test_R2_website_update_after_interview_in_same_batch_is_rejected(settings, acme_fixtures):
    seen: list[dict] = []

    def apply(messages):
        res = last_tool_result(messages)
        assert res["status"] == "answered"
        return tool_call(
            "apply_profile_updates",
            {
                "updates": [
                    {
                        "field_path": "customer.target_customer",
                        "value": res["answer"],
                        "evidence": {"kind": "interview", "question_id": res["qid"]},
                    },
                    {
                        "field_path": "customer.target_customer",
                        "value": "Startups and enterprises",
                        "evidence": {
                            "kind": "website",
                            "source_url": f"{SITE}/",
                            "excerpt": "Acme Vault helps startups and enterprises protect sensitive data in use",
                        },
                    },
                ]
            },
        )

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call("ask_user", CONFLICT_QUESTION),
        apply,
        _recorder(seen, tool_call("finalize_profile", {})),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(
        settings,
        steps,
        scraper=FixtureScraper(acme_fixtures),
        answers=["Large regulated enterprises such as banks"],
    )
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    res = seen[0]
    assert res["applied"] == ["customer.target_customer"]
    assert [r["reason_code"] for r in res["rejected"]] == ["USER_ANSWER_PROTECTED"]
    brain = _brain(settings, outcome.run_id)
    assert brain["customer"]["target_customer"] == "Large regulated enterprises such as banks"
    store = RunStore(settings.runs_dir / outcome.run_id)
    live = [
        e
        for e in store.list_evidence(include_superseded=False)
        if e["field_path"] == "customer.target_customer"
    ]
    assert [e["kind"] for e in live] == ["interview"]
    assert "customer.target_customer" in store.interview_bases()


# ---------------------------------------------------------------------------------------
# 3. re-saving an unchanged list with evidence for one item keeps the other items' rows
# ---------------------------------------------------------------------------------------


def test_R3_partial_evidence_refresh_keeps_sibling_item_rows(settings, acme_fixtures):
    prof = copy.deepcopy(DRAFT_PROFILE)
    prof["customer"]["buyers"] = ["Startups", "Banks"]
    ev = [
        *DRAFT_EVIDENCE,
        {"field_path": "customer.buyers[0]", "source_url": f"{SITE}/", "excerpt": HOME_EXCERPT},
        {
            "field_path": "customer.buyers[1]",
            "source_url": f"{SITE}/customers",
            "excerpt": BANKS_EXCERPT,
        },
    ]
    fresh = {"field_path": "customer.buyers[0]", "source_url": f"{SITE}/", "excerpt": TEAMS_EXCERPT}
    seen: list[dict] = []
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": prof, "evidence": ev}),
        tool_call("save_profile_draft", {"profile": prof, "evidence": [fresh]}),
        _recorder(seen, tool_call("finalize_profile", {})),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    assert seen[0]["evidence_accepted"] == 1
    assert "customer.buyers[1]" not in seen[0]["grounding"]["ungrounded"]
    assert _brain(settings, outcome.run_id)["customer"]["buyers"] == ["Startups", "Banks"]
    store = RunStore(settings.runs_dir / outcome.run_id)
    rows = sorted(
        (e["field_path"], e["excerpt"])
        for e in store.list_evidence(include_superseded=False)
        if e["field_path"].startswith("customer.buyers")
    )
    assert rows == [("customer.buyers[0]", TEAMS_EXCERPT), ("customer.buyers[1]", BANKS_EXCERPT)]


# ---------------------------------------------------------------------------------------
# 4. discovery never scrapes past the page budget, and never a non-homepage URL
# ---------------------------------------------------------------------------------------


def test_R4_three_discover_calls_with_budget_one_scrape_once(settings, acme_fixtures):
    small = settings.model_copy(update={"max_pages": 1})
    scraper = FixtureScraper(acme_fixtures)
    seen: list[dict] = []
    steps = [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        _recorder(seen, tool_call("discover_pages", {"start_url": f"{SITE}/"})),
        _recorder(seen, tool_call("discover_pages", {"start_url": f"{SITE}/"})),
        _recorder(seen, AIMessage(content="done")),
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(small, steps, scraper=scraper)
    outcome = runner.start(f"{SITE}/")
    store = RunStore(small.runs_dir / outcome.run_id)
    assert [c for c in scraper.calls if c[0] == "scrape"] == [("scrape", f"{SITE}/")]
    assert store.unique_pages_scraped() == 1
    assert all(r["ok"] and r["candidates"] for r in seen)
    assert all(r["page_budget_remaining"] == 0 for r in seen)


def test_R4_discover_after_budget_exhausted_does_not_fetch(settings, acme_fixtures):
    small = settings.model_copy(update={"max_pages": 1})
    scraper = FixtureScraper(acme_fixtures)
    seen: list[dict] = []
    steps = [
        tool_call("scrape_pages", {"urls": [f"{SITE}/product"]}),
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        _recorder(seen, tool_call("discover_pages", {"start_url": f"{SITE}/"})),
        _recorder(seen, AIMessage(content="done")),
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(small, steps, scraper=scraper)
    outcome = runner.start(f"{SITE}/")
    store = RunStore(small.runs_dir / outcome.run_id)
    # the only live scrape is the explicit one; discovery ran from the site map alone
    assert [c for c in scraper.calls if c[0] == "scrape"] == [("scrape", f"{SITE}/product")]
    assert store.unique_pages_scraped() == 1
    assert store.get_counter("scrape_attempts") == 1
    assert all(r["ok"] and r["source"] == "map" for r in seen), seen
    assert f"{SITE}/customers" in {c["url"] for c in seen[0]["candidates"]}


def test_R4_discover_never_fetches_a_non_homepage_url(settings, acme_fixtures):
    scraper = FixtureScraper(acme_fixtures)
    seen: list[dict] = []
    steps = [
        tool_call("discover_pages", {"start_url": f"{SITE}/product"}),
        _recorder(seen, AIMessage(content="done")),
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=scraper)
    runner.start(f"{SITE}/")
    scraped = [u for kind, u in scraper.calls if kind == "scrape"]
    assert f"{SITE}/product" not in scraped
    assert seen[0]["ok"] and seen[0]["candidates"]


def test_R4_discover_without_homepage_fetch_uses_the_map_only(acme_fixtures):
    scraper = FixtureScraper(acme_fixtures)
    result = discover(f"{SITE}/", scraper, timeout_ms=1000, check_dns=False)
    assert [c for c in scraper.calls if c[0] == "scrape"] == []
    assert result.homepage is None and result.homepage_error is None
    assert result.source == "map" and f"{SITE}/product" in {c.url for c in result.candidates}


# ---------------------------------------------------------------------------------------
# 5. the export writes all three artifacts or none of them
# ---------------------------------------------------------------------------------------

OUTPUTS = ("company_brain.json", "evidence.json", "report.md")


def _fail_evidence_publish(monkeypatch):
    real = os.replace

    def flaky(src, dst, *args, **kwargs):
        if Path(dst).name == "evidence.json":
            raise OSError("disk full")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", flaky)


def test_R5_failed_evidence_write_leaves_no_partial_outputs(settings, acme_fixtures, monkeypatch):
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    with pytest.MonkeyPatch.context() as mp:
        _fail_evidence_publish(mp)
        outcome = runner.start(f"{SITE}/")
    assert outcome.status == "interrupted"
    run_dir = settings.runs_dir / outcome.run_id
    assert [n for n in OUTPUTS if (run_dir / n).exists()] == []
    assert not [p for p in run_dir.iterdir() if p.name.startswith(".tmp-")]
    store = RunStore(run_dir)
    assert store.get_run().status == "interrupted"
    assert not store.has_warning("UNGROUNDED_OMITTED")  # the finalize transaction rolled back
    out = runner.export(outcome.run_id)
    assert out.status == "interrupted"
    assert [n for n in OUTPUTS if (run_dir / n).exists()] == list(OUTPUTS)


def test_R5_failed_re_export_restores_previous_outputs(settings, acme_fixtures, monkeypatch):
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    run_dir = settings.runs_dir / outcome.run_id
    before = {n: (run_dir / n).read_bytes() for n in OUTPUTS}
    # a newer draft, so a re-export would produce a different company_brain.json
    store = RunStore(run_dir)
    changed = copy.deepcopy(store.latest_profile())
    changed["product"]["positioning"] = ""
    store.save_draft(changed, "draft")
    store.close()
    with pytest.MonkeyPatch.context() as mp:
        _fail_evidence_publish(mp)
        with pytest.raises(OSError):
            runner.export(outcome.run_id)
    after = {n: (run_dir / n).read_bytes() for n in OUTPUTS if (run_dir / n).exists()}
    assert after == before  # the previous set survived intact: no half-replaced outputs
    assert not [p for p in run_dir.iterdir() if p.name.startswith(".tmp-")]
    assert RunStore(run_dir).get_run().status == "complete"
    runner.export(outcome.run_id)  # without the fault the re-export goes through
    assert _brain(settings, outcome.run_id)["product"]["positioning"] == ""
