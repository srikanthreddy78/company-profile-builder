"""Regression tests for the third review round: finalize atomicity and omission ordering,
draft/update evidence handling, per-item list grounding, provider rejections, no-usable-pages
runs, discovery/scrape edge cases, interview accounting, and the offline test harness."""

from __future__ import annotations

import copy
import json
import socket
from pathlib import Path

import httpx
import openai
import pytest
import requests
from langchain_core.messages import AIMessage, ToolMessage

import profile_builder.agent.finalize as finalize_mod
import profile_builder.agent.ingest as ingest_mod
import profile_builder.agent.tools.web_tools as web_tools_mod
import profile_builder.config as config_mod
from profile_builder.agent.finalize import omit_many
from profile_builder.retrieval.index import HybridIndex
from profile_builder.schema import json_schema
from profile_builder.security import url_cache_name
from profile_builder.state.run_store import RunStore, shift_path_set, shifted_path
from profile_builder.web.scraper import (
    FixtureScraper,
    ScrapedPage,
    TransientScrapeError,
)
from tests.conftest import (
    DRAFT_EVIDENCE,
    DRAFT_PROFILE,
    SITE,
    ScriptedChatModel,
    ScriptExhausted,
    finalize_steps,
    last_tool_result,
    make_runner,
    tool_call,
)

HOME_EXCERPT = "helps startups and enterprises protect sensitive data"
BANKS_EXCERPT = "used by Fortune 500 banks and healthcare networks"
TEAMS_EXCERPT = "Trusted by security teams at fast-moving startups and global enterprises alike."
CLAIMS = [
    {"source_url": f"{SITE}/", "excerpt": HOME_EXCERPT, "claim": "startups"},
    {"source_url": f"{SITE}/customers", "excerpt": BANKS_EXCERPT, "claim": "banks"},
]


def _prefix():
    return [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call(
            "scrape_pages", {"urls": [f"{SITE}/product", f"{SITE}/customers", f"{SITE}/about"]}
        ),
    ]


def _brain(settings, run_id):
    return json.loads((settings.runs_dir / run_id / "company_brain.json").read_text())


def _evidence_doc(settings, run_id):
    return json.loads((settings.runs_dir / run_id / "evidence.json").read_text())


def _three_buyers():
    prof = copy.deepcopy(DRAFT_PROFILE)
    prof["customer"]["buyers"] = ["Startups", "Banks", "Security teams"]
    ev = [
        *DRAFT_EVIDENCE,
        {"field_path": "customer.buyers[0]", "source_url": f"{SITE}/", "excerpt": HOME_EXCERPT},
        {
            "field_path": "customer.buyers[1]",
            "source_url": f"{SITE}/customers",
            "excerpt": BANKS_EXCERPT,
        },
        {"field_path": "customer.buyers[2]", "source_url": f"{SITE}/", "excerpt": TEAMS_EXCERPT},
    ]
    return prof, ev


# ---------------------------------------------------------------------------------------
# F1: re-saving an unchanged draft without evidence keeps the grounding
# ---------------------------------------------------------------------------------------


def test_F1_resave_same_profile_without_evidence_keeps_grounding(settings, acme_fixtures):
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": []}),
        record,
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    assert seen[0]["grounding"]["ungrounded"] == ["company.name"]  # unchanged after the resave
    brain = _brain(settings, outcome.run_id)
    assert brain["product"]["name"] == "Acme Vault", brain["product"]
    assert brain["customer"]["use_cases"] == DRAFT_PROFILE["customer"]["use_cases"]


def test_F1b_thin_redraft_keeps_product_name(settings, acme_fixtures):
    thin = {
        **DRAFT_PROFILE,
        "content_evidence": {k: [] for k in DRAFT_PROFILE["content_evidence"]},
        "brand": {k: [] for k in DRAFT_PROFILE["brand"]},
    }
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call("save_profile_draft", {"profile": thin, "evidence": []}),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    brain = _brain(settings, outcome.run_id)
    assert brain["product"]["name"] == "Acme Vault", brain["product"]
    assert brain["brand"]["preferred_terms"] == ["confidential computing"]


def test_F1c_changed_value_without_new_evidence_loses_stale_grounding(settings, acme_fixtures):
    changed = copy.deepcopy(DRAFT_PROFILE)
    changed["product"]["positioning"] = "A brand new positioning the page never stated"
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call("save_profile_draft", {"profile": changed, "evidence": []}),
        record,
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert "product.positioning" in seen[0]["grounding"]["ungrounded"]
    assert _brain(settings, outcome.run_id)["product"]["positioning"] == ""


# ---------------------------------------------------------------------------------------
# F2 / F6: apply_profile_updates never crashes and never leaves placeholders behind
# ---------------------------------------------------------------------------------------


def test_F2_evidence_string_is_a_soft_rejection(settings, acme_fixtures):
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call(
            "apply_profile_updates",
            {
                "updates": [
                    {"field_path": "customer.buyers", "value": ["CISOs"], "evidence": "website"},
                    {"field_path": "customer.buyers", "value": ["CISOs"], "evidence": None},
                    {
                        "field_path": "customer.buyers",
                        "value": ["CISOs"],
                        "evidence": {"kind": "x"},
                    },
                    {"field_path": "nope.field", "value": "x", "evidence": {"kind": "website"}},
                ]
            },
        ),
        record,
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete", outcome
    res = seen[0]
    assert res["ok"] is False and res["applied"] == []
    assert [r["reason_code"] for r in res["rejected"]] == [
        "BAD_EVIDENCE_KIND",
        "BAD_EVIDENCE_KIND",
        "BAD_EVIDENCE_KIND",
        "BAD_PATH",
    ]


def _append_feature_with_bad_evidence():
    return tool_call(
        "apply_profile_updates",
        {
            "updates": [
                {
                    "field_path": "customer.buyers",
                    "value": ["Security teams"],
                    "evidence": {
                        "kind": "website",
                        "source_url": f"{SITE}/",
                        "excerpt": TEAMS_EXCERPT,
                    },
                },
                {  # append a feature; the excerpt is not verbatim → soft rejection, no leftover
                    "field_path": "product.features_and_capabilities[2]",
                    "value": {
                        "name": "Audit logging",
                        "description": "Every access is logged",
                        "how_it_works": "",
                        "customer_benefit": "",
                    },
                    "evidence": {
                        "kind": "website",
                        "source_url": f"{SITE}/product",
                        "excerpt": "This sentence does not appear on the page at all.",
                    },
                },
            ]
        },
    )


def test_F6_rejected_append_is_rolled_back(settings, acme_fixtures):
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        _append_feature_with_bad_evidence(),
        record,
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    res = seen[0]
    assert res["ok"] is True and res["applied"] == ["customer.buyers"], res
    assert [r["reason_code"] for r in res["rejected"]] == ["NOT_VERBATIM"]
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert store.get_counter("repair_attempts") == 0
    brain = _brain(settings, outcome.run_id)
    assert len(brain["product"]["features_and_capabilities"]) == 2
    assert brain["customer"]["buyers"] == ["Security teams"]


def test_F6b_repeated_bad_updates_do_not_kill_the_run(settings, acme_fixtures):
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        _append_feature_with_bad_evidence(),
        _append_feature_with_bad_evidence(),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete", outcome


def test_F6c_invalid_value_is_rejected_per_update(settings, acme_fixtures):
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call(
            "apply_profile_updates",
            {
                "updates": [
                    {  # a filler value is a validation error → INVALID_VALUE, not a repair
                        "field_path": "product.positioning",
                        "value": "unknown",
                        "evidence": {
                            "kind": "website",
                            "source_url": f"{SITE}/product",
                            "excerpt": "Acme Vault protects data during computation, not only at rest.",
                        },
                    },
                    {
                        "field_path": "customer.buyers[0]",
                        "value": "Security teams",
                        "evidence": {
                            "kind": "website",
                            "source_url": f"{SITE}/",
                            "excerpt": TEAMS_EXCERPT,
                        },
                    },
                ]
            },
        ),
        record,
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    res = seen[0]
    assert res["applied"] == ["customer.buyers[0]"]
    assert res["rejected"][0]["reason_code"] == "INVALID_VALUE"
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert store.get_counter("repair_attempts") == 0


def test_apply_updates_website_kind_appends_with_verbatim_excerpt(settings, acme_fixtures):
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return tool_call(
            "apply_profile_updates",
            {
                "updates": [
                    {  # negative: paraphrased excerpt → rejected, list unchanged
                        "field_path": "customer.buyers[1]",
                        "value": "Global enterprises",
                        "evidence": {
                            "kind": "website",
                            "source_url": f"{SITE}/",
                            "excerpt": "Trusted by security teams at fast-moving start-ups and big enterprises",
                        },
                    }
                ]
            },
        )

    def record2(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call(
            "apply_profile_updates",
            {
                "updates": [
                    {
                        "field_path": "customer.buyers[0]",  # [len] of the empty list
                        "value": "Security teams",
                        "evidence": {
                            "kind": "website",
                            "source_url": f"{SITE}/",
                            "excerpt": TEAMS_EXCERPT,
                        },
                    }
                ]
            },
        ),
        record,
        record2,
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    assert seen[0]["ok"] is True and seen[0]["applied"] == ["customer.buyers[0]"]
    assert seen[1]["ok"] is False and seen[1]["rejected"][0]["reason_code"] == "NOT_VERBATIM"
    store = RunStore(settings.runs_dir / outcome.run_id)
    rows = [e for e in store.list_evidence(include_superseded=False) if "buyers" in e["field_path"]]
    assert [(e["field_path"], e["excerpt"]) for e in rows] == [
        ("customer.buyers[0]", TEAMS_EXCERPT)
    ]
    assert _brain(settings, outcome.run_id)["customer"]["buyers"] == ["Security teams"]


# ---------------------------------------------------------------------------------------
# F3: finalize before any draft is a tool error, not a dead run
# ---------------------------------------------------------------------------------------


def test_F3_finalize_before_draft_is_a_tool_error(settings, acme_fixtures):
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return tool_call(
            "save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}
        )

    steps = [*_prefix(), tool_call("finalize_profile", {}), record, *finalize_steps()]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete", outcome
    assert seen[0] == {
        "ok": False,
        "error": "no draft yet; call save_profile_draft first",
        "error_code": "NO_DRAFT",
    }


def test_finalize_refusal_carries_not_ready_code(settings, acme_fixtures):
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call("finalize_profile", {}),
        record,
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    assert seen[0]["error_code"] == "NOT_READY" and seen[0]["ungrounded"] == ["company.name"]


# ---------------------------------------------------------------------------------------
# F4: a permanent homepage error is recorded and the run continues with the map
# ---------------------------------------------------------------------------------------


def test_F4_homepage_404_continues_with_map(settings, acme_fixtures_mutable):
    fx = acme_fixtures_mutable
    (fx / "pages" / url_cache_name(f"{SITE}/")).unlink()  # homepage → HTTP_404 permanent
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return tool_call(
            "scrape_pages", {"urls": [f"{SITE}/product", f"{SITE}/customers", f"{SITE}/about"]}
        )

    steps = [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        record,
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(fx))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete", outcome
    res = seen[0]
    assert res["ok"] is True and f"{SITE}/" not in {c["url"] for c in res["candidates"]}
    assert f"{SITE}/product" in {c["url"] for c in res["candidates"]}
    store = RunStore(settings.runs_dir / outcome.run_id)
    home = store.get_page(f"{SITE}/")
    assert home is not None and home.status == "failed" and home.error_code == "HTTP_404"
    assert store.has_warning("PAGE_SKIPPED_HTTP_404")


def test_F4b_homepage_404_without_map_is_an_actionable_tool_error(settings, acme_fixtures_mutable):
    fx = acme_fixtures_mutable
    (fx / "pages" / url_cache_name(f"{SITE}/")).unlink()
    (fx / "map.json").unlink()
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return AIMessage(content="giving up")

    steps = [tool_call("discover_pages", {"start_url": f"{SITE}/"}), record, AIMessage(content="x")]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(fx))
    outcome = runner.start(f"{SITE}/")
    assert seen[0]["ok"] is False and seen[0]["error_code"] == "HOMEPAGE_UNAVAILABLE"
    assert "www" in seen[0]["error"]  # actionable: suggests checking the URL variant
    assert outcome.status == "failed" and "no usable pages" in (outcome.message or "")


# ---------------------------------------------------------------------------------------
# F5: two unresolved conflicts on one list; finalize is atomic
# ---------------------------------------------------------------------------------------


def _two_conflicts_steps():
    prof, ev = _three_buyers()
    return [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": prof, "evidence": ev}),
        tool_call(
            "note_conflict", {"field_path": "customer.buyers[0]", "claims": CLAIMS, "summary": "c0"}
        ),
        tool_call(
            "note_conflict", {"field_path": "customer.buyers[1]", "claims": CLAIMS, "summary": "c1"}
        ),
        *finalize_steps(),
    ]


def test_F5_two_open_conflicts_same_list_finalize_cleanly(settings, acme_fixtures):
    runner, _, _ = make_runner(
        settings, _two_conflicts_steps(), scraper=FixtureScraper(acme_fixtures)
    )
    outcome = runner.start(f"{SITE}/")
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert outcome.status == "complete", (
        outcome,
        [w["message"] for w in store.list_warnings()][-3:],
    )
    assert _brain(settings, outcome.run_id)["customer"]["buyers"] == ["Security teams"]
    # both conflicts keep their original paths as a record of what was omitted
    assert {(c["field_path"], c["status"]) for c in store.list_conflicts()} == {
        ("customer.buyers[0]", "omitted"),
        ("customer.buyers[1]", "omitted"),
    }
    paths = {
        e["field_path"]
        for e in store.list_evidence(include_superseded=False)
        if e["field_path"].startswith("customer.buyers")
    }
    assert paths == {"customer.buyers[0]"}
    runner.export(outcome.run_id)
    assert _brain(settings, outcome.run_id)["customer"]["buyers"] == ["Security teams"]


def test_F5b_failed_output_write_rolls_the_store_back(settings, acme_fixtures, monkeypatch):
    real_write = finalize_mod.write_outputs
    calls = {"n": 0}

    def flaky_write(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(finalize_mod, "write_outputs", flaky_write)
    runner, _, _ = make_runner(
        settings, _two_conflicts_steps(), scraper=FixtureScraper(acme_fixtures)
    )
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "interrupted" and outcome.exit_code == 3
    store = RunStore(settings.runs_dir / outcome.run_id)
    # nothing of the failed finalize stuck: conflicts open, evidence unshifted, no warning
    assert [c["status"] for c in store.list_conflicts()] == ["open", "open"]
    assert not store.has_warning("CONFLICT_UNRESOLVED_OMITTED")
    assert store.latest_draft()[1]["customer"]["buyers"] == ["Startups", "Banks", "Security teams"]
    assert not (store.run_dir / "company_brain.json").exists()
    out = runner.export(outcome.run_id)
    brain = _brain(settings, outcome.run_id)
    assert brain["customer"]["buyers"] == ["Security teams"], (out, brain["customer"]["buyers"])
    assert out.status == "interrupted"  # export never changes the status of an unfinished run


def test_transaction_rolls_back_and_nests(tmp_path):
    store = RunStore(tmp_path / "run")
    store.create_run("pb-20260101-aaaaaa", f"{SITE}/", None, {})
    with pytest.raises(RuntimeError), store.transaction():
        store.add_warning("X", "inside")
        with store.transaction():  # nested: joins the outer transaction
            store.add_warning("Y", "nested")
        raise RuntimeError("boom")
    assert store.list_warnings() == []
    with store.transaction():
        store.add_warning("Z", "committed")
    assert [w["code"] for w in store.list_warnings()] == ["Z"]


def test_shift_list_paths_keeps_conflict_records_and_skips_clashes(tmp_path):
    store = RunStore(tmp_path / "run")
    store.upsert_conflict("customer.buyers[0]", CLAIMS, "a")
    store.upsert_conflict("customer.buyers[2]", CLAIMS, "c")
    store.set_conflict_status("customer.buyers[0]", "omitted", "gone")
    store.add_evidence("customer.buyers[1]", "website", source_url="u", excerpt="e1")
    store.add_evidence("customer.buyers[2]", "website", source_url="u", excerpt="e2")
    store.shift_list_paths("customer.buyers", 1)
    assert {(c["field_path"], c["status"]) for c in store.list_conflicts()} == {
        ("customer.buyers[0]", "omitted"),  # omitted rows are never renumbered
        ("customer.buyers[1]", "open"),  # open row followed its item
    }
    assert [e["field_path"] for e in store.list_evidence()] == ["customer.buyers[1]"]
    # an open row whose target path exists is left alone instead of clobbering the record:
    # [1] → [0] would hit the omitted record, and [2] → [1] would then hit the stuck [1]
    store.upsert_conflict("customer.buyers[2]", CLAIMS, "d")
    store.shift_list_paths("customer.buyers", 0)
    assert {(c["field_path"], c["status"]) for c in store.list_conflicts()} == {
        ("customer.buyers[0]", "omitted"),
        ("customer.buyers[1]", "open"),
        ("customer.buyers[2]", "open"),
    }
    # resolved rows are records too: [2] → [1] is skipped because the resolved [1] is kept
    store.set_conflict_status("customer.buyers[1]", "resolved", "done")
    store.shift_list_paths("customer.buyers", 1)
    assert {(c["field_path"], c["status"]) for c in store.list_conflicts()} == {
        ("customer.buyers[0]", "omitted"),
        ("customer.buyers[1]", "resolved"),
        ("customer.buyers[2]", "open"),
    }
    assert shifted_path("customer.buyers[2].x", "customer.buyers", 0) == "customer.buyers[1].x"
    assert shifted_path("customer.buyers[0]", "customer.buyers", 0) is None
    assert shifted_path("customer.users[5]", "customer.buyers", 0) == "customer.users[5]"
    assert shift_path_set({"a.b[0]", "a.b[1]", "a.c[1]"}, "a.b", 0) == {"a.b[0]", "a.c[1]"}


# ---------------------------------------------------------------------------------------
# F7: a whole-list answer resolves conflicts on its items
# ---------------------------------------------------------------------------------------


def test_F7_item_conflict_resolved_by_whole_list_answer(settings, acme_fixtures):
    prof = copy.deepcopy(DRAFT_PROFILE)
    prof["customer"]["buyers"] = ["Startups"]
    ev = [
        *DRAFT_EVIDENCE,
        {"field_path": "customer.buyers[0]", "source_url": f"{SITE}/", "excerpt": HOME_EXCERPT},
    ]
    q = {
        "question": "Who buys Acme Vault?",
        "why_unclear": "pages disagree",
        "field_paths": ["customer.buyers"],
        "kind": "conflict",
    }

    def apply(messages):
        res = last_tool_result(messages)
        return tool_call(
            "apply_profile_updates",
            {
                "updates": [
                    {
                        "field_path": "customer.buyers",
                        "value": [res["answer"]],
                        "evidence": {"kind": "interview", "question_id": res["qid"]},
                    }
                ]
            },
        )

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": prof, "evidence": ev}),
        tool_call(
            "note_conflict",
            {"field_path": "customer.buyers[0]", "claims": CLAIMS, "summary": "who"},
        ),
        tool_call("ask_user", q),
        apply,
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(
        settings, steps, scraper=FixtureScraper(acme_fixtures), answers=["CISOs at banks"]
    )
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    store = RunStore(settings.runs_dir / outcome.run_id)
    brain = _brain(settings, outcome.run_id)
    assert brain["customer"]["buyers"] == ["CISOs at banks"], (
        brain["customer"]["buyers"],
        [c["status"] for c in store.list_conflicts()],
    )
    assert [c["status"] for c in store.list_conflicts()] == ["resolved"]
    # interview evidence for a list is stored per item
    rows = [e for e in store.list_evidence(include_superseded=False) if e["kind"] == "interview"]
    assert [e["field_path"] for e in rows] == ["customer.buyers[0]"]


def test_note_conflict_reopens_a_resolved_conflict(tmp_path):
    store = RunStore(tmp_path / "run")
    store.upsert_conflict("customer.buyers", CLAIMS, "first")
    store.resolve_conflict("customer.buyers", "user said banks")
    assert store.list_conflicts("resolved")
    store.upsert_conflict("customer.buyers", CLAIMS, "new claims arrived")
    (c,) = store.list_conflicts()
    assert (
        c["status"] == "open" and c["resolution"] is None and c["summary"] == "new claims arrived"
    )


# ---------------------------------------------------------------------------------------
# F8 / interview accounting
# ---------------------------------------------------------------------------------------

PRODUCT_Q = {
    "question": "Which product should the profile focus on?",
    "why_unclear": "two products",
    "field_paths": ["product.name"],
    "kind": "product_selection",
}
GAP_Q = {
    "question": "Who signs off on the purchase?",
    "why_unclear": "not stated",
    "field_paths": ["customer.buyers"],
    "kind": "gap",
}


def test_F8_preset_product_is_not_counted_as_a_question(settings, acme_fixtures):
    settings = settings.model_copy(update={"max_questions": 1})
    seen = []

    def after_preset(messages):
        seen.append(last_tool_result(messages))
        return tool_call("ask_user", GAP_Q)

    def after_gap(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call("ask_user", PRODUCT_Q),
        after_preset,
        after_gap,
        *finalize_steps(),
    ]
    runner, _, asked = make_runner(
        settings, steps, scraper=FixtureScraper(acme_fixtures), answers=["CISOs"]
    )
    outcome = runner.start(f"{SITE}/", product="Acme Vault")
    assert outcome.status == "complete"
    assert asked == ["CISOs"], (asked, seen)
    assert seen[0]["status"] == "answered" and seen[0]["answer"] == "Acme Vault"
    assert seen[0]["questions_remaining"] == 1
    assert seen[1]["status"] == "answered" and seen[1]["answer"] == "CISOs"
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert [q["status"] for q in store.list_questions()] == ["preset", "answered"]
    assert store.questions_asked() == 1
    assert _evidence_doc(settings, outcome.run_id)["product_focus"] == "Acme Vault"


def test_product_selection_is_asked_and_sets_the_run_focus(settings, acme_fixtures):
    seen = []

    def after_product(messages):
        seen.append(last_tool_result(messages))
        return tool_call(
            "save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}
        )

    steps = [*_prefix(), tool_call("ask_user", PRODUCT_Q), after_product, *finalize_steps()]
    runner, _, asked = make_runner(
        settings, steps, scraper=FixtureScraper(acme_fixtures), answers=["Acme Keys"]
    )
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete" and asked == ["Acme Keys"]
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert store.get_run().product_focus == "Acme Keys"
    assert seen[0]["status"] == "answered" and seen[0]["questions_remaining"] == 2
    (q,) = store.list_questions()
    assert q["kind"] == "product_selection" and q["ordinal"] == 1 and store.questions_asked() == 1
    assert _evidence_doc(settings, outcome.run_id)["product_focus"] == "Acme Keys"


def test_F12_rejected_ask_user_calls_do_not_eat_the_question(settings, acme_fixtures):
    settings = settings.model_copy(update={"max_questions": 1})
    bad = {**GAP_Q, "question": "1) Who buys? 2) Who uses? 3) Which alternatives?"}
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return tool_call("ask_user", {**bad, "why_unclear": "y"})

    def record2(messages):
        seen.append(last_tool_result(messages))
        return tool_call("ask_user", GAP_Q)

    def record3(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call("ask_user", bad),
        record,
        record2,
        record3,
        *finalize_steps(),
    ]
    runner, _, asked = make_runner(
        settings, steps, scraper=FixtureScraper(acme_fixtures), answers=["CISOs"]
    )
    outcome = runner.start(f"{SITE}/")
    assert asked == ["CISOs"], (asked, seen)
    assert outcome.status == "complete"
    assert seen[0]["error_code"] == "MULTI_QUESTION" and seen[1]["error_code"] == "MULTI_QUESTION"
    assert seen[2]["status"] == "answered"


def test_solo_ask_user_guard_rejects_a_bundled_call(settings, acme_fixtures):
    bundled = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_pages", "args": {"query": "buyers"}, "id": "bundle_search"},
            {"name": "ask_user", "args": GAP_Q, "id": "bundle_ask"},
        ],
    )
    seen = []

    def record(messages):
        tool_msgs = {m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)}
        seen.append(tool_msgs["bundle_ask"])
        seen.append(tool_msgs["bundle_search"])
        return tool_call("finalize_profile", {})

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        bundled,
        record,
        *finalize_steps(),
    ]
    runner, _, asked = make_runner(
        settings, steps, scraper=FixtureScraper(acme_fixtures), answers=["should not be asked"]
    )
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete" and asked == []
    ask_msg, search_msg = seen
    assert ask_msg.status == "error" and "only tool call" in ask_msg.content
    assert search_msg.status != "error"
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert store.list_questions() == []


def test_replayed_skipped_answer_returns_marker(settings, acme_fixtures):
    """Crash between answer-commit and checkpoint: the replayed ask_user must return the
    same SKIPPED marker the normal path returns, not a null answer."""
    from tests.test_exit_resume_interview import two_question_steps

    scraper = FixtureScraper(acme_fixtures)
    runner1, _, _ = make_runner(settings, two_question_steps(), scraper=scraper, answers=[])
    outcome1 = runner1.start(f"{SITE}/")
    assert outcome1.status == "paused"
    store = RunStore(settings.runs_dir / outcome1.run_id)
    store.answer_question(store.pending_question()["qid"], "skipped", None)
    seen = []

    def after_replay(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    model2 = ScriptedChatModel(steps=[after_replay, *finalize_steps()])
    runner2, _, asked = make_runner(settings, [], scraper=scraper, model=model2, answers=["x"])
    outcome2 = runner2.resume(outcome1.run_id)
    assert outcome2.status == "complete" and asked == []
    assert seen[0]["status"] == "skipped" and seen[0]["answer"] == "SKIPPED"
    assert seen[0]["note"] == "answer already recorded"


# ---------------------------------------------------------------------------------------
# F9 / F10: omission ordering keeps evidence and neighbours intact
# ---------------------------------------------------------------------------------------


def _ghost_feature_steps(ghost):
    prof = copy.deepcopy(DRAFT_PROFILE)
    prof["product"]["features_and_capabilities"] = [
        ghost,
        DRAFT_PROFILE["product"]["features_and_capabilities"][0],
    ]
    ev = [e for e in DRAFT_EVIDENCE if "features_and_capabilities" not in e["field_path"]]
    ev.append(
        {
            "field_path": "product.features_and_capabilities[1]",
            "source_url": f"{SITE}/product",
            "excerpt": "Runtime encryption protects data while it is processed in memory. It uses Intel SGX and AMD SEV enclaves.",
        }
    )
    return [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": prof, "evidence": ev}),
        *finalize_steps(),
    ]


def test_F9_empty_feature_removal_reindexes_evidence(settings, acme_fixtures):
    ghost = {
        "name": "",
        "description": "Unsupported capability text",
        "how_it_works": "",
        "customer_benefit": "",
    }
    runner, _, _ = make_runner(
        settings, _ghost_feature_steps(ghost), scraper=FixtureScraper(acme_fixtures)
    )
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    expected = [DRAFT_PROFILE["product"]["features_and_capabilities"][0]]
    assert _brain(settings, outcome.run_id)["product"]["features_and_capabilities"] == expected
    evdoc = _evidence_doc(settings, outcome.run_id)
    assert evdoc["grounding"]["ungrounded"] == [], evdoc["grounding"]["ungrounded"]
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert {e["field_path"] for e in store.list_evidence() if "features" in e["field_path"]} == {
        "product.features_and_capabilities[0]"
    }
    runner.export(outcome.run_id)
    assert _brain(settings, outcome.run_id)["product"]["features_and_capabilities"] == expected


def test_F10_fully_populated_ghost_feature_does_not_damage_neighbour(settings, acme_fixtures):
    ghost = {
        "name": "Ghost",
        "description": "ghost desc",
        "how_it_works": "ghost how",
        "customer_benefit": "ghost benefit",
    }
    runner, _, _ = make_runner(
        settings, _ghost_feature_steps(ghost), scraper=FixtureScraper(acme_fixtures)
    )
    outcome = runner.start(f"{SITE}/")
    feats = _brain(settings, outcome.run_id)["product"]["features_and_capabilities"]
    assert feats == [DRAFT_PROFILE["product"]["features_and_capabilities"][0]], feats


def test_F10_omit_many_is_order_independent(tmp_path):
    store = RunStore(tmp_path / "run")
    base = "product.features_and_capabilities"
    paths = [
        f"{base}[0].name",
        f"{base}[0].description",
        f"{base}[0].how_it_works",
        f"{base}[0].customer_benefit",
    ]
    for order in (paths, list(reversed(paths)), paths[2:] + paths[:2]):
        data = copy.deepcopy(DRAFT_PROFILE)
        data["product"]["features_and_capabilities"][0] = {
            "name": "Ghost",
            "description": "ghost desc",
            "how_it_works": "ghost how",
            "customer_benefit": "ghost benefit",
        }
        shifts = omit_many(store, data, order)
        feats = data["product"]["features_and_capabilities"]
        assert (
            len(feats) == 1 and feats[0] == DRAFT_PROFILE["product"]["features_and_capabilities"][1]
        )
        assert shifts == [(base, 0)]
    # sub-field blanking happens before deletion; items go highest index first
    data = copy.deepcopy(DRAFT_PROFILE)
    data["customer"]["buyers"] = ["a", "b", "c"]
    shifts = omit_many(
        None, data, ["customer.buyers[2]", "customer.buyers[0]", f"{base}[1].customer_benefit"]
    )
    assert data["customer"]["buyers"] == ["b"]
    assert data["product"]["features_and_capabilities"][1]["customer_benefit"] == ""
    assert shifts == [("customer.buyers", 2), ("customer.buyers", 0)]
    # whole-field omissions blank strings and lists
    omit_many(None, data, ["product.positioning", "customer.use_cases"])
    assert data["product"]["positioning"] == "" and data["customer"]["use_cases"] == []


# ---------------------------------------------------------------------------------------
# F11: discovery keeps homepage links across calls (cached homepage carries its links)
# ---------------------------------------------------------------------------------------


def test_F11_second_discover_keeps_homepage_links_without_map(settings, acme_fixtures_mutable):
    fx = acme_fixtures_mutable
    (fx / "map.json").unlink()
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return tool_call("discover_pages", {"start_url": f"{SITE}/"})

    def record2(messages):
        seen.append(last_tool_result(messages))
        return AIMessage(content="done")

    steps = [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        record,
        record2,
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(fx))
    outcome = runner.start(f"{SITE}/")
    first, second = seen
    assert first["source"] == "homepage_links"  # map fallback to the homepage's links
    assert f"{SITE}/product" in {c["url"] for c in first["candidates"]}
    assert len(second["candidates"]) == len(first["candidates"]), (first, second)
    assert second["source"] == first["source"]
    cached = ScrapedPage.from_json(
        (settings.runs_dir / outcome.run_id / "pages" / url_cache_name(f"{SITE}/")).read_text()
    )
    assert f"{SITE}/product" in cached.links


# ---------------------------------------------------------------------------------------
# F13: export keeps the status of an unfinished run; resume continues
# ---------------------------------------------------------------------------------------


def test_F13_export_of_paused_run_keeps_status_and_resume_completes(settings, acme_fixtures):
    from tests.test_exit_resume_interview import two_question_steps

    scraper = FixtureScraper(acme_fixtures)
    runner1, _, _ = make_runner(
        settings, two_question_steps(), scraper=scraper, answers=["Regulated enterprises", "exit"]
    )
    o1 = runner1.start(f"{SITE}/")
    assert o1.status == "paused"
    exported = runner1.export(o1.run_id)
    assert exported.status == "paused" and exported.resume_hint
    store = RunStore(settings.runs_dir / o1.run_id)
    assert store.get_run().status == "paused" and store.get_run().output_path
    assert _evidence_doc(settings, o1.run_id)["status"] == "partial"
    model2 = ScriptedChatModel(steps=two_question_steps()[6:])
    runner2, _, asked = make_runner(settings, [], scraper=scraper, model=model2, answers=["CISOs"])
    o2 = runner2.resume(o1.run_id)
    assert o2.status == "complete", o2
    assert asked == ["CISOs"]
    assert _brain(settings, o1.run_id)["customer"]["buyers"] == ["CISOs"]


# ---------------------------------------------------------------------------------------
# Repair accounting: a crash replay of the same invalid call is not a second attempt
# ---------------------------------------------------------------------------------------

BAD_PROFILE = {
    **DRAFT_PROFILE,
    "customer": {**DRAFT_PROFILE["customer"], "target_customer": "unknown"},
}


def test_replayed_invalid_draft_does_not_double_count(settings, acme_fixtures):
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": BAD_PROFILE, "evidence": []}, "same-call"),
        tool_call("save_profile_draft", {"profile": BAD_PROFILE, "evidence": []}, "same-call"),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete", outcome


def test_repeated_invalid_draft_in_a_new_call_still_fails(settings, acme_fixtures):
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": BAD_PROFILE, "evidence": []}),
        tool_call("save_profile_draft", {"profile": BAD_PROFILE, "evidence": []}),
        AIMessage(content="unreachable"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "failed" and outcome.exit_code == 1


# ---------------------------------------------------------------------------------------
# G1: per-item list evidence
# ---------------------------------------------------------------------------------------


def test_G1_list_excerpt_grounds_only_the_items_it_mentions(settings, acme_fixtures):
    prof = copy.deepcopy(DRAFT_PROFILE)
    prof["customer"]["buyers"] = ["Security teams", "Hospital CIOs", "Retail chains"]
    ev = [
        *DRAFT_EVIDENCE,
        {"field_path": "customer.buyers", "source_url": f"{SITE}/", "excerpt": TEAMS_EXCERPT},
    ]
    seen = []

    def record_draft(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    def record_refusal(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": prof, "evidence": ev}),
        record_draft,
        record_refusal,
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    draft, refusal = seen
    assert {(r["field_path"], r["reason_code"]) for r in draft["evidence_rejected"]} >= {
        ("customer.buyers[1]", "NO_OVERLAP"),
        ("customer.buyers[2]", "NO_OVERLAP"),
    }
    assert {"customer.buyers[1]", "customer.buyers[2]"} <= set(refusal["ungrounded"])
    assert "customer.buyers[0]" not in refusal["ungrounded"]
    store = RunStore(settings.runs_dir / outcome.run_id)
    rows = {
        e["field_path"]
        for e in store.list_evidence()
        if e["field_path"].startswith("customer.buyers")
    }
    assert rows == {"customer.buyers[0]"}
    assert _brain(settings, outcome.run_id)["customer"]["buyers"] == ["Security teams"]
    assert store.has_warning("UNGROUNDED_OMITTED")


def test_G1_base_row_does_not_cover_list_items():
    from profile_builder.workflow.gaps import covering_paths, grounding_report

    assert covering_paths("customer.buyers[1]") == ["customer.buyers[1]"]
    assert covering_paths("product.features_and_capabilities[0].how_it_works") == [
        "product.features_and_capabilities[0].how_it_works",
        "product.features_and_capabilities[0]",
    ]
    profile = copy.deepcopy(DRAFT_PROFILE)
    profile["customer"]["buyers"] = ["a", "b"]
    report = grounding_report(profile, {"customer.buyers", "customer.buyers[0]"})
    assert (
        "customer.buyers[1]" in report["ungrounded"]
        and "customer.buyers[0]" not in report["ungrounded"]
    )
    report = grounding_report(profile, {"product.features_and_capabilities[0]"})
    assert not [
        p for p in report["ungrounded"] if p.startswith("product.features_and_capabilities[0]")
    ]


def test_evidence_rejection_codes_for_source_and_length(settings, acme_fixtures):
    ev = [
        *DRAFT_EVIDENCE,
        {
            "field_path": "product.name",
            "source_url": f"{SITE}/pricing",
            "excerpt": "Acme Vault pricing is based on the number of protected workloads.",
        },
        {
            "field_path": "product.description",
            "source_url": f"{SITE}/product",
            "excerpt": "Acme Vault " * 120,
        },
        {"field_path": "customer.buyers", "source_url": f"{SITE}/", "excerpt": TEAMS_EXCERPT},
        {"field_path": "nope.field", "source_url": f"{SITE}/", "excerpt": TEAMS_EXCERPT},
    ]
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": ev}),
        record,
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    runner.start(f"{SITE}/")
    codes = {(r["field_path"], r["reason_code"]) for r in seen[0]["evidence_rejected"]}
    assert codes == {
        ("company.name", "NOT_VERBATIM"),
        ("product.name", "SOURCE_NOT_FETCHED"),
        ("product.description", "TOO_LONG"),
        ("customer.buyers", "FIELD_EMPTY"),
        ("nope.field", "BAD_PATH"),
    }


def test_unknown_keys_are_reported_by_the_tool(settings, acme_fixtures):
    prof = copy.deepcopy(DRAFT_PROFILE)
    prof["extra_top"] = 1
    prof["content_evidence"]["proprietary_insights_or_examples_additional"] = ["x"]
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return tool_call("finalize_profile", {})

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": prof, "evidence": DRAFT_EVIDENCE}),
        record,
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    assert set(seen[0]["ignored_unknown_keys"]) == {
        "extra_top",
        "content_evidence.proprietary_insights_or_examples_additional",
    }
    assert RunStore(settings.runs_dir / outcome.run_id).has_warning("UNKNOWN_KEYS_IGNORED")


# ---------------------------------------------------------------------------------------
# G2: provider rejections fail fast with an actionable message
# ---------------------------------------------------------------------------------------


def _auth_error():
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.AuthenticationError(
        "Incorrect API key provided", response=httpx.Response(401, request=request), body=None
    )


def test_G2_authentication_error_is_not_retried(settings, acme_fixtures):
    model = ScriptedChatModel(steps=_prefix(), fail_first_n=1, failure=_auth_error())
    runner, model, _ = make_runner(settings, [], scraper=FixtureScraper(acme_fixtures), model=model)
    outcome = runner.start(f"{SITE}/")
    assert model.calls == 1
    assert outcome.status == "failed" and outcome.exit_code == 1
    assert outcome.message.startswith(
        "OpenAI rejected the request (check OPENAI_API_KEY / model id)"
    )
    assert "Incorrect API key" in outcome.message
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert store.get_run().status == "failed" and store.has_warning("RUN_FAILED")


def test_retry_predicate_classifies_provider_errors():
    from profile_builder.agent.middleware import is_provider_rejection, is_retryable_model_error

    assert is_provider_rejection(_auth_error()) and not is_retryable_model_error(_auth_error())
    assert is_retryable_model_error(RuntimeError("503 upstream"))
    assert not is_provider_rejection(TimeoutError())


def test_unexpected_error_interrupts_then_resume_completes(settings, acme_fixtures):
    settings = settings.model_copy(update={"max_retries": 0})

    def boom(messages):
        raise RuntimeError("connection reset by peer")

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        boom,
    ]
    scraper = FixtureScraper(acme_fixtures)
    runner1, _, _ = make_runner(settings, steps, scraper=scraper)
    o1 = runner1.start(f"{SITE}/")
    assert o1.status == "interrupted" and o1.exit_code == 3 and o1.resume_hint
    store = RunStore(settings.runs_dir / o1.run_id)
    assert store.get_run().status == "interrupted" and store.has_warning("RUN_INTERRUPTED")
    model2 = ScriptedChatModel(steps=finalize_steps())
    runner2, _, _ = make_runner(settings, [], scraper=scraper, model=model2)
    o2 = runner2.resume(o1.run_id)
    assert o2.status == "complete", o2
    assert _brain(settings, o1.run_id)["product"]["name"] == "Acme Vault"
    assert len([c for c in scraper.calls if c[0] == "scrape"]) == 4  # nothing re-fetched


def test_model_call_cap_exports_partial_with_hint(settings, acme_fixtures, monkeypatch):
    monkeypatch.setattr(config_mod, "MODEL_CALL_BASE", 3)
    monkeypatch.setattr(config_mod, "MODEL_CALLS_PER_PAGE", 0)
    monkeypatch.setattr(config_mod, "MODEL_CALLS_PER_QUESTION", 0)
    assert settings.max_model_calls == 3
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        AIMessage(content="never reached: the cap ends the run"),
    ]
    runner, model, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "partial", outcome
    assert model.calls == 3
    assert "--max-questions" in (outcome.resume_hint or "")
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert store.has_warning("LIMIT_MODEL_CALLS_REACHED")
    assert _evidence_doc(settings, outcome.run_id)["status"] == "partial"


# ---------------------------------------------------------------------------------------
# G3: no usable pages
# ---------------------------------------------------------------------------------------


def test_G3_all_pages_404_without_draft_fails_with_diagnosis(settings, tmp_path):
    empty = tmp_path / "empty-site"
    empty.mkdir()
    steps = [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call("scrape_pages", {"urls": [f"{SITE}/product", f"{SITE}/about"]}),
        AIMessage(content="nothing works"),
        AIMessage(content="still nothing"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(empty))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "failed" and "no usable pages were fetched" in outcome.message
    store = RunStore(settings.runs_dir / outcome.run_id)
    (w,) = [w for w in store.list_warnings() if w["code"] == "NO_USABLE_PAGES"]
    assert {(p["url"], p["error_code"]) for p in w["details"]["pages"]} == {
        (f"{SITE}/", "HTTP_404"),
        (f"{SITE}/product", "HTTP_404"),
        (f"{SITE}/about", "HTTP_404"),
    }
    assert "no usable pages" in runner.console.export_text()
    assert not (store.run_dir / "company_brain.json").exists()


def test_G3_all_pages_404_with_interview_exports_partial_interview_only(settings, tmp_path):
    empty = tmp_path / "empty-site"
    empty.mkdir()
    minimal = {
        **DRAFT_PROFILE,
        "company": {"name": "Acme", "website_url": f"{SITE}/"},
        "product": {
            **DRAFT_PROFILE["product"],
            "features_and_capabilities": [],
            "differentiators": [],
        },
        "customer": {
            k: ([] if isinstance(v, list) else "") for k, v in DRAFT_PROFILE["customer"].items()
        },
        "content_evidence": {k: [] for k in DRAFT_PROFILE["content_evidence"]},
        "brand": {k: [] for k in DRAFT_PROFILE["brand"]},
    }
    q = {
        "question": "Who is the target customer?",
        "why_unclear": "the website could not be fetched",
        "field_paths": ["customer.target_customer"],
        "kind": "gap",
    }

    def apply(messages):
        res = last_tool_result(messages)
        return tool_call(
            "apply_profile_updates",
            {
                "updates": [
                    {
                        "field_path": "customer.target_customer",
                        "value": res["answer"],
                        "evidence": {"kind": "interview", "question_id": res["qid"]},
                    }
                ]
            },
        )

    steps = [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call("save_profile_draft", {"profile": minimal, "evidence": []}),
        tool_call("ask_user", q),
        apply,
        *finalize_steps(),
    ]
    runner, _, asked = make_runner(
        settings, steps, scraper=FixtureScraper(empty), answers=["Regional banks"]
    )
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "partial", outcome
    assert asked == ["Regional banks"]
    evdoc = _evidence_doc(settings, outcome.run_id)
    assert evdoc["status"] == "partial" and evdoc["interview_only"] is True
    brain = _brain(settings, outcome.run_id)
    assert brain["customer"]["target_customer"] == "Regional banks"
    assert brain["product"]["name"] == ""  # website claims without pages are omitted
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert store.has_warning("NO_USABLE_PAGES")


# ---------------------------------------------------------------------------------------
# Scrape edge cases: budgets, redirects, empty / truncated pages, batches
# ---------------------------------------------------------------------------------------


def test_fetch_attempt_budget_and_url_cap(settings, acme_fixtures, monkeypatch):
    monkeypatch.setattr(
        web_tools_mod, "SCRAPE_ATTEMPTS_PER_PAGE", 1
    )  # cap = max_pages (4) attempts
    settings = settings.model_copy(update={"max_retries": 2})
    scraper = FixtureScraper(
        acme_fixtures,
        failures={f"{SITE}/product": [TransientScrapeError("timeout", code="TIMEOUT")] * 10},
    )
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        urls = [f"{SITE}/product", f"{SITE}/customers", f"{SITE}/about", f"{SITE}/pricing"]
        return tool_call("scrape_pages", {"urls": urls + [f"{SITE}/p{i}" for i in range(8)]})

    def record2(messages):
        seen.append(last_tool_result(messages))
        return AIMessage(content="done")

    steps = [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call("scrape_pages", {"urls": [f"{SITE}/product", f"{SITE}/customers"]}),
        record,
        record2,
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=scraper)
    outcome = runner.start(f"{SITE}/")
    store = RunStore(settings.runs_dir / outcome.run_id)
    first, second = seen
    by_url = {r.get("url"): r for r in first["results"]}
    # home(1) + product(2) + customers(3) + product retry(4) = the cap; 3rd attempt refuses
    assert by_url[f"{SITE}/product"]["error_code"] == "FETCH_BUDGET_EXHAUSTED"
    assert by_url[f"{SITE}/customers"]["status"] == "already_fetched"  # fetched on attempt 1
    assert store.has_warning("LIMIT_SCRAPE_ATTEMPTS_REACHED")
    assert len([c for c in scraper.calls if c == ("scrape", f"{SITE}/product")]) == 2
    assert len(store.fetched_urls()) == 2 and store.unique_pages_scraped() == 2  # budget left
    assert second["results"][0]["error_code"] == "TOO_MANY_URLS"
    assert len([r for r in second["results"] if "url" in r]) == 10  # 12 requested, 10 considered
    assert f"{SITE}/p7" not in {r.get("url") for r in second["results"]}


def test_private_ip_redirect_empty_and_truncated_pages(
    settings, acme_fixtures_mutable, monkeypatch
):
    monkeypatch.setattr(ingest_mod, "MAX_PAGE_CHARS", 400)
    fx = acme_fixtures_mutable
    redirect, empty, long_ = f"{SITE}/go", f"{SITE}/empty", f"{SITE}/long"
    FixtureScraper.write_page(
        fx,
        ScrapedPage(
            url=redirect,
            final_url="http://10.0.0.5/admin",
            title="R",
            markdown="body " * 100,
            http_status=200,
        ),
    )
    FixtureScraper.write_page(
        fx, ScrapedPage(url=empty, final_url=empty, title="E", markdown="tiny", http_status=200)
    )
    FixtureScraper.write_page(
        fx,
        ScrapedPage(
            url=long_,
            final_url=long_,
            title="L",
            markdown="# Long\n\n"
            + "Acme protects data in use with confidential computing enclaves. " * 40,
            http_status=200,
        ),
    )
    seen = []

    def record(messages):
        seen.append(last_tool_result(messages))
        return AIMessage(content="done")

    steps = [
        tool_call("scrape_pages", {"urls": [redirect, empty, long_]}),
        record,
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(fx))
    outcome = runner.start(f"{SITE}/")
    store = RunStore(settings.runs_dir / outcome.run_id)
    by_url = {r["url"]: r for r in seen[0]["results"]}
    assert (
        by_url[redirect]["status"] == "rejected"
        and store.get_page(redirect).error_code == "REDIRECT_REJECTED"
    )
    assert store.has_warning("REDIRECT_REJECTED") and redirect not in store.fetched_urls()
    assert (
        by_url[empty]["status"] == "skipped" and store.get_page(empty).error_code == "EMPTY_CONTENT"
    )
    assert store.has_warning("EMPTY_CONTENT")
    assert by_url[long_]["status"] == "fetched" and by_url[long_]["chars"] == 400
    assert store.has_warning("PAGE_TRUNCATED") and store.get_page(long_).char_count == 400
    # all three attempts consumed unique-page budget (something came back for each)
    assert store.unique_pages_scraped() == 3


def test_embedding_failure_does_not_abort_indexing(settings, acme_fixtures, tmp_path):
    class FlakyEmbeddings:
        def __init__(self):
            self.docs = 0

        def embed_documents(self, texts):
            self.docs += 1
            if self.docs == 1:
                raise RuntimeError("embedding provider down")
            from profile_builder.retrieval.index import HashEmbeddings

            return HashEmbeddings().embed_documents(texts)

        def embed_query(self, text):
            raise RuntimeError("embedding provider down")

    store = RunStore(tmp_path / "idx")
    index = HybridIndex(store, FlakyEmbeddings(), embedding_model="fake")
    kept, _ = index.index_page(
        "https://a.example/one",
        "# One\n\nBanks use confidential computing enclaves for fraud analytics.",
        title="One",
    )
    assert kept == 1 and index.last_embedding_error.startswith("RuntimeError")
    index.index_page(
        "https://a.example/two",
        "# Two\n\nHospitals share patient data with research partners securely.",
        title="Two",
    )
    assert index.last_embedding_error is None
    hits = index.search("fraud analytics banks")  # mixed corpus + failing query embedding → BM25
    assert hits and hits[0].url == "https://a.example/one"
    # end to end: the page is indexed BM25-only and the run records the warning
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(
        settings, steps, scraper=FixtureScraper(acme_fixtures), embeddings=FlakyEmbeddings()
    )
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    run_store = RunStore(settings.runs_dir / outcome.run_id)
    assert run_store.has_warning("EMBEDDINGS_UNAVAILABLE")
    assert len(run_store.fetched_urls()) == 4


# ---------------------------------------------------------------------------------------
# Store details, schema, harness
# ---------------------------------------------------------------------------------------


def test_add_evidence_dedupes_and_dead_method_is_gone(tmp_path):
    store = RunStore(tmp_path / "run")
    for _ in range(3):
        store.add_evidence("customer.buyers[0]", "website", source_url="u", excerpt="same quote")
    store.add_evidence("customer.buyers[0]", "website", source_url="u", excerpt="other quote")
    store.add_evidence("customer.buyers[0]", "interview", question_id="q1", answer="a")
    store.add_evidence("customer.buyers[0]", "interview", question_id="q1", answer="a")
    assert len(store.list_evidence()) == 3
    assert not hasattr(RunStore, "supersede_evidence")


def test_G5_schema_requires_every_property():
    schema = json_schema()

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                assert node["required"] == list(node["properties"]), node.get("title")
            for v in node.values():
                check(v)

    check(schema)
    assert schema["required"] == [
        "artifact",
        "version",
        "company",
        "product",
        "customer",
        "content_evidence",
        "brand",
    ]
    assert schema["$defs"]["FeatureCapability"]["required"] == [
        "name",
        "description",
        "how_it_works",
        "customer_benefit",
    ]


def test_G4_tests_are_offline():
    with pytest.raises(RuntimeError, match="network access is not allowed"):
        requests.get("https://example.com/", timeout=1)
    with pytest.raises(RuntimeError, match="network access is not allowed"):
        requests.Session().get("https://example.com/", timeout=1)
    with pytest.raises(RuntimeError, match="network access is not allowed"):
        socket.getaddrinfo("example.com", 443)
    assert not (Path.cwd() / ".env").exists()  # tests never run in the repo root


def test_harness_tool_call_ids_are_unique_and_exhaustion_is_loud():
    ids = {tool_call("search_pages", {"query": "x"}).tool_calls[0]["id"] for _ in range(50)}
    assert len(ids) == 50
    model = ScriptedChatModel(steps=[])
    with pytest.raises(ScriptExhausted):
        model._generate([])
    assert not issubclass(ScriptExhausted, Exception)
    assert last_tool_result(
        [
            AIMessage(content="", tool_calls=[{"name": "a", "args": {}, "id": "1"}]),
            ToolMessage(content="not json", tool_call_id="1", status="error"),
            ToolMessage(content='{"ok": true}', tool_call_id="other"),
        ]
    ) == {"_raw": "not json", "_status": "error"}
