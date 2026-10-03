"""Regression tests for the second external review: grounding at export, user-answer
precedence, idempotent re-export, crash-after-commit resume, resumable partial runs,
transient 5xx classification, page accounting, redaction, exit codes, short facts, stages."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage

from profile_builder.logging_setup import read_events
from profile_builder.retrieval.chunking import chunk_markdown
from profile_builder.state.run_store import RunStore
from profile_builder.web.scraper import FixtureScraper, ScrapedPage, TransientScrapeError
from profile_builder.workflow.runner import RunOutcome
from tests.conftest import (
    DRAFT_EVIDENCE,
    DRAFT_PROFILE,
    SITE,
    ScriptedChatModel,
    finalize_steps,
    last_tool_result,
    make_runner,
    tool_call,
)

Q_SEGMENT = {
    "question": "Which segment should this profile prioritize?",
    "why_unclear": "pages disagree",
    "field_paths": ["customer.target_customer"],
    "kind": "conflict",
}


def _prefix():
    return [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call(
            "scrape_pages", {"urls": [f"{SITE}/product", f"{SITE}/customers", f"{SITE}/about"]}
        ),
    ]


# 1. grounding is enforced at export -----------------------------------------------------------


def test_ungrounded_claims_are_omitted_from_export(settings, acme_fixtures):
    fabricated = {
        **DRAFT_PROFILE,
        "customer": {**DRAFT_PROFILE["customer"], "buyers": ["Martian procurement officers"]},
    }
    seen: list[dict] = []

    def after_first_finalize(messages: list[BaseMessage]) -> AIMessage:
        res = last_tool_result(messages)
        seen.append(res)
        return tool_call("finalize_profile", {}, "second")

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": fabricated, "evidence": DRAFT_EVIDENCE}),
        tool_call("finalize_profile", {}, "first"),
        after_first_finalize,
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    assert seen[0]["ok"] is False and "customer.buyers[0]" in seen[0]["ungrounded"]
    store = RunStore(settings.runs_dir / outcome.run_id)
    brain = json.loads((store.run_dir / "company_brain.json").read_text())
    assert brain["customer"]["buyers"] == []  # fabricated claim did not ship
    assert brain["product"]["name"] == "Acme Vault"  # grounded fields did
    assert store.has_warning("UNGROUNDED_OMITTED")
    ev = json.loads((store.run_dir / "evidence.json").read_text())
    assert ev["grounding"]["ungrounded"] == []


# 2. user corrections take precedence ----------------------------------------------------------


def test_user_answer_survives_later_draft_and_blocks_website_override(settings, acme_fixtures):
    def apply_answer(messages):
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

    later_draft = {
        **DRAFT_PROFILE,
        "customer": {**DRAFT_PROFILE["customer"], "target_customer": "Small bakeries in Ohio"},
    }
    results: list[dict] = []

    def record(messages):
        results.append(last_tool_result(messages))
        return tool_call(
            "apply_profile_updates",
            {
                "updates": [
                    {
                        "field_path": "customer.target_customer",
                        "value": "Startups everywhere",
                        "evidence": {
                            "kind": "website",
                            "source_url": f"{SITE}/",
                            "excerpt": "Acme Vault helps startups and enterprises protect sensitive data in use",
                        },
                    }
                ]
            },
            "web_override",
        )

    def record2(messages):
        results.append(last_tool_result(messages))
        return tool_call("finalize_profile", {}, "fin")

    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call("ask_user", Q_SEGMENT),
        apply_answer,
        tool_call(
            "save_profile_draft", {"profile": later_draft, "evidence": DRAFT_EVIDENCE}, "later"
        ),
        record,
        record2,
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(
        settings, steps, scraper=FixtureScraper(acme_fixtures), answers=["Only banks"]
    )
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    assert "customer.target_customer" in results[0]["protected_by_user_answers"]
    assert results[1]["rejected"][0]["reason_code"] == "USER_ANSWER_PROTECTED"
    store = RunStore(settings.runs_dir / outcome.run_id)
    brain = json.loads((store.run_dir / "company_brain.json").read_text())
    assert brain["customer"]["target_customer"] == "Only banks"
    active = [
        e
        for e in store.list_evidence(include_superseded=False)
        if e["field_path"] == "customer.target_customer"
    ]
    assert [e["kind"] for e in active] == ["interview"]


# 3. re-export is idempotent and evidence indexes follow their items ------------------------


def test_reexport_does_not_delete_more_items(settings, acme_fixtures):
    two_buyers = {
        **DRAFT_PROFILE,
        "customer": {**DRAFT_PROFILE["customer"], "buyers": ["Disputed buyer", "Security teams"]},
    }
    evidence = [
        *DRAFT_EVIDENCE,
        {
            "field_path": "customer.buyers[0]",
            "source_url": f"{SITE}/",
            "excerpt": "Trusted by security teams at fast-moving startups and global enterprises alike.",
        },
        {
            "field_path": "customer.buyers[1]",
            "source_url": f"{SITE}/",
            "excerpt": "Trusted by security teams at fast-moving startups and global enterprises alike.",
        },
    ]
    steps = [
        *_prefix(),
        tool_call("save_profile_draft", {"profile": two_buyers, "evidence": evidence}),
        tool_call(
            "note_conflict",
            {
                "field_path": "customer.buyers[0]",
                "claims": [
                    {
                        "source_url": f"{SITE}/",
                        "excerpt": "helps startups and enterprises protect sensitive data",
                        "claim": "startups",
                    },
                    {
                        "source_url": f"{SITE}/customers",
                        "excerpt": "used by Fortune 500 banks and healthcare networks",
                        "claim": "banks",
                    },
                ],
                "summary": "who buys is disputed",
            },
        ),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    store = RunStore(settings.runs_dir / outcome.run_id)
    brain = json.loads((store.run_dir / "company_brain.json").read_text())
    assert brain["customer"]["buyers"] == ["Security teams"]
    assert [c["status"] for c in store.list_conflicts()] == ["omitted"]
    # evidence for the surviving item moved from [1] to [0]
    paths = {
        e["field_path"]
        for e in store.list_evidence(include_superseded=False)
        if e["field_path"].startswith("customer.buyers")
    }
    assert paths == {"customer.buyers[0]"}
    # re-export twice: nothing else changes
    for _ in range(2):
        out = runner.export(outcome.run_id)
        assert out.status == "complete"
        brain = json.loads((store.run_dir / "company_brain.json").read_text())
        assert brain["customer"]["buyers"] == ["Security teams"]


# 4. crash after the answer was committed but before the checkpoint advanced ----------------


def test_resume_does_not_reask_committed_answer(settings, acme_fixtures):
    from tests.test_exit_resume_interview import two_question_steps

    scraper = FixtureScraper(acme_fixtures)
    runner1, _, _ = make_runner(
        settings, two_question_steps(), scraper=scraper, answers=[]
    )  # EOF at Q1 → paused
    outcome1 = runner1.start(f"{SITE}/")
    assert outcome1.status == "paused"
    store = RunStore(settings.runs_dir / outcome1.run_id)
    pending = store.pending_question()
    # simulate: the tool committed the answer, then the process died before the checkpoint
    store.answer_question(pending["qid"], "answered", "Regulated enterprises")
    model2 = ScriptedChatModel(steps=two_question_steps()[4:])
    runner2, _, asked = make_runner(settings, [], scraper=scraper, model=model2, answers=["CISOs"])
    outcome2 = runner2.resume(outcome1.run_id)
    assert outcome2.status == "complete", outcome2
    assert asked == ["CISOs"]  # Q1 was NOT shown again; only Q2 was asked
    brain = json.loads((store.run_dir / "company_brain.json").read_text())
    assert brain["customer"]["target_customer"] == "Regulated enterprises"


# 5. partial runs resume after the budget is raised ------------------------------------------


def test_partial_run_resumes_with_higher_budget(settings, acme_fixtures):
    tight = settings.model_copy(update={"budget_usd": 0.002})  # ~4 scripted model calls
    steps = [
        *_prefix(),
        tool_call("search_pages", {"query": "customers"}),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call("ask_user", Q_SEGMENT),
        AIMessage(content="unused"),
        *finalize_steps(),
    ]
    scraper = FixtureScraper(acme_fixtures)
    runner1, model1, _ = make_runner(tight, steps, scraper=scraper, answers=["Only banks"])
    outcome1 = runner1.start(f"{SITE}/")
    assert outcome1.status == "partial", outcome1
    assert "--budget-usd" in (outcome1.resume_hint or ""), outcome1  # README promises a hint
    store = RunStore(tight.runs_dir / outcome1.run_id)
    assert store.has_warning("BUDGET_EXCEEDED")
    used = model1.cursor
    model2 = ScriptedChatModel(steps=[*steps[used:]])
    runner2, _, _asked = make_runner(
        tight, [], scraper=scraper, model=model2, answers=["Only banks"]
    )
    outcome2 = runner2.resume(outcome1.run_id, budget_usd=5.0)
    assert outcome2.status == "complete", outcome2
    assert store.get_run().settings["budget_usd"] == 5.0


# 6. transient 5xx failures are retried on a later call ---------------------------------------


def test_http_503_page_is_refetched_later(settings, acme_fixtures):
    scraper = FixtureScraper(
        acme_fixtures,
        failures={f"{SITE}/product": [TransientScrapeError("503", code="HTTP_503")] * 3},
    )
    missing = f"{SITE}/does-not-exist"
    seen: list[dict] = []

    def after_first_scrape(messages):
        # retries exhausted: the page is recorded as a *transient* failure, not given up on
        store = RunStore(next(settings.runs_dir.glob("pb-*")))
        rec = store.get_page(f"{SITE}/product")
        assert rec is not None and rec.status == "failed" and rec.error_code == "HTTP_503"
        assert store.get_page(missing).error_code == "HTTP_404"
        seen.append(last_tool_result(messages))
        return tool_call("scrape_pages", {"urls": [f"{SITE}/product", missing]})

    def after_second_scrape(messages):
        seen.append(last_tool_result(messages))
        return AIMessage(content="done")

    steps = [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call("scrape_pages", {"urls": [f"{SITE}/product", missing, f"{SITE}/customers"]}),
        after_first_scrape,
        after_second_scrape,
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=scraper)
    outcome = runner.start(f"{SITE}/")
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert f"{SITE}/product" in store.fetched_urls()
    assert seen[0]["_status"] == "error"  # the retried call ended as a tool error
    by_url = {r.get("url"): r for r in seen[1]["results"]}
    assert by_url[f"{SITE}/product"]["status"] == "fetched"
    assert by_url[missing]["status"] == "failed" and by_url[missing]["error_code"] == "HTTP_404"
    assert "already failed" in by_url[missing]["error"]
    # the batch was not starved by the failing URL: customers was fetched in the first call
    assert len([c for c in scraper.calls if c == ("scrape", f"{SITE}/customers")]) == 1
    assert len([c for c in scraper.calls if c == ("scrape", missing)]) == 1


# 7. page budget counts unique scrapes; discovery reuses the cached homepage ----------------


def test_duplicates_count_toward_budget_and_homepage_fetched_once(settings, acme_fixtures_mutable):
    from profile_builder.security import url_cache_name

    dup = f"{SITE}/about-copy"
    about_page = ScrapedPage.from_json(
        (acme_fixtures_mutable / "pages" / url_cache_name(f"{SITE}/about")).read_text()
    )
    FixtureScraper.write_page(
        acme_fixtures_mutable,
        ScrapedPage(
            url=dup,
            final_url=dup,
            title="About copy",
            markdown=about_page.markdown,
            http_status=200,
        ),
    )
    small = settings.model_copy(update={"max_pages": 3})
    scraper = FixtureScraper(acme_fixtures_mutable)
    steps = [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call("scrape_pages", {"urls": [f"{SITE}/about", dup, f"{SITE}/product"]}),
        AIMessage(content="done"),
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(small, steps, scraper=scraper)
    outcome = runner.start(f"{SITE}/")
    store = RunStore(small.runs_dir / outcome.run_id)
    assert (
        len([c for c in scraper.calls if c == ("scrape", f"{SITE}/")]) == 1
    )  # homepage fetched once
    assert store.get_page(dup).error_code == "DUPLICATE_CONTENT"
    assert store.unique_pages_scraped() == 3  # home + about + duplicate
    assert f"{SITE}/product" not in store.fetched_urls() and store.has_warning(
        "LIMIT_PAGES_REACHED"
    )


# 8. secrets in transient errors never reach the database ------------------------------------


def test_transient_error_text_is_redacted_before_storage(settings, acme_fixtures):
    secret = "sk-proj-SYNTHETICSECRET1234567890abcdef"
    scraper = FixtureScraper(
        acme_fixtures,
        failures={
            f"{SITE}/product": [
                TransientScrapeError(f"upstream said Bearer {secret}", code="NETWORK")
            ]
            * 3
        },
    )
    steps = [*_prefix(), AIMessage(content="done"), AIMessage(content="done")]
    runner, _, _ = make_runner(settings, steps, scraper=scraper)
    outcome = runner.start(f"{SITE}/")
    store = RunStore(settings.runs_dir / outcome.run_id)
    page = store.get_page(f"{SITE}/product")
    assert page is not None and secret not in (page.error or "")
    for name in ("events.jsonl", "run.log"):
        assert secret not in (store.run_dir / name).read_text()


# 9. exit codes --------------------------------------------------------------------------------


def test_exit_codes():
    assert RunOutcome("pb-20260101-aaaaaa", "complete").exit_code == 0
    assert RunOutcome("pb-20260101-aaaaaa", "partial").exit_code == 0
    assert RunOutcome("pb-20260101-aaaaaa", "paused").exit_code == 0
    assert RunOutcome("pb-20260101-aaaaaa", "failed").exit_code == 1
    assert RunOutcome("pb-20260101-aaaaaa", "interrupted").exit_code == 3


# 10. short facts survive chunking ------------------------------------------------------------


def test_short_facts_and_small_lists_are_kept():
    chunks, _ = chunk_markdown(
        "# Pricing\n\nPlans start at $99 per month.\n\n## Platforms\n\n- Intel SGX\n- AMD SEV\n- Intel TDX\n- NVIDIA H100\n",
        page_title="Pricing",
    )
    text = " ".join(c.text for c in chunks)
    assert "$99 per month" in text and "NVIDIA H100" in text
    nav = "\n".join(
        ["Home", "Products", "Solutions", "Customers", "Pricing", "About", "Careers", "Contact"]
    )
    chunks, _ = chunk_markdown(
        f"# Site\n\n{nav}\n\nWe protect data in use with confidential computing for regulated industries."
    )
    assert all("Careers" not in c.text for c in chunks)


# 11. stage labels on model-call telemetry -----------------------------------------------------


def test_model_call_events_carry_real_stages(settings, acme_fixtures):
    from tests.conftest import happy_path_steps

    runner, _, _ = make_runner(
        settings, happy_path_steps(), scraper=FixtureScraper(acme_fixtures), answers=["Banks"]
    )
    outcome = runner.start(f"{SITE}/")
    stages = {
        e["stage"]
        for e in read_events(settings.runs_dir / outcome.run_id)
        if e.get("event") == "model_call"
    }
    assert {"discover", "scrape", "draft"} <= stages, stages
