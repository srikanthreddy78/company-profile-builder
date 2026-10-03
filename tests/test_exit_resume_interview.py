"""Exit during the interview and resume in a fresh process without losing committed
answers or re-asking them; crash replay of ask_user is idempotent."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage

from profile_builder.state.run_store import RunStore
from profile_builder.web.scraper import FixtureScraper
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

Q1 = {
    "question": "Which customer segment should this profile prioritize?",
    "why_unclear": "Homepage and customers page disagree.",
    "field_paths": ["customer.target_customer"],
    "kind": "conflict",
}
Q2 = {
    "question": "Who typically signs off on the purchase (buyers)?",
    "why_unclear": "No page names the decision maker.",
    "field_paths": ["customer.buyers"],
    "kind": "gap",
}


def two_question_steps():
    def apply_first(messages: list[BaseMessage]) -> AIMessage:
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

    def apply_second(messages: list[BaseMessage]) -> AIMessage:
        res = last_tool_result(messages)
        if res.get("status") == "answered":
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
        return tool_call("finalize_profile", {})

    return [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call(
            "scrape_pages", {"urls": [f"{SITE}/product", f"{SITE}/customers", f"{SITE}/about"]}
        ),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        tool_call("ask_user", Q1, "ask1"),
        apply_first,
        tool_call("ask_user", Q2, "ask2"),
        apply_second,
        *finalize_steps(),
    ]


def test_exit_then_resume_keeps_committed_answer(settings, acme_fixtures):
    scraper = FixtureScraper(acme_fixtures)
    # Process 1: answer Q1, then exit when Q2 appears.
    runner1, _model1, _asked1 = make_runner(
        settings, two_question_steps(), scraper=scraper, answers=["Regulated enterprises", "exit"]
    )
    outcome1 = runner1.start(f"{SITE}/")
    assert outcome1.status == "paused" and outcome1.resume_hint
    run_dir = settings.runs_dir / outcome1.run_id
    store = RunStore(run_dir)
    qs = store.list_questions()
    assert [q["status"] for q in qs] == ["answered", "pending"]
    assert qs[0]["answer"] == "Regulated enterprises"
    assert store.get_run().status == "paused"
    assert not (run_dir / "company_brain.json").exists()

    # Process 2: a brand-new model/agent instance resumes from the SQLite checkpoint. The
    # scripted model continues from the step after the pending ask_user (the agent replays
    # the pending tool call itself, so the model is not asked for it again).
    model2 = ScriptedChatModel(steps=two_question_steps()[6:])
    runner2, _, asked2 = make_runner(settings, [], scraper=scraper, model=model2, answers=["CISOs"])
    outcome2 = runner2.resume(outcome1.run_id)
    assert outcome2.status == "complete", outcome2
    store = RunStore(run_dir)
    qs = store.list_questions()
    assert [q["status"] for q in qs] == ["answered", "answered"]
    assert asked2 == ["CISOs"]  # Q1 was NOT re-asked
    brain = json.loads((run_dir / "company_brain.json").read_text())
    assert brain["customer"]["target_customer"] == "Regulated enterprises"
    assert brain["customer"]["buyers"] == ["CISOs"]
    # pages were not re-fetched on resume
    assert len([c for c in scraper.calls if c[0] == "scrape"]) == 4


def test_skip_and_idk_are_recorded_as_unresolved(settings, acme_fixtures):
    scraper = FixtureScraper(acme_fixtures)
    steps = two_question_steps()

    # replace apply_first with a step that tolerates a skipped answer
    def after_q1(messages):
        res = last_tool_result(messages)
        assert res["status"] == "skipped" and res["answer"] == "SKIPPED"
        return steps[5]

    steps[4] = after_q1
    runner, _, _ = make_runner(settings, steps, scraper=scraper, answers=["skip", "I don't know"])
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert [q["status"] for q in store.list_questions()] == ["skipped", "unknown"]
    ev = json.loads((store.run_dir / "evidence.json").read_text())
    assert {q["status"] for q in ev["questions"]} == {"skipped", "unknown"}


def test_non_interactive_mode_skips_everything(settings, acme_fixtures):
    runner, _, _ = make_runner(
        settings, two_question_steps(), scraper=FixtureScraper(acme_fixtures), non_interactive=True
    )
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert all(q["status"] == "skipped" for q in store.list_questions())


def test_question_limit_is_enforced(settings, acme_fixtures):
    settings = settings.model_copy(update={"max_questions": 1})
    steps = two_question_steps()

    def after_q2(messages):
        res = last_tool_result(messages)
        assert res["ok"] is False and "limit" in res["error"]
        return tool_call("finalize_profile", {})

    steps[6] = after_q2
    runner, _, asked = make_runner(
        settings,
        steps,
        scraper=FixtureScraper(acme_fixtures),
        answers=["Regulated enterprises", "should-not-be-asked"],
    )
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete" and asked == ["Regulated enterprises"]
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert store.has_warning("LIMIT_QUESTIONS_REACHED") and store.questions_asked() == 1


def test_multi_question_prompts_are_rejected():
    from profile_builder.agent.tools import is_multi_question

    assert is_multi_question(
        "Please answer each:\n1) Who buys?\n2) Who uses?\n3) Which alternatives?"
    )
    assert is_multi_question(
        "Who buys it? Who uses it? Which alternatives matter? Any terms to avoid?"
    )
    assert not is_multi_question(
        "Which customer segment should this profile prioritize (banks or startups)?"
    )
    assert not is_multi_question("Who signs off on the purchase? For example a CISO or a CIO.")
