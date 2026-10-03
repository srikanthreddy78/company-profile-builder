"""Settings precedence and derived limits: one variable drives everything."""

from __future__ import annotations

from profile_builder.agent.prompts import render_system_prompt
from profile_builder.config import (
    MODEL_CALL_BASE,
    MODEL_CALLS_PER_PAGE,
    MODEL_CALLS_PER_QUESTION,
    Settings,
)
from profile_builder.models import MODEL_TIERS


def test_defaults_and_derived_limits():
    s = Settings(_env_file=None)
    assert s.max_pages == 10 and s.max_questions == 5
    assert s.model_id == MODEL_TIERS["fast"]
    assert s.budget_usd is None  # unlimited by default
    assert s.max_model_calls == MODEL_CALL_BASE + 3 * MODEL_CALLS_PER_PAGE * 10 // 3 + MODEL_CALLS_PER_QUESTION * 5


def test_env_precedence(monkeypatch):
    monkeypatch.setenv("PROFILE_BUILDER_MAX_PAGES", "15")
    monkeypatch.setenv("PROFILE_BUILDER_TIER", "quality")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-1234567890")
    s = Settings(_env_file=None)
    assert s.max_pages == 15
    assert s.model_id == MODEL_TIERS["quality"]
    assert s.has_openai()
    flagged = s.with_overrides(max_pages=20, model="gpt-5.4-mini", budget_usd=None)
    assert flagged.max_pages == 20 and flagged.model_id == "gpt-5.4-mini"
    assert flagged.budget_usd is None  # None overrides are ignored


def test_max_pages_propagates_everywhere():
    small = Settings(_env_file=None, max_pages=10, max_questions=5)
    big = Settings(_env_file=None, max_pages=15, max_questions=5)
    assert big.max_scrape_calls > small.max_scrape_calls
    assert big.max_model_calls == small.max_model_calls + 5 * MODEL_CALLS_PER_PAGE
    prompt = render_system_prompt(max_pages=big.max_pages, max_questions=big.max_questions, product_focus=None)
    assert "at most 15 unique pages" in prompt and "at most 5 interview" in prompt


def test_snapshot_roundtrip_keeps_limits_but_not_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-abcdefghijkl")
    s = Settings(_env_file=None, max_pages=7, max_questions=2, budget_usd=0.5, tier="quality")
    snap = s.snapshot()
    assert "openai_api_key" not in snap and snap["model_id"] == MODEL_TIERS["quality"]
    restored = Settings.from_snapshot(snap)
    assert restored.max_pages == 7 and restored.max_questions == 2 and restored.budget_usd == 0.5
    assert restored.has_openai()  # secrets come from the environment, not the snapshot
    extended = Settings.from_snapshot(snap, max_questions=8)
    assert extended.max_questions == 8 and extended.max_pages == 7
