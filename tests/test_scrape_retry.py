"""Transient failures (429 / timeout) are retried with a cap; permanent ones are not."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from profile_builder.logging_setup import read_events
from profile_builder.state.run_store import RunStore
from profile_builder.web.scraper import (
    FixtureScraper,
    PermanentScrapeError,
    TransientScrapeError,
    classify_firecrawl_error,
)
from tests.conftest import SITE, make_runner, tool_call


def _steps():
    return [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call("scrape_pages", {"urls": [f"{SITE}/product", f"{SITE}/customers"]}),
        AIMessage(content="done"),
        AIMessage(content="done"),
    ]


def test_429_then_success_is_retried_and_logged(settings, acme_fixtures):
    scraper = FixtureScraper(
        acme_fixtures,
        failures={
            f"{SITE}/product": [
                TransientScrapeError("rate limited (429)", code="RATE_LIMITED"),
                TransientScrapeError("timeout", code="TIMEOUT"),
            ],
        },
    )
    settings = settings.model_copy(update={"max_retries": 2})
    runner, _, _ = make_runner(settings, _steps(), scraper=scraper)
    outcome = runner.start(f"{SITE}/")
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert f"{SITE}/product" in store.fetched_urls()
    assert f"{SITE}/customers" in store.fetched_urls()
    scrape_calls = [c for c in scraper.calls if c == ("scrape", f"{SITE}/product")]
    assert len(scrape_calls) == 3  # first attempt + 2 retries
    events = read_events(store.run_dir)
    tool_attempts = [
        e["attempt"]
        for e in events
        if e.get("event") == "tool_call" and e.get("tool") == "scrape_pages"
    ]
    assert max(tool_attempts) == 3
    assert any(e.get("status") == "transient" for e in events if e.get("event") == "page_fetch")


def test_retries_are_capped_and_run_continues(settings, acme_fixtures):
    scraper = FixtureScraper(
        acme_fixtures,
        failures={
            f"{SITE}/product": [TransientScrapeError("timeout", code="TIMEOUT")] * 10,
        },
    )
    settings = settings.model_copy(update={"max_retries": 2})
    runner, _, _ = make_runner(settings, _steps(), scraper=scraper)
    outcome = runner.start(f"{SITE}/")
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert len([c for c in scraper.calls if c == ("scrape", f"{SITE}/product")]) == 3
    assert f"{SITE}/product" not in store.fetched_urls()
    failed = store.get_page(f"{SITE}/product")
    assert failed is not None and failed.status == "failed" and failed.error_code == "TIMEOUT"
    assert outcome.status == "failed"  # no draft existed → no fabricated profile


def test_404_is_not_retried(settings, acme_fixtures):
    scraper = FixtureScraper(acme_fixtures)
    steps = [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call("scrape_pages", {"urls": [f"{SITE}/does-not-exist", f"{SITE}/product"]}),
        AIMessage(content="done"),
        AIMessage(content="done"),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=scraper)
    outcome = runner.start(f"{SITE}/")
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert len([c for c in scraper.calls if c == ("scrape", f"{SITE}/does-not-exist")]) == 1
    assert store.has_warning("PAGE_SKIPPED_HTTP_404")
    assert f"{SITE}/product" in store.fetched_urls()


def test_firecrawl_error_classification():
    import requests
    from firecrawl.v2.utils import error_handler as eh

    assert isinstance(classify_firecrawl_error(eh.RateLimitError("429")), TransientScrapeError)
    assert isinstance(classify_firecrawl_error(eh.RequestTimeoutError("408")), TransientScrapeError)
    assert isinstance(classify_firecrawl_error(eh.InternalServerError("500")), TransientScrapeError)
    assert isinstance(
        classify_firecrawl_error(requests.exceptions.ConnectTimeout()), TransientScrapeError
    )
    assert isinstance(
        classify_firecrawl_error(eh.PaymentRequiredError("402")), PermanentScrapeError
    )
    assert isinstance(classify_firecrawl_error(eh.UnauthorizedError("401")), PermanentScrapeError)
    assert classify_firecrawl_error(eh.WebsiteNotSupportedError("403")).code == "BLOCKED"


def test_model_transient_error_is_retried(settings, acme_fixtures):
    from tests.conftest import ScriptedChatModel

    scraper = FixtureScraper(acme_fixtures)
    model = ScriptedChatModel(steps=_steps(), fail_first_n=1, failure=RuntimeError("503 upstream"))
    runner, model, _ = make_runner(settings, [], scraper=scraper, model=model)
    outcome = runner.start(f"{SITE}/")
    store = RunStore(settings.runs_dir / outcome.run_id)
    assert model.calls >= 2
    events = read_events(store.run_dir)
    model_events = [e for e in events if e.get("event") == "model_call"]
    assert any(e.get("status") == "error" for e in model_events) and any(
        e.get("status") == "ok" for e in model_events
    )
