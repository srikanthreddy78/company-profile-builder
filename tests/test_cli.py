"""CLI commands against a run produced offline: status, inspect, logs, export, schema, list, doctor."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from profile_builder.cli import app
from profile_builder.web.scraper import FixtureScraper
from tests.conftest import SITE, happy_path_steps, make_runner

cli = CliRunner()


def _env(settings) -> dict[str, str]:
    return {
        "PROFILE_BUILDER_RUNS_DIR": str(settings.runs_dir),
        "OPENAI_API_KEY": "",
        "FIRECRAWL_API_KEY": "",
        "COLUMNS": "160",
    }


def test_schema_command(tmp_path: Path):
    out = tmp_path / "s.json"
    result = cli.invoke(app, ["schema", "--out", str(out)])
    assert result.exit_code == 0 and json.loads(out.read_text())["title"] == "company_brain"
    result = cli.invoke(app, ["schema"])
    assert result.exit_code == 0 and "features_and_capabilities" in result.output


def test_status_inspect_logs_export_list(settings, acme_fixtures, monkeypatch):
    runner, _, _ = make_runner(
        settings,
        happy_path_steps(),
        scraper=FixtureScraper(acme_fixtures),
        answers=["Regulated enterprises"],
    )
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    for k, v in _env(settings).items():
        monkeypatch.setenv(k, v)
    monkeypatch.chdir(settings.runs_dir.parent)  # no .env in cwd

    result = cli.invoke(app, ["status", "--run-id", outcome.run_id])
    assert result.exit_code == 0 and "complete" in result.output and "Interview" in result.output

    result = cli.invoke(app, ["inspect", "--run-id", outcome.run_id, "--field", "customer."])
    assert (
        result.exit_code == 0
        and "customer.target_customer" in result.output
        and "interview" in result.output
    )
    assert "superseded" in result.output  # original website evidence still shown

    result = cli.invoke(app, ["logs", "--run-id", outcome.run_id, "--json", "--tail", "20"])
    assert result.exit_code == 0
    rows = [json.loads(line) for line in result.output.strip().splitlines()]
    assert len(rows) == 20 and any(r.get("event") == "run_finished" for r in rows)
    assert all(r["run_id"] == outcome.run_id for r in rows)
    result = cli.invoke(app, ["logs", "--run-id", outcome.run_id, "--level", "WARNING", "--json"])
    warnings = [json.loads(line) for line in result.output.strip().splitlines()]
    assert result.exit_code == 0 and all(w["level"] == "WARNING" for w in warnings)
    assert {w.get("code") for w in warnings} >= {
        "EVIDENCE_REJECTED",
        "USER_CORRECTION_SUPERSEDES_SITE",
    }
    result = cli.invoke(app, ["logs", "--run-id", outcome.run_id, "--level", "WARNING"])
    assert result.exit_code == 0 and "events" in result.output

    result = cli.invoke(app, ["export", "--run-id", outcome.run_id])
    assert (
        result.exit_code == 0
        and (settings.runs_dir / outcome.run_id / "company_brain.json").exists()
    )

    result = cli.invoke(app, ["list"])
    assert result.exit_code == 0 and outcome.run_id in result.output

    result = cli.invoke(app, ["status", "--run-id", "pb-20260101-zzzzzz"])
    assert result.exit_code == 2
    result = cli.invoke(app, ["status", "--run-id", "../../etc"])
    assert result.exit_code != 0


def test_doctor_without_keys(settings, monkeypatch):
    for k, v in _env(settings).items():
        monkeypatch.setenv(k, v)
    monkeypatch.chdir(settings.runs_dir.parent)
    result = cli.invoke(app, ["doctor"])
    assert result.exit_code == 1 and "missing" in result.output


def test_start_rejects_private_url(settings, monkeypatch):
    for k, v in _env(settings).items():
        monkeypatch.setenv(k, v)
    monkeypatch.chdir(settings.runs_dir.parent)
    result = cli.invoke(
        app,
        ["start", "--url", "http://169.254.169.254/latest", "--fixtures", str(settings.runs_dir)],
    )
    assert result.exit_code == 2 and "rejected" in result.output
