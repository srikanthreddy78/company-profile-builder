"""Regression tests for the sixth review round (publish-journal edge cases): publish and
recovery are serialized by a per-run lock so a concurrent command never rolls back a publish
in progress, removing the journal is part of the commit (a failure there rolls back like any
other), and a failure while staging or backing up still cleans up and raises the original
error instead of an ``UnboundLocalError``."""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
from pathlib import Path

import pytest

import profile_builder.workflow.export as export_mod
from profile_builder.config import PUBLISH_LOCK_FILENAME
from profile_builder.state.run_store import RunStore
from profile_builder.web.scraper import FixtureScraper
from profile_builder.workflow.export import (
    PUBLISH_JOURNAL_FILENAME,
    _publish_lock,
    recover_interrupted_publish,
    write_all_or_nothing,
)
from tests.conftest import (
    DRAFT_EVIDENCE,
    DRAFT_PROFILE,
    SITE,
    finalize_steps,
    make_runner,
    tool_call,
)

OUTPUTS = ("company_brain.json", "evidence.json", "report.md")


def _artifacts(run_dir: Path, tag: str) -> list[tuple[Path, str, int]]:
    return [
        (run_dir / "company_brain.json", f'{{"tag": "{tag}"}}\n', 0o644),
        (run_dir / "evidence.json", f'{{"evidence": "{tag}"}}\n', 0o600),
        (run_dir / "report.md", f"# report {tag}\n", 0o600),
    ]


def _publish(run_dir: Path, tag: str) -> None:
    write_all_or_nothing(_artifacts(run_dir, tag), journal=run_dir / PUBLISH_JOURNAL_FILENAME)


def _contents(run_dir: Path) -> dict[str, bytes]:
    return {n: (run_dir / n).read_bytes() for n in OUTPUTS if (run_dir / n).exists()}


def _names(run_dir: Path) -> list[str]:
    return sorted(p.name for p in run_dir.iterdir())


def _kill_on_second_rename(monkeypatch: pytest.MonkeyPatch) -> None:
    """The second output rename raises KeyboardInterrupt: the process 'dies' with
    company_brain.json replaced and the other two outputs still old (or missing)."""
    real = os.replace

    def crash(src, dst, *args, **kwargs):
        if Path(dst).name == "evidence.json":
            raise KeyboardInterrupt
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", crash)


def _half_published(run_dir: Path) -> dict[str, bytes]:
    """Publish v1, then kill a v2 publish between renames; returns the v1 contents."""
    _publish(run_dir, "v1")
    before = _contents(run_dir)
    with pytest.MonkeyPatch.context() as mp:
        _kill_on_second_rename(mp)
        with pytest.raises(KeyboardInterrupt):
            _publish(run_dir, "v2")
    assert (run_dir / PUBLISH_JOURNAL_FILENAME).exists()
    assert _contents(run_dir) != before
    return before


def _record_events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []
    real = export_mod.event

    def recording(logger, name, message, *args, **kwargs):
        seen.append(name)
        real(logger, name, message, *args, **kwargs)

    monkeypatch.setattr(export_mod, "event", recording)
    return seen


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------------------
# 1. publish and recovery are serialized by a per-run lock
# ---------------------------------------------------------------------------------------


def test_R1_recovery_defers_to_an_active_publisher(run_dir, monkeypatch):
    before = _half_published(run_dir)
    seen = _record_events(monkeypatch)
    with _publish_lock(run_dir, blocking=True) as held:
        assert held is True
        during = _contents(run_dir)
        # a concurrent `status`/`export`/... must not treat the live publish as abandoned
        assert recover_interrupted_publish(run_dir) is False
        assert (run_dir / PUBLISH_JOURNAL_FILENAME).exists()
        assert _contents(run_dir) == during
        assert seen == []
    # once the publisher is gone the journal is processed as before
    assert recover_interrupted_publish(run_dir) is True
    assert seen == ["publish_recovered"]
    assert _contents(run_dir) == before
    assert _names(run_dir) == sorted(OUTPUTS)  # no journal, temp, backup or lock file left


def test_R1_non_blocking_acquire_reports_a_holder_and_the_lock_file_is_removed(run_dir):
    with _publish_lock(run_dir, blocking=True) as outer:
        assert outer is True
        assert (run_dir / PUBLISH_LOCK_FILENAME).exists()
        with _publish_lock(run_dir, blocking=False) as inner:
            assert inner is False
        assert (run_dir / PUBLISH_LOCK_FILENAME).exists()  # a refused acquire never deletes it
    assert not (run_dir / PUBLISH_LOCK_FILENAME).exists()
    with _publish_lock(run_dir, blocking=False) as again:
        assert again is True
    assert _names(run_dir) == []


def test_R1_publish_waits_for_the_lock_holder(run_dir):
    done = threading.Event()
    errors: list[BaseException] = []

    def publish() -> None:
        try:
            _publish(run_dir, "v1")
        except BaseException as exc:  # pragma: no cover - surfaced by the assertion below
            errors.append(exc)
        finally:
            done.set()

    with _publish_lock(run_dir, blocking=True):
        worker = threading.Thread(target=publish, daemon=True)
        worker.start()
        assert not done.wait(0.3)  # blocked: nothing staged, renamed or journaled meanwhile
        assert _contents(run_dir) == {}
    assert done.wait(5) and not errors
    worker.join(5)
    assert _names(run_dir) == sorted(OUTPUTS)
    assert b"v1" in (run_dir / "report.md").read_bytes()


def test_R1_recovery_runs_when_the_lock_is_merely_stale(run_dir):
    """A lock file left by a killed process (no holder) must not block recovery."""
    before = _half_published(run_dir)
    (run_dir / PUBLISH_LOCK_FILENAME).write_text("")
    assert recover_interrupted_publish(run_dir) is True
    assert _contents(run_dir) == before
    assert _names(run_dir) == sorted(OUTPUTS)


def test_R1_journal_records_pid_and_start_time(run_dir):
    _half_published(run_dir)
    data = json.loads((run_dir / PUBLISH_JOURNAL_FILENAME).read_text())
    assert data["version"] == 1
    assert data["pid"] == os.getpid()
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", data["started_at"])
    assert [e["target"] for e in data["targets"]] == list(OUTPUTS)


# ---------------------------------------------------------------------------------------
# 2. removing the journal is part of the commit
# ---------------------------------------------------------------------------------------


def _fail_journal_unlink_once(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """The first attempt to remove a publish journal raises; later attempts succeed."""
    real = os.unlink
    state = {"failed": 0}

    def flaky(path, *args, **kwargs):
        if Path(path).name == PUBLISH_JOURNAL_FILENAME and not state["failed"]:
            state["failed"] += 1
            raise OSError("input/output error")
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", flaky)
    return state


def test_R2_journal_unlink_failure_rolls_back_a_fresh_publish(run_dir, monkeypatch):
    state = _fail_journal_unlink_once(monkeypatch)
    with pytest.raises(OSError, match="input/output error"):
        _publish(run_dir, "v1")
    assert state["failed"] == 1
    assert _names(run_dir) == []  # the new files are gone with the journal, temps, backups
    _publish(run_dir, "v1")  # a following publish succeeds
    assert _names(run_dir) == sorted(OUTPUTS)
    assert recover_interrupted_publish(run_dir) is False


def test_R2_journal_unlink_failure_restores_the_previous_set(run_dir, monkeypatch):
    _publish(run_dir, "v1")
    before = _contents(run_dir)
    state = _fail_journal_unlink_once(monkeypatch)
    with pytest.raises(OSError, match="input/output error"):
        _publish(run_dir, "v2")
    assert state["failed"] == 1
    assert _contents(run_dir) == before
    assert _names(run_dir) == sorted(OUTPUTS)
    # nothing is left for recovery to misinterpret, and the next publish goes through
    assert recover_interrupted_publish(run_dir) is False
    _publish(run_dir, "v2")
    assert _names(run_dir) == sorted(OUTPUTS)
    assert all(b"v2" in (run_dir / n).read_bytes() for n in OUTPUTS)


def _complete_run(settings, acme_fixtures):
    steps = [
        tool_call("discover_pages", {"start_url": f"{SITE}/"}),
        tool_call(
            "scrape_pages", {"urls": [f"{SITE}/product", f"{SITE}/customers", f"{SITE}/about"]}
        ),
        tool_call("save_profile_draft", {"profile": DRAFT_PROFILE, "evidence": DRAFT_EVIDENCE}),
        *finalize_steps(),
    ]
    runner, _, _ = make_runner(settings, steps, scraper=FixtureScraper(acme_fixtures))
    outcome = runner.start(f"{SITE}/")
    assert outcome.status == "complete"
    return runner, outcome


def test_R2_export_with_journal_unlink_failure_leaves_store_and_files_unchanged(
    settings, acme_fixtures, monkeypatch
):
    runner, outcome = _complete_run(settings, acme_fixtures)
    run_dir = settings.runs_dir / outcome.run_id
    before = _contents(run_dir)
    store = RunStore(run_dir)
    changed = json.loads(json.dumps(store.latest_profile()))
    changed["product"]["positioning"] = ""
    store.save_draft(changed, "draft")
    drafts = store.draft_count()
    store.close()
    state = _fail_journal_unlink_once(monkeypatch)
    with pytest.raises(OSError, match="input/output error"):
        runner.export(outcome.run_id)
    assert state["failed"] == 1
    assert _contents(run_dir) == before
    assert not (run_dir / PUBLISH_JOURNAL_FILENAME).exists()
    assert not any(n.startswith(".tmp-") or n == PUBLISH_LOCK_FILENAME for n in _names(run_dir))
    store = RunStore(run_dir)
    assert store.get_run().status == "complete"
    assert store.draft_count() == drafts  # finalize's save_draft rolled back with the write
    store.close()
    # the next export publishes the changed draft as a unit
    out = runner.export(outcome.run_id)
    assert out.status == "complete"
    brain = json.loads((run_dir / "company_brain.json").read_text())
    assert brain["product"]["positioning"] == ""
    assert RunStore(run_dir).draft_count() == drafts + 1
    assert not (run_dir / PUBLISH_JOURNAL_FILENAME).exists()


# ---------------------------------------------------------------------------------------
# 3. an early staging/backup failure cleans up and raises the original error
# ---------------------------------------------------------------------------------------


def _stage_fails_for(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    real = export_mod._stage

    def flaky(target: Path, text: str, mode: int) -> Path:
        if target.name == name:
            raise OSError("disk full while staging")
        return real(target, text, mode)

    monkeypatch.setattr(export_mod, "_stage", flaky)


def test_R3_staging_failure_on_a_fresh_run_raises_the_original_error(run_dir, monkeypatch):
    _stage_fails_for(monkeypatch, "evidence.json")
    with pytest.raises(OSError, match="disk full while staging"):
        _publish(run_dir, "v1")
    assert _names(run_dir) == []


def test_R3_staging_failure_keeps_the_previous_outputs(run_dir, monkeypatch):
    _publish(run_dir, "v1")
    before = _contents(run_dir)
    _stage_fails_for(monkeypatch, "evidence.json")
    with pytest.raises(OSError, match="disk full while staging"):
        _publish(run_dir, "v2")
    assert _contents(run_dir) == before
    assert _names(run_dir) == sorted(OUTPUTS)  # no .tmp-*, .tmp-bak-*, journal or lock file
    assert recover_interrupted_publish(run_dir) is False


def test_R3_backup_failure_keeps_the_previous_outputs(run_dir, monkeypatch):
    _publish(run_dir, "v1")
    before = _contents(run_dir)
    real = shutil.copy2

    def flaky(src, dst, *args, **kwargs):
        if Path(src).name == "evidence.json":
            raise OSError("disk full while backing up")
        return real(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", flaky)
    with pytest.raises(OSError, match="disk full while backing up"):
        _publish(run_dir, "v2")
    assert _contents(run_dir) == before
    assert _names(run_dir) == sorted(OUTPUTS)
