"""The daily check's state file: what it reported, its incidents and its last run."""

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ccdrift.state import load_state, new_state, record_run, save_state


def test_a_missing_state_file_reads_as_a_fresh_state(tmp_path):
    assert load_state(tmp_path / "state.json") == {
        "version": 2, "incidents": [], "settings": [], "blank_cache": [], "reported": {}}


def test_a_version_1_state_file_keeps_what_it_reported(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps(
        {"reported": {"cache_ratio": ["2026-08-18"]}, "blank_cache": ["2026-09-01"]}))
    assert load_state(tmp_path / "state.json") == {
        "version": 2, "incidents": [], "settings": [], "blank_cache": ["2026-09-01"],
        "reported": {"cache_ratio": ["2026-08-18"]}}


def test_a_state_file_that_holds_no_object_is_unreadable(tmp_path):
    (tmp_path / "state.json").write_text("[]")
    with pytest.raises(ValueError):
        load_state(tmp_path / "state.json")


def test_saving_replaces_the_state_file_in_one_step(tmp_path, monkeypatch):
    # Status lines read the file often and must never see half of it.
    path = tmp_path / "state.json"
    save_state(path, new_state())
    replaced = []
    real_replace = os.replace

    def replace(src, dst):
        replaced.append((Path(src).name, Path(dst).name))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    save_state(path, {**new_state(), "blank_cache": ["2026-09-01"]})
    assert replaced == [("state.json.tmp", "state.json")]
    assert load_state(path)["blank_cache"] == ["2026-09-01"]
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_each_run_is_recorded_with_its_start_and_outcome():
    state = new_state()
    started = datetime(2026, 9, 16, 9, 0, 2, tzinfo=timezone(timedelta(hours=3)))
    record_run(state, started, None)
    record_run(state, started + timedelta(days=1), "RuntimeError: no transcripts")
    assert state["last_run"] == {"started": "2026-09-17T09:00:02+03:00", "ok": False,
                                 "error": "RuntimeError: no transcripts"}
    assert state["last_ok"] == "2026-09-16T09:00:02+03:00"
