"""The daily check's state file: what it reported, its incidents and its last run."""

import json
import os
import stat
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ccdrift.cli import main
from ccdrift.state import load_state, new_state, record_run, save_state, state_lock


def test_a_missing_state_file_reads_as_a_fresh_state(tmp_path):
    assert load_state(tmp_path / "state.json") == {
        "version": 2, "incidents": [], "settings": [], "blank_cache": [], "reported": {},
        "field_gaps": [], "new_fields": [], "new_attachments": [], "hook_failures": [], "context_changes": [], "early_warnings": [],
        "loop_warnings": [], "failed_requests": [], "cut_short": [], "hook_changes": [], "unreadable": [],
        "no_responses": [], "thinking_rises": [], "withheld": None, "context_rule": 2,
        "runs": []}


def test_a_version_1_state_file_keeps_what_it_reported(tmp_path):
    (tmp_path / "state.json").write_text(json.dumps(
        {"reported": {"cache_ratio": ["2026-08-18"]}, "blank_cache": ["2026-09-01"]}))
    assert load_state(tmp_path / "state.json") == {
        "version": 2, "incidents": [], "settings": [], "blank_cache": ["2026-09-01"],
        "reported": {"cache_ratio": ["2026-08-18"]},
        "field_gaps": [], "new_fields": [], "new_attachments": [], "hook_failures": [], "context_changes": [], "early_warnings": [],
        "loop_warnings": [], "failed_requests": [], "cut_short": [], "hook_changes": [], "unreadable": [],
        "no_responses": [], "thinking_rises": [], "withheld": None, "context_rule": 1,
        "runs": []}


def test_a_state_file_that_holds_no_object_is_unreadable(tmp_path):
    (tmp_path / "state.json").write_text("[]")
    with pytest.raises(ValueError):
        load_state(tmp_path / "state.json")


@pytest.mark.parametrize("version", [3, "2", 2.0, None, True])
def test_a_state_file_from_a_newer_ccdrift_or_with_a_bad_version_is_unreadable(tmp_path, version):
    # Loaded as version 2, it would be overwritten by this ccdrift's next check.
    (tmp_path / "state.json").write_text(json.dumps({"version": version, "incidents": []}))
    with pytest.raises(ValueError, match="version"):
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
    assert [(src.startswith("state.json.") and src.endswith(".tmp"), dst) for src, dst in replaced] == \
        [(True, "state.json")]
    assert load_state(path)["blank_cache"] == ["2026-09-01"]
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_only_the_owner_can_read_the_state_even_one_saved_over_a_file_others_could(tmp_path):
    # The state names versions, incidents and hook streams, which name projects and tools.
    # It is private because the temporary file it is written through is created that way;
    # nothing tested it, found in the audit of 2026-09-25.
    path = tmp_path / "state.json"
    path.write_text("{}")
    path.chmod(0o644)
    umask = os.umask(0o022)
    try:
        save_state(path, new_state())
    finally:
        os.umask(umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_two_saves_at_once_dont_share_a_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    real_replace = os.replace
    saves = []

    def replace(src, dst):
        if not saves:
            saves.append(src)
            save_state(path, {**new_state(), "blank_cache": ["2026-09-02"]})
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    save_state(path, {**new_state(), "blank_cache": ["2026-09-01"]})
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


def test_successful_runs_keep_their_local_dates_the_newest_14():
    state = new_state()
    started = datetime(2026, 9, 1, 23, 30, tzinfo=timezone(timedelta(hours=3)))
    for day in range(16):
        record_run(state, started + timedelta(days=day), None)
        record_run(state, started + timedelta(days=day, minutes=10), None)
    record_run(state, started + timedelta(days=16), "RuntimeError: no transcripts")
    assert state["runs"] == [f"2026-09-{day:02d}" for day in range(3, 17)]


def test_the_state_reaches_the_disk_before_it_replaces_the_last_one(tmp_path, monkeypatch):
    # A rename can reach the disk before the data it points at: after a power cut the
    # state would read back empty, and load_state refuses it on every run from then on.
    calls = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        calls.append("fsync" if os.fstat(fd).st_size > 0 else "fsync of nothing written yet")
        real_fsync(fd)

    def replace(src, dst):
        calls.append("replace")
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    save_state(tmp_path / "state.json", new_state())
    assert calls[-2:] == ["fsync", "replace"]
    assert load_state(tmp_path / "state.json") == new_state()



def test_a_second_command_waits_for_the_state_lock_saying_so_and_goes_on_once_it_is_released(tmp_path, capfd):
    fcntl = pytest.importorskip("fcntl")
    path = tmp_path / "state.json"
    held = open(tmp_path / "state.json.lock", "a")
    fcntl.flock(held, fcntl.LOCK_EX)
    entered = threading.Event()

    def second():
        with state_lock(path):
            entered.set()

    waiting = threading.Thread(target=second)
    waiting.start()
    assert not entered.wait(0.3)
    held.close()  # releases the lock
    waiting.join(5)
    assert entered.is_set()
    assert f"Waiting for another ccdrift command to finish with {path}..." in capfd.readouterr().err


def test_an_incident_added_while_a_check_holds_the_state_is_kept_along_with_the_checks_own_change(tmp_path, capsys):
    # The check reads the state, works for a while and saves it; `ccdrift incident add` run
    # meanwhile must wait for it, or one of them saves over the other's change.
    path = tmp_path / "state.json"
    save_state(path, new_state())
    loaded, adding = threading.Event(), threading.Event()

    def check():
        with state_lock(path):
            state = load_state(path)
            loaded.set()
            adding.wait(5)
            time.sleep(0.3)
            state["blank_cache"] = ["2026-09-01"]
            save_state(path, state)

    checking = threading.Thread(target=check)
    checking.start()
    loaded.wait(5)
    codes = []

    def add():
        adding.set()
        codes.append(main(["incident", "add", "cache", "2026-08-16..2026-09-04", "--state", str(path)]))

    incident = threading.Thread(target=add)
    incident.start()
    checking.join(5)
    incident.join(5)
    state = load_state(path)
    assert (codes, state["blank_cache"], [i["start"] for i in state["incidents"]]) == (
        [0], ["2026-09-01"], ["2026-08-16"])


def test_a_save_that_fails_leaves_the_last_state_and_no_temporary_file(tmp_path):
    path = tmp_path / "state.json"
    save_state(path, {**new_state(), "blank_cache": ["2026-09-01"]})
    with pytest.raises(TypeError):
        save_state(path, {**new_state(), "blank_cache": {"not", "json"}})
    assert load_state(path)["blank_cache"] == ["2026-09-01"]
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_a_state_from_before_the_withheld_total_was_kept_reads_as_nothing_withheld(tmp_path):
    path = tmp_path / "state.json"
    old = new_state()
    del old["withheld"]
    path.write_text(json.dumps(old))
    assert load_state(path)["withheld"] is None
