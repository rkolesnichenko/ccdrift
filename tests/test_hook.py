"""The Claude Code plugin's hook: when a session start begins a check, and the check it begins."""

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

import ccdrift.check
import ccdrift.hook
from ccdrift.check import run_check
from ccdrift.cli import main
from ccdrift.hook import CHECK_INTERVAL, check_due, run_hook_check
from ccdrift.logs import default_source
from ccdrift.state import ccdrift_home, load_state, new_state, save_state

NOW = datetime(2026, 10, 6, 18, 0, tzinfo=timezone.utc)


def ran_before(tmp_path, ago, now=NOW):
    path = tmp_path / "state.json"
    save_state(path, {**new_state(), "last_run": {"started": (now - ago).isoformat(), "ok": True, "error": None}})
    return path


def test_the_interval_is_the_hourly_schedules():
    assert CHECK_INTERVAL == timedelta(minutes=60)


@pytest.mark.parametrize("ago, due", [(timedelta(minutes=59), False), (timedelta(minutes=60), True),
                                      (timedelta(minutes=61), True), (timedelta(days=3), True)],
                         ids=["59min", "60min", "61min", "3days"])
def test_a_check_is_due_once_the_last_began_an_hour_ago_or_more(tmp_path, ago, due):
    assert check_due(ran_before(tmp_path, ago), NOW) is due


def test_a_check_is_due_when_none_has_run(tmp_path):
    assert check_due(tmp_path / "state.json", NOW) is True
    save_state(tmp_path / "state.json", new_state())
    assert check_due(tmp_path / "state.json", NOW) is True


@pytest.mark.parametrize("text", ["{", "[]", json.dumps({"version": 99}),
                                  json.dumps({"version": 3, "last_run": {"started": "noon"}}),
                                  json.dumps({"version": 3, "last_run": {"started": "2026-10-06T17:30:00"}})],
                         ids=["not-json", "not-an-object", "newer-ccdrift", "bad-time", "time-without-zone"])
def test_a_check_is_due_when_the_state_cant_be_read_so_the_check_reports_it(tmp_path, text):
    # The last is a time without its zone, which can't be set against an aware one.
    (tmp_path / "state.json").write_text(text)
    assert check_due(tmp_path / "state.json", NOW) is True


def runs(monkeypatch):
    calls = []
    monkeypatch.setattr(ccdrift.check, "_run_on_state", lambda *args: calls.append(args) or ([], None))
    return calls


def within(seconds, call):
    out = []
    worker = threading.Thread(target=lambda: out.append(call()), daemon=True)
    worker.start()
    worker.join(seconds)
    return out


def test_the_plugins_check_gives_up_at_once_saying_nothing_while_another_command_holds_the_state(
        tmp_path, monkeypatch, capsys):
    fcntl = pytest.importorskip("fcntl")
    calls = runs(monkeypatch)
    path = tmp_path / "state.json"
    held = open(tmp_path / "state.json.lock", "a")
    fcntl.flock(held, fcntl.LOCK_EX)
    try:
        out = within(5, lambda: run_check(tmp_path, path, due=lambda: True))
    finally:
        held.close()
    assert out == [0]
    assert calls == [] and not path.exists()
    assert capsys.readouterr().out == ""


def test_the_plugins_check_asks_whether_it_is_still_due_only_once_it_holds_the_state_and_skips_if_not(
        tmp_path, monkeypatch, capsys):
    fcntl = pytest.importorskip("fcntl")
    calls = runs(monkeypatch)
    path = tmp_path / "state.json"
    asked = []

    def due():
        probe = open(tmp_path / "state.json.lock", "a")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            probe.close()
        asked.append(True)
        return False

    assert run_check(tmp_path, path, due=due) == 0
    assert asked == [True] and calls == []
    assert capsys.readouterr().out == ""


def test_the_plugins_check_runs_when_still_due_once_it_holds_the_state(tmp_path, monkeypatch, capsys):
    calls = runs(monkeypatch)
    assert run_check(tmp_path, tmp_path / "state.json", due=lambda: True) == 0
    assert len(calls) == 1
    assert "no alerts" in capsys.readouterr().out


def test_hook_check_runs_the_check_with_notifications_and_due_while_the_last_began_an_hour_ago(
        tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(ccdrift.check, "run_check",
                        lambda source, state_path, **options: seen.update(source=source, state=state_path,
                                                                          **options) or 0)
    now = datetime.now().astimezone()
    path = ran_before(tmp_path, timedelta(minutes=59), now)
    assert run_hook_check(path) == 0
    assert (seen["source"], seen["state"], seen["notify_user"]) == (default_source(), path, True)
    assert set(seen) == {"source", "state", "notify_user", "due"}
    assert seen["due"]() is False
    ran_before(tmp_path, timedelta(minutes=61), now)
    assert seen["due"]() is True


def test_ccdrift_hook_check_checks_the_state_in_ccdrift_home(monkeypatch):
    seen = []
    monkeypatch.setattr(ccdrift.hook, "run_hook_check", lambda path: seen.append(path) or 0)
    assert main(["hook", "check"]) == 0
    assert seen == [ccdrift_home() / "check-state.json"]
