"""The Claude Code plugin's hook: what a session start shows, when it begins a check, and the
check it begins."""

import io
import json
import os
import stat
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import ccdrift.check
import ccdrift.hook
from ccdrift.check import run_check
from ccdrift.cli import main
from ccdrift.hook import CHECK_INTERVAL, check_due, run_hook_check, run_session_start, start_check
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


# What Claude Code sends a SessionStart hook; the path, folder and id must go nowhere.
PAYLOAD = {"session_id": "SESSION-ID-1", "hook_event_name": "SessionStart", "source": "startup",
           "transcript_path": "/Users/someone/.claude/projects/-Users-someone-secret-project/abc.jsonl",
           "cwd": "/Users/someone/secret-project", "model": "claude-opus-5-5", "permission_mode": "default"}
LINE = "ccdrift: cache ratio down since 09-14"


def stdin_of(payload):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return io.TextIOWrapper(io.BytesIO(data))


def following(tmp_path, ago=timedelta(minutes=5)):
    """A state whose last check began `ago` before NOW and follows an open incident."""
    path = ran_before(tmp_path, ago)
    state = load_state(path)
    state["last_ok"] = state["last_run"]["started"]
    state["incidents"] = [{"metric": "cache_ratio", "start": "2026-09-14", "end": None, "status": "open",
                           "source": "check", "closed_by": None, "recovered_from": None, "opened_on": "2026-09-14",
                           "closed_on": None, "versions": [], "cost": 0}]
    save_state(path, state)
    return path


def starts(monkeypatch):
    started = []
    monkeypatch.setattr(ccdrift.hook, "start_check", lambda path: started.append(path))
    return started


@pytest.mark.parametrize("source", ["startup", "resume"])
def test_a_startup_or_a_resume_shows_the_status_lines_verdict_as_the_only_output(tmp_path, monkeypatch, capsys,
                                                                                source):
    starts(monkeypatch)
    assert run_session_start(following(tmp_path), stdin_of({**PAYLOAD, "source": source}), NOW) == 0
    assert capsys.readouterr().out == json.dumps({"systemMessage": LINE}) + "\n"


@pytest.mark.parametrize("source", ["clear", "compact", "fork"])
def test_a_session_start_in_the_middle_of_work_shows_nothing(tmp_path, monkeypatch, capsys, source):
    starts(monkeypatch)
    assert run_session_start(following(tmp_path), stdin_of({**PAYLOAD, "source": source}), NOW) == 0
    assert capsys.readouterr().out == ""


def test_nothing_is_shown_when_nothing_needs_attention(tmp_path, monkeypatch, capsys):
    starts(monkeypatch)
    path = ran_before(tmp_path, timedelta(minutes=5))
    state = load_state(path)
    save_state(path, {**state, "last_ok": state["last_run"]["started"]})
    assert run_session_start(path, stdin_of(PAYLOAD), NOW) == 0
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("data", [b"", b"{", b"\xff\xfe", b"[1, 2]", b'"startup"', b'{"source": 7}', b"{}",
                                  b"x" * (70 * 1024)],
                         ids=["empty", "cut-off", "not-utf8", "a-list", "a-string", "source-not-text", "no-source",
                              "over-64k"])
def test_a_payload_that_doesnt_say_where_the_session_came_from_counts_as_a_startup(tmp_path, monkeypatch, capsys,
                                                                                    data):
    starts(monkeypatch)
    assert run_session_start(following(tmp_path), stdin_of(data), NOW) == 0
    assert capsys.readouterr().out == json.dumps({"systemMessage": LINE}) + "\n"


@pytest.mark.parametrize("source", ["startup", "resume", "clear", "compact", "fork"])
@pytest.mark.parametrize("ago, begun", [(timedelta(minutes=59), False), (timedelta(minutes=61), True)],
                         ids=["59min", "61min"])
def test_every_session_start_begins_a_check_when_one_is_due_and_only_then(tmp_path, monkeypatch, source, ago, begun):
    started = starts(monkeypatch)
    path = following(tmp_path, ago)
    run_session_start(path, stdin_of({**PAYLOAD, "source": source}), NOW)
    assert started == ([path] if begun else [])


def test_a_check_starts_detached_with_this_python_its_output_appended_to_a_private_log(tmp_path, monkeypatch):
    seen = []

    class Popen:
        def __init__(self, argv, **options):
            seen.append((argv, options, options["stdout"].name, options["stdout"].mode))

    monkeypatch.setattr(subprocess, "Popen", Popen)
    log = tmp_path / "check.log"
    log.write_text("earlier run\n")
    log.chmod(0o644)
    start_check(tmp_path / "state.json")
    [(argv, options, name, mode)] = seen
    assert argv == [sys.executable, "-m", "ccdrift", "hook", "check"]
    assert options["start_new_session"] is True and options["stdin"] == subprocess.DEVNULL
    assert options["stderr"] is options["stdout"]
    assert (name, mode) == (str(log), "a")
    assert log.read_text() == "earlier run\n"
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


def test_a_check_starts_in_ccdrifts_home_so_the_projects_own_modules_cant_stand_in_for_ccdrift(tmp_path, monkeypatch):
    # Claude Code runs the hook in the project's folder, and `python -m` searches the current
    # folder first: a repository holding a ccdrift/ or a json.py of its own would run as the check.
    seen = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **options: seen.append(options))
    project = tmp_path / "project"
    (project / "ccdrift").mkdir(parents=True)
    (project / "ccdrift" / "__main__.py").write_text("raise SystemExit('the project ran')\n")
    monkeypatch.chdir(project)
    start_check(tmp_path / "home" / "check-state.json")
    [options] = seen
    assert options["cwd"] == tmp_path / "home"


def test_a_hook_that_fails_says_so_in_one_line_logs_the_traceback_privately_and_still_exits_0(
        tmp_path, monkeypatch, capsys):
    def broken(path):
        raise OSError("no room left")

    monkeypatch.setattr(ccdrift.hook, "start_check", broken)
    path = following(tmp_path, timedelta(hours=2))
    assert run_session_start(path, stdin_of(PAYLOAD), NOW) == 0
    out = capsys.readouterr().out
    assert out.count("\n") == 1
    assert json.loads(out) == {"systemMessage": "ccdrift: the session-start hook failed (OSError); "
                                                "the traceback is in check.log"}
    log = tmp_path / "check.log"
    assert "OSError: no room left" in log.read_text()
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    for private in ("SESSION-ID-1", "secret-project", "abc.jsonl"):
        assert private not in out and private not in log.read_text()


def test_hook_session_start_loads_neither_pandas_nor_numpy(tmp_path):
    # Claude's first response waits for it, as a status line refresh waits for `status --short`.
    root = Path(__file__).resolve().parents[1]
    home = tmp_path / "home"
    home.mkdir()
    following(tmp_path, timedelta(minutes=5)).rename(home / "check-state.json")
    code = ("import sys\n"
            "from ccdrift.cli import main\n"
            "main(['hook', 'session-start'])\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('pandas', 'numpy')))\n")
    # The state's last check is minutes old, so no check starts; a transcript folder of its own
    # keeps one that did from reading the real ones.
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "CCDRIFT_HOME": str(home),
           "CLAUDE_CONFIG_DIR": str(tmp_path / "claude")}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            input=json.dumps(PAYLOAD), env=env)
    [shown, modules] = result.stdout.splitlines()
    assert json.loads(shown)["systemMessage"].startswith("ccdrift: ") and modules == "[]", result.stderr
    assert not (home / "check.log").exists()


def test_ccdrift_hook_session_start_reads_standard_input_and_the_state_in_ccdrift_home(monkeypatch):
    seen = []
    monkeypatch.setattr(ccdrift.hook, "run_session_start", lambda path, stdin: seen.append((path, stdin)) or 0)
    assert main(["hook", "session-start"]) == 0
    assert seen == [(ccdrift_home() / "check-state.json", sys.stdin)]
