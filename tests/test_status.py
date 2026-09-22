"""ccdrift status: how the last check went and what it follows, from the state file alone."""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from ccdrift.cli import main
from ccdrift.state import new_state
from ccdrift.status import run_status, short_status, status_report

EEST = timezone(timedelta(hours=3))
NOW = datetime(2026, 9, 20, 12, 0, tzinfo=EEST)


def incident(metric="cache_ratio", start="2026-09-14", status="open", **fields):
    return {"metric": metric, "start": start, "end": None, "status": status, "source": "check",
            "closed_by": None, "recovered_from": None, "opened_on": start, "closed_on": None,
            "versions": ["2.1.273 (since 09-15)"], "cost": 550_000, **fields}


def state_file(tmp_path, **state):
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"version": 2, **state}))
    return path


def ran(ok=True, when="2026-09-20T09:00:02+03:00", error=None):
    return {"last_run": {"started": when, "ok": ok, "error": error}}


def loop_warning(stream, at="2026-09-20T08:40:00+00:00", misses=7, sessions=2):
    return {"stream": stream, "at": at, "since": "2026-09-20T08:00:00+00:00", "misses": misses, "turns": 180,
            "sessions": sessions, "base_rate": 0.0018, "tokens": 3_200_000, "versions": [],
            "reported_on": "2026-09-20"}


@pytest.mark.parametrize("state, expected", [
    ({}, "ccdrift: no check yet"),
    ({**ran(ok=False, error="RuntimeError: boom"), "last_ok": "2026-09-19T09:00:01+03:00"},
     "ccdrift: check failed 09-20 09:00"),
    ({**ran(when="2026-09-16T09:00:00+03:00"), "last_ok": "2026-09-16T09:00:00+03:00"},
     "ccdrift: no check for 4 days"),
    ({**ran(), "last_ok": "2026-09-20T09:00:02+03:00",
      "incidents": [incident(), incident("haiku_fraction", "2026-09-18")]},
     "ccdrift: cache ratio down since 09-14; Haiku share up since 09-18"),
    ({**ran(), "last_ok": "2026-09-20T09:00:02+03:00", "incidents": [incident(status="recovered")]}, ""),
])
def test_short_status_says_the_most_pressing_thing_or_nothing(tmp_path, state, expected):
    assert short_status(state_file(tmp_path, **state), NOW) == expected


def test_short_status_never_fails_a_status_line(tmp_path, capsys):
    (tmp_path / "state.json").write_text("not json")
    assert run_status(tmp_path / "state.json", short=True, now=NOW) == 0
    assert capsys.readouterr().out == "ccdrift: can't read state\n"


@pytest.mark.parametrize("state", [
    ran(),  # ok, but no top-level "last_ok" at all
    {**ran(), "last_ok": "2026-09-20T09:00:02"},  # last_ok has no UTC offset
    {**ran(), "last_ok": "2026-09-20T09:00:02+03:00",
     "incidents": [{"metric": "cache_ratio", "start": "2026-09-14"}]},  # incident missing "status"
    {**ran(), "last_ok": "2026-09-20T09:00:02+03:00", "version": 3},  # written by a newer ccdrift
])
def test_short_status_treats_a_malformed_state_as_unreadable(tmp_path, state):
    assert short_status(state_file(tmp_path, **state), NOW) == "ccdrift: can't read state"


def test_short_status_treats_an_early_warning_time_out_of_range_as_unreadable(tmp_path, capsys):
    # Shown in UTC-5, this `since` falls before year 1, which raised OverflowError.
    warning = {"at": "2026-09-20T08:40:00+00:00", "since": "0001-01-01T00:00:00+05:00", "misses": 3, "turns": 7,
               "base_rate": 0.005, "versions": [], "reported_on": "2026-09-20"}
    path = state_file(tmp_path, **ran(), last_ok="2026-09-20T09:00:02+03:00", early_warnings=[warning])
    now = datetime(2026, 9, 20, 4, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert run_status(path, short=True, now=now) == 0
    assert capsys.readouterr().out == "ccdrift: can't read state\n"


def test_run_status_short_never_fails_on_a_malformed_state(tmp_path, capsys):
    path = state_file(tmp_path, **ran())  # ok, but no top-level "last_ok"
    assert run_status(path, short=True, now=NOW) == 0
    assert capsys.readouterr().out == "ccdrift: can't read state\n"


def test_short_status_prints_nothing_when_all_is_well(tmp_path, capsys):
    path = state_file(tmp_path, **ran(), last_ok="2026-09-20T09:00:02+03:00")
    assert run_status(path, short=True, now=NOW) == 0
    assert capsys.readouterr().out == ""


def test_status_shows_the_last_check_incidents_and_setting_changes(tmp_path, capsys):
    path = state_file(tmp_path, **ran(), last_ok="2026-09-20T09:00:02+03:00", incidents=[
        incident(),
        incident(start="2026-08-18", status="recovered", end="2026-09-03", closed_by="check",
                 recovered_from="2026-09-04", closed_on="2026-09-07", versions=[], cost=17_556_103),
        incident(start="2026-07-01", status="recovered", end="2026-07-05", closed_by="check",
                 recovered_from="2026-07-06", closed_on="2026-07-09"),
    ], settings=[{"setting": "cache_tier", "model": "claude-opus-5", "from": "1h", "to": "5m",
                  "since": "2026-09-17", "days": ["2026-09-17", "2026-09-18"], "reported_on": "2026-09-19"}])
    assert run_status(path, now=NOW) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Last check: 2026-09-20 09:00, ok",
        "Open incidents:",
        "  cache  2026-09-14..now           open; ~550k tokens re-cached; on 2.1.273 (since 09-15)",
        "Closed in the last 30 days:",
        "  cache  2026-08-18..2026-09-03    back to normal from 2026-09-04; ~18M tokens re-cached",
        "Setting changes in the last 30 days:",
        "  cache tier for claude-opus-5: 1h -> 5m from 2026-09-17",
        "Other changes in the last 30 days: none",
    ]


def test_status_lists_other_changes_of_the_last_30_days(tmp_path, capsys):
    path = state_file(
        tmp_path, **ran(), last_ok="2026-09-20T09:00:02+03:00",
        context_changes=[{"since": "2026-09-10", "from": 128_000.0, "to": 54_000.0,
                          "days": ["2026-09-10", "2026-09-12"], "reported_on": "2026-09-13"}],
        hook_failures=[{"since": "2026-09-16", "days": ["2026-09-16", "2026-09-17"], "runs": [15, 14],
                        "failed": [12, 14], "reported_on": "2026-09-18"},
                       {"since": "2026-08-01", "days": ["2026-08-01", "2026-08-02"], "runs": [10, 10],
                        "failed": [10, 10], "reported_on": "2026-08-03"}],
        field_gaps=[{"field": "effort", "version": "2.1.280", "share_before": 1.0, "share": 0.0,
                     "responses": 312, "reported_on": "2026-09-19"},
                    {"field": "version", "version": "unknown", "share_before": 1.0, "share": 0.0,
                     "responses": 60, "reported_on": "2026-09-19"}],
        early_warnings=[{"at": "2026-09-20T08:40:00+00:00", "since": "2026-09-20T08:00:00+00:00", "misses": 3,
                         "turns": 7, "base_rate": 0.005, "versions": [], "reported_on": "2026-09-20"}],
        loop_warnings=[loop_warning("subagent", misses=1, sessions=1)])
    run_status(path, now=NOW)
    lines = capsys.readouterr().out.splitlines()
    assert lines[lines.index("Other changes in the last 30 days:"):] == [
        "Other changes in the last 30 days:",
        "  session start ~130k -> ~54k tokens from 2026-09-10",
        "  stop hooks failing from 2026-09-16: 12 of 15 runs on 2026-09-16, 14 of 14 on 2026-09-17",
        "  effort not logged on 2.1.280: 0% of 312 responses, 100% before",
        "  version not logged: 0% of 60 responses, 100% before",
        "  cache misses rising at 2026-09-20 08:40 UTC: 3 of 7 new-prompt turns (usually 0.5%)",
        "  subagent cache misses rising at 2026-09-20 08:40 UTC: 1 of 180 turns in 1 session (usually 0.18%), "
        "~3.2M tokens rewritten",
    ]


def test_status_lists_a_new_field_under_other_changes():
    state = new_state()
    state["last_run"] = {"started": "2026-09-19T10:00:00", "ok": True, "error": None}
    state["new_fields"].append({"paths": ["advisorModel"], "version": "2.1.276", "share": 0.94,
                                "responses": 1602, "reported_on": "2026-09-19"})
    assert "1 new field on 2.1.276: advisorModel" in status_report(state, datetime(2026, 9, 20, 10, 0))


def test_status_before_the_first_check_says_how_to_set_it_up(tmp_path, capsys):
    assert run_status(tmp_path / "state.json", now=NOW) == 0
    assert capsys.readouterr().out == "The daily check hasn't run yet. `ccdrift schedule install` sets it up.\n"


def test_status_shows_a_failing_check_and_when_it_last_worked(tmp_path, capsys):
    path = state_file(tmp_path, **ran(ok=False, error="RuntimeError: boom"), last_ok="2026-09-19T09:00:01+03:00")
    run_status(path, now=NOW)
    assert capsys.readouterr().out.splitlines()[:2] == [
        "Last check: 2026-09-20 09:00, failed: RuntimeError: boom",
        "Last successful check: 2026-09-19 09:00",
    ]


def test_status_command_reads_the_state_option(tmp_path, capsys):
    path = state_file(tmp_path)
    assert main(["status", "--short", "--state", str(path)]) == 0
    assert capsys.readouterr().out == "ccdrift: no check yet\n"


@pytest.mark.parametrize("reported_on, expected", [
    ("2026-09-17", "ccdrift: hooks failing since 09-16"),
    ("2026-09-16", ""),
])
def test_short_status_shows_failing_hooks_for_three_days(tmp_path, reported_on, expected):
    failure = {"since": "2026-09-16", "days": ["2026-09-16", "2026-09-17"], "runs": [10, 10], "failed": [9, 10],
               "reported_on": reported_on}
    path = state_file(tmp_path, **ran(), last_ok="2026-09-20T09:00:02+03:00", hook_failures=[failure])
    assert short_status(path, NOW) == expected


@pytest.mark.parametrize("hours_later, expected", [(0, "ccdrift: cache misses rising since 11:00"), (24, "")])
def test_short_status_shows_rising_cache_misses_for_a_day(tmp_path, hours_later, expected):
    warning = {"at": "2026-09-20T08:40:00+00:00", "since": "2026-09-20T08:00:00+00:00", "misses": 3, "turns": 7,
               "base_rate": 0.005, "versions": [], "reported_on": "2026-09-20"}
    path = state_file(tmp_path, **ran(), last_ok="2026-09-20T09:00:02+03:00", early_warnings=[warning])
    assert short_status(path, NOW + timedelta(hours=hours_later)) == expected


@pytest.mark.parametrize("warnings, hours_later, expected", [
    ([loop_warning("subagent"), loop_warning("main")], 0, "ccdrift: tool-loop cache misses rising since 11:00"),
    ([loop_warning("subagent")], 0, "ccdrift: subagent cache misses rising since 11:00"),
    ([loop_warning("subagent")], 24, ""),
])
def test_short_status_shows_rising_tool_loop_misses_for_a_day_main_thread_first(tmp_path, warnings, hours_later,
                                                                                expected):
    path = state_file(tmp_path, **ran(), last_ok="2026-09-20T09:00:02+03:00", loop_warnings=warnings)
    assert short_status(path, NOW + timedelta(hours=hours_later)) == expected


def test_status_short_loads_neither_pandas_nor_numpy(tmp_path):
    # It runs on every status line refresh; importing pandas took ~0.3 s of it.
    root = Path(__file__).resolve().parents[1]
    code = ("import sys\n"
            "from ccdrift.cli import main\n"
            f"main(['status', '--short', '--state', {str(tmp_path / 'state.json')!r}])\n"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('pandas', 'numpy')))\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(root / "src")})
    assert result.stdout.splitlines() == ["ccdrift: no check yet", "[]"], result.stderr
