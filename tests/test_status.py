"""ccdrift status: how the last check went and what it follows, from the state file alone."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from ccdrift.cli import main
from ccdrift.status import run_status, short_status

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
])
def test_short_status_treats_a_malformed_state_as_unreadable(tmp_path, state):
    assert short_status(state_file(tmp_path, **state), NOW) == "ccdrift: can't read state"


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
    ]


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
