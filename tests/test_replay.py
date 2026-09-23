"""Replaying incident detection over past days: what a first check records, and what
`ccdrift replay` shows without recording anything."""

from datetime import date

import pytest

from ccdrift.cli import main
from ccdrift.detector import DetectorConfig
from ccdrift.incidents import add_incident, dismiss_incident
from ccdrift.logs import parse_source
from ccdrift.replay import first_run, replay_incidents, run_replay
from ccdrift.state import new_state, save_state
from ccdrift.texts import history_message, replayed_line
from tests.helpers import main_thread_days

# Haiku on 12 of 60 responses a day from Sep 15 to 17 on 2.1.233, then none on 2.1.259.
REGRESSION = [{}] * 14 + [{"haiku": 12, "version": "2.1.233"}] * 3 + [{"version": "2.1.259"}] * 5
TODAY = date(2026, 10, 10)


@pytest.mark.parametrize("state, expected", [
    ({}, True),
    ({"last_run": {"started": "2026-09-01T09:00:00+00:00", "ok": False, "error": "boom"}}, True),
    ({"last_ok": "2026-09-01T09:00:00+00:00"}, False),
    ({"incidents": [{"metric": "cache_ratio"}]}, False),
    ({"reported": {"cache_ratio": ["2026-08-18"]}}, False),
])
def test_a_first_run_is_a_state_no_check_has_run_on_that_holds_nothing(state, expected):
    assert first_run({**new_state(), **state}) is expected


def replayed(tmp_path, days, today):
    main_thread_days(tmp_path / "logs", days)
    state = new_state()
    events = replay_incidents(parse_source(tmp_path / "logs"), today, DetectorConfig(), state)
    return state, [(day, event.kind) for day, event in events]


def test_the_replay_follows_a_regression_from_weeks_ago_on_the_days_the_check_would_have(tmp_path):
    # On Oct 10 the flag from Sep 15 is too old for a first check to open; replayed day by
    # day, the Sep 18 check flags it and the Sep 23 check closes it.
    state, events = replayed(tmp_path, REGRESSION, TODAY)
    assert events == [("2026-09-18", "flag"), ("2026-09-23", "recovered")]
    assert state["incidents"] == [{
        "metric": "haiku_fraction", "start": "2026-09-15", "end": "2026-09-19", "status": "recovered",
        "source": "replay", "closed_by": "check", "recovered_from": "2026-09-20", "opened_on": "2026-09-18",
        "closed_on": "2026-09-23", "versions": ["2.1.233 (since 09-15)"], "cost": 36}]


def test_a_replayed_regression_still_going_stays_open(tmp_path):
    state, events = replayed(tmp_path, [{}] * 14 + [{"haiku": 12}] * 10, date(2026, 10, 1))
    assert events == [("2026-09-18", "flag")]
    found = state["incidents"][0]
    assert (found["status"], found["end"], found["cost"]) == ("open", None, 120)


def test_a_replayed_regression_that_lasts_30_days_closes_as_persistent_on_its_30th_day(tmp_path):
    state, events = replayed(tmp_path, [{}] * 14 + [{"haiku": 12}] * 40, date(2026, 11, 1))
    assert events == [("2026-09-18", "flag"), ("2026-10-15", "persistent")]
    found = state["incidents"][0]
    assert (found["status"], found["end"], found["closed_on"]) == ("persistent", "2026-10-14", "2026-10-15")


def test_a_replay_of_too_little_history_finds_nothing(tmp_path):
    assert replayed(tmp_path, [{"haiku": 12}] * 3, date(2026, 9, 10)) == (new_state(), [])


def cache_incident(status, **fields):
    return {"metric": "cache_ratio", "start": "2026-08-18", "end": "2026-09-03", "status": status,
            "source": "replay", "recovered_from": "2026-09-04", "versions": ["2.1.235 (since 08-19)"],
            "cost": 15_950_524, **fields}


def test_the_summary_names_each_incident_with_how_it_ended_what_it_cost_and_its_versions():
    haiku = {"metric": "haiku_fraction", "start": "2026-09-10", "end": None, "status": "open", "source": "replay",
             "recovered_from": None, "versions": [], "cost": 120}
    assert history_message([cache_incident("recovered"), haiku], "2026-08-06") == (
        "Replaying your history from 2026-08-06 found 2 incidents ccdrift would have followed: cache ratio down "
        "2026-08-18..2026-09-03, back to normal from 2026-09-04, ~16M tokens re-cached, on Claude Code 2.1.235 "
        "(since 08-19); Haiku share up since 2026-09-10, still going, ~120 extra Haiku responses so far. "
        "`ccdrift incident list` has the details; `ccdrift incident dismiss` puts a false alarm's days back in "
        "the baseline.")
    assert replayed_line(cache_incident("persistent", end="2026-09-16", recovered_from=None)) == (
        "cache ratio down from 2026-08-18, still changed after 30 days, ~16M tokens re-cached, on Claude Code "
        "2.1.235 (since 08-19)")


def test_replay_prints_the_alerts_the_check_would_have_sent_and_the_incidents_it_found(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", REGRESSION)
    assert run_replay(tmp_path / "logs", tmp_path / "state.json", today=TODAY) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Replaying the check day by day from 2026-09-02 to 2026-10-10 (UTC) on an empty state, incidents only.",
        "Nothing is recorded and no alert is sent.",
        "",
        "2026-09-18  ccdrift flag: Haiku share on the main thread up from 2026-09-15, on Claude Code 2.1.233 "
        "(since 09-15). ~36 extra Haiku responses so far.",
        "2026-09-23  ccdrift: back to normal: Haiku share on the main thread back to normal from 2026-09-20, on "
        "Claude Code 2.1.259 (since 09-18). The incident from 2026-09-15: ~36 extra Haiku responses.",
        "",
        "Incidents the replay found:",
        "  haiku  2026-09-15..2026-09-19    back to normal from 2026-09-20; ~36 extra Haiku responses; on 2.1.233 "
        "(since 09-15)",
        "    not recorded: `ccdrift incident add haiku 2026-09-15..2026-09-19` records it",
    ]


@pytest.mark.parametrize("dismissed, expected", [
    (False, "    recorded: haiku 2026-09-14..2026-09-17"),
    (True, "    dismissed: haiku 2026-09-14..2026-09-17"),
])
def test_replay_says_whether_each_incident_it_found_is_recorded(tmp_path, capsys, dismissed, expected):
    main_thread_days(tmp_path / "logs", REGRESSION)
    state = new_state()
    add_incident(state["incidents"], "haiku_fraction", "2026-09-14", "2026-09-17", TODAY)
    if dismissed:
        dismiss_incident(state["incidents"], "haiku_fraction", "2026-09-14", TODAY)
    save_state(tmp_path / "state.json", state)
    run_replay(tmp_path / "logs", tmp_path / "state.json", today=TODAY)
    assert capsys.readouterr().out.splitlines()[-1] == expected


@pytest.mark.parametrize("days, today, expected", [
    ([{}] * 14 + [{"haiku": 12}] * 10, date(2026, 10, 1),
     "    not recorded: `ccdrift incident add haiku 2026-09-15..2026-09-30` records its days so far"),
    ([{}] * 14 + [{"haiku": 12}] * 40, date(2026, 11, 1),
     "    not recorded: after 30 days the check takes the new level as normal, so its days need no record"),
])
def test_replay_says_how_to_record_an_open_incident_and_that_a_persistent_one_needs_no_record(
        tmp_path, capsys, days, today, expected):
    main_thread_days(tmp_path / "logs", days)
    run_replay(tmp_path / "logs", tmp_path / "state.json", today=today)
    assert capsys.readouterr().out.splitlines()[-1] == expected


def test_replay_of_a_clean_history_says_so(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 20)
    run_replay(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 10, 1))
    assert capsys.readouterr().out.splitlines()[-1] == (
        "No incidents: the check would have sent no incident alert over these days.")


def test_replay_before_a_complete_day_says_there_is_nothing_to_replay(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{}])
    assert run_replay(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 1)) == 0
    assert capsys.readouterr().out == "Nothing to replay: no complete UTC day with main-thread activity yet.\n"


def test_replay_writes_no_state_and_no_history_store(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", REGRESSION)
    run_replay(tmp_path / "logs", tmp_path / "state.json", today=TODAY)
    assert [path.name for path in tmp_path.iterdir()] == ["logs"]


def test_replay_exits_2_without_transcripts_and_1_on_an_unreadable_state(tmp_path, capsys):
    (tmp_path / "logs").mkdir()
    assert run_replay(tmp_path / "logs", tmp_path / "state.json", today=TODAY) == 2
    (tmp_path / "state.json").write_text("not json")
    assert run_replay(tmp_path / "logs", tmp_path / "state.json", today=TODAY) == 1
    assert "Can't read the state file" in capsys.readouterr().err


def test_the_replay_command_reads_the_source_and_state_options(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    assert main(["replay", "--source", str(tmp_path / "logs"), "--state", str(tmp_path / "state.json")]) == 0
    assert capsys.readouterr().out.startswith("Replaying the check day by day from 2026-09-02 to ")
