"""ccdrift incident: listing incidents and changing them by hand."""

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from ccdrift.cli import main
from ccdrift.incidents import add_incident, close_incident, dismiss_incident, incident_line
from ccdrift.state import load_state
from tests.helpers import main_thread_days

TODAY = date(2026, 9, 16)


def opened(start="2026-09-10"):
    return {"metric": "cache_ratio", "start": start, "end": None, "status": "open", "source": "check",
            "closed_by": None, "recovered_from": None, "opened_on": start, "closed_on": None,
            "versions": ["2.1.273 (since 09-09)"], "cost": 550_000}


def test_a_past_incident_added_by_hand_stays_out_of_the_baseline():
    incidents = []
    incident = add_incident(incidents, "cache_ratio", "2026-08-16", "2026-09-04", TODAY)
    assert (incident["status"], incident["source"], incident["end"]) == ("recovered", "user", "2026-09-04")
    assert incidents == [incident]


@pytest.mark.parametrize("start, end, reason", [
    ("2026-09-04", "2026-08-16", "is before"),
    ("2026-09-10", "2026-09-16", "isn't over yet"),
    ("2026-09-01", "2026-09-12", "overlaps the cache incident from 2026-09-10"),
])
def test_an_incident_that_cant_be_added_says_why(start, end, reason):
    with pytest.raises(ValueError, match=reason):
        add_incident([opened()], "cache_ratio", start, end, TODAY)


def test_closing_by_hand_ends_the_open_incident_yesterday():
    incident = close_incident([opened()], "cache_ratio", TODAY)
    assert (incident["status"], incident["closed_by"], incident["end"]) == ("recovered", "user", "2026-09-15")
    with pytest.raises(ValueError, match="no cache incident is open"):
        close_incident([incident], "cache_ratio", TODAY)


def test_dismissing_an_open_incident_gives_it_an_end():
    # Left open-ended, a dismissed incident would overlap every later flag.
    incident = dismiss_incident([opened()], "cache_ratio", "2026-09-10", TODAY)
    assert (incident["status"], incident["end"], incident["closed_on"]) == ("dismissed", "2026-09-15", "2026-09-16")
    with pytest.raises(ValueError, match="no haiku incident starts on 2026-09-10"):
        dismiss_incident([incident], "haiku_fraction", "2026-09-10", TODAY)


def test_an_incident_reads_as_one_line():
    assert incident_line(opened()) == \
        "cache  2026-09-10..now           open; ~550k tokens re-cached; on 2.1.273 (since 09-09)"
    recovered = {**opened(), "end": "2026-09-12", "status": "recovered", "closed_by": "check",
                 "recovered_from": "2026-09-13", "versions": []}
    assert incident_line(recovered, cost=0) == \
        "cache  2026-09-10..2026-09-12    back to normal from 2026-09-13; no tokens re-cached"
    # A check-opened incident closed by hand: closed_by wins over the "check" source.
    closed_by_hand = {**opened(), "end": "2026-09-12", "status": "recovered", "closed_by": "user",
                      "closed_on": "2026-09-16", "versions": []}
    assert incident_line(closed_by_hand, cost=0) == \
        "cache  2026-09-10..2026-09-12    closed by hand on 2026-09-16; no tokens re-cached"
    persistent = {**opened(), "status": "persistent"}
    assert incident_line(persistent) == \
        "cache  2026-09-10..now           still changed after 30 days; ~550k tokens re-cached; " \
        "on 2.1.273 (since 09-09)"


def test_incident_add_through_cli_saves_the_state(tmp_path, capsys):
    state = tmp_path / "state.json"
    assert main(["incident", "add", "cache", "2026-08-16..2026-09-04", "--state", str(state)]) == 0
    assert load_state(state)["incidents"][0]["start"] == "2026-08-16"
    assert capsys.readouterr().out == "cache  2026-08-16..2026-09-04    added by hand; no tokens re-cached\n"


def test_incident_commands_change_nothing_on_bad_input(tmp_path, capsys):
    state = tmp_path / "state.json"
    assert main(["incident", "add", "cache", "2026-08-16", "--state", str(state)]) == 2
    assert main(["incident", "close", "haiku", "--state", str(state)]) == 2
    assert main(["incident", "dismiss", "cache", "someday", "--state", str(state)]) == 2
    assert not state.exists()
    assert capsys.readouterr().err.count("Nothing changed:") == 3


def test_incident_close_through_cli_uses_the_last_complete_utc_day(tmp_path, capsys):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"version": 2, "incidents": [opened("2026-01-01")]}))
    assert main(["incident", "close", "cache", "--state", str(state)]) == 0
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    today = datetime.now(timezone.utc).date().isoformat()
    assert load_state(state)["incidents"][0]["end"] == yesterday
    assert capsys.readouterr().out == (
        f"cache  2026-01-01..{yesterday}    closed by hand on {today}; ~550k tokens re-cached; "
        "on 2.1.273 (since 09-09)\n")


def test_incident_dismiss_through_cli_saves_the_state(tmp_path, capsys):
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"version": 2, "incidents": [opened()]}))
    assert main(["incident", "dismiss", "cache", "2026-09-10", "--state", str(state)]) == 0
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    saved = load_state(state)["incidents"][0]
    assert saved["status"] == "dismissed"
    assert saved["end"] == max("2026-09-10", yesterday)
    assert capsys.readouterr().out == (
        f"cache  2026-09-10..{saved['end']}    dismissed; ~550k tokens re-cached; on 2.1.273 (since 09-09)\n")


def test_incident_list_shows_each_incident_with_its_cost_newest_first(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"version": 2, "incidents": [
        {**opened("2026-08-01"), "end": "2026-08-05", "status": "dismissed", "closed_by": "user", "versions": []},
        opened("2026-09-02")]}))
    assert main(["incident", "list", "--source", str(tmp_path / "logs"), "--state", str(state)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Incidents, newest first:",
        "  cache  2026-09-02..now           open; no tokens re-cached; on 2.1.273 (since 09-09)",
        "  cache  2026-08-01..2026-08-05    dismissed; no tokens re-cached",
    ]


def test_incident_list_says_so_when_there_are_none(tmp_path, capsys):
    assert main(["incident", "list", "--source", str(tmp_path), "--state", str(tmp_path / "state.json")]) == 0
    assert capsys.readouterr().out == "No incidents recorded.\n"
