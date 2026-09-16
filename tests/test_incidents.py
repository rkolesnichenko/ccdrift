"""Incidents: a flag followed from its first deviant day until the metric recovers."""

import random
from datetime import date

import pandas as pd
import pytest

from ccdrift.detector import DetectorConfig
from ccdrift.incidents import (Event, approx, describe, exclusions, incident_cost, update_incidents,
                               versions_text)
from ccdrift.logs import judged_turns
from ccdrift.state import new_state
from tests.helpers import HAIKU, MOSTLY_CLEAN_CACHE, QUIET, daily_turns, nth_day, prompt_turn_days


def run(state, days, today):
    """One check run on `today` over daily turn fixtures: (kind, incident start) per event."""
    df = daily_turns(days)
    if "main_thread" not in df:
        df["main_thread"] = True
    events = update_incidents(judged_turns(df, today), state, today, DetectorConfig())
    return [(event.kind, event.incident["start"]) for event in events]


def test_a_new_flag_opens_an_incident_from_its_first_deviant_day():
    state = new_state()
    assert run(state, [QUIET] * 14 + [HAIKU] * 3, date(2026, 9, 18)) == [("flag", "2026-09-15")]
    assert [(i["metric"], i["status"], i["end"]) for i in state["incidents"]] == [("haiku_fraction", "open", None)]


def test_an_open_incident_is_not_reported_again():
    state = new_state()
    run(state, [QUIET] * 14 + [HAIKU] * 3, date(2026, 9, 18))
    assert run(state, [QUIET] * 14 + [HAIKU] * 4, date(2026, 9, 19)) == []


def test_a_day_in_progress_waits_until_it_is_over():
    # A day still in progress holds only part of its turns.
    assert run(new_state(), [QUIET] * 14 + [HAIKU] * 3, date(2026, 9, 17)) == []


def test_flags_from_weeks_ago_open_nothing():
    # Installed on Sep 15, the check would otherwise open with the August cache
    # regression, four weeks old by then.
    assert run(new_state(), [QUIET] * 14 + [HAIKU] * 3, date(2026, 10, 10)) == []


def test_subagent_haiku_opens_nothing():
    # Subagent Haiku comes in bursts: across all turns, 3 of 5 synthetic logs
    # were falsely flagged.
    quiet = {"is_haiku": [0.0] * 440, "main_thread": [True] * 400 + [False] * 40}
    burst = {"is_haiku": [0.0] * 400 + [1.0] * 40, "main_thread": [True] * 400 + [False] * 40}
    assert run(new_state(), [quiet] * 14 + [burst] * 3, date(2026, 9, 18)) == []


def test_effort_opens_nothing():
    # Effort swings more from day to day than a 70% cut in thinking moves it,
    # so an effort flag in one user's logs says more about the work than Claude.
    days = [{"thinking_fraction": [0.5] * 400}] * 14 + [{"thinking_fraction": [0.1] * 400}] * 3
    assert run(new_state(), days, date(2026, 9, 18)) == []


def test_a_long_shift_stays_open_while_it_lasts():
    # Against a rolling baseline the shift would look normal after two weeks.
    state = new_state()
    run(state, [QUIET] * 14 + [HAIKU] * 3, date(2026, 9, 18))
    assert run(state, [QUIET] * 14 + [HAIKU] * 25, date(2026, 10, 10)) == []
    assert state["incidents"][0]["status"] == "open"


def test_an_incident_recovers_once_three_pooled_windows_in_a_row_are_normal():
    # On the real caching regression, single days closed it at Sep 4 "from Sep 2"
    # although Sep 3 still missed; pooling 3 days closed it at Sep 6, "from Sep 4".
    state = new_state()
    days = [QUIET] * 14 + [HAIKU] * 6 + [QUIET] * 6
    assert run(state, days, date(2026, 9, 27)) == [("flag", "2026-09-15"), ("recovered", "2026-09-15")]
    incident = state["incidents"][0]
    assert (incident["status"], incident["end"], incident["recovered_from"], incident["closed_by"]) == \
        ("recovered", "2026-09-22", "2026-09-23", "check")


def test_a_cache_incident_is_flagged_and_recovers_like_a_haiku_incident():
    # The cache ratio's harmful direction is "down", the opposite of Haiku share's
    # "up": this exercises that branch of recovery, and the cache metric's own
    # z-threshold (3.0, not the 3.5 default). 14 mostly-clean baseline days (an
    # occasional cold turn, as real logs run), then 6 days at a 20% miss rate,
    # then 6 clean days again.
    state = new_state()
    days = prompt_turn_days(MOSTLY_CLEAN_CACHE + [8] * 6 + [0] * 6, random.Random(0))
    assert run(state, days, date(2026, 9, 27)) == [("flag", "2026-09-15"), ("recovered", "2026-09-15")]
    incident = state["incidents"][0]
    assert (incident["status"], incident["end"], incident["recovered_from"], incident["closed_by"]) == \
        ("recovered", "2026-09-22", "2026-09-23", "check")


def test_an_incident_is_followed_from_run_to_run():
    state = new_state()
    days = [QUIET] * 14 + [HAIKU] * 6 + [QUIET] * 6
    seen = []
    for d in range(17, 27):
        today = date.fromisoformat(nth_day(d))
        seen += [(today.isoformat(), kind) for kind, _ in run(state, days, today)]
    assert seen == [("2026-09-18", "flag"), ("2026-09-26", "recovered")]


def test_an_incident_open_for_30_days_becomes_the_new_normal():
    state = new_state()
    run(state, [QUIET] * 14 + [HAIKU] * 3, date(2026, 9, 18))
    assert run(state, [QUIET] * 14 + [HAIKU] * 30, date(2026, 10, 15)) == [("persistent", "2026-09-15")]
    assert (state["incidents"][0]["status"], state["incidents"][0]["end"]) == ("persistent", "2026-10-14")
    assert run(state, [QUIET] * 14 + [HAIKU] * 31, date(2026, 10, 16)) == []


def test_a_persistent_incident_ends_on_the_last_complete_day_even_without_turns():
    # Data stops at Oct 14 but today is Oct 16, so the last complete day (Oct 15)
    # has no turns of its own; the incident still ends there, not on the last day
    # that happens to have data.
    state = new_state()
    run(state, [QUIET] * 14 + [HAIKU] * 3, date(2026, 9, 18))
    assert run(state, [QUIET] * 14 + [HAIKU] * 30, date(2026, 10, 16)) == [("persistent", "2026-09-15")]
    assert state["incidents"][0]["end"] == "2026-10-15"


def test_a_dismissed_incident_opens_nothing_again():
    state = new_state()
    state["incidents"].append({"metric": "haiku_fraction", "start": "2026-09-15", "end": "2026-09-17",
                               "status": "dismissed", "source": "check", "closed_by": "user"})
    assert run(state, [QUIET] * 14 + [HAIKU] * 3, date(2026, 9, 18)) == []


def test_a_flag_a_version_1_state_file_reported_opens_nothing():
    state = new_state()
    state["reported"] = {"haiku_fraction": ["2026-09-15"]}
    assert run(state, [QUIET] * 14 + [HAIKU] * 3, date(2026, 9, 18)) == []


def test_only_open_and_recovered_incidents_leave_the_baseline():
    incidents = [
        {"metric": "cache_ratio", "start": nth_day(1), "end": nth_day(2), "status": "recovered"},
        {"metric": "cache_ratio", "start": nth_day(4), "end": None, "status": "open"},
        {"metric": "haiku_fraction", "start": nth_day(0), "end": nth_day(5), "status": "dismissed"},
    ]
    assert exclusions([nth_day(i) for i in range(6)], incidents) == {
        "cache_ratio": [False, True, True, False, True, True], "haiku_fraction": [False] * 6}


def prompt_turns(day, turns, misses=0, writes_per_miss=0, haiku=0, version="2.1.226"):
    """Judged new-prompt turns of one day: `misses` of them missed the cache and
    wrote `writes_per_miss` tokens each; `haiku` of them came from Haiku."""
    return pd.DataFrame({
        "day": day, "main_thread": True, "prompt_within_ttl": True, "version": version,
        "is_miss": [True] * misses + [False] * (turns - misses),
        "cache_creation": [float(writes_per_miss)] * misses + [10.0] * (turns - misses),
        "is_haiku": [1.0] * haiku + [0.0] * (turns - haiku)})


def test_a_cache_incident_costs_the_writes_of_misses_above_the_usual_miss_rate():
    turns = pd.concat([prompt_turns(nth_day(0), 100, 1, 1000), prompt_turns(nth_day(1), 100, 1, 1000),
                       prompt_turns(nth_day(2), 100, 5, 2000), prompt_turns(nth_day(3), 50)], ignore_index=True)
    incident = {"metric": "cache_ratio", "start": nth_day(2), "end": None, "status": "open"}
    # The usual 1% accounts for 1 of the 5 misses: 4/5 of their 10,000 tokens.
    assert incident_cost(turns, incident, [incident], DetectorConfig()) == pytest.approx(8000)


def test_a_haiku_incident_costs_the_haiku_responses_above_the_usual_share():
    turns = pd.concat([prompt_turns(nth_day(0), 100, haiku=1), prompt_turns(nth_day(1), 100, haiku=1),
                       prompt_turns(nth_day(2), 200, haiku=12)], ignore_index=True)
    incident = {"metric": "haiku_fraction", "start": nth_day(2), "end": None, "status": "open"}
    assert incident_cost(turns, incident, [incident], DetectorConfig()) == pytest.approx(10)


def test_versions_are_those_behind_a_fifth_of_the_responses_with_the_day_each_was_first_seen():
    turns = pd.DataFrame({
        "day": [nth_day(0)] * 10 + [nth_day(1)] * 10 + [nth_day(2)] * 10,
        "version": ["2.1.226"] * 17 + ["2.1.233"] * 11 + ["2.1.234", "2.1.226"]})
    assert versions_text(turns, [nth_day(2)]) == ["2.1.233 (since 09-02)"]
    assert versions_text(turns, [nth_day(1), nth_day(2)]) == ["2.1.233 (since 09-02)", "2.1.226 (since 09-01)"]


@pytest.mark.parametrize("value, text", [(17_556_103, "18M"), (2_753_531, "2.8M"), (552_373, "550k"),
                                         (36, "36"), (0, "0")])
def test_costs_read_to_two_significant_figures(value, text):
    assert approx(value) == text


def cache_incident_turns():
    return pd.concat([prompt_turns(nth_day(0), 100, 1, 1000), prompt_turns(nth_day(1), 100, 1, 1000),
                      prompt_turns(nth_day(2), 100, 5, 2000, version="2.1.233"),
                      prompt_turns(nth_day(3), 100, version="2.1.259")], ignore_index=True)


def test_a_flag_alert_names_the_versions_and_the_cost_so_far():
    incident = {"metric": "cache_ratio", "start": nth_day(2), "end": None, "status": "open",
                "versions": [], "cost": 0}
    event = Event("flag", incident, days=[nth_day(2)], run=[nth_day(2)], z=[-3.84])
    assert describe(event, cache_incident_turns(), [incident], DetectorConfig()) == (
        "flag", "ccdrift flag",
        "Cache read ratio on new prompts down from 2026-09-03, on Claude Code 2.1.233 (since 09-03). "
        "~8k tokens re-cached so far.",
        ["days 2026-09-03; z = -3.8"])
    assert (incident["versions"], incident["cost"]) == (["2.1.233 (since 09-03)"], 8000)


def test_a_recovery_alert_names_the_versions_and_what_the_incident_cost():
    incident = {"metric": "cache_ratio", "start": nth_day(2), "end": nth_day(2), "status": "recovered",
                "recovered_from": nth_day(3), "versions": [], "cost": 0}
    event = Event("recovered", incident, days=[nth_day(3)])
    assert describe(event, cache_incident_turns(), [incident], DetectorConfig())[1:3] == (
        "ccdrift: back to normal",
        "Cache read ratio on new prompts back to normal from 2026-09-04, on Claude Code 2.1.259 (since 09-04). "
        "The incident from 2026-09-03: ~8k tokens re-cached.")


def test_a_persistent_alert_says_the_change_is_now_the_new_normal():
    incident = {"metric": "haiku_fraction", "start": "2026-09-15", "end": "2026-10-14", "status": "persistent",
                "versions": [], "cost": 0}
    turns = pd.concat([prompt_turns(nth_day(i), 100) for i in range(3)], ignore_index=True)
    assert describe(Event("persistent", incident, days=[]), turns, [incident], DetectorConfig())[1:3] == (
        "ccdrift: change persists",
        "Haiku share on the main thread still up 30 days after 2026-09-15. ccdrift now treats it as the "
        "new normal; `ccdrift incident list` has the details.")


def test_agent_sdk_haiku_opens_nothing():
    # Agent SDK sessions are the user's own scripts.
    quiet = {"is_haiku": [0.0] * 440, "entrypoint": ["cli"] * 400 + ["sdk-py"] * 40}
    burst = {"is_haiku": [0.0] * 400 + [1.0] * 40, "entrypoint": ["cli"] * 400 + ["sdk-py"] * 40}
    assert run(new_state(), [quiet] * 14 + [burst] * 3, date(2026, 9, 18)) == []
