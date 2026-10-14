"""`ccdrift calibrate`: the plants, the measurement on a history and what it prints."""

from datetime import date

import pytest

from ccdrift.calibrate import calibration, plant, plant_starts, plant_subagent, subagent_openings
from ccdrift.logs import parse_all
from tests.helpers import main_thread_days, nth_day, tool_loop_days


def main_turns(tmp_path, days=2):
    main_thread_days(tmp_path, [{}] * days)
    return parse_all(tmp_path).responses


def test_a_cache_plant_moves_its_share_of_each_new_prompts_cache_read_into_writes_from_its_start_on(tmp_path):
    turns = main_turns(tmp_path)
    planted = plant(turns, "cache", 0.2, start=60)
    first, second = planted.iloc[:60], planted.iloc[60:]
    assert (first["cache_read"] == 900).all() and (first["cache_creation"] == 100).all()
    judged = second[second["prompt_within_ttl"]]
    assert len(judged) == 59
    assert (judged["cache_read"] == 720).all() and (judged["cache_creation"] == 280).all()
    assert judged["prompt_cache_read_ratio"].round(3).eq(0.72).all()
    opener = second[~second["prompt_within_ttl"]]
    assert (opener["cache_read"] == 900).all()


def test_a_haiku_plant_moves_its_share_of_the_turns_from_its_start_on_to_haiku_the_same_way_for_a_seed(tmp_path):
    turns = main_turns(tmp_path)
    planted = plant(turns, "haiku", 0.25, seed=3, start=60)
    again = plant(turns, "haiku", 0.25, seed=3, start=60)
    assert planted["is_haiku"].iloc[:60].sum() == 0
    assert planted["is_haiku"].iloc[60:].sum() == 15
    assert planted["model"].tolist() == again["model"].tolist()


def test_a_plant_of_a_kind_it_doesnt_know_is_refused(tmp_path):
    with pytest.raises(ValueError, match="^effort$"):
        plant(main_turns(tmp_path), "effort", 0.2)


def test_a_subagent_plant_makes_its_share_of_a_days_loop_turns_read_only_what_the_transcript_opened_with(tmp_path):
    tool_loop_days(tmp_path, 2, per_day=100, subagent=True)
    responses = parse_all(tmp_path).responses
    loops = responses[responses["loop_turn"].astype(bool)]
    opening = subagent_openings(responses)
    assert opening.tolist() == [0.0, 0.0]
    planted = plant_subagent(loops, opening, [nth_day(1)], 0.1, seed=0)
    missed = planted[planted["is_loop_miss"]]
    assert set(missed["day"].astype(str)) == {nth_day(1)}
    assert len(missed) == round(0.1 * int((loops["day"].astype(str) == nth_day(1)).sum()))
    assert (missed["cache_read"] == 0).all()
    assert (planted["cache_read"] + planted["cache_creation"]).tolist() == (
        loops["cache_read"] + loops["cache_creation"]).tolist()


@pytest.mark.parametrize("count, starts", [(10, []), (11, [7]), (13, [7, 8, 9])], ids=["10-days", "11-days", "13-days"])
def test_a_start_leaves_seven_judged_days_before_it_and_three_after(count, starts):
    days = [nth_day(i) for i in range(count)]
    assert plant_starts(days) == [days[i] for i in starts]


def test_starts_are_ten_at_most_spread_evenly_from_the_first_that_fits_to_the_last():
    days = [nth_day(i) for i in range(40)]
    chosen = plant_starts(days)
    assert len(set(chosen)) == 10 and chosen == sorted(chosen)
    assert (chosen[0], chosen[-1]) == (days[7], days[36])


def calibrated(tmp_path, days, incidents=(), loops=0):
    """The calibration of a history with one main-thread session a day from Sep 1, 150
    responses each as `days` sets them, and when `loops` is set a subagent of it with that
    many tool-loop turns a day, judged the day after the last."""
    main_thread_days(tmp_path, days, per_day=150)
    if loops:
        tool_loop_days(tmp_path, len(days), per_day=loops, subagent=True)
    result = calibration(parse_all(tmp_path).responses, list(incidents), date.fromisoformat(nth_day(len(days))))
    return result, {metric["metric"]: metric for metric in result["metrics"]}


STEP = [{"misses": 30 if i >= 12 else 0} for i in range(20)]


def test_a_steady_history_passes_every_metric_each_start_catching_a_planted_five_percent(tmp_path):
    result, metrics = calibrated(tmp_path, [{}] * 12, loops=301)
    assert (result["first"], result["last"], result["main_days"], result["subagent_days"]) == (
        nth_day(0), nth_day(11), 12, 12)
    for name, runs in (("cache_ratio", 2), ("haiku_fraction", 2), ("subagent_cache", 10)):
        metric = metrics[name]
        assert (metric["verdict"], metric["failed"], metric["flagged"], metric["left_out"]) == ("pass", [], [], [])
        assert (metric["days"], metric["starts"], metric["runs"], metric["caught"][0.05]) == (
            12, [nth_day(7), nth_day(8)], runs, runs)


def test_a_history_too_short_for_any_start_is_not_enough_data_and_still_reports_what_flags(tmp_path):
    _, metrics = calibrated(tmp_path, [{"misses": 30 if i >= 6 else 0} for i in range(10)])
    cache = metrics["cache_ratio"]
    assert (cache["verdict"], cache["failed"], cache["flagged"], cache["runs"]) == (
        "not_enough_data", ["quiet"], [nth_day(6)], 0)
    assert (metrics["haiku_fraction"]["verdict"], metrics["haiku_fraction"]["failed"]) == ("not_enough_data", [])


def test_days_that_swing_more_than_a_planted_five_percent_fail_reach_and_give_the_smallest_change_caught(tmp_path):
    _, metrics = calibrated(tmp_path, [{"misses": 0 if i % 2 else 4} for i in range(12)])
    cache = metrics["cache_ratio"]
    assert (cache["verdict"], cache["failed"], cache["flagged"]) == ("fail", ["reach"], [])
    assert cache["caught"][0.05] == 0
    assert cache["floors"] == {nth_day(7): 0.15, nth_day(8): 0.15}


def test_a_step_ccdrift_never_recorded_fails_quiet_and_names_the_day_it_began(tmp_path):
    _, metrics = calibrated(tmp_path, STEP)
    cache = metrics["cache_ratio"]
    assert (cache["verdict"], cache["flagged"], cache["left_out"]) == ("fail", [nth_day(12)], [])
    # Plants from the step on add no flag of their own, so half the starts catch nothing.
    assert (cache["failed"], cache["caught"][0.05], cache["runs"]) == (["quiet", "reach"], 5, 10)


def test_subagent_read_back_whose_days_swing_more_than_a_planted_five_percent_fails_reach(tmp_path):
    main_thread_days(tmp_path, [{}] * 12, per_day=150)
    tool_loop_days(tmp_path, 12, per_day=301, subagent=True, misses=10, miss_days=range(0, 12, 2))
    subagent = calibration(parse_all(tmp_path).responses, [], date.fromisoformat(nth_day(12)))["metrics"][2]
    assert (subagent["verdict"], subagent["failed"], subagent["flagged"]) == ("fail", ["reach"], [])
    assert (subagent["caught"], subagent["runs"]) == ({0.02: 0, 0.05: 0}, 10)
    assert subagent["floors"] == {nth_day(7): None, nth_day(8): None}


def test_subagent_flags_the_history_raises_unplanted_catch_no_plant(tmp_path):
    main_thread_days(tmp_path, [{}] * 14, per_day=150)
    tool_loop_days(tmp_path, 14, per_day=301, subagent=True, misses=20, miss_days=range(9, 14))
    subagent = calibration(parse_all(tmp_path).responses, [], date.fromisoformat(nth_day(14)))["metrics"][2]
    assert (subagent["flagged"], subagent["failed"]) == ([nth_day(9)], ["quiet", "reach"])
    # The two starts on flagged days add no flag of their own.
    assert (subagent["starts"], subagent["caught"][0.05], subagent["runs"]) == (
        [nth_day(7), nth_day(8), nth_day(9), nth_day(10)], 10, 20)


@pytest.mark.parametrize("caught, verdict", [(9, "pass"), (8, "fail")], ids=["9-of-10", "8-of-10"])
def test_subagent_read_back_needs_a_planted_five_percent_caught_in_nine_runs_of_ten(tmp_path, monkeypatch, caught,
                                                                                     verdict):
    monkeypatch.setattr("ccdrift.calibrate.reach_subagent", lambda *args: {
        "starts": [nth_day(7), nth_day(8)], "runs": 10, "caught": {0.02: 0, 0.05: caught}, "floors": {}})
    tool_loop_days(tmp_path, 1, per_day=10, subagent=True)
    subagent = calibration(parse_all(tmp_path).responses, [], date.fromisoformat(nth_day(1)))["metrics"][2]
    assert subagent["verdict"] == verdict


@pytest.mark.parametrize("status, end", [("open", None), ("recovered", nth_day(19)), ("persistent", nth_day(19))],
                         ids=["open", "recovered", "persistent"])
def test_a_step_ccdrift_follows_as_an_incident_is_left_out_and_the_days_before_it_pass(tmp_path, status, end):
    incident = {"metric": "cache_ratio", "start": nth_day(12), "end": end, "status": status}
    _, metrics = calibrated(tmp_path, STEP, [incident])
    cache = metrics["cache_ratio"]
    assert (cache["verdict"], cache["flagged"], cache["left_out"], cache["days"]) == (
        "pass", [], [(nth_day(12), end)], 12)


def test_a_dismissed_incidents_days_stay_in_and_its_flag_counts_as_a_false_alarm(tmp_path):
    incident = {"metric": "cache_ratio", "start": nth_day(12), "end": nth_day(19), "status": "dismissed"}
    _, metrics = calibrated(tmp_path, STEP, [incident])
    cache = metrics["cache_ratio"]
    assert (cache["flagged"], cache["left_out"], cache["days"]) == ([nth_day(12)], [], 20)
    assert "quiet" in cache["failed"]


def test_an_incident_of_another_metric_leaves_this_ones_days_alone(tmp_path):
    incident = {"metric": "haiku_fraction", "start": nth_day(12), "end": None, "status": "open"}
    _, metrics = calibrated(tmp_path, STEP, [incident])
    assert (metrics["cache_ratio"]["flagged"], metrics["cache_ratio"]["days"]) == ([nth_day(12)], 20)
    assert (metrics["haiku_fraction"]["left_out"], metrics["haiku_fraction"]["days"]) == ([(nth_day(12), None)], 12)


def test_subagent_days_with_too_few_loop_turns_are_neither_judged_nor_starts(tmp_path):
    result, metrics = calibrated(tmp_path, [{}] * 12, loops=100)
    subagent = metrics["subagent_cache"]
    assert (subagent["verdict"], subagent["days"], subagent["starts"], subagent["runs"]) == (
        "not_enough_data", 0, [], 0)
    assert result["subagent_days"] == 12
