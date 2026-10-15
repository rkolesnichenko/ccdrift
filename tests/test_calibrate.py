"""`ccdrift calibrate`: the plants, the measurement on a history and what it prints."""

import json
from datetime import date

import pytest

import ccdrift.calibrate
from ccdrift import early, loops
from ccdrift.calibrate import (cache_miss_warning, calibrate_json, calibrate_lines, calibration, metric_lines, plant,
                               plant_starts, plant_subagent, run_calibrate, subagent_openings, tool_loop_warning,
                               warning_lines)
from ccdrift.cli import main
from ccdrift.early import MIN_P0
from ccdrift.history import History
from ccdrift.logs import parse_all
from ccdrift.loops import LOOP_SETTINGS, LoopSetting
from ccdrift.state import STATE_VERSION
from ccdrift.texts import floors_text
from ccdrift.warning_gates import alarms_per_turn
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


@pytest.mark.parametrize("status, end", [("open", None), ("recovered", nth_day(19))], ids=["open", "recovered"])
def test_a_step_ccdrift_follows_as_an_incident_is_left_out_and_the_days_before_it_pass(tmp_path, status, end):
    incident = {"metric": "cache_ratio", "start": nth_day(12), "end": end, "status": status}
    _, metrics = calibrated(tmp_path, STEP, [incident])
    cache = metrics["cache_ratio"]
    assert (cache["verdict"], cache["flagged"], cache["left_out"], cache["days"]) == (
        "pass", [], [(nth_day(12), end)], 20)


def test_a_persistent_incidents_days_stay_in_as_the_new_normal_the_check_takes_them_for(tmp_path):
    # The check keeps them in the baseline once the level persists, so the days after one are
    # judged against the new level, and the step itself is the incident, not a false alarm.
    incident = {"metric": "cache_ratio", "start": nth_day(10), "end": nth_day(29), "status": "persistent"}
    _, metrics = calibrated(tmp_path, [{"misses": 30 if i >= 10 else 0} for i in range(40)], [incident])
    cache = metrics["cache_ratio"]
    assert (cache["flagged"], cache["days"], cache["left_out"], cache["settled"]) == (
        [], 40, [], [(nth_day(10), nth_day(29))])
    assert "quiet" not in cache["failed"]
    # No plant starts where its days would meet the step's.
    assert cache["starts"] == [nth_day(d) for d in range(30, 37)]


def test_a_plant_whose_flag_would_join_an_incidents_last_days_is_missed_as_the_check_misses_it(tmp_path):
    # The check reports no flag run that meets a recorded incident, so a drop starting the day
    # after one whose last days are still deviant merges into it and is never reported.
    incident = {"metric": "cache_ratio", "start": nth_day(8), "end": nth_day(10), "status": "recovered"}
    _, metrics = calibrated(tmp_path, [{"misses": 30 if 8 <= i <= 10 else 0} for i in range(20)], [incident])
    cache = metrics["cache_ratio"]
    # No start's days meet the incident, and the two right after it catch nothing.
    assert cache["starts"] == [nth_day(d) for d in range(11, 17)]
    assert (cache["failed"], cache["flagged"], cache["caught"][0.05], cache["runs"]) == (["reach"], [], 4, 6)
    assert [cache["floors"][nth_day(d)] for d in range(11, 17)] == [None, None, 0.05, 0.05, 0.05, 0.05]


def test_the_days_after_an_incident_are_judged_against_the_days_before_it_as_the_check_judges_them(tmp_path):
    # Ten days of misses stay out of the baselines, so a plant after them is measured against
    # the clean days before; in the baselines they would hide every planted 5%.
    incident = {"metric": "cache_ratio", "start": nth_day(8), "end": nth_day(17), "status": "recovered"}
    _, metrics = calibrated(tmp_path, [{"misses": 30 if 8 <= i <= 17 else 0} for i in range(30)], [incident])
    cache = metrics["cache_ratio"]
    assert (cache["starts"][0], cache["starts"][-1], cache["caught"][0.05], cache["runs"]) == (
        nth_day(18), nth_day(26), 7, 9)
    # The two starts right after it join its last days, as test_a_plant_whose_flag... shows.
    assert [cache["floors"][day] for day in cache["starts"]] == [None, None] + [0.05] * 7


def test_a_subagent_plant_starts_off_incidents_and_is_judged_on_the_history_up_to_its_fourth_day(tmp_path):
    # Judged on the whole history, a plant just before the incident would join its flag run.
    main_thread_days(tmp_path, [{}] * 24, per_day=150)
    tool_loop_days(tmp_path, 24, per_day=301, subagent=True, misses=20, miss_days=range(15, 18))
    incident = {"metric": "subagent_cache", "start": nth_day(15), "end": nth_day(17), "status": "recovered"}
    subagent = calibration(parse_all(tmp_path).responses, [incident], date.fromisoformat(nth_day(24)))["metrics"][2]
    assert subagent["starts"] == [nth_day(d) for d in (7, 8, 9, 10, 11, 18, 19, 20)]
    assert (subagent["caught"][0.05], subagent["runs"]) == (30, 40)


def test_no_plant_starts_on_or_joins_a_dismissed_incident_whose_flags_the_check_never_reports(tmp_path):
    incident = {"metric": "cache_ratio", "start": nth_day(8), "end": nth_day(10), "status": "dismissed"}
    _, metrics = calibrated(tmp_path, [{"misses": 30 if 8 <= i <= 10 else 0} for i in range(20)], [incident])
    cache = metrics["cache_ratio"]
    assert cache["starts"] == [nth_day(d) for d in range(11, 17)]
    assert [cache["floors"][nth_day(d)] for d in range(11, 13)] == [None, None]


def test_incidents_that_leave_no_start_clear_of_them_are_not_enough_data_on_a_long_history(tmp_path):
    incident = {"metric": "cache_ratio", "start": nth_day(10), "end": None, "status": "open"}
    _, metrics = calibrated(tmp_path, [{}] * 20, [incident])
    cache = metrics["cache_ratio"]
    assert (cache["verdict"], cache["days"], cache["starts"]) == ("not_enough_data", 20, [])
    assert metric_lines(cache)[0] == (
        "Cache ratio: NOT ENOUGH DATA, 20 days judged, but none can start a planted change: it needs 7 judged "
        "days before it and 3 after it, all within 14 days and meeting no incident ccdrift follows")


def test_no_plant_starts_across_a_break_longer_than_the_days_the_check_reports_a_flag_from(tmp_path):
    # The check reports only flags begun in the last 14 days, so a plant whose days straddle a
    # long break could be caught by no check at all.
    main_thread_days(tmp_path, [{}] * 10, per_day=150)
    main_thread_days(tmp_path, [{}] * 6, per_day=150, first_day=30)
    cache = calibration(parse_all(tmp_path).responses, [], date.fromisoformat(nth_day(36)))["metrics"][0]
    assert cache["starts"] == [nth_day(30), nth_day(31), nth_day(32)]


def test_every_metric_starts_its_plants_off_every_incident_and_within_the_days_the_check_reports_from(
        tmp_path, monkeypatch):
    asked = []
    real = ccdrift.calibrate.plant_starts
    monkeypatch.setattr("ccdrift.calibrate.plant_starts",
                        lambda days, **options: asked.append(options) or real(days, **options))
    incidents = [{"metric": metric, "start": nth_day(3), "end": nth_day(4), "status": "dismissed"}
                 for metric in ("cache_ratio", "haiku_fraction", "subagent_cache")]
    main_thread_days(tmp_path, [{}] * 12, per_day=150)
    tool_loop_days(tmp_path, 12, per_day=301, subagent=True)
    calibration(parse_all(tmp_path).responses, incidents, date.fromisoformat(nth_day(12)))
    assert asked == [{"avoid": [(nth_day(3), nth_day(4))], "within": 14}] * 3


def test_a_dismissed_incidents_days_stay_in_and_its_flag_counts_as_a_false_alarm(tmp_path):
    incident = {"metric": "cache_ratio", "start": nth_day(12), "end": nth_day(19), "status": "dismissed"}
    _, metrics = calibrated(tmp_path, STEP, [incident])
    cache = metrics["cache_ratio"]
    assert (cache["flagged"], cache["dismissed"], cache["left_out"], cache["days"]) == (
        [nth_day(12)], [nth_day(12)], [], 20)
    assert "quiet" in cache["failed"]


def test_an_incident_of_another_metric_leaves_this_ones_days_alone(tmp_path):
    incident = {"metric": "haiku_fraction", "start": nth_day(12), "end": None, "status": "open"}
    _, metrics = calibrated(tmp_path, STEP, [incident])
    assert (metrics["cache_ratio"]["flagged"], metrics["cache_ratio"]["days"]) == ([nth_day(12)], 20)
    assert (metrics["haiku_fraction"]["left_out"], metrics["haiku_fraction"]["days"]) == ([(nth_day(12), None)], 20)


def test_subagent_days_with_too_few_loop_turns_are_neither_judged_nor_starts(tmp_path):
    result, metrics = calibrated(tmp_path, [{}] * 12, loops=100)
    subagent = metrics["subagent_cache"]
    assert (subagent["verdict"], subagent["days"], subagent["starts"], subagent["runs"]) == (
        "not_enough_data", 0, [], 0)
    assert result["subagent_days"] == 12


def run(tmp_path, capsys, days, state=None, **options):
    """`ccdrift calibrate` on a history like `calibrated`'s in a project folder named
    secret-client, judged the day after its last, with what it printed."""
    logs = tmp_path / "logs"
    main_thread_days(logs / "secret-client", days, per_day=150)
    tool_loop_days(logs / "secret-client", len(days), per_day=301, subagent=True)
    code = run_calibrate(logs, state or tmp_path / "home" / "state.json", today=date.fromisoformat(nth_day(len(days))),
                         **options)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_calibrate_prints_what_it_judged_each_verdict_with_what_it_rests_on_and_what_pass_means(tmp_path, capsys):
    code, out, err = run(tmp_path, capsys, [{}] * 12)
    assert code == 0
    assert err == "Planting changes in your history and replaying the detector over each; this can take a minute.\n"
    assert out.splitlines() == [
        "Calibration on your history, 2026-09-01 to 2026-09-12: 12 days judged on the main thread, 12 with "
        "subagent loops.",
        "",
        "Cache ratio: PASS",
        "  A planted 5% drop is caught from 2 of 2 starts, 2026-09-08 to 2026-09-09.",
        "  Smallest drop caught: 5% from every start.",
        "  Nothing flags.",
        "Haiku share: PASS",
        "  A planted 5% rise is caught from 2 of 2 starts, 2026-09-08 to 2026-09-09.",
        "  Smallest rise caught: 5% from every start.",
        "  Nothing flags.",
        "Subagent read-back: PASS",
        "  A planted 5% drop is caught in 10 of 10 runs (100%), against a bar of 90%. A planted 2%: 10 of 10.",
        "  Nothing flags.",
        "",
        "Early warnings, turn by turn:",
        "Cache-miss warning: NOT ENOUGH DATA, 4 days judged, and a rate a week needs 7",
        "Tool-loop warning: NOT ENOUGH DATA, no week has 1,000 tool-loop turns in the 14 days before it",
        "Subagent tool-loop warning: no setting ships, so subagents get no early warning.",
        "",
        "Not judged yet: session start, failed requests, cut-short responses, hooks, thinking and spawn models.",
        "PASS means the shipped setting clears the bar it cleared on the logs it was tuned on: a planted 5% caught "
        "from every start (90% of runs for subagent read-back), and no flag outside the incidents ccdrift follows.",
        "For the warnings, the bars their gates in lab/ set: false cache-miss warnings no more than once in 6 weeks at "
        "your volume and a planted 5% miss rate caught within a median of 150 new-prompt turns; no tool-loop alarm at "
        "the most sensitive usual rate and a planted 2% miss rate caught in 90% of runs within a median of 300."]


def result_of(metric="cache_ratio", verdict="fail", failed=("quiet", "reach"), **fields):
    defaults = {"days": 20, "starts": [nth_day(7), nth_day(8), nth_day(9)], "runs": 3,
                "caught": {0.05: 1, 0.02: 0}, "floors": {nth_day(7): 0.05, nth_day(8): 0.2, nth_day(9): None},
                "flagged": [nth_day(12)], "dismissed": [], "left_out": [(nth_day(2), nth_day(4)), (nth_day(15), None)],
                "settled": []}
    return {"metric": metric, "verdict": verdict, "failed": list(failed), **defaults, **fields}


def test_a_failed_metric_says_which_half_failed_what_it_still_catches_and_how_to_label_a_flag():
    assert metric_lines(result_of()) == [
        "Cache ratio: FAIL, it flags days outside the incidents ccdrift follows, and a planted 5% drop is missed",
        "  A planted 5% drop is caught from 1 of 3 starts, 2026-09-08 to 2026-09-10.",
        "  Smallest drop caught: 5% from 1 start, 20% from 1 start, none up to 70% from 1 start.",
        "  It flags from 2026-09-13 with no incident recorded. `ccdrift incident dismiss` marks a false alarm, "
        "`ccdrift incident add` records a real one.",
        "  Left out as incidents ccdrift follows: 2026-09-03 to 2026-09-05, 2026-09-16 on, still open."]
    assert metric_lines(result_of(failed=["quiet"]))[0] == (
        "Cache ratio: FAIL, it flags days outside the incidents ccdrift follows")
    assert metric_lines(result_of("haiku_fraction", failed=["reach"]))[0] == (
        "Haiku share: FAIL, a planted 5% rise is missed")


def test_a_flag_the_user_dismissed_is_told_as_theirs_and_a_settled_level_as_the_new_normal():
    lines = metric_lines(result_of(failed=["quiet"], flagged=[nth_day(3), nth_day(12)], dismissed=[nth_day(12)],
                                   left_out=[], settled=[(nth_day(15), nth_day(20))]))
    assert lines[3:] == [
        "  It flags from 2026-09-04 with no incident recorded. `ccdrift incident dismiss` marks a false alarm, "
        "`ccdrift incident add` records a real one.",
        "  It flags from 2026-09-13, which you dismissed as a false alarm; dismissed flags count against it.",
        "  Kept in as the new normal, as the check keeps them: 2026-09-16 to 2026-09-21."]


def test_a_metric_with_one_start_or_none_says_so():
    one = metric_lines(result_of(verdict="pass", failed=(), starts=[nth_day(7)], runs=1, caught={0.05: 1},
                                 floors={nth_day(7): 0.05}, flagged=[], left_out=[]))
    assert one == ["Cache ratio: PASS", "  A planted 5% drop is caught from 1 of 1 start, 2026-09-08.",
                   "  Smallest drop caught: 5% from every start.", "  Nothing flags."]
    none = metric_lines(result_of("subagent_cache", verdict="not_enough_data", failed=(), days=4, starts=[], runs=0,
                                  caught={0.02: 0, 0.05: 0}, floors={}, flagged=[], left_out=[]))
    assert none == ["Subagent read-back: NOT ENOUGH DATA, 4 days judged where a planted change needs 11: 7 before "
                    "its start and 3 after it", "  Nothing flags."]


def test_subagent_read_back_reports_runs_rather_than_starts():
    lines = metric_lines(result_of("subagent_cache", failed=["reach"], runs=50, caught={0.02: 30, 0.05: 41},
                                   flagged=[], left_out=[]))
    assert lines == ["Subagent read-back: FAIL, a planted 5% drop is missed",
                     "  A planted 5% drop is caught in 41 of 50 runs (82%), against a bar of 90%. A planted 2%: 30 "
                     "of 50.", "  Nothing flags."]


@pytest.mark.parametrize("floors, text", [({"a": 0.05, "b": 0.05}, "5% from every start"),
                                          ({"a": 0.15, "b": 0.05, "c": 0.15}, "5% from 1 start, 15% from 2 starts"),
                                          ({"a": None, "b": None}, "none up to 70% from 2 starts"),
                                          ({"a": 0.05, "b": None}, "5% from 1 start, none up to 70% from 1 start")],
                         ids=["every", "two-sizes", "none", "one-and-none"])
def test_the_smallest_sizes_caught_are_told_smallest_first_with_how_many_starts_each(floors, text):
    assert floors_text(floors, 0.7) == text


def test_a_history_with_no_complete_day_yet_says_so():
    lines = calibrate_lines({"first": None, "last": None, "main_days": 0, "subagent_days": 0, "metrics": [],
                             "warnings": []})
    assert lines[0] == "Calibration on your history: no complete UTC day judged yet."


def test_the_json_holds_days_counts_and_the_bar_and_names_no_path_session_or_project(tmp_path, capsys):
    code, out, _ = run(tmp_path, capsys, [{}] * 12, as_json=True)
    payload = json.loads(out)
    assert code == 0
    assert set(payload) == {"first", "last", "main_days", "subagent_days", "metrics", "warnings", "bar",
                            "subagent_share", "not_judged"}
    assert (payload["bar"], payload["subagent_share"]) == (0.05, 0.9)
    assert payload["not_judged"] == ["context_changes", "failed_requests", "cut_short", "hook_changes",
                                     "thinking_rises", "model_mismatches"]
    assert [(w["warning"], w["verdict"]) for w in payload["warnings"]] == [
        ("cache_miss", "not_enough_data"), ("tool_loop", "not_enough_data"), ("tool_loop", "none_ships")]
    cache = payload["metrics"][0]
    assert cache["caught"]["0.05"] == 2 and cache["left_out"] == []
    assert "secret-client" not in out and str(tmp_path) not in out
    assert not any(f'"s{d}"' in out for d in range(12))


def test_with_no_state_it_creates_nothing_in_ccdrifts_home(tmp_path, capsys):
    home = tmp_path / "home"
    assert run(tmp_path, capsys, [{}] * 12)[0] == 0
    assert not home.exists()


def test_a_state_it_cant_read_exits_1_and_is_left_as_it_was(tmp_path, capsys):
    state = tmp_path / "home" / "state.json"
    state.parent.mkdir()
    state.write_text(json.dumps({"version": STATE_VERSION + 1}))
    before = state.read_bytes()
    code, out, err = run(tmp_path, capsys, [{}] * 12, state=state)
    assert (code, out) == (1, "")
    assert "newer ccdrift" in err
    assert state.read_bytes() == before


def test_it_judges_the_days_the_check_reads_not_the_whole_store(tmp_path, capsys):
    # The check reads the last 90 days of a store, so a long one doesn't slow it down.
    logs, home = tmp_path / "logs", tmp_path / "home"
    main_thread_days(logs / "p", [{}] * 120, per_day=20)
    home.mkdir()
    with History(home / "history.sqlite") as history:
        history.update(logs)
    assert run_calibrate(logs, home / "state.json", as_json=True, today=date.fromisoformat(nth_day(120))) == 0
    payload = json.loads(capsys.readouterr().out)
    assert (payload["first"], payload["last"], payload["main_days"]) == (nth_day(30), nth_day(119), 90)


def test_an_unusable_store_exits_1_and_a_source_with_no_transcripts_exits_2(tmp_path, capsys):
    home = tmp_path / "home"
    home.mkdir()
    (home / "history.sqlite").write_text("not a database")
    code, _, err = run(tmp_path, capsys, [{}] * 12, state=home / "state.json")
    assert code == 1 and "Move it aside" in err
    (tmp_path / "empty").mkdir()
    assert run_calibrate(tmp_path / "empty", tmp_path / "other" / "state.json") == 2
    assert "No Claude Code transcripts found" in capsys.readouterr().err


def test_the_command_line_passes_its_source_state_and_json_through(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ccdrift.calibrate, "run_calibrate", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    paths = ["--source", str(tmp_path / "logs"), "--state", str(tmp_path / "state.json")]
    assert main(["calibrate", "--json", *paths]) == 0
    assert main(["calibrate", *paths]) == 0
    assert calls == [((tmp_path / "logs", tmp_path / "state.json"), {"as_json": True}),
                     ((tmp_path / "logs", tmp_path / "state.json"), {"as_json": False})]


@pytest.fixture
def h4(monkeypatch):
    """The cache-miss histories built to fall on one side of the bar or the other at h = 4's
    alarm rate, whatever h ships: they test the judging, the gate tests the setting."""
    monkeypatch.setattr(early, "THRESHOLD", 4.0)


def cache_miss(tmp_path, days, per_day, incidents=()):
    """The cache-miss warning on main-thread days from Sep 1, judged on the day after the last."""
    main_thread_days(tmp_path, days, per_day=per_day)
    return cache_miss_warning(parse_all(tmp_path).responses, list(incidents), date.fromisoformat(nth_day(len(days))))


@pytest.mark.usefixtures("h4")
@pytest.mark.parametrize("per_day, misses, verdict, false_a_week, turns_a_week, at_floor", [
    (51, 1, "pass", 0.11, 350, 0),
    (201, 4, "fail", 0.43, 1400, 1),
])
def test_the_cache_miss_warning_is_judged_quiet_by_its_false_warnings_at_this_historys_rate_and_volume(
        tmp_path, per_day, misses, verdict, false_a_week, turns_a_week, at_floor):
    # A miss every other day keeps the usual rate at 1% either way; four times the turns a
    # week are four times the false warnings, past the bar of one in 6 weeks. At 200 turns a
    # day the first day judged rests on one day without a miss, and counts at the floor.
    result = cache_miss(tmp_path, [{"misses": misses * (i % 2)} for i in range(28)], per_day)
    assert (result["verdict"], result["usual_rate"], result["false_a_week"], result["turns_a_week"],
            result["days_at_floor"]) == (verdict, 0.01, pytest.approx(false_a_week, abs=5e-4), turns_a_week, at_floor)
    assert result["failed"] == ([] if verdict == "pass" else ["quiet"])
    # Unrounded, as the verdict compares it: rounded, a rate just past the bar could equal it.
    assert (round(result["false_a_week"], 3) != result["false_a_week"], result["bar_false_a_week"]) == (True, 1 / 6)


@pytest.mark.parametrize("status, verdict, days, usual_rate, alarms", [
    ("recovered", "pass", 15, 0.01, []),
    ("dismissed", "fail", 18, 0.0327, [nth_day(12), nth_day(13), nth_day(14)]),
])
def test_a_cache_incidents_days_leave_the_cache_miss_warning_as_the_check_leaves_them_unless_dismissed(
        tmp_path, status, verdict, days, usual_rate, alarms):
    incident = {"metric": "cache_ratio", "start": nth_day(12), "end": nth_day(14), "status": status}
    result = cache_miss(tmp_path, [{"misses": 30 if 12 <= i <= 14 else i % 2} for i in range(28)], 51, [incident])
    assert (result["verdict"], result["days"], result["usual_rate"], result["alarms"]) == (
        verdict, days, usual_rate, alarms)


@pytest.mark.usefixtures("h4")
@pytest.mark.parametrize("metric, days, false_a_week", [("cache_ratio", 24, 0.22), ("haiku_fraction", 62, 0.221)])
def test_days_of_new_prompts_the_cache_miss_warning_leaves_out_leave_its_weeks_too(tmp_path, metric, days,
                                                                                    false_a_week):
    # A cache incident's thirty days, and the week after it without a usual rate, are days the
    # warning is off: taking their weeks with none of their turns once halved the rate and
    # passed a history that fails. Another metric's incident leaves nothing out.
    incident = {"metric": metric, "start": nth_day(20), "end": nth_day(49), "status": "recovered"}
    result = cache_miss(tmp_path, [{"misses": 1}] * 70, 101, [incident])
    assert (result["verdict"], result["days"], result["turns_a_week"], result["false_a_week"]) == (
        "fail", days, 700, pytest.approx(false_a_week, abs=5e-4))


def test_days_without_new_prompts_count_as_weeks_the_cache_miss_warning_was_on(tmp_path):
    # A day's work every other day: 16 days judged over 31, 200 turns each.
    for day in range(0, 40, 2):
        main_thread_days(tmp_path, [{"misses": 2}], per_day=201, first_day=day)
    result = cache_miss_warning(parse_all(tmp_path).responses, [], date.fromisoformat(nth_day(40)))
    assert (result["days"], result["weeks"], result["turns_a_week"]) == (16, 4.4, 723)


def test_the_cache_miss_warning_needs_a_week_of_judged_days_for_a_rate_a_week(tmp_path):
    result = cache_miss(tmp_path, [{"misses": i % 2} for i in range(14)], 51)
    assert (result["verdict"], result["days"]) == ("not_enough_data", 4)
    assert warning_lines(result) == [
        "Cache-miss warning: NOT ENOUGH DATA, 4 days judged, and a rate a week needs 7"]


def test_a_usual_miss_rate_under_the_floor_the_check_judges_at_counts_at_the_floor(tmp_path):
    # No misses at all once simulated none, and passed any volume.
    result = cache_miss(tmp_path, [{}] * 28, 51)
    assert (result["usual_rate"], result["days"], result["days_at_floor"], result["false_a_week"]) == (
        0, 18, 18, pytest.approx(alarms_per_turn(early.THRESHOLD, MIN_P0) * 350))


@pytest.mark.parametrize("rate, at_floor", [(0.001, 28), (MIN_P0, 0)], ids=["under", "at"])
def test_a_usual_miss_rate_between_none_and_the_floor_counts_at_the_floor_and_one_at_it_as_itself(
        tmp_path, monkeypatch, rate, at_floor):
    monkeypatch.setattr(ccdrift.calibrate, "prompt_base_rate", lambda *args: rate)
    result = cache_miss(tmp_path, [{}] * 28, 51)
    assert (result["days"], result["days_at_floor"], result["false_a_week"]) == (
        28, at_floor, pytest.approx(alarms_per_turn(early.THRESHOLD, MIN_P0) * 350))


def test_neither_warning_judges_todays_turns_until_the_day_is_complete(tmp_path):
    main_thread_days(tmp_path / "cache", [{"misses": i % 2} for i in range(27)] + [{"misses": 30}], per_day=51)
    responses = parse_all(tmp_path / "cache").responses
    assert [cache_miss_warning(responses, [], date.fromisoformat(nth_day(day)))["alarms"] for day in (27, 28)] == [
        [], [nth_day(27)]]
    tool_loop_days(tmp_path / "loop", 25, per_day=150, misses=2, miss_days=[24])
    responses = parse_all(tmp_path / "loop").responses
    assert [tool_loop_warning(responses, "main", [], date.fromisoformat(nth_day(day)))["alarms"] for day in (24, 25)] == [
        [], [nth_day(24)]]


def test_the_cache_miss_warning_has_not_enough_data_without_200_new_prompt_turns_in_the_14_days_before_a_day(
        tmp_path):
    result = cache_miss(tmp_path, [{}] * 28, 11)
    assert (result["verdict"], result["days"]) == ("not_enough_data", 0)


@pytest.mark.usefixtures("h4")
def test_a_planted_miss_rate_the_cache_miss_warning_cant_catch_within_a_median_of_150_turns_fails_its_reach(
        tmp_path):
    # 20 turns a day against a usual rate of 2.5%: a planted 5% gains too slowly for a week.
    result = cache_miss(tmp_path, [{"misses": int(i % 2 == 0)} for i in range(28)], 21)
    assert (result["verdict"], result["failed"], result["caught"], result["runs"], result["median_turns"]) == (
        "fail", ["reach"], 15, 50, None)


@pytest.mark.usefixtures("h4")
def test_the_days_the_history_itself_raises_a_cache_miss_warning_are_listed_and_never_decide_the_verdict(tmp_path):
    result = cache_miss(tmp_path, [{"misses": 3 if i == 20 else i % 2} for i in range(28)], 51)
    assert (result["verdict"], result["alarms"]) == ("pass", [nth_day(20)])


def test_no_cache_miss_warning_shipping_is_said_rather_than_judged(tmp_path, monkeypatch):
    monkeypatch.setattr(early, "THRESHOLD", None)
    assert cache_miss(tmp_path, [{}] * 2, 51) == {"warning": "cache_miss", "setting": None, "verdict": "none_ships"}


def tool_loop(tmp_path, days, incidents=(), stream="main", **options):
    """The tool-loop warning of `stream` on tool-loop days from Sep 1, judged on the day after the last."""
    tool_loop_days(tmp_path, days, subagent=stream == "subagent", **options)
    return tool_loop_warning(parse_all(tmp_path).responses, stream, list(incidents), date.fromisoformat(nth_day(days)))


@pytest.mark.parametrize("status, verdict, days, alarms, dismissed", [
    (None, "fail", 11, [nth_day(20)], []),
    ("recovered", "pass", 8, [], []),
    ("dismissed", "fail", 11, [nth_day(20)], [nth_day(20)]),
])
def test_a_tool_loop_alarm_at_the_floor_fails_quiet_unless_a_cache_incident_ccdrift_follows_covers_its_day(
        tmp_path, status, verdict, days, alarms, dismissed):
    # Two misses in a row against the 0.2% floor at p1 = 2% add 2 × ln(0.02/0.002) = 4.6, past h = 3.
    incidents = [{"metric": "cache_ratio", "start": nth_day(19), "end": nth_day(21), "status": status}] if status else []
    result = tool_loop(tmp_path, 24, incidents, per_day=150, misses=2, miss_days=[20])
    assert (result["verdict"], result["days"], result["alarms"], result["dismissed"]) == (verdict, days, alarms, dismissed)
    assert result["failed"] == (["quiet"] if verdict == "fail" else [])
    # A usual rate of none is under the floor, so the check runs at the floor too.
    assert result["own_alarms"] == alarms


def test_tool_loop_alarms_at_the_floor_are_told_beside_those_the_check_raises_at_the_users_own_rate(tmp_path):
    # Two misses in a row every other day keep the usual rate at 0.5%: past h = 3 at the floor,
    # 2 × ln(0.02/0.005) = 2.8 at the user's own rate.
    result = tool_loop(tmp_path, 24, per_day=200, misses=2, miss_days=range(0, 24, 2))
    assert (result["verdict"], result["alarms"], result["own_alarms"]) == (
        "fail", [nth_day(day) for day in range(12, 24, 2)], [])


def test_the_check_at_the_users_own_rate_can_alarm_on_a_day_the_floor_doesnt(tmp_path):
    # Two misses at day 25's end and one at day 26's start: the floor alarms on the second and
    # starts again, while at the usual 0.5% each counts for less and only the third crosses h.
    result = tool_loop(tmp_path, 28, per_day=200, misses=1, miss_days=range(28), miss_turns={25: [199], 26: [1]})
    assert (result["alarms"], result["own_alarms"]) == ([nth_day(25)], [nth_day(26)])


def test_an_incident_of_another_metric_leaves_a_tool_loop_alarm_counted(tmp_path):
    incident = {"metric": "haiku_fraction", "start": nth_day(19), "end": nth_day(21), "status": "recovered"}
    result = tool_loop(tmp_path, 24, [incident], per_day=150, misses=2, miss_days=[20])
    assert (result["verdict"], result["days"], result["alarms"]) == ("fail", 11, [nth_day(20)])


def test_the_tool_loop_warning_has_not_enough_data_without_1000_turns_in_the_14_days_before_a_window(tmp_path):
    result = tool_loop(tmp_path, 24, per_day=50)
    assert (result["verdict"], result["days"]) == ("not_enough_data", 0)


def test_a_tool_loop_plant_caught_in_under_90_percent_of_runs_fails_reach_however_fast_the_rest(tmp_path):
    # The last starts have a day or two of turns left to catch it in.
    result = tool_loop(tmp_path, 20, per_day=80)
    assert (result["failed"], result["caught"], result["runs"], result["median_turns"]) == (["reach"], 30, 35, 89)


def test_a_tool_loop_plant_caught_every_time_but_in_a_median_past_300_turns_fails_reach(tmp_path):
    # Days 0 to 3 put the usual rate of most starts at the 1% it is held under, where a
    # planted 2% gains slowly.
    result = tool_loop(tmp_path, 24, per_day=300, misses=30, miss_days=range(4))
    assert (result["failed"], result["caught"], result["runs"], result["median_turns"]) == (["reach"], 50, 50, 413)


def test_a_stream_with_no_setting_is_said_to_ship_none_and_is_judged_once_one_ships(tmp_path, monkeypatch):
    assert tool_loop(tmp_path / "none", 24, stream="subagent", per_day=150) == {
        "warning": "tool_loop", "stream": "subagent", "setting": None, "verdict": "none_ships"}
    monkeypatch.setitem(LOOP_SETTINGS, "subagent", LoopSetting(0.02, 3.0, 1))
    result = tool_loop(tmp_path / "set", 24, stream="subagent", per_day=150)
    assert (result["setting"], result["verdict"], result["days"]) == (
        {"p1": 0.02, "h": 3.0, "min_sessions": 1}, "pass", 11)


def test_the_calibration_judges_the_cache_miss_warning_and_every_tool_loop_stream(tmp_path):
    main_thread_days(tmp_path, [{}] * 2)
    warnings = calibration(parse_all(tmp_path).responses, [], date.fromisoformat(nth_day(2)))["warnings"]
    assert [(w["warning"], w.get("stream"), w["verdict"]) for w in warnings] == [
        ("cache_miss", None, "not_enough_data"), ("tool_loop", "main", "not_enough_data"),
        ("tool_loop", "subagent", "none_ships")]


def test_a_median_past_every_catch_is_null_in_the_json(tmp_path):
    main_thread_days(tmp_path, [{"misses": int(i % 2 == 0)} for i in range(28)], per_day=21)
    result = calibration(parse_all(tmp_path).responses, [], date.fromisoformat(nth_day(28)))
    assert json.loads(calibrate_json(result))["warnings"][0]["median_turns"] is None


def cache_result(verdict="fail", failed=("quiet", "reach"), **fields):
    defaults = {"days": 18, "weeks": 2.6, "turns_a_week": 1400, "usual_rate": 0.0043, "days_at_floor": 0,
                "false_a_week": 0.418, "alarms": [nth_day(7), nth_day(9)], "starts": [nth_day(3), nth_day(5), nth_day(9)],
                "runs": 15, "caught": 4, "median_turns": None}
    return {"warning": "cache_miss", "setting": {"h": 4.0, "p1": 0.05}, "verdict": verdict, "failed": list(failed),
            **defaults, **fields}


def test_a_failed_cache_miss_warning_says_which_half_failed_and_what_its_history_raised():
    assert warning_lines(cache_result()) == [
        "Cache-miss warning: FAIL, it would warn falsely 0.42 times a week at your volume, and a planted 5% miss rate "
        "isn't caught within a median of 150 turns",
        "  False warnings: 0.42 a week expected at your usual miss rate (0.43% median) and 1,400 new-prompt turns a "
        "week, over 2.6 weeks judged, against a bar of 1 every 6 weeks (0.17).",
        "  Your history alarmed on 2026-09-08, 2026-09-10 outside the cache incidents ccdrift follows.",
        "  A planted 5% miss rate is not caught in a median: 4 of 15 runs from 3 starts, 2026-09-04 to 2026-09-10, "
        "against a bar of 150 turns."]


def test_a_passing_cache_miss_warning_says_what_it_rests_on():
    assert warning_lines(cache_result(verdict="pass", failed=[], false_a_week=0.159, turns_a_week=463, alarms=[],
                                      caught=15, median_turns=26.0)) == [
        "Cache-miss warning: PASS",
        "  False warnings: 0.16 a week expected at your usual miss rate (0.43% median) and 463 new-prompt turns a "
        "week, over 2.6 weeks judged, against a bar of 1 every 6 weeks (0.17).",
        "  Your history raised none.",
        "  A planted 5% miss rate is caught in a median 26 turns, 15 of 15 runs from 3 starts, 2026-09-04 to "
        "2026-09-10, against a bar of 150 turns."]


@pytest.mark.parametrize("verdict, failed, rate, head", [
    ("fail", ["quiet"], 0.1668, "Cache-miss warning: FAIL, it would warn falsely 0.1668 times a week at your volume"),
    ("pass", [], 0.1666, "Cache-miss warning: PASS"),
], ids=["just-past", "just-under"])
def test_a_rate_that_rounds_to_the_bar_is_told_to_the_digit_that_sets_them_apart(verdict, failed, rate, head):
    assert warning_lines(cache_result(verdict=verdict, failed=failed, false_a_week=rate, alarms=[]))[:2] == [
        head,
        f"  False warnings: {rate} a week expected at your usual miss rate (0.43% median) and 1,400 new-prompt turns "
        "a week, over 2.6 weeks judged, against a bar of 1 every 6 weeks (0.1667)."]


def test_a_rate_at_the_bar_passes_and_prints_as_the_bar():
    assert warning_lines(cache_result(verdict="pass", failed=[], false_a_week=1 / 6, alarms=[]))[:2] == [
        "Cache-miss warning: PASS",
        "  False warnings: 0.17 a week expected at your usual miss rate (0.43% median) and 1,400 new-prompt turns a "
        "week, over 2.6 weeks judged, against a bar of 1 every 6 weeks (0.17)."]


def test_days_whose_usual_miss_rate_counts_at_the_floor_are_counted_under_the_rate():
    assert warning_lines(cache_result(verdict="pass", failed=[], false_a_week=0.159, days_at_floor=2))[1:3] == [
        "  False warnings: 0.16 a week expected at your usual miss rate (0.43% median) and 1,400 new-prompt turns a "
        "week, over 2.6 weeks judged, against a bar of 1 every 6 weeks (0.17).",
        "  On 2 judged days the usual miss rate was under 0.2%, the floor the check judges against, and counts at it."]
    assert warning_lines(cache_result(verdict="pass", failed=[], false_a_week=0.159, days_at_floor=1))[2] == (
        "  On 1 judged day the usual miss rate was under 0.2%, the floor the check judges against, and counts at it.")


def test_a_cache_miss_warning_with_one_half_failed_names_only_that_half():
    assert warning_lines(cache_result(failed=["reach"], false_a_week=0.02, alarms=[]))[:3] == [
        "Cache-miss warning: FAIL, a planted 5% miss rate isn't caught within a median of 150 turns",
        "  False warnings: 0.02 a week expected at your usual miss rate (0.43% median) and 1,400 new-prompt turns a "
        "week, over 2.6 weeks judged, against a bar of 1 every 6 weeks (0.17).",
        "  Your history raised none."]


def loop_result(verdict="fail", failed=("quiet",), **fields):
    defaults = {"days": 11, "alarms": [nth_day(20), nth_day(21)], "dismissed": [nth_day(21)],
                "own_alarms": [nth_day(21)], "starts": [nth_day(13)], "runs": 5, "caught": 5, "median_turns": 91.5}
    return {"warning": "tool_loop", "stream": "main", "setting": {"p1": 0.02, "h": 3.0, "min_sessions": 1},
            "verdict": verdict, "failed": list(failed), **defaults, **fields}


def test_a_tool_loop_warning_that_alarms_says_on_which_days_which_were_dismissed_and_which_the_check_would_raise():
    assert warning_lines(loop_result()) == [
        "Tool-loop warning: FAIL, it alarms outside the cache incidents ccdrift follows",
        "  It alarms on 2026-09-21 with no cache incident recorded, at the most sensitive usual rate, 0.2%.",
        "  It alarms on 2026-09-22, which you dismissed as a false alarm; dismissed alarms count against it.",
        "  At your own usual rate, as the check runs, it alarms on 2026-09-22.",
        "  A planted 2% miss rate is caught in 5 of 5 runs (100%), in a median 91.5 turns, against 90% within a "
        "median of 300."]


def test_tool_loop_alarms_the_check_wouldnt_raise_at_the_users_own_rate_are_said_to_be_none_of_its():
    assert warning_lines(loop_result(dismissed=[], own_alarms=[]))[1:3] == [
        "  It alarms on 2026-09-21, 2026-09-22 with no cache incident recorded, at the most sensitive usual rate, 0.2%.",
        "  At your own usual rate, as the check runs, it raises none."]


def test_an_alarm_only_the_users_own_rate_raises_is_told_beside_a_quiet_floor():
    assert warning_lines(loop_result(verdict="pass", failed=[], alarms=[], dismissed=[], own_alarms=[nth_day(26)]))[1:3] == [
        "  No alarm on 11 days judged at the most sensitive usual rate, 0.2%.",
        "  At your own usual rate, as the check runs, it alarms on 2026-09-27."]


def test_a_quiet_tool_loop_warning_that_misses_its_plant_says_how_often_it_was_caught():
    assert warning_lines(loop_result(failed=["reach"], alarms=[], dismissed=[], own_alarms=[], caught=2,
                                     median_turns=None)) == [
        "Tool-loop warning: FAIL, a planted 2% miss rate isn't caught in 90% of runs within a median of 300 turns",
        "  No alarm on 11 days judged at the most sensitive usual rate, 0.2%.",
        "  A planted 2% miss rate is caught in 2 of 5 runs (40%), not in a median, against 90% within a median of "
        "300."]


def test_a_warning_without_the_turns_to_judge_or_a_setting_says_so_in_one_line():
    assert warning_lines({"warning": "cache_miss", "setting": {"h": 4.0, "p1": 0.05}, "verdict": "not_enough_data",
                          "failed": [], "days": 0}) == [
        "Cache-miss warning: NOT ENOUGH DATA, no week has 200 new-prompt turns in the 14 days before it"]
    assert warning_lines({"warning": "cache_miss", "setting": None, "verdict": "none_ships"}) == [
        "Cache-miss warning: no setting ships, so new prompts get no early warning."]


def test_the_tool_loop_warnings_not_enough_data_line_counts_the_tool_loops_own_base_days(monkeypatch):
    monkeypatch.setattr(loops, "BASE_DAYS", 21)
    assert warning_lines({"warning": "tool_loop", "stream": "main", "setting": {"p1": 0.02, "h": 3.0, "min_sessions": 1},
                          "verdict": "not_enough_data", "failed": [], "days": 0}) == [
        "Tool-loop warning: NOT ENOUGH DATA, no week has 1,000 tool-loop turns in the 21 days before it"]
