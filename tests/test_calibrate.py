"""`ccdrift calibrate`: the plants, the measurement on a history and what it prints."""

import json
from datetime import date

import pytest

import ccdrift.calibrate
from ccdrift.calibrate import (calibrate_lines, calibration, metric_lines, plant, plant_starts, plant_subagent,
                               run_calibrate, subagent_openings)
from ccdrift.cli import main
from ccdrift.logs import parse_all
from ccdrift.state import STATE_VERSION
from ccdrift.texts import floors_text
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
        "Not judged yet: session start, the cache-miss and tool-loop warnings, failed requests, cut-short "
        "responses, hooks, thinking and spawn models.",
        "PASS means the shipped setting clears the bar it cleared on the logs it was tuned on: a planted 5% caught "
        "from every start (90% of runs for subagent read-back), and no flag outside the incidents ccdrift follows."]


def result_of(metric="cache_ratio", verdict="fail", failed=("quiet", "reach"), **fields):
    defaults = {"days": 20, "starts": [nth_day(7), nth_day(8), nth_day(9)], "runs": 3,
                "caught": {0.05: 1, 0.02: 0}, "floors": {nth_day(7): 0.05, nth_day(8): 0.2, nth_day(9): None},
                "flagged": [nth_day(12)], "left_out": [(nth_day(2), nth_day(4)), (nth_day(15), None)]}
    return {"metric": metric, "verdict": verdict, "failed": list(failed), **defaults, **fields}


def test_a_failed_metric_says_which_half_failed_what_it_still_catches_and_how_to_label_a_flag():
    assert metric_lines(result_of()) == [
        "Cache ratio: FAIL, it flags days with no incident recorded, and a planted 5% drop is missed",
        "  A planted 5% drop is caught from 1 of 3 starts, 2026-09-08 to 2026-09-10.",
        "  Smallest drop caught: 5% from 1 start, 20% from 1 start, none up to 70% from 1 start.",
        "  It flags from 2026-09-13 with no incident recorded. `ccdrift incident dismiss` marks a false alarm, "
        "`ccdrift incident add` records a real one.",
        "  Left out as incidents ccdrift follows: 2026-09-03 to 2026-09-05, 2026-09-16 on, still open."]
    assert metric_lines(result_of(failed=["quiet"]))[0] == "Cache ratio: FAIL, it flags days with no incident recorded"
    assert metric_lines(result_of("haiku_fraction", failed=["reach"]))[0] == (
        "Haiku share: FAIL, a planted 5% rise is missed")


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
    lines = calibrate_lines({"first": None, "last": None, "main_days": 0, "subagent_days": 0, "metrics": []})
    assert lines[0] == "Calibration on your history: no complete UTC day judged yet."


def test_the_json_holds_days_counts_and_the_bar_and_names_no_path_session_or_project(tmp_path, capsys):
    code, out, _ = run(tmp_path, capsys, [{}] * 12, as_json=True)
    payload = json.loads(out)
    assert code == 0
    assert set(payload) == {"first", "last", "main_days", "subagent_days", "metrics", "bar", "subagent_share",
                            "not_judged"}
    assert (payload["bar"], payload["subagent_share"]) == (0.05, 0.9)
    assert payload["not_judged"] == ["context_changes", "early_warnings", "loop_warnings", "failed_requests",
                                     "cut_short", "hook_changes", "thinking_rises", "model_mismatches"]
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
