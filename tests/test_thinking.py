"""Thinking that rises: the metric, and the rule the check alerts on (G16 measured it)."""

from datetime import date, timedelta

import pandas as pd

from ccdrift.state import new_state
from ccdrift.thinking import COUNT_COLUMNS, DAYS, RATIO, counted, thinking_counts, thinking_rises
from tests.helpers import nth_day


def history(levels, model="claude-opus-5", responses=100, start=0):
    """Day counts for `model`, one day per level from nth_day(start): every response logs a
    count, and the day's thinking is level x responses."""
    return pd.DataFrame([{"model": model, "day": nth_day(start + i), "responses": responses, "logged": responses,
                          "thinking": level * responses} for i, level in enumerate(levels)], columns=COUNT_COLUMNS)


def replayed(counts, **setting):
    """Every rise the rule reports, judging each day the morning after it as the check does."""
    state, found = new_state(), []
    for day in sorted(set(counts["day"].astype(str))):
        found += thinking_rises(counts, state, date.fromisoformat(day) + timedelta(days=1), **setting)
    return found


def test_the_level_is_the_mean_over_the_responses_that_logged_a_count():
    turns = pd.DataFrame({"day": [nth_day(0)] * 4, "model": ["claude-opus-5"] * 4,
                          "thinking_logged": [100.0, 300.0, 0.0, None]})
    assert thinking_counts(turns).to_dict("records") == [
        {"model": "claude-opus-5", "day": nth_day(0), "responses": 4, "logged": 3, "thinking": 400.0}]
    assert counted(pd.DataFrame([{"model": "m", "day": nth_day(0), "responses": 100, "logged": 95,
                                  "thinking": 9500.0}]))["level"].tolist() == [100.0]


def test_a_day_counts_only_with_enough_responses_and_logged_counts():
    counts = pd.DataFrame([{"model": "m", "day": nth_day(0), "responses": 49, "logged": 49, "thinking": 1.0},
                           {"model": "m", "day": nth_day(1), "responses": 100, "logged": 89, "thinking": 1.0},
                           {"model": "m", "day": nth_day(2), "responses": 50, "logged": 45, "thinking": 1.0}])
    assert counted(counts)["day"].tolist() == [nth_day(2)]


def test_the_rule_ships_at_twice_the_usual_level_on_a_single_day():
    # G16 on the owner's logs, the episode ending 2026-09-17: the one setting that passes.
    assert (RATIO, DAYS) == (2, 1)
    assert [r["since"] for r in replayed(history([250] * 8 + [500]))] == [nth_day(8)]
    assert replayed(history([250] * 8 + [499])) == []


def test_a_run_is_reported_once_on_its_last_day_with_its_days_and_the_extra_thinking():
    counts = history([250] * 8 + [1500, 1800, 2000])
    state = new_state()
    today = date.fromisoformat(nth_day(11))
    [rise] = thinking_rises(counts, state, today, ratio=3, days=2)
    assert rise == {"model": "claude-opus-5", "since": nth_day(8), "on": nth_day(9), "days": [nth_day(8), nth_day(9)],
                    "median": 250.0, "levels": [1500.0, 1800.0], "extra": 280_000, "reported_on": nth_day(11)}
    assert state["thinking_rises"] == [rise]
    assert thinking_rises(counts, state, today, ratio=3, days=2) == []


def test_a_shorter_run_is_not_reported():
    assert replayed(history([250] * 8 + [1500, 250, 250]), ratio=3, days=2) == []


def test_a_flat_history_reports_nothing():
    assert replayed(history([144, 388, 250, 300, 211, 246, 335, 381, 192, 241, 416, 155])) == []


def test_a_model_with_too_few_baseline_days_is_not_judged():
    # The 2000 has 4 counted days before it, one short of a baseline, so it isn't judged at all.
    assert replayed(history([250] * 4 + [2000])) == []


def test_two_models_are_judged_apart_so_a_switch_to_a_model_that_thinks_more_is_not_a_rise():
    counts = pd.concat([history([250] * 10), history([2000] * 6, model="claude-opus-5-5", start=10)],
                       ignore_index=True)
    assert replayed(counts) == []


def test_a_second_run_after_a_day_that_isnt_raised_is_reported_again():
    found = replayed(history([250] * 8 + [1500, 1500, 250, 1500, 1500]), ratio=3, days=2)
    assert [(r["since"], r["on"]) for r in found] == [(nth_day(8), nth_day(9)), (nth_day(11), nth_day(12))]


def test_a_day_that_doesnt_count_neither_raises_nor_ends_a_run():
    counts = history([250] * 8 + [1500, 2000, 1500])
    quiet = counts["day"] == nth_day(9)
    counts.loc[quiet, "responses"] = 10
    counts.loc[quiet, "logged"] = 10
    assert [(r["since"], r["on"]) for r in replayed(counts, ratio=3, days=2)] == [(nth_day(8), nth_day(10))]


def test_a_raised_day_stays_in_the_baseline_so_a_rise_that_lasts_becomes_the_level():
    # Twelve days at 1500 make 1500 the level, so after one quiet day 1500 is no rise again. Held
    # out of the baseline, the raised days would leave it at 250 and report a second rise.
    found = replayed(history([250] * 6 + [1500] * 12 + [250] + [1500] * 2), ratio=3)
    assert [r["since"] for r in found] == [nth_day(6)]


def test_a_baseline_under_100_tokens_a_response_raises_nothing_and_at_100_it_can():
    # A model that barely thinks would alarm on a handful of tokens; the owner's lowest counted
    # day is 144, so the floor silences nothing real.
    assert replayed(history([99] * 8 + [800])) == []
    assert [r["since"] for r in replayed(history([100] * 8 + [800]))] == [nth_day(8)]
    assert replayed(history([0] * 8 + [500])) == []


def test_only_days_before_today_are_judged():
    counts = history([250] * 8 + [1500])
    assert thinking_rises(counts, new_state(), date.fromisoformat(nth_day(8))) == []


def test_a_rise_older_than_two_weeks_is_neither_reported_nor_recorded():
    # The first check after an upgrade reads 90 days; a rise finished long ago is history.
    counts = history([250] * 8 + [1500] + [250] * 3)
    state = new_state()
    assert thinking_rises(counts, state, date.fromisoformat(nth_day(8)) + timedelta(days=15)) == []
    assert state["thinking_rises"] == []
    assert thinking_rises(counts, state, date.fromisoformat(nth_day(8)) + timedelta(days=14)) != []
