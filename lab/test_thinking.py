"""G16: the thinking-rise rule, its metric, and the gate that judges it."""

from datetime import date, timedelta

import pandas as pd

from lab.thinking import COUNT_COLUMNS, counted, thinking_counts, thinking_rises
from tests.helpers import nth_day


def history(levels, model="claude-opus-5", responses=100, start=0):
    """Day counts for `model`, one day per level from nth_day(start): every response logs a
    count, and the day's thinking is level x responses. None is a day with no responses."""
    return pd.DataFrame([{"model": model, "day": nth_day(start + i), "responses": responses, "logged": responses,
                          "thinking": level * responses}
                         for i, level in enumerate(levels) if level is not None], columns=COUNT_COLUMNS)


def rises(counts, ratio=3, days=2, today=None):
    today = today or date.fromisoformat(max(counts["day"].astype(str))) + timedelta(days=1)
    return thinking_rises(counts, {}, today, ratio=ratio, days=days)


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


def test_a_run_of_raised_days_is_reported_once_on_its_last_day():
    counts = history([250] * 8 + [1500, 1800, 2000])
    state = {}
    today = date.fromisoformat(nth_day(11))
    [rise] = thinking_rises(counts, state, today, ratio=3, days=2)
    assert rise == {"model": "claude-opus-5", "since": nth_day(8), "on": nth_day(9), "median": 250.0,
                    "levels": [1500.0, 1800.0], "reported_on": nth_day(11)}
    assert thinking_rises(counts, state, today, ratio=3, days=2) == []


def test_a_shorter_run_is_not_reported():
    assert rises(history([250] * 8 + [1500, 250, 250]), days=2) == []


def test_a_flat_history_reports_nothing():
    assert rises(history([144, 388, 250, 300, 211, 246, 335, 381, 192, 241]), ratio=2, days=1) == []


def test_a_model_with_too_few_baseline_days_is_not_judged():
    # Only the last 2000 has 5 counted days before it, so no run of 2 judged days can form.
    assert rises(history([250] * 3 + [2000, 2000, 2000]), days=2) == []


def test_two_models_are_judged_apart_so_a_switch_to_a_model_that_thinks_more_is_not_a_rise():
    counts = pd.concat([history([250] * 10), history([2000] * 6, model="claude-opus-5-5", start=10)],
                       ignore_index=True)
    assert rises(counts, ratio=2, days=1) == []


def test_a_second_run_after_a_day_that_isnt_raised_is_reported_again():
    found = rises(history([250] * 8 + [1500, 1500, 250, 1500, 1500]), days=2)
    assert [(r["since"], r["on"]) for r in found] == [(nth_day(8), nth_day(9)), (nth_day(11), nth_day(12))]


def test_a_day_that_doesnt_count_neither_raises_nor_ends_a_run():
    counts = history([250] * 8 + [1500, 2000, 1500])
    quiet = counts["day"] == nth_day(9)
    counts.loc[quiet, "responses"] = 10
    counts.loc[quiet, "logged"] = 10
    assert [(r["since"], r["on"]) for r in rises(counts, days=2)] == [(nth_day(8), nth_day(10))]


def test_a_raised_day_stays_in_the_baseline_so_a_rise_that_lasts_becomes_the_level():
    # Twenty days at 1500 make 1500 the level, so after one quiet day 1500 is no rise again. Held
    # out of the baseline, the raised days would leave it at 250 and report a second rise.
    found = rises(history([250] * 6 + [1500] * 20 + [250] + [1500] * 2), ratio=3, days=1)
    assert [r["since"] for r in found] == [nth_day(6)]


def test_a_baseline_of_no_thinking_raises_nothing():
    assert rises(history([0] * 8 + [500, 500]), ratio=2, days=1) == []


def test_only_days_before_today_are_judged():
    counts = history([250] * 8 + [1500, 1500])
    assert thinking_rises(counts, {}, date.fromisoformat(nth_day(9)), ratio=3, days=2) == []
