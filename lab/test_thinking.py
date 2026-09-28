"""G16: the thinking-rise rule, its metric, and the gate that judges it."""

from datetime import date, timedelta

import pandas as pd

from lab.thinking import (COUNT_COLUMNS, counted, gate, level_lines, plant, plant_run, plant_starts, replay,
                          rows, thinking_counts, thinking_rises)
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


def test_the_plant_multiplies_only_the_next_three_counted_days_of_that_model():
    counts = pd.concat([history([250] * 10), history([400] * 10, model="claude-sonnet-5")], ignore_index=True)
    quiet = (counts["model"] == "claude-opus-5") & (counts["day"] == nth_day(6))
    counts.loc[quiet, "responses"] = 10
    counts.loc[quiet, "logged"] = 10
    assert plant_run(counts, "claude-opus-5", nth_day(5)) == [nth_day(5), nth_day(7), nth_day(8)]
    planted = plant(counts, "claude-opus-5", nth_day(5), 3)
    changed = planted[planted["thinking"] != counts["thinking"]]
    assert list(zip(changed["model"], changed["day"])) == [("claude-opus-5", nth_day(5)), ("claude-opus-5", nth_day(7)),
                                                           ("claude-opus-5", nth_day(8))]
    assert (changed["thinking"] == 3 * 250 * 100).all()


def test_a_plant_starts_on_a_counted_day_with_a_baseline_before_it_and_room_after_it():
    assert plant_starts(history([250] * 10), "claude-opus-5") == [nth_day(5), nth_day(6), nth_day(7)]


def test_the_replay_judges_each_day_the_morning_after_and_reports_each_rise_once():
    found = replay(history([250] * 8 + [1500, 1800, 2000]), ratio=3, days=2)
    assert [(r["since"], r["on"], r["reported_on"]) for r in found] == [(nth_day(8), nth_day(9), nth_day(10))]


def gate_history():
    """14 days swinging 200-300, and the same with a three-day episode at 2000 after them."""
    clean = history([200, 300] * 7)
    full = pd.concat([clean, history([2000] * 3, start=14)], ignore_index=True)
    return clean, full, [nth_day(14), nth_day(15), nth_day(16)]


def test_a_setting_passes_quiet_on_a_clean_history_catching_every_plant_and_the_episode():
    clean, full, episode = gate_history()
    [steady, jumpy] = rows(clean, full, episode, ratios=(2, 1.1), days_grid=(1,), plants=(1.5, 3))
    assert (steady["alarms"], steady["caught"][3], steady["starts"], steady["episode"]) == ([], 7, 7, nth_day(14))
    assert steady["caught"][1.5] < 7 and steady["passes"]
    assert jumpy["alarms"] and not jumpy["passes"]


def test_a_plant_the_clean_history_already_reports_is_not_a_catch():
    # At 1.1 every 300 day starts a rise, and a plant on the day after it only lengthens that
    # rise, which the clean replay reported first: no plant is caught.
    clean, full, episode = gate_history()
    [jumpy] = rows(clean, full, episode, ratios=(1.1,), days_grid=(1,), plants=(3,))
    assert (len(jumpy["alarms"]), jumpy["caught"][3], jumpy["starts"]) == (5, 0, 7)


def test_each_setting_names_the_plant_starts_it_missed():
    # Found in the G16 review: a count of catches hid which plants were missed, so a plant lost
    # to a run the clean history had already reported read like a rise the rule couldn't see.
    clean, full, episode = gate_history()
    [steady, jumpy] = rows(clean, full, episode, ratios=(2, 1.1), days_grid=(1,), plants=(1.5, 3))
    assert steady["missed"][3] == [] and len(steady["missed"][1.5]) == 7 - steady["caught"][1.5]
    assert jumpy["missed"][3] == [nth_day(day) for day in range(5, 12)]


def test_each_counted_day_is_printed_with_its_level_against_the_median_before_it():
    # Found in the G16 review: the findings quoted levels the gate never printed, two of them wrong.
    lines = level_lines(history([200, 300, 250, 250, 1000, 1500]), "claude-opus-5")  # median 250, mean 400
    assert lines[0] == "  2026-09-01 100 responses, level 200, not judged: 0 counted days before"
    assert lines[-1] == "  2026-09-06 100 responses, level 1500, 6.00x the median of the 5 counted days before"


def test_the_episode_counts_only_when_reported_by_its_third_counted_day():
    clean, _, _ = gate_history()
    full = pd.concat([clean, history([2000] * 4, start=14)], ignore_index=True)
    episode = [nth_day(day) for day in range(14, 18)]
    found = rows(clean, full, episode, ratios=(2,), days_grid=(3, 4), plants=(3,))
    assert [row["episode"] for row in found] == [nth_day(16), None]


def test_the_gate_names_the_passing_settings():
    clean, full, episode = gate_history()
    ok, notes = gate(rows(clean, full, episode, ratios=(2, 1.1), days_grid=(1,), plants=(3,)))
    assert ok and notes == ["passing settings: ratio=2 days=1"]


def test_the_gate_fails_when_no_setting_passes_and_names_the_closest():
    clean, full, episode = gate_history()
    # 1.1 alarms on every 300 day; 6 alarms on none and catches no plant of x3: fewer false
    # alarms come first, so 6 is the closest.
    ok, notes = gate(rows(clean, full, episode, ratios=(1.1, 6), days_grid=(1,), plants=(3,)))
    assert not ok
    assert notes == ["no setting passes; closest: ratio=6 days=1: 0 false alarm(s), plants of x3 or more caught "
                     "0 of 7, episode on 2026-09-15"]


def test_a_setting_that_raises_a_false_alarm_fails_even_catching_everything_else():
    clean = history([200, 300] * 6 + [800, 300])
    full = pd.concat([clean, history([2000] * 3, start=14)], ignore_index=True)
    [row] = rows(clean, full, [nth_day(14), nth_day(15), nth_day(16)], ratios=(2,), days_grid=(1,), plants=(3,))
    assert (row["alarms"], row["caught"][3], row["starts"], row["episode"]) == (["claude-opus-5 2026-09-13"], 7, 7,
                                                                               nth_day(14))
    assert not row["passes"]
