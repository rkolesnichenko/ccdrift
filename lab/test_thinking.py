"""G16: the thinking-rise rule, its metric, and the gate that judges it."""

import pandas as pd

from ccdrift.thinking import COUNT_COLUMNS
from lab.thinking import gate, level_lines, plant, plant_run, plant_starts, replay, rows
from tests.helpers import nth_day


def history(levels, model="claude-opus-5", responses=100, start=0):
    """Day counts for `model`, one day per level from nth_day(start): every response logs a
    count, and the day's thinking is level x responses. None is a day with no responses."""
    return pd.DataFrame([{"model": model, "day": nth_day(start + i), "responses": responses, "logged": responses,
                          "thinking": level * responses}
                         for i, level in enumerate(levels) if level is not None], columns=COUNT_COLUMNS)


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


def test_the_gate_judges_the_setting_ccdrift_ships():
    clean, full, episode = gate_history()
    ok, notes = gate(rows(clean, full, episode, ratios=(2, 1.1), days_grid=(1,), plants=(3,)))
    assert ok and notes == ["ships ratio=2 days=1: 0 false alarm(s), plants of x3 or more caught 7 of 7, "
                            "episode on 2026-09-15"]


def test_the_gate_fails_the_shipped_setting_and_names_the_ones_that_pass():
    clean, full, episode = gate_history()
    ok, notes = gate(rows(clean, full, episode, ratios=(1.1, 2), days_grid=(1,), plants=(3,)),
                     {"ratio": 1.1, "days": 1})
    assert not ok
    assert notes == ["ships ratio=1.1 days=1: 5 false alarm(s) (claude-opus-5 2026-09-06, claude-opus-5 2026-09-08, "
                     "claude-opus-5 2026-09-10, claude-opus-5 2026-09-12, claude-opus-5 2026-09-14), plants of x3 or "
                     "more caught 0 of 7, episode missed", "passing settings: ratio=2 days=1"]


def test_the_gate_says_when_no_setting_passes_or_the_shipped_one_is_not_in_the_grid():
    clean, full, episode = gate_history()
    table = rows(clean, full, episode, ratios=(1.1, 6), days_grid=(1,), plants=(3,))
    assert gate(table, {"ratio": 6, "days": 1})[1][-1] == "no setting in the grid passes"
    assert gate(table) == (False, ["the setting ccdrift ships isn't in the grid"])


def test_a_setting_that_raises_a_false_alarm_fails_even_catching_everything_else():
    clean = history([200, 300] * 6 + [800, 300])
    full = pd.concat([clean, history([2000] * 3, start=14)], ignore_index=True)
    [row] = rows(clean, full, [nth_day(14), nth_day(15), nth_day(16)], ratios=(2,), days_grid=(1,), plants=(3,))
    assert (row["alarms"], row["caught"][3], row["starts"], row["episode"]) == (["claude-opus-5 2026-09-13"], 7, 7,
                                                                               nth_day(14))
    assert not row["passes"]
