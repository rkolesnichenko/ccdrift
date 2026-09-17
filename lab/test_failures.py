"""The G10/G11 gates: how the failure rules are judged on a history."""

import pandas as pd

from ccdrift.failures import FAILURE_DAY_COLUMNS, cut_short, failing_requests
from lab.failures import cut_rows, gate, plant, plant_days, replay, request_rows
from tests.helpers import nth_day


def counts_of(rows, responses=1000):
    """rows: (failed requests, responses cut short) per day, one day each from Sep 1. A
    day holds `responses` responses, enough that a planted 1% is over the floor."""
    return pd.DataFrame([{"day": nth_day(i), "responses": responses, "requests": requests, "truncated": cut,
                          "refused": 0, "overloaded": requests, "stream": 0, "other": 0, "retry": 0, "slept": 0}
                         for i, (requests, cut) in enumerate(rows)], columns=FAILURE_DAY_COLUMNS)


def test_replay_judges_each_day_as_the_check_would_the_morning_after():
    counts = counts_of([(0, 0)] * 6 + [(9, 0)])
    assert replay(counts, failing_requests) == [nth_day(6)]
    assert replay(counts_of([(0, 0)] * 7), failing_requests) == []


def test_planting_a_bad_day_makes_the_day_it_names_fail_or_cut_short():
    counts = counts_of([(0, 0)] * 7)
    planted = plant(counts, nth_day(5), requests=10)
    assert [int(value) for value in planted["requests"]] == [0] * 5 + [10, 0]
    assert int(plant(counts, nth_day(6), share=0.01)["truncated"].iloc[-1]) == 10
    assert replay(planted, failing_requests) == [nth_day(5)]
    assert replay(plant(counts, nth_day(6), share=0.05), cut_short) == [nth_day(6)]
    # Planting only ever makes a day worse: a day already cut short more than the share
    # asks for keeps its own count.
    already = counts_of([(0, 0)] * 6 + [(0, 50)])
    assert int(plant(already, nth_day(6), share=0.01)["truncated"].iloc[-1]) == 50


def test_a_burst_is_planted_on_the_last_judgeable_days_oldest_first():
    # A rule judges a day only against the active days in the 14 before it, and only
    # when there are at least 5 of them, so the first days of a corpus can carry no plant.
    counts = counts_of([(0, 0)] * 12)
    assert plant_days(counts) == [nth_day(i) for i in range(7, 12)]
    assert plant_days(counts, how_many=2) == [nth_day(10), nth_day(11)]
    # Days too quiet to be active are neither planted on nor counted among the days before.
    assert plant_days(counts_of([(0, 0)] * 8, responses=10)) == []
    gapped = counts_of([(0, 0)] * 26)
    assert plant_days(gapped[gapped["day"].isin([nth_day(i) for i in range(5)] + [nth_day(25)])]) == []


def test_the_grid_rows_say_how_often_each_setting_alerts_and_how_many_bursts_it_catches():
    rows = request_rows(counts_of([(0, 0)] * 6 + [(4, 0)]), days=7)
    tight = next(row for row in rows if (row["floor"], row["ratio"]) == (5, 2))
    loose = next(row for row in rows if (row["floor"], row["ratio"]) == (3, 2))
    assert (tight["alerts"], tight["caught"], tight["plants"]) == (0, 2, 2)
    assert (loose["alerts"], loose["caught"], loose["plants"]) == (1, 2, 2)


def test_a_setting_only_catches_when_the_planted_day_itself_alerts():
    # An early real spike (20 failed requests) sits far enough before every planted day
    # that a setting's window over the days before one never sees it. A setting whose
    # floor the spike alone clears must not be credited with catching a plant just
    # because the spike alerted somewhere earlier in the history.
    rows = [(0, 0)] * 6 + [(20, 0)] + [(0, 0)] * 20
    result = request_rows(counts_of(rows), days=27)
    floor_12 = next(row for row in result if (row["floor"], row["ratio"]) == (12, 2))
    assert (floor_12["caught"], floor_12["plants"], floor_12["alerts"]) == (0, 5, 1)
    assert floor_12["passes"] is False
    floor_5 = next(row for row in result if (row["floor"], row["ratio"]) == (5, 2))
    assert (floor_5["caught"], floor_5["plants"]) == (5, 5)


def test_a_setting_passes_when_it_stays_within_the_budget_and_catches_every_burst():
    rows = cut_rows(counts_of([(0, 0)] * 6 + [(0, 1)]), days=7)
    passed, notes = gate(rows, {"floor": 5, "share": 0.005})
    assert passed, notes
    assert "0 alert(s) over 7 days" in notes[0]
    assert "planted bursts caught: 2 of 2" in notes[0]
    # Two bad days far enough apart to alert twice are more than 40 days of history allow.
    often = [(0, 0)] * 6 + [(0, 20)] + [(0, 0)] * 23 + [(0, 20)] + [(0, 0)] * 9
    noisy, notes = gate(cut_rows(counts_of(often), days=40), {"floor": 5, "share": 0.005})
    assert not noisy and "2 alert(s) over 40 days" in notes[0]
