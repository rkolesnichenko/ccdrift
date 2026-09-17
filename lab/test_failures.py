"""The G10/G11 gates: how the failure rules are judged on a history."""

import pandas as pd

from ccdrift.failures import FAILURE_DAY_COLUMNS, cut_short, failing_requests
from lab.failures import cut_rows, gate, plant, replay, request_rows
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


def test_planting_a_bad_day_makes_the_last_day_fail_or_cut_short():
    counts = counts_of([(0, 0)] * 7)
    assert int(plant(counts, requests=10)["requests"].iloc[-1]) == 10
    assert int(plant(counts, share=0.01)["truncated"].iloc[-1]) == 10
    assert replay(plant(counts, requests=10), failing_requests) == [nth_day(6)]
    assert replay(plant(counts, share=0.05), cut_short) == [nth_day(6)]


def test_the_grid_rows_say_how_often_each_setting_alerts_and_whether_it_catches_a_burst():
    rows = request_rows(counts_of([(0, 0)] * 6 + [(4, 0)]), days=7)
    tight = next(row for row in rows if (row["floor"], row["ratio"]) == (5, 2))
    loose = next(row for row in rows if (row["floor"], row["ratio"]) == (3, 2))
    assert (tight["alerts"], tight["catches"]) == (0, True)
    assert (loose["alerts"], loose["catches"]) == (1, True)


def test_a_setting_passes_when_it_stays_within_the_budget_and_catches_the_burst():
    rows = cut_rows(counts_of([(0, 0)] * 6 + [(0, 1)]), days=7)
    passed, notes = gate(rows, {"floor": 5, "share": 0.005})
    assert passed, notes
    assert "0 alert(s) over 7 days" in notes[0]
    # Two bad days far enough apart to alert twice are more than 40 days of history allow.
    often = [(0, 0)] * 6 + [(0, 20)] + [(0, 0)] * 23 + [(0, 20)] + [(0, 0)] * 9
    noisy, notes = gate(cut_rows(counts_of(often), days=40), {"floor": 5, "share": 0.005})
    assert not noisy and "2 alert(s) over 40 days" in notes[0]
