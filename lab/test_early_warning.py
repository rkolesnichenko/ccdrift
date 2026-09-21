"""The G3 spike on hand-made new-prompt turns."""

import random

import pandas as pd

from lab.early_warning import (MAX_FALSE_PER_WEEK, base_rate, evaluate, false_alarm_rate,
                               prompt_turns, rate_sweep)
from tests.helpers import nth_day


def turns_frame(days=40, per_day=60, incident=(), incident_rate=0.07, seed=3):
    """CLI main-thread new-prompt turns: one miss in 200 on clean days, `incident_rate` on incident days."""
    rng = random.Random(seed)
    rows = []
    for d in range(days):
        day = nth_day(d)
        for k in range(per_day):
            rate = incident_rate if day in incident else 0.005
            rows.append({"timestamp": pd.Timestamp(f"{day}T08:00:00Z") + pd.Timedelta(minutes=10 * k), "day": day,
                         "main_thread": True, "entrypoint": "cli", "prompt_within_ttl": True,
                         "cache_read": 900.0, "cache_creation": 100.0, "is_miss": rng.random() < rate})
    return pd.DataFrame(rows)


def test_prompt_turns_keep_cli_main_thread_new_prompts_in_time_order():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-09-01T10:00Z", "2026-09-01T09:00Z", "2026-09-01T11:00Z",
                                                    "2026-09-01T12:00Z"]),
                       "day": ["2026-09-01"] * 4, "main_thread": [True, True, False, True],
                       "entrypoint": ["cli", "cli", "cli", "sdk-py"], "prompt_within_ttl": [True] * 4,
                       "cache_read": [900.0] * 4, "cache_creation": [100.0] * 4,
                       "is_miss": [False, True, False, False]})
    assert prompt_turns(df)["is_miss"].tolist() == [True, False]


def test_base_rate_uses_the_14_days_before_outside_incidents():
    turns = prompt_turns(turns_frame(days=20, incident=[nth_day(i) for i in range(10, 13)], incident_rate=1.0))
    rate = base_rate(turns, nth_day(15), excluded=(nth_day(10), nth_day(12)))
    assert rate is not None and rate < 0.02


def test_evaluate_finds_a_threshold_that_passes_on_a_clear_regression():
    incident = [nth_day(i) for i in range(20, 26)]
    table = evaluate(turns_frame(incident=incident), (incident[0], incident[-1]))
    assert list(table.columns) == ["h", "false_alarms", "weeks", "rate", "planted_median",
                                   "planted_caught", "planted_runs", "real_alarm", "passes"]
    assert table["passes"].any()


def test_the_false_alarm_rate_falls_as_the_threshold_rises():
    # Seeded, so this pins the measurement the gate reads rather than resampling it.
    rates = [false_alarm_rate(h, 0.0043) for h in (2, 3, 4, 5, 6)]
    assert rates == sorted(rates, reverse=True)
    assert rates[-1] < rates[0]


def test_the_bar_separates_the_shipped_threshold_from_the_noisy_ones():
    # What the gate now decides on: h = 4 clears the bar on every usual rate the check
    # runs at, h = 3 clears none of them. A single alarm on a short corpus decides nothing.
    sweep = rate_sweep().set_index("h")
    assert (sweep.loc[4] <= MAX_FALSE_PER_WEEK).all()
    assert (sweep.loc[3] > MAX_FALSE_PER_WEEK).all()
