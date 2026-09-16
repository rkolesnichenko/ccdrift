"""The weekly digest: when it is due and what it says."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from ccdrift.digest import digest_due, weekly_digest
from ccdrift.state import new_state

EEST = timezone(timedelta(hours=3))
MONDAY = datetime(2026, 9, 21, 9, 30, tzinfo=EEST)  # ISO week 2026-W39


@pytest.mark.parametrize("now, digest_week, runs, expected", [
    (MONDAY, None, ["2026-09-18"], date(2026, 9, 14)),
    (MONDAY.replace(hour=8, minute=59), None, ["2026-09-18"], None),        # before 09:00
    (MONDAY, "2026-W39", ["2026-09-18"], None),                             # already sent this week
    (MONDAY, None, ["2026-09-21"], None),                                   # no run before this week
    (MONDAY + timedelta(days=2), "2026-W38", ["2026-09-19"], date(2026, 9, 14)),  # Monday was missed
])
def test_the_digest_is_due_on_the_first_run_after_monday_9_of_a_week(now, digest_week, runs, expected):
    state = {**new_state(), "runs": runs, **({"digest_week": digest_week} if digest_week else {})}
    assert digest_due(state, now) == expected


def week_turns(haiku=0):
    """70 judged responses over Sep 14-20, 10 a day: 2.1.261 for 3 days, then 2.1.270;
    every one a new-prompt turn reading 99% from the cache but one that misses."""
    rows = [{"day": (date(2026, 9, 14) + timedelta(days=d)).isoformat(), "version": "2.1.261" if d < 3 else "2.1.270",
             "prompt_within_ttl": True, "cache_read_ratio": 0.99, "is_miss": False, "is_haiku": 0.0}
            for d in range(7) for _ in range(10)]
    rows[5].update(cache_read_ratio=0.0, is_miss=True)
    for row in rows[:haiku]:
        row["is_haiku"] = 1.0
    return pd.DataFrame(rows)


def test_a_quiet_week_reads_as_one_line():
    state = {**new_state(), "runs": [(date(2026, 9, 14) + timedelta(days=d)).isoformat() for d in range(7)],
             "settings": [{"reported_on": "2026-09-16"}, {"reported_on": "2026-09-10"}]}
    assert weekly_digest(week_turns(), state, date(2026, 9, 14)) == (
        "Week of 09-14: 70 responses on 2.1.261–2.1.270; cache ratio 0.976 (1.4% misses); no Haiku; "
        "no open incidents; 1 setting change; check ran on 7 of 7 days.")


def test_the_digest_counts_haiku_open_incidents_and_missed_runs():
    state = {**new_state(), "runs": ["2026-09-15", "2026-09-17"], "incidents": [{"status": "open"}],
             "settings": [{"reported_on": "2026-09-15"}, {"reported_on": "2026-09-19"}]}
    assert weekly_digest(week_turns(haiku=1), state, date(2026, 9, 14)) == (
        "Week of 09-14: 70 responses on 2.1.261–2.1.270; cache ratio 0.976 (1.4% misses); Haiku 1.4% of responses; "
        "1 open incident; 2 setting changes; check ran on 2 of 7 days.")


def test_a_week_without_responses_says_so():
    assert weekly_digest(week_turns().iloc[0:0], new_state(), date(2026, 9, 14)) == (
        "Week of 09-14: no responses; no open incidents; no setting changes; check ran on 0 of 7 days.")
