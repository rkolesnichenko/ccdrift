"""The weekly digest: when it is due and what it says."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from ccdrift.digest import digest_due, weekly_digest
from ccdrift.state import new_state
from ccdrift.texts import digest_text

EEST = timezone(timedelta(hours=3))
MONDAY = datetime(2026, 9, 21, 9, 30, tzinfo=EEST)  # ISO week 2026-W39
SYDNEY = timezone(timedelta(hours=10))


@pytest.mark.parametrize("now, digest_week, runs, expected", [
    (MONDAY, None, ["2026-09-18"], date(2026, 9, 14)),
    (MONDAY.replace(hour=8, minute=59), None, ["2026-09-18"], None),        # before 09:00
    (MONDAY, "2026-W39", ["2026-09-18"], None),                             # already sent this week
    (MONDAY, None, ["2026-09-21"], None),                                   # no run before this week
    (MONDAY + timedelta(days=2), "2026-W38", ["2026-09-19"], date(2026, 9, 14)),  # Monday was missed
    # The week is summed by UTC day, and at 09:30 in Sydney Sunday has half an hour left in UTC.
    (MONDAY.replace(tzinfo=SYDNEY), None, ["2026-09-18"], None),
    (MONDAY.replace(hour=10, minute=0, tzinfo=SYDNEY), None, ["2026-09-18"], date(2026, 9, 14)),
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
        "no tool-loop turns, no subagent loop turns; no open incidents; 1 setting change; no new fields; "
        "check ran on 7 of 7 days.")


def test_the_digest_counts_haiku_open_incidents_and_missed_runs():
    state = {**new_state(), "runs": ["2026-09-15", "2026-09-17"], "incidents": [{"status": "open"}],
             "settings": [{"reported_on": "2026-09-15"}, {"reported_on": "2026-09-19"}]}
    assert weekly_digest(week_turns(haiku=1), state, date(2026, 9, 14)) == (
        "Week of 09-14: 70 responses on 2.1.261–2.1.270; cache ratio 0.976 (1.4% misses); Haiku 1.4% of responses; "
        "no tool-loop turns, no subagent loop turns; 1 open incident; 2 setting changes; no new fields; "
        "check ran on 2 of 7 days.")


def test_the_digest_counts_the_weeks_tool_loop_misses():
    loops = pd.DataFrame({"loop_turns": [900, 980, 5000], "loop_misses": [1, 1, 9],
                          "subagent_loop_turns": [6400, 7000, 0], "subagent_loop_misses": [30, 6, 0]},
                         index=["2026-09-14", "2026-09-20", "2026-09-21"])
    assert "; no Haiku; tool-loop misses 2 of 1,880, subagent 36 of 13,400; " in weekly_digest(
        week_turns(), new_state(), date(2026, 9, 14), loops)


def test_a_week_without_responses_says_so():
    assert weekly_digest(week_turns().iloc[0:0], new_state(), date(2026, 9, 14)) == (
        "Week of 09-14: no responses; no open incidents; no setting changes; no new fields; "
        "check ran on 0 of 7 days.")


def test_the_weekly_summary_counts_the_new_fields_reported_that_week():
    state = new_state()
    state["new_fields"].append({"paths": ["advisorModel", "perTurnEffort"], "version": "2.1.276",
                                "share": 1.0, "responses": 120, "reported_on": "2026-09-09"})
    assert "2 new fields" in weekly_digest(pd.DataFrame(), state, date(2026, 9, 7))


def test_the_weekly_summary_says_no_new_fields_when_none_arrived():
    assert "no new fields" in weekly_digest(pd.DataFrame(), new_state(), date(2026, 9, 7))


def test_the_digest_line_is_worded_from_plain_numbers_the_status_path_can_import():
    # texts imports nothing heavy; the counting stays with pandas in digest.week_summary.
    summary = {"week_start": "2026-09-14", "responses": 70, "versions": ["2.1.261", "2.1.270"],
               "prompts": (0.976, 1 / 70), "haiku": 0.0, "loops": {"loop": (1880, 2), "subagent_loop": (0, 0)},
               "failures": (3, 1), "open_incidents": 1, "setting_changes": 0, "new_fields": 2, "ran": 6}
    assert digest_text(summary) == (
        "Week of 09-14: 70 responses on 2.1.261–2.1.270; cache ratio 0.976 (1.4% misses); no Haiku; "
        "tool-loop misses 2 of 1,880, no subagent loop turns; 3 failed requests, 1 response cut short; "
        "1 open incident; no setting changes; 2 new fields; check ran on 6 of 7 days.")
