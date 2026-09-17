"""The G8 and G9 spike on hand-made tool-loop turns."""

import pandas as pd
import pytest

from ccdrift.loops import LoopSetting, loop_turns
from lab.loop_cache import COLUMNS, base_rate, choose, evaluate
from tests.helpers import nth_day


def loop_frame(days=30, per_day=200, trickle_from=None):
    """Tool-loop turns 30 s apart from 08:00 UTC, `per_day` a day on the main thread and
    as many in subagents, spread over 4 sessions a day, one in 1,000 a miss. From day
    `trickle_from` on, every other subagent turn runs in one long session instead, and
    every 33rd of those misses."""
    rows = []
    for d in range(days):
        day = nth_day(d)
        for stream in ("main", "subagent"):
            for k in range(per_day):
                session, missed = f"{stream}-{day}-{k % 4}", (d * per_day + k) % 1000 == 999
                if stream == "subagent" and trickle_from is not None and d >= trickle_from and k % 2 == 0:
                    session, missed = "trickle", k % 66 == 2
                rows.append({"timestamp": pd.Timestamp(f"{day}T08:00:00Z") + pd.Timedelta(seconds=30 * k),
                             "day": day, "main_thread": stream == "main", "entrypoint": "cli", "loop_turn": True,
                             "is_loop_miss": missed, "session_id": session, "version": "2.1.280",
                             "cache_creation": 100_000.0})
    return pd.DataFrame(rows)


def test_the_usual_rate_needs_1000_turns_in_the_14_days_before():
    turns = loop_turns(loop_frame(days=20, per_day=100), "main")
    assert base_rate(turns, nth_day(15)) == pytest.approx(1 / 1400)  # days 1-14 hold the miss of day 9
    assert base_rate(turns, nth_day(5)) is None                       # 500 turns


def test_a_stream_passes_when_a_planted_rise_is_caught_without_false_alarms():
    table = evaluate(loop_frame())
    assert list(table.columns) == COLUMNS
    assert len(table) == 72
    assert table.groupby("stream")["passes"].any().to_dict() == {"main": True, "subagent": True}
    assert set(table["days_judged"]) == {19}


def test_misses_trickling_from_one_long_session_alarm_only_without_the_sessions_rule():
    table = evaluate(loop_frame(trickle_from=20)).set_index(["stream", "p1", "h", "min_sessions"])
    assert table.loc[("subagent", 0.05, 2, 1), "false_alarms"] > 0
    assert table.loc[("subagent", 0.05, 2, 2), "false_alarms"] == 0


def test_the_spike_picks_the_fastest_passing_combination_then_one_session_then_the_larger_h_and_p1():
    table = pd.DataFrame([
        {"stream": "main", "p1": 0.01, "h": 3, "min_sessions": 1, "planted_median": 90.0, "passes": False},
        {"stream": "main", "p1": 0.02, "h": 3, "min_sessions": 2, "planted_median": 120.0, "passes": True},
        {"stream": "main", "p1": 0.02, "h": 4, "min_sessions": 1, "planted_median": 120.0, "passes": True},
        {"stream": "main", "p1": 0.05, "h": 4, "min_sessions": 1, "planted_median": 120.0, "passes": True},
        {"stream": "subagent", "p1": 0.05, "h": 8, "min_sessions": 1, "planted_median": 60.0, "passes": False},
    ])
    assert choose(table, "main") == LoopSetting(p1=0.05, h=4.0, min_sessions=1)
    assert choose(table, "subagent") is None
