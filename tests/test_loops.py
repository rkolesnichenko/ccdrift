"""Tool-loop cache misses: the turns judged and the alarms that count."""

import pandas as pd

from ccdrift.early import alarm_runs, clamp_rate
from ccdrift.loops import LoopSetting, loop_turns, qualifying_alarms


def test_loop_turns_keep_each_streams_tool_loop_turns_outside_sdk_sessions_in_time_order():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-09-01T10:02Z", "2026-09-01T10:01Z", "2026-09-01T10:03Z",
                                     "2026-09-01T10:04Z", "2026-09-01T10:05Z"]),
        "day": ["2026-09-01"] * 5, "main_thread": [True, True, False, True, True],
        "entrypoint": ["cli", "cli", "cli", "sdk-py", "cli"], "loop_turn": [True, True, True, True, False],
        "is_loop_miss": [True, False, False, False, False], "session_id": ["a", "b", "c", "d", "e"],
        "version": ["2.1.280"] * 5, "cache_creation": [100.0] * 5})
    assert loop_turns(df, "main")["session_id"].tolist() == ["b", "a"]
    assert loop_turns(df, "subagent")["session_id"].tolist() == ["c"]


def test_a_frame_without_loop_columns_has_no_loop_turns():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-09-01T10:00Z"]), "day": ["2026-09-01"],
                       "main_thread": [True]})
    assert loop_turns(df, "main").empty


def test_the_usual_rate_can_be_held_below_a_lower_p1():
    # Against a usual rate above p1 every miss would lower the sum and no rise could show.
    # Held at 1%, each miss adds ln(0.02/0.01) = 0.69, so the sixth passes h = 4.
    misses = [True] * 6
    assert alarm_runs(misses, base_rate=0.3, h=4, p1=0.02) == []
    assert alarm_runs(misses, base_rate=0.3, h=4, p1=0.02, high=0.01) == [(0, 5)]
    assert clamp_rate(0.3, high=0.01) == 0.01


def every_tenth_turn_misses(sessions):
    """60 turns, every tenth a miss, with the session of each turn."""
    return pd.DataFrame({"is_loop_miss": [(i + 1) % 10 == 0 for i in range(60)], "session_id": sessions})


def test_an_alarm_counts_only_when_its_misses_come_from_enough_sessions():
    # At p1 = 5% against 0.2%, a miss adds ln(25) = 3.22 and the 9 hits before the next
    # take 0.44, so every second miss passes h = 4: rises of turns 9-19, 29-39 and 49-59.
    one_session = every_tenth_turn_misses(["a"] * 60)
    two_sessions = every_tenth_turn_misses(["a" if (i // 10) % 2 == 0 else "b" for i in range(60)])
    rises = [(9, 19), (29, 39), (49, 59)]
    assert qualifying_alarms(one_session, 0.002, LoopSetting(p1=0.05, h=4, min_sessions=1)) == rises
    assert qualifying_alarms(one_session, 0.002, LoopSetting(p1=0.05, h=4, min_sessions=2)) == []
    assert qualifying_alarms(two_sessions, 0.002, LoopSetting(p1=0.05, h=4, min_sessions=2)) == rises
