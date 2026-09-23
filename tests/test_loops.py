"""Tool-loop cache misses: the turns judged and the alarms that count."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd

import ccdrift.loops
from ccdrift.early import alarm_runs, clamp_rate
from ccdrift.loops import COUNT_COLUMNS, LoopSetting, loop_counts, loop_turns, loop_warning, qualifying_alarms
from ccdrift.state import new_state
from ccdrift.texts import loop_message
from tests.helpers import nth_day


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


def test_tool_loop_counts_of_a_table_without_loop_turns_are_empty():
    counts = loop_counts(pd.DataFrame(), date(2026, 9, 4))
    assert counts.empty and list(counts.columns) == COUNT_COLUMNS


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


NOW = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
SETTING = LoopSetting(p1=0.02, h=3, min_sessions=1)
# One miss in the 1,400 turns of the 14 days before the week, then 4 in a row on Sep 21 from 10:00.
RISING = {(2, 0), (20, 24), (20, 25), (20, 26), (20, 27)}


def loop_frame(misses, stream="main", days=21, per_day=100, sessions=4):
    """CLI tool-loop turns of `stream` from Sep 1, every 5 minutes from 08:00 UTC, in
    `sessions` sessions a day, on 2.1.270 and, the last day, on 2.1.280; `misses` holds
    the (day, turn) pairs that miss, each writing 200,000 tokens (1,000 otherwise)."""
    return pd.DataFrame([
        {"timestamp": pd.Timestamp(f"{nth_day(d)}T08:00:00Z") + pd.Timedelta(minutes=5 * k), "day": nth_day(d),
         "main_thread": stream == "main", "entrypoint": "cli", "loop_turn": True, "is_loop_miss": (d, k) in misses,
         "session_id": f"s{d}-{k % sessions}", "version": "2.1.280" if d == days - 1 else "2.1.270",
         "cache_creation": 200_000.0 if (d, k) in misses else 1_000.0}
        for d in range(days) for k in range(per_day)])


def test_tool_loop_misses_rising_today_warn_once_saying_how_many_missed_and_what_they_rewrote():
    # Each miss adds ln(0.02/0.002) = 2.30, so the misses at 10:05 and 10:15 pass h = 3;
    # the second alarm follows the first at once, so the rise runs from 10:00.
    state = new_state()
    warning = loop_warning(loop_frame(RISING), "main", state, NOW, SETTING)
    assert warning == {"stream": "main", "at": "2026-09-21T10:15:00+00:00", "since": "2026-09-21T10:00:00+00:00",
                       "misses": 4, "turns": 4, "sessions": 4, "base_rate": 0.0007, "tokens": 800_000,
                       "versions": ["2.1.280 (since 09-21)"], "reported_on": "2026-09-21"}
    assert loop_message(warning, NOW) == (
        "4 of the last 4 tool-loop turns missed the cache (usually 0.07%), since 10:00, in 4 sessions, rewriting "
        "~800k tokens, on Claude Code 2.1.280 (since 09-21). `ccdrift report` shows whether it lasts.")
    assert state["loop_warnings"] == [warning]
    assert loop_warning(loop_frame(RISING), "main", state, NOW, SETTING) is None


def test_a_rise_that_mixes_hits_and_misses_counts_its_turns_but_only_the_misses_sessions_and_tokens():
    # The usual 0.07% is held at 0.2%, so a miss adds ln(0.02/0.002) = 2.30 and a hit ln(0.98/0.998) = -0.018:
    # the miss at 10:00 (turn 24, session s20-0) stands at 2.30, the hit at 10:05 (s20-1) at 2.28,
    # and the miss at 10:10 (turn 26, s20-2) at 4.59, past h = 3. The rise runs over those 3 turns,
    # whose 2 misses come from 2 sessions and wrote 400,000 tokens; the hit's 1,000 don't count.
    warning = loop_warning(loop_frame({(2, 0), (20, 24), (20, 26)}), "main", new_state(), NOW, SETTING)
    assert warning == {"stream": "main", "at": "2026-09-21T10:10:00+00:00", "since": "2026-09-21T10:00:00+00:00",
                       "misses": 2, "turns": 3, "sessions": 2, "base_rate": 0.0007, "tokens": 400_000,
                       "versions": ["2.1.280 (since 09-21)"], "reported_on": "2026-09-21"}


def test_a_main_thread_warning_doesnt_silence_subagents():
    state = new_state()
    loop_warning(loop_frame(RISING), "main", state, NOW, SETTING)
    warning = loop_warning(loop_frame(RISING, stream="subagent"), "subagent", state, NOW, SETTING)
    assert (warning["stream"], warning["misses"]) == ("subagent", 4)
    assert loop_warning(loop_frame(RISING), "subagent", new_state(), NOW, SETTING) is None


def test_an_alarm_more_than_a_day_old_is_not_news():
    assert loop_warning(loop_frame(RISING), "main", new_state(), NOW + timedelta(days=1), SETTING) is None


def test_too_few_turns_before_the_week_give_no_warning():
    # 70 turns a day: 980 in the 14 days before the week.
    assert loop_warning(loop_frame(RISING, per_day=70), "main", new_state(), NOW, SETTING) is None


def test_misses_from_one_session_dont_warn_when_the_setting_needs_two():
    one_session = loop_frame(RISING, sessions=1)
    assert loop_warning(one_session, "main", new_state(), NOW, SETTING._replace(min_sessions=2)) is None
    assert loop_warning(one_session, "main", new_state(), NOW, SETTING)["sessions"] == 1


def test_a_stream_whose_gate_failed_gets_no_warning(monkeypatch):
    monkeypatch.setitem(ccdrift.loops.LOOP_SETTINGS, "main", None)
    assert loop_warning(loop_frame(RISING), "main", new_state(), NOW) is None


def test_the_message_says_one_session_and_leaves_out_unknown_versions():
    warning = {"stream": "subagent", "at": "2026-09-20T23:50:00+00:00", "since": "2026-09-20T23:40:00+00:00",
               "misses": 1, "turns": 3, "sessions": 1, "base_rate": 0.0018, "tokens": 182_000, "versions": [],
               "reported_on": "2026-09-21"}
    assert loop_message(warning, NOW) == (
        "1 of the last 3 subagent tool-loop turns missed the cache (usually 0.18%), since 09-20 23:40, in 1 session, "
        "rewriting ~180k tokens. `ccdrift report` shows whether it lasts.")
