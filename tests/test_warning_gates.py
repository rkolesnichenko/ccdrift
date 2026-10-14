"""The early warnings' gates, measured on hand-made turns."""

import functools

import pandas as pd
import pytest

from ccdrift.loops import LoopSetting
from ccdrift.warning_gates import (alarms_per_turn, loop_alarm_days, loop_base_rate, loop_plants, prompt_alarm_days,
                                   prompt_base_rate, prompt_plants)
from tests.helpers import nth_day


def prompt_frame(days, per_day=60, missed=lambda day, k: False):
    """New-prompt turns ten minutes apart from 08:00 UTC; `missed(day, k)` says which miss."""
    return pd.DataFrame([{"timestamp": pd.Timestamp(f"{nth_day(d)}T08:00:00Z") + pd.Timedelta(minutes=10 * k),
                          "day": nth_day(d), "is_miss": missed(d, k)} for d in range(days) for k in range(per_day)])


def loop_frame(days, per_day=100, missed=lambda day, k: False):
    """Main-thread tool-loop turns 30 s apart from 08:00 UTC, in one session a day."""
    return pd.DataFrame([{"timestamp": pd.Timestamp(f"{nth_day(d)}T08:00:00Z") + pd.Timedelta(seconds=30 * k),
                          "day": nth_day(d), "session_id": f"s{d}", "version": "2.1.280", "cache_creation": 1000.0,
                          "is_loop_miss": missed(d, k)} for d in range(days) for k in range(per_day)])


def test_the_usual_new_prompt_rate_leaves_out_every_span_it_is_given():
    turns = prompt_frame(20, per_day=20, missed=lambda day, k: day in (3, 9))
    both = [(nth_day(3), nth_day(3)), (nth_day(9), nth_day(9))]
    assert prompt_base_rate(turns, nth_day(15), both) == 0.0
    assert prompt_base_rate(turns, nth_day(15), both[:1]) == pytest.approx(20 / 260)
    assert prompt_base_rate(turns, nth_day(10), both) is None  # 160 turns


def test_new_prompt_alarms_are_dated_one_per_alarm_outside_every_span():
    # Day 22's first 10 turns miss against a usual rate of 0, held at the 0.2% floor: a miss
    # adds ln(0.05/0.002) = 3.22, so every second one passes h = 4, five alarms. Day 5 missing
    # throughout would put the usual rate at 60 of 840, held at 2.5%, where the 10 raise one;
    # day 25 missing throughout would alarm on its own. Both are left out.
    turns = prompt_frame(30, missed=lambda day, k: day in (5, 25) or (day == 22 and k < 10))
    spans = [(nth_day(5), nth_day(5)), (nth_day(25), nth_day(25))]
    assert prompt_alarm_days(turns, spans, 4.0) == [nth_day(22)] * 5
    assert prompt_alarm_days(turns, spans[1:], 4.0) == [nth_day(22)]


def test_new_prompt_plants_start_on_up_to_ten_days_with_a_usual_rate_spread_evenly():
    # A usual rate needs 200 turns in the 14 days before, so days 4 to 39 can start one.
    starts, runs = prompt_plants(prompt_frame(40), [], 4.0)
    assert starts == [nth_day(d) for d in (4, 8, 12, 16, 20, 23, 27, 31, 35, 39)]
    assert len(runs) == 50


def test_the_simulated_false_alarm_rate_is_per_turn_and_judged_against_the_clamped_rate():
    # 84 alarms over 200 streams of 1,000 turns at the owner's usual rate. A stream missing at
    # 5%, past the 2.5% the check clamps the usual rate to, alarms far more often.
    assert alarms_per_turn(4.0, 0.0043) == pytest.approx(84 / 200_000)
    assert alarms_per_turn(4.0, 0.05) > 20 * alarms_per_turn(4.0, 0.025)


def test_tool_loop_alarms_come_with_the_days_judged_those_with_a_usual_rate():
    # A usual rate needs 1,000 turns in the 14 days before a window's first day, so days 16
    # to 19 are judged. Two misses in a row against the 0.2% floor at p1 = 2% add
    # 2 × ln(0.02/0.002) = 4.6, past h = 3.
    turns = loop_frame(20, missed=lambda day, k: day == 18 and k in (10, 11))
    usual = functools.partial(loop_base_rate, turns)
    judged, alarms = loop_alarm_days(turns, usual, LoopSetting(0.02, 3.0, 1))
    assert judged == [nth_day(d) for d in range(16, 20)]
    assert alarms == [nth_day(18)]


def test_tool_loop_plants_start_on_every_day_with_a_usual_rate_up_to_ten():
    turns = loop_frame(20)
    starts, stretches = loop_plants(turns, functools.partial(loop_base_rate, turns))
    assert starts == [nth_day(d) for d in range(10, 20)]
    assert len(stretches) == 50
