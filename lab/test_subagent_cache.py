"""G17: the gate on the subagent cache metric's cutoff."""

from datetime import date

import pandas as pd

from ccdrift.logs import judged_subagent_loops, parse_all
from lab.subagent_cache import (BAR, SMALL, first_check, flagged, gate, openings, plant, plant_starts, rows)
from tests.helpers import nth_day, tool_loop_days


def subagent_loops(tmp_path, days=20, **kw):
    """Subagent tool loops of `days` days through the parser, and each transcript's opening read."""
    tool_loop_days(tmp_path, days, subagent=True, **kw)
    responses = parse_all(tmp_path).responses
    return judged_subagent_loops(responses, date.fromisoformat(nth_day(days))).reset_index(drop=True), openings(responses)


def test_a_transcripts_opening_is_what_its_first_response_read(tmp_path):
    tool_loop_days(tmp_path, 2, subagent=True)
    assert openings(parse_all(tmp_path).responses).tolist() == [0.0, 0.0]


def test_a_plant_turns_the_share_asked_for_of_each_days_hits_into_prefix_reads_that_write_the_rest(tmp_path):
    # Day 1 already misses on its last 50 turns; the plant adds its share among the rest.
    loops, opening = subagent_loops(tmp_path, days=4, misses=50, miss_days=[1])
    planted = plant(loops, opening, [nth_day(1), nth_day(2)], 0.05, seed=0)
    misses = planted.groupby("day")["is_loop_miss"].sum()
    per_day = loops.groupby("day").size()
    assert misses.tolist() == [0, 50 + round(0.05 * per_day.iloc[1]), round(0.05 * per_day.iloc[2]), 0]
    hit = planted["is_loop_miss"]
    assert (planted.loc[hit, "loop_readback"] == 0.0).all()
    pd.testing.assert_series_equal(planted["cache_read"] + planted["cache_creation"],
                                   loops["cache_read"] + loops["cache_creation"])


def test_a_plant_repeats_with_its_seed_and_moves_with_another(tmp_path):
    loops, opening = subagent_loops(tmp_path, days=2)
    first, again = (plant(loops, opening, [nth_day(0)], 0.05, seed=1) for _ in range(2))
    other = plant(loops, opening, [nth_day(0)], 0.05, seed=2)
    assert first["is_loop_miss"].tolist() == again["is_loop_miss"].tolist()
    assert first["is_loop_miss"].tolist() != other["is_loop_miss"].tolist()


def test_plants_start_on_evenly_spread_days_with_room_before_and_after():
    days = [nth_day(i) for i in range(30)]
    assert plant_starts(days, starts=3) == [nth_day(7), nth_day(17), nth_day(26)]
    assert plant_starts(days[:12], starts=3) == [nth_day(7), nth_day(8)]


def test_clean_loops_raise_no_alarm_and_a_planted_drop_is_caught(tmp_path):
    loops, opening = subagent_loops(tmp_path)
    [row] = [r for r in rows(loops, opening, z_grid=(3.5,), plants=(BAR,), seeds=2, starts=2)
             if r["variant"] == "readback"]
    assert row["alarms"] == [] and row["caught"] == {BAR: 4} and row["runs"] == 4 and row["passes"]


def test_a_cutoff_passes_only_on_plants_of_the_bars_size(tmp_path):
    loops, opening = subagent_loops(tmp_path)
    # At 2.5 the planted 2% is caught; with no plant of BAR's size to judge, the cutoff doesn't pass.
    [row] = [r for r in rows(loops, opening, z_grid=(2.5,), plants=(SMALL,), seeds=1, starts=1)
             if r["variant"] == "readback"]
    assert row["caught"] == {SMALL: 1} and not row["passes"]


def test_a_cutoff_that_flags_the_logs_fails_whatever_it_catches(tmp_path):
    # Misses on the last three days are a drop the logs already hold, after the one plant.
    tool_loop_days(tmp_path, 20, subagent=True, misses=20, miss_days=[17, 18, 19])
    responses = parse_all(tmp_path).responses
    loops = judged_subagent_loops(responses, date.fromisoformat(nth_day(20))).reset_index(drop=True)
    assert flagged(loops, "readback", 3.5)[0] == [nth_day(17), nth_day(18), nth_day(19)]
    [row] = [r for r in rows(loops, openings(responses), z_grid=(3.5,), plants=(BAR,), seeds=1, starts=1)
             if r["variant"] == "readback"]
    assert row["caught"] == {BAR: 1} and not row["passes"]


def test_the_flags_the_logs_already_raise_catch_no_plant(tmp_path):
    # Misses on days 7 to 10 are a drop the logs already hold, on the days the one plant covers.
    tool_loop_days(tmp_path, 20, subagent=True, misses=20, miss_days=[7, 8, 9, 10])
    responses = parse_all(tmp_path).responses
    loops = judged_subagent_loops(responses, date.fromisoformat(nth_day(20))).reset_index(drop=True)
    assert flagged(loops, "readback", 3.5)[0] == [nth_day(7), nth_day(8), nth_day(9), nth_day(10)]
    [row] = [r for r in rows(loops, openings(responses), z_grid=(3.5,), plants=(BAR,), seeds=1, starts=1)
             if r["variant"] == "readback"]
    assert row["caught"] == {BAR: 0}


def row(z, passes, small=10):
    return {"variant": "readback", "z": z, "alarms": [], "deviant": [], "passes": passes,
            "caught": {SMALL: small, BAR: 10}, "runs": 10}


def test_the_shipped_cutoff_passes_only_with_every_stricter_one():
    assert gate([row(3.0, True), row(3.5, True), row(4.0, True, small=9)], 3.5)[0]
    ok, notes = gate([row(3.0, True), row(3.5, True), row(4.0, False)], 3.5)
    assert not ok and notes[-1] == "passing cutoffs: 3, 3.5"


def test_the_gate_names_the_strictest_cutoff_that_catches_every_small_plant():
    _, notes = gate([row(3.0, True), row(3.5, True), row(4.0, True, small=9)], 3.0)
    assert f"strictest cutoff catching every planted {SMALL:g}: 3.5" in notes


def test_the_gate_fails_when_ccdrift_ships_no_cutoff_or_one_outside_the_grid():
    assert gate([row(3.5, True)], None) == (False, ["ccdrift ships no cutoff", "passing cutoffs: 3.5"])
    assert gate([row(3.5, True)], 2.0) == (False, ["the shipped cutoff 2 isn't in the grid"])


def test_a_first_check_opens_only_flags_starting_within_its_last_two_weeks(tmp_path):
    tool_loop_days(tmp_path, 30, subagent=True, misses=20, miss_days=[26, 27, 28])
    loops = judged_subagent_loops(parse_all(tmp_path).responses, date.fromisoformat(nth_day(29))).reset_index(drop=True)
    assert first_check(loops, 3.5, date.fromisoformat(nth_day(29))) == [nth_day(26)]
    assert first_check(loops, 3.5, date.fromisoformat(nth_day(26 + 15))) == []
