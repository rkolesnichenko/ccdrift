"""`ccdrift calibrate`: the plants, the measurement on a history and what it prints."""

import pytest

from ccdrift.calibrate import plant, plant_starts, plant_subagent, subagent_openings
from ccdrift.logs import parse_all
from tests.helpers import main_thread_days, nth_day, tool_loop_days


def main_turns(tmp_path, days=2):
    main_thread_days(tmp_path, [{}] * days)
    return parse_all(tmp_path).responses


def test_a_cache_plant_moves_its_share_of_each_new_prompts_cache_read_into_writes_from_its_start_on(tmp_path):
    turns = main_turns(tmp_path)
    planted = plant(turns, "cache", 0.2, start=60)
    first, second = planted.iloc[:60], planted.iloc[60:]
    assert (first["cache_read"] == 900).all() and (first["cache_creation"] == 100).all()
    judged = second[second["prompt_within_ttl"]]
    assert len(judged) == 59
    assert (judged["cache_read"] == 720).all() and (judged["cache_creation"] == 280).all()
    assert judged["prompt_cache_read_ratio"].round(3).eq(0.72).all()
    opener = second[~second["prompt_within_ttl"]]
    assert (opener["cache_read"] == 900).all()


def test_a_haiku_plant_moves_its_share_of_the_turns_from_its_start_on_to_haiku_the_same_way_for_a_seed(tmp_path):
    turns = main_turns(tmp_path)
    planted = plant(turns, "haiku", 0.25, seed=3, start=60)
    again = plant(turns, "haiku", 0.25, seed=3, start=60)
    assert planted["is_haiku"].iloc[:60].sum() == 0
    assert planted["is_haiku"].iloc[60:].sum() == 15
    assert planted["model"].tolist() == again["model"].tolist()


def test_a_plant_of_a_kind_it_doesnt_know_is_refused(tmp_path):
    with pytest.raises(ValueError, match="^effort$"):
        plant(main_turns(tmp_path), "effort", 0.2)


def test_a_subagent_plant_makes_its_share_of_a_days_loop_turns_read_only_what_the_transcript_opened_with(tmp_path):
    tool_loop_days(tmp_path, 2, per_day=100, subagent=True)
    responses = parse_all(tmp_path).responses
    loops = responses[responses["loop_turn"].astype(bool)]
    opening = subagent_openings(responses)
    assert opening.tolist() == [0.0, 0.0]
    planted = plant_subagent(loops, opening, [nth_day(1)], 0.1, seed=0)
    missed = planted[planted["is_loop_miss"]]
    assert set(missed["day"].astype(str)) == {nth_day(1)}
    assert len(missed) == round(0.1 * int((loops["day"].astype(str) == nth_day(1)).sum()))
    assert (missed["cache_read"] == 0).all()
    assert (planted["cache_read"] + planted["cache_creation"]).tolist() == (
        loops["cache_read"] + loops["cache_creation"]).tolist()


@pytest.mark.parametrize("count, starts", [(10, []), (11, [7]), (13, [7, 8, 9])], ids=["10-days", "11-days", "13-days"])
def test_a_start_leaves_seven_judged_days_before_it_and_three_after(count, starts):
    days = [nth_day(i) for i in range(count)]
    assert plant_starts(days) == [days[i] for i in starts]


def test_starts_are_ten_at_most_spread_evenly_from_the_first_that_fits_to_the_last():
    days = [nth_day(i) for i in range(40)]
    chosen = plant_starts(days)
    assert len(set(chosen)) == 10 and chosen == sorted(chosen)
    assert (chosen[0], chosen[-1]) == (days[7], days[36])
