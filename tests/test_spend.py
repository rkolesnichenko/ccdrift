"""Partitioning the history by where its tokens went."""

from datetime import date

import pandas as pd
import pytest

from ccdrift.logs import parse_source
from ccdrift.prices import Price
from ccdrift.spend import DIMENSIONS, spend_rows, spend_turns, total_tokens
from tests.helpers import at, line, text, write

TODAY = date(2026, 9, 10)


def corpus(tmp_path):
    """Four responses: two main thread, two in subagents, with attribution that overlaps
    across dimensions the way Claude Code's does."""
    write(tmp_path / "proj-a" / "s1.jsonl", [
        line("m1", text(40), ts=at(0), entrypoint="cli", branch="main"),
        line("m2", text(40), ts=at(60), entrypoint="cli", branch="main",
             skill="superpowers:writing-plans", plugin="superpowers"),
    ])
    write(tmp_path / "proj-b" / "s2.jsonl", [
        line("a1", text(40), ts=at(120), entrypoint="cli", sidechain=True, agent_type="general-purpose",
             branch="topic"),
        line("a2", text(40), ts=at(180), entrypoint="cli", sidechain=True, agent_type="Explore",
             branch="topic", mcp_server="context7"),
    ])
    return spend_turns(parse_source(tmp_path), TODAY)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_every_dimension_accounts_for_all_the_tokens(tmp_path, dimension):
    turns = corpus(tmp_path)
    rows = spend_rows(turns, dimension, {})
    assert rows["tokens"].sum() == pytest.approx(total_tokens(turns))
    assert rows["share"].sum() == pytest.approx(1.0)


def test_a_response_carrying_a_plugin_and_a_skill_is_counted_once_in_each(tmp_path):
    turns = corpus(tmp_path)
    skills = spend_rows(turns, "skill", {})
    plugins = spend_rows(turns, "plugin", {})
    assert skills.loc[skills["bucket"] == "superpowers:writing-plans", "responses"].iloc[0] == 1
    assert plugins.loc[plugins["bucket"] == "superpowers", "responses"].iloc[0] == 1


def test_responses_a_dimension_does_not_name_get_their_own_bucket(tmp_path):
    rows = spend_rows(corpus(tmp_path), "skill", {})
    assert rows.loc[rows["bucket"] == "no skill", "responses"].iloc[0] == 3


def test_the_thread_dimension_splits_main_from_subagents(tmp_path):
    rows = spend_rows(corpus(tmp_path), "thread", {})
    assert dict(zip(rows["bucket"], rows["responses"])) == {"main thread": 2, "subagent": 2}


def test_buckets_come_out_largest_first_with_ties_broken_by_name(tmp_path):
    rows = spend_rows(corpus(tmp_path), "branch", {})
    tokens = rows["tokens"].tolist()
    assert tokens == sorted(tokens, reverse=True)
    pd.testing.assert_frame_equal(rows, spend_rows(corpus(tmp_path), "branch", {}))


def test_a_bucket_is_priced_only_when_every_model_in_it_is(tmp_path):
    turns = corpus(tmp_path)
    priced = {"claude-opus-5": Price(5e-6, 25e-6, 0.0, 0.0, 9)}
    rows = spend_rows(turns, "thread", priced)
    assert rows["dollars"].notna().all()
    assert spend_rows(turns, "thread", {})["dollars"].isna().all()


def test_the_day_still_in_progress_is_left_out(tmp_path):
    write(tmp_path / "p" / "s1.jsonl", [line("m1", text(40), ts=at(0), entrypoint="cli")])
    assert spend_turns(parse_source(tmp_path / "p"), date(2026, 9, 1)).empty


def test_agent_sdk_sessions_are_left_out(tmp_path):
    write(tmp_path / "p" / "s1.jsonl", [line("m1", text(40), ts=at(0), entrypoint="sdk-py")])
    assert spend_turns(parse_source(tmp_path / "p"), TODAY).empty
