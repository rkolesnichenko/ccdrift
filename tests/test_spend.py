"""Partitioning the history by where its tokens went."""

import json
from datetime import date

import pandas as pd
import pytest

from ccdrift.logs import parse_source
from ccdrift.prices import Price
from ccdrift.spend import DIMENSIONS, run_spend, spend_rows, spend_turns, total_tokens
from ccdrift.state import new_state, save_state
from tests.helpers import at, line, text, write

TODAY = date(2026, 9, 10)


def corpus(tmp_path):
    """Four responses: two main thread, two in subagents, with attribution that overlaps
    across dimensions the way Claude Code's does. a2 carries a different model from the
    other three, so the model dimension genuinely splits and the subagent thread bucket
    holds two models, one priced and one not."""
    write(tmp_path / "proj-a" / "s1.jsonl", [
        line("m1", text(40), ts=at(0), entrypoint="cli", branch="main"),
        line("m2", text(40), ts=at(60), entrypoint="cli", branch="main",
             skill="superpowers:writing-plans", plugin="superpowers"),
    ])
    write(tmp_path / "proj-b" / "s2.jsonl", [
        line("a1", text(40), ts=at(120), entrypoint="cli", sidechain=True, agent_type="general-purpose",
             branch="topic"),
        line("a2", text(40), ts=at(180), entrypoint="cli", sidechain=True, agent_type="Explore",
             branch="topic", mcp_server="context7", model="claude-haiku-4-5"),
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


def test_the_project_dimension_reads_back_as_a_path_not_its_raw_encoding(tmp_path):
    rows = spend_rows(corpus(tmp_path), "project", {})
    assert set(rows["bucket"]) == {"/proj/a", "/proj/b"}
    assert "proj-a" not in rows["bucket"].tolist() and "proj-b" not in rows["bucket"].tolist()


def test_buckets_come_out_largest_first_with_ties_broken_by_name(tmp_path):
    rows = spend_rows(corpus(tmp_path), "branch", {})
    tokens = rows["tokens"].tolist()
    assert tokens == sorted(tokens, reverse=True)
    pd.testing.assert_frame_equal(rows, spend_rows(corpus(tmp_path), "branch", {}))


def test_the_model_dimension_splits_two_models_into_two_buckets(tmp_path):
    rows = spend_rows(corpus(tmp_path), "model", {})
    assert dict(zip(rows["bucket"], rows["responses"])) == {"claude-opus-5": 3, "claude-haiku-4-5": 1}


def test_a_bucket_is_priced_only_when_every_model_in_it_is(tmp_path):
    turns = corpus(tmp_path)
    # The subagent bucket holds both claude-opus-5 (a1) and claude-haiku-4-5 (a2); pricing
    # only the former must leave the whole bucket unpriced, not just a1's own dollars.
    opus_only = {"claude-opus-5": Price(5e-6, 25e-6, 0.0, 0.0, 9)}
    rows = spend_rows(turns, "thread", opus_only)
    assert pd.notna(rows.loc[rows["bucket"] == "main thread", "dollars"].iloc[0])
    assert pd.isna(rows.loc[rows["bucket"] == "subagent", "dollars"].iloc[0])

    both_priced = {**opus_only, "claude-haiku-4-5": Price(1e-6, 5e-6, 0.0, 0.0, 9)}
    assert spend_rows(turns, "thread", both_priced)["dollars"].notna().all()
    assert spend_rows(turns, "thread", {})["dollars"].isna().all()


def test_the_day_still_in_progress_is_left_out(tmp_path):
    write(tmp_path / "p" / "s1.jsonl", [line("m1", text(40), ts=at(0), entrypoint="cli")])
    assert spend_turns(parse_source(tmp_path / "p"), date(2026, 9, 1)).empty


def test_agent_sdk_sessions_are_left_out(tmp_path):
    write(tmp_path / "p" / "s1.jsonl", [line("m1", text(40), ts=at(0), entrypoint="sdk-py")])
    assert spend_turns(parse_source(tmp_path / "p"), TODAY).empty


def test_the_default_view_leads_with_thread_then_agent(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    corpus(tmp_path / "logs")
    assert run_spend(tmp_path / "logs", state, today=TODAY) == 0
    out = capsys.readouterr().out
    assert out.index("By thread") < out.index("By agent")


def test_a_named_dimension_shows_only_that_one(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    corpus(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, by="skill", today=TODAY)
    out = capsys.readouterr().out
    assert "By skill" in out and "By agent" not in out


def test_a_history_with_no_cost_records_reports_tokens_and_no_dollars(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    corpus(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, today=TODAY)
    out = capsys.readouterr().out
    assert "tokens" in out and "$" not in out


def test_a_source_with_no_transcripts_says_so(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    (tmp_path / "empty").mkdir()
    assert run_spend(tmp_path / "empty", state, today=TODAY) == 2


def test_the_json_holds_every_dimension_but_the_ones_that_name_your_folders(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    corpus(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, as_json=True, today=TODAY)
    payload = json.loads(capsys.readouterr().out)
    assert "thread" in payload["dimensions"] and "skill" in payload["dimensions"]
    assert "project" not in payload["dimensions"] and "branch" not in payload["dimensions"]


def test_the_json_names_no_project_and_no_branch_even_when_asked_for_one(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    corpus(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, by="branch", as_json=True, today=TODAY)
    out = capsys.readouterr().out
    assert "topic" not in out and "proj-a" not in out
    assert json.loads(out)["withheld"] == ["branch"]


def test_the_json_says_what_it_could_not_price(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    corpus(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, as_json=True, today=TODAY)
    payload = json.loads(capsys.readouterr().out)
    assert payload["dollars"] is None and payload["priced_models"] == []
