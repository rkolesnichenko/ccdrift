"""Subagent spawns: the model each Agent call asked for, the one Claude Code resolved and the
one the subagent was served."""

import sqlite3

from ccdrift.history import History
from ccdrift.logs import parse_all
from tests.helpers import PRIVATE_TEXT, agent_call, agent_result, at, line, text, tool_use, write


def spawning(root, name="s1", sidechain=False, sid="s1"):
    """A session whose main thread asks for Sonnet once and names no model once, and whose
    first subagent answers twice."""
    write(root / "p" / f"{name}.jsonl", [
        line("m1", agent_call("t1", model="sonnet"), ts=at(0), sid=sid, version="2.1.289", entrypoint="cli"),
        agent_result(at(1), "t1", "a1", resolved="claude-sonnet-5", sid=sid),
        line("m2", agent_call("t2"), ts=at(2), sid=sid, version="2.1.289", entrypoint="cli"),
        agent_result(at(3), "t2", "a2", resolved="claude-opus-5[1m]", sid=sid),
    ])
    write(root / "p" / name / "subagents" / "agent-a1.jsonl", [
        line("x1", text(20), ts=at(0.5), sid=sid, sidechain=True, model="claude-sonnet-5", agent_id="a1"),
        line("x2", text(20), ts=at(0.7), sid=sid, sidechain=True, model="claude-sonnet-5", agent_id="a1"),
    ])


def test_an_agent_calls_result_is_a_spawn_with_the_model_asked_for_and_the_one_resolved(tmp_path):
    spawning(tmp_path)
    spawns = parse_all(tmp_path).spawns
    rows = spawns[["agent_id", "requested", "resolved", "version", "is_sidechain", "day", "session_id"]]
    # pandas 3 reads a missing text value as NaN, pandas 2 as None.
    rows = rows.astype(object).where(rows.notna(), None)
    assert rows.to_dict("records") == [
        {"agent_id": "a1", "requested": "sonnet", "resolved": "claude-sonnet-5", "version": "2.1.289",
         "is_sidechain": False, "day": "2026-09-01", "session_id": "s1"},
        {"agent_id": "a2", "requested": None, "resolved": "claude-opus-5[1m]", "version": "2.1.289",
         "is_sidechain": False, "day": "2026-09-01", "session_id": "s1"}]


def test_each_response_carries_the_agent_id_of_the_subagent_it_came_from(tmp_path):
    spawning(tmp_path)
    responses = parse_all(tmp_path).responses
    assert sorted(zip(responses["source_file"], responses["agent_id"].fillna("-"))) == [
        ("p/s1.jsonl", "-"), ("p/s1.jsonl", "-"),
        ("p/s1/subagents/agent-a1.jsonl", "a1"), ("p/s1/subagents/agent-a1.jsonl", "a1")]


def test_a_spawn_keeps_nothing_of_the_calls_prompt_or_description(tmp_path):
    spawning(tmp_path)
    assert PRIVATE_TEXT not in parse_all(tmp_path).spawns.to_json()
    path = tmp_path / "history.sqlite"
    with History(path) as history:
        history.update(tmp_path / "p")
    db = sqlite3.connect(path)
    try:
        assert PRIVATE_TEXT not in repr(db.execute("SELECT * FROM spawns").fetchall())
    finally:
        db.close()


def test_a_result_that_is_not_a_record_or_answers_another_tool_is_no_spawn(tmp_path):
    refused = agent_result(at(1), "t1", "a1")
    refused["toolUseResult"] = "Error: the agent could not start"
    bash = agent_result(at(3), "t2", "a2", resolved="claude-sonnet-5")
    write(tmp_path / "p" / "s1.jsonl", [
        line("m1", agent_call("t1", model="sonnet"), ts=at(0)), refused,
        line("m2", tool_use("t2", "Bash"), ts=at(2)), bash])
    assert parse_all(tmp_path).spawns.empty


def test_a_subagent_that_spawns_one_of_its_own_marks_the_spawn_as_on_a_side_thread(tmp_path):
    write(tmp_path / "p" / "s1" / "subagents" / "agent-a1.jsonl", [
        line("x1", agent_call("t5", model="haiku"), ts=at(0), sidechain=True, agent_id="a1"),
        agent_result(at(1), "t5", "a9", resolved="claude-haiku-4-5", sidechain=True)])
    spawns = parse_all(tmp_path).spawns
    assert spawns[["agent_id", "requested", "is_sidechain"]].to_dict("records") == [
        {"agent_id": "a9", "requested": "haiku", "is_sidechain": True}]


def test_a_spawn_copied_into_a_second_transcript_counts_once_from_the_path_that_sorts_first(tmp_path):
    spawning(tmp_path, name="s1")
    spawning(tmp_path, name="s2", sid="s2")
    spawns = parse_all(tmp_path).spawns
    assert spawns[["agent_id", "source_file"]].to_dict("records") == [
        {"agent_id": "a1", "source_file": "p/s1.jsonl"}, {"agent_id": "a2", "source_file": "p/s1.jsonl"}]


def test_the_store_keeps_spawns_and_agent_ids_as_the_transcripts_read_them(tmp_path):
    spawning(tmp_path / "logs")
    spawning(tmp_path / "logs", name="s2", sid="s2")
    with History(tmp_path / "history.sqlite") as history:
        history.update(tmp_path / "logs")
        stored, responses = history.spawns(), history.responses()
    read = parse_all(tmp_path / "logs")
    columns = ["agent_id", "requested", "resolved", "version", "entrypoint", "is_sidechain", "day", "source_file",
               "session_id", "timestamp"]
    assert stored[columns].to_dict("records") == read.spawns[columns].to_dict("records")
    assert sorted(responses["agent_id"].dropna()) == sorted(read.responses["agent_id"].dropna()) == ["a1", "a1"]
