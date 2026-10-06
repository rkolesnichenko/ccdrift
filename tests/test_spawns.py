"""Subagent spawns: the model each Agent call asked for, the one Claude Code resolved and the
one the subagent was served."""

import sqlite3
from datetime import date

import pandas as pd

from ccdrift.history import History
from ccdrift.logs import parse_all
from ccdrift.spawns import (JOINED_COLUMNS, alias_history, honours, judge, mismatches, model_mismatch_alerts, model_name,
                            spawn_models)
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


def spawn(agent, requested, resolved, served, version="2.1.289", day="2026-09-01", responses=1):
    """One row as spawn_models gives it."""
    return {"agent_id": agent, "requested": requested, "resolved": resolved, "version": version, "day": day,
            "timestamp": pd.Timestamp(f"{day}T10:00:00Z"), "served": tuple(served),
            "responses": responses if served else 0}


def joined(*rows):
    return pd.DataFrame(list(rows), columns=list(JOINED_COLUMNS))


def test_a_model_compares_in_lowercase_without_its_one_million_context_marker():
    assert model_name("Claude-Opus-5[1m]") == "claude-opus-5"
    assert [model_name(raw) for raw in ("", "  ", None, 3)] == [None, None, None, None]


def test_an_alias_is_honoured_by_any_model_of_its_family_and_a_full_id_only_by_itself():
    assert honours("opus", "claude-opus-5") and honours("opus", "claude-opus-5-5")
    assert honours("haiku", "claude-haiku-4-5-20251001")
    assert not honours("opus", "claude-sonnet-5")
    assert honours("claude-opus-5-5", "claude-opus-5-5")
    assert not honours("claude-opus-5", "claude-opus-5-5")


def test_a_spawn_resolved_to_another_model_than_it_asked_for_is_not_honoured():
    assert judge("sonnet", "claude-opus-5", ("claude-opus-5",)) == ["not_honoured"]


def test_a_spawn_served_another_model_than_the_one_resolved_differs_but_the_one_million_marker_does_not():
    assert judge("opus", "claude-opus-5-5[1m]", ("claude-sonnet-5",)) == ["served_differs"]
    assert judge("opus", "claude-opus-5[1m]", ("claude-opus-5",)) == []


def test_with_no_resolved_model_logged_the_request_is_judged_against_what_was_served():
    assert judge("haiku", None, ("claude-sonnet-5",)) == ["not_honoured"]
    assert judge("haiku", None, ("claude-haiku-4-5",)) == []


def test_a_spawn_naming_no_model_or_inherit_is_judged_only_on_what_was_served():
    assert judge(None, "claude-opus-5", ("claude-opus-5",)) == []
    assert judge("inherit", "claude-opus-5", ("claude-opus-5",)) == []
    assert judge("inherit", "claude-opus-5", ("claude-haiku-4-5",)) == ["served_differs"]


def test_a_spawn_with_no_served_response_is_not_judged():
    assert judge("sonnet", "claude-opus-5", ()) == []


def test_a_spawn_served_two_models_differs_when_either_is_not_the_one_resolved():
    assert judge("haiku", "claude-haiku-4-5", ("claude-haiku-4-5", "claude-sonnet-5")) == ["served_differs"]


def test_each_spawn_carries_the_models_its_responses_were_served_and_how_many(tmp_path):
    spawning(tmp_path)
    tables = parse_all(tmp_path)
    found = spawn_models(tables.spawns, tables.responses)
    assert found[["agent_id", "served", "responses"]].to_dict("records") == [
        {"agent_id": "a1", "served": ("claude-sonnet-5",), "responses": 2},
        {"agent_id": "a2", "served": (), "responses": 0}]


def test_a_mismatch_names_what_was_asked_resolved_and_served_with_none_for_no_request():
    found = mismatches(joined(spawn("a1", None, "claude-opus-5", ["claude-haiku-4-5", "claude-opus-5"]),
                              spawn("a2", "Sonnet", "claude-opus-5[1m]", ["claude-opus-5"], version="2.1.290")))
    assert found.to_dict("records") == [
        {"kind": "served_differs", "requested": "none", "resolved": "claude-opus-5",
         "served": "claude-haiku-4-5, claude-opus-5", "version": "2.1.289", "day": "2026-09-01"},
        {"kind": "not_honoured", "requested": "sonnet", "resolved": "claude-opus-5", "served": "claude-opus-5",
         "version": "2.1.290", "day": "2026-09-01"}]


def test_a_mismatch_with_no_version_logged_names_none_however_pandas_reads_the_gap():
    found = mismatches(joined(spawn("a1", "sonnet", "claude-opus-5", ["claude-opus-5"], version=float("nan"))))
    assert found["version"].tolist() == [None]


def test_the_history_shows_each_alias_moving_to_a_new_model_with_its_days_versions_and_spawns():
    rows = joined(spawn("a1", "opus", "claude-opus-5[1m]", ["claude-opus-5"], version="2.1.99", day="2026-09-01"),
                  spawn("a2", "opus", "claude-opus-5", ["claude-opus-5"], version="2.1.280", day="2026-09-02"),
                  spawn("a3", "opus", "claude-opus-5-5", ["claude-opus-5-5"], version="2.1.288", day="2026-09-03"),
                  spawn("a4", None, "claude-sonnet-5", [], day="2026-09-03"),
                  spawn("a5", "haiku", "claude-haiku-4-5", ["claude-haiku-4-5"], day="2026-08-31"))
    assert alias_history(rows, ["2026-09-01", "2026-09-02", "2026-09-03"]) == {
        "aliases": [{"alias": "none", "models": [{"model": "claude-sonnet-5", "first": "2026-09-03",
                                                  "last": "2026-09-03", "versions": ["2.1.289"], "spawns": 1}]},
                    {"alias": "opus", "models": [
                        {"model": "claude-opus-5", "first": "2026-09-01", "last": "2026-09-02",
                         "versions": ["2.1.99", "2.1.280"], "spawns": 2},
                        {"model": "claude-opus-5-5", "first": "2026-09-03", "last": "2026-09-03",
                         "versions": ["2.1.288"], "spawns": 1}]}],
        "spawns": 4, "judged": 3, "no_response": 1, "mismatches": []}


def test_a_mismatch_alerts_once_per_kind_models_and_version_and_again_when_the_version_moves():
    state = {"model_mismatches": []}
    first = joined(spawn("a1", "sonnet", "claude-opus-5", ["claude-opus-5"], day="2026-09-07"),
                   spawn("a2", "sonnet", "claude-opus-5", ["claude-opus-5"], day="2026-09-08"))
    new = model_mismatch_alerts(first, state, date(2026, 9, 8))
    assert new == [{"kind": "not_honoured", "requested": "sonnet", "resolved": "claude-opus-5",
                    "served": "claude-opus-5", "version": "2.1.289", "first_day": "2026-09-07",
                    "reported_on": "2026-09-08"}]
    assert state["model_mismatches"] == new
    assert model_mismatch_alerts(first, state, date(2026, 9, 9)) == []
    moved = joined(spawn("a3", "sonnet", "claude-opus-5", ["claude-opus-5"], version="2.1.290", day="2026-09-09"))
    assert [m["version"] for m in model_mismatch_alerts(moved, state, date(2026, 9, 9))] == ["2.1.290"]


def test_a_mismatch_older_than_the_alert_window_is_left_to_the_report():
    state = {"model_mismatches": []}
    rows = joined(spawn("a1", "sonnet", "claude-opus-5", ["claude-opus-5"], day="2026-09-01"),
                  spawn("a2", "haiku", "claude-opus-5", ["claude-opus-5"], day="2026-09-02"))
    assert [m["requested"] for m in model_mismatch_alerts(rows, state, date(2026, 9, 8))] == ["haiku"]
