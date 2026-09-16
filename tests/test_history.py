"""ccdrift's history store: the responses it keeps after Claude Code deletes transcripts."""

import os
import sqlite3

import pandas as pd
import pytest

import ccdrift.history
from ccdrift.history import History, HistoryError, load_history, load_turns
from ccdrift.logs import parse_all, parse_durations, parse_source
from tests.helpers import (at, compact_boundary, damage_responses_table, line, prompt, response,
                           stop_hook_summary, text, thinking, tool_result, turn_duration, write)


def transcripts(folder):
    """A main-thread transcript with a prompt turn, a tool loop, a turn duration
    and a later prompt, and a Haiku subagent."""
    write(folder / "p" / "s1.jsonl", [
        prompt(at(0)),
        *response("m1", thinking(400), text(40), ts=at(0), cache_creation=1000, version="2.1.226",
                  entrypoint="cli", effort="xhigh", cache_1h=1000, thinking_logged=120),
        tool_result(at(30)), line("m2", text(40), ts=at(30), cache_read=1000),
        turn_duration(at(40), 40000, 2, uuid="d1"),
        prompt(at(600)), line("m3", text(40), ts=at(600), cache_read=900, cache_creation=100, cache_5m=100),
    ])
    write(folder / "p" / "s1" / "subagents" / "agent-a.jsonl",
          [line("a1", text(40), ts=at(100), sidechain=True, model="claude-haiku-4-5")])


def test_history_holds_the_same_responses_as_the_transcripts(tmp_path):
    transcripts(tmp_path / "logs")
    stored = load_turns(tmp_path / "logs", tmp_path / "state.json")
    pd.testing.assert_frame_equal(stored, parse_source(tmp_path / "logs"), check_like=True)


def test_history_holds_the_same_turn_durations_as_the_transcripts(tmp_path):
    transcripts(tmp_path / "logs")
    with History(tmp_path / "history.sqlite") as history:
        history.update(tmp_path / "logs")
        stored = history.durations()
    pd.testing.assert_frame_equal(stored, parse_durations(tmp_path / "logs"), check_like=True)


def test_history_reads_only_transcripts_that_changed(tmp_path):
    transcripts(tmp_path / "logs")
    with History(tmp_path / "history.sqlite") as history:
        assert history.update(tmp_path / "logs") == 2
        assert history.update(tmp_path / "logs") == 0
        with (tmp_path / "logs" / "p" / "s1.jsonl").open("a") as fh:
            fh.write('{"type": "user"}\n')
        assert history.update(tmp_path / "logs") == 1


def test_history_keeps_responses_after_claude_code_deletes_their_transcript(tmp_path):
    transcripts(tmp_path / "logs")
    load_turns(tmp_path / "logs", tmp_path / "state.json")
    (tmp_path / "logs" / "p" / "s1" / "subagents" / "agent-a.jsonl").unlink()
    assert len(load_turns(tmp_path / "logs", tmp_path / "state.json")) == 4


def test_a_transcript_that_grew_replaces_its_rows(tmp_path):
    write(tmp_path / "logs" / "s1.jsonl", [line("m1", text(40), ts=at(0), out=3)])
    load_turns(tmp_path / "logs", tmp_path / "state.json")
    write(tmp_path / "logs" / "s1.jsonl", [line("m1", text(40), ts=at(0), out=3),
                                           line("m1", text(400), ts=at(1), out=250),
                                           line("m2", text(40), ts=at(60))])
    assert load_turns(tmp_path / "logs", tmp_path / "state.json")["output_tokens"].tolist() == [250.0, 100.0]


@pytest.mark.parametrize("first, second", [("b.jsonl", "a.jsonl"), ("a.jsonl", "b.jsonl")])
def test_a_response_in_two_transcripts_belongs_to_the_path_that_sorts_first(tmp_path, first, second):
    # Resuming a session copies earlier responses into a new transcript, which
    # the store can read before or after the original.
    rec = line("m1", text(40), ts=at(0))
    write(tmp_path / "logs" / first, [rec])
    load_turns(tmp_path / "logs", tmp_path / "state.json")
    write(tmp_path / "logs" / second, [rec])
    assert load_turns(tmp_path / "logs", tmp_path / "state.json")["source_file"].tolist() == ["a.jsonl"]


def test_history_built_from_another_folder_is_left_alone(tmp_path, capsys):
    transcripts(tmp_path / "logs")
    write(tmp_path / "other" / "s9.jsonl", [line("m9", text(40), ts=at(0))])
    load_turns(tmp_path / "logs", tmp_path / "state.json")
    assert load_turns(tmp_path / "other", tmp_path / "state.json")["source_file"].tolist() == ["s9.jsonl"]
    assert f"it was built from {(tmp_path / 'logs').resolve()}" in capsys.readouterr().err
    with History(tmp_path / "history.sqlite") as history:
        assert len(history.responses()) == 4


def test_history_takes_the_first_folder_that_holds_transcripts(tmp_path, capsys):
    # A first run pointed at an empty folder mustn't tie the store to it.
    (tmp_path / "empty").mkdir()
    load_turns(tmp_path / "empty", tmp_path / "state.json")
    transcripts(tmp_path / "logs")
    assert len(load_turns(tmp_path / "logs", tmp_path / "state.json")) == 4
    assert capsys.readouterr().err == ""


def test_a_new_parser_version_reads_every_transcript_again(tmp_path, monkeypatch):
    transcripts(tmp_path / "logs")
    with History(tmp_path / "history.sqlite") as history:
        history.update(tmp_path / "logs")
    monkeypatch.setattr(ccdrift.history, "PARSER_VERSION", 3)
    with History(tmp_path / "history.sqlite") as history:
        assert history.update(tmp_path / "logs") == 2
        assert history.update(tmp_path / "logs") == 0


def test_a_store_from_a_newer_ccdrift_is_refused(tmp_path):
    History(tmp_path / "history.sqlite").close()
    db = sqlite3.connect(tmp_path / "history.sqlite")
    db.execute("UPDATE meta SET value = '99' WHERE key = 'schema_version'")
    db.commit()
    db.close()
    with pytest.raises(HistoryError, match="newer ccdrift"):
        History(tmp_path / "history.sqlite")


def test_a_store_from_a_newer_ccdrift_with_an_incompatible_schema_is_refused_untouched(tmp_path):
    # A future ccdrift may reshape the tables entirely; this version must recognize
    # the newer schema_version before running its own DDL against them.
    path = tmp_path / "history.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    db.execute("INSERT INTO meta VALUES ('schema_version', '3')")
    db.execute("CREATE TABLE responses (key INTEGER PRIMARY KEY, payload BLOB)")
    db.commit()
    db.close()
    with pytest.raises(HistoryError, match="newer ccdrift"):
        History(path)
    db = sqlite3.connect(path)
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    db.close()
    assert tables == {"meta", "responses"}


def test_an_unusable_store_says_how_to_rebuild_it(tmp_path):
    transcripts(tmp_path / "logs")
    (tmp_path / "history.sqlite").write_text("not a database")
    with pytest.raises(HistoryError, match="Move it aside"):
        load_turns(tmp_path / "logs", tmp_path / "state.json")


def test_a_store_whose_rows_cant_be_read_says_how_to_rebuild_it(tmp_path):
    transcripts(tmp_path / "logs")
    load_turns(tmp_path / "logs", tmp_path / "state.json")
    damage_responses_table(tmp_path / "history.sqlite")
    with pytest.raises(HistoryError, match="Move it aside"):
        load_turns(tmp_path / "logs", tmp_path / "state.json")


def test_a_store_with_a_schema_version_that_isnt_a_number_is_unusable(tmp_path):
    History(tmp_path / "history.sqlite").close()
    db = sqlite3.connect(tmp_path / "history.sqlite")
    db.execute("UPDATE meta SET value = 'one' WHERE key = 'schema_version'")
    db.commit()
    db.close()
    with pytest.raises(HistoryError, match="Move it aside"):
        History(tmp_path / "history.sqlite")


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a read-only file")
def test_a_store_that_cant_record_its_schema_version_is_unusable(tmp_path):
    path = tmp_path / "history.sqlite"
    History(path).close()
    db = sqlite3.connect(path)
    db.execute("DELETE FROM meta WHERE key = 'schema_version'")
    db.commit()
    db.close()
    path.chmod(0o444)
    with pytest.raises(HistoryError, match="Move it aside"):
        History(path)


def test_a_store_that_cant_be_created_doesnt_suggest_moving_it_aside(tmp_path):
    # Moving a store aside can't help when its folder can't be made.
    (tmp_path / "home").write_text("a file where the folder should be")
    with pytest.raises(HistoryError, match="Can't open the history store") as raised:
        load_turns(tmp_path / "logs", tmp_path / "home" / "state.json")
    assert "Move it aside" not in str(raised.value)


V1_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE files (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, size INTEGER, mtime_ns INTEGER, session_id TEXT);
CREATE TABLE responses (key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    model TEXT, version TEXT, entrypoint TEXT, effort TEXT, speed TEXT, service_tier TEXT,
    is_sidechain INTEGER, new_prompt INTEGER, after_compaction INTEGER,
    input_tokens INTEGER, output_tokens INTEGER, cache_creation INTEGER, cache_read INTEGER,
    cache_1h INTEGER, cache_5m INTEGER, thinking_logged INTEGER,
    signature_chars INTEGER, visible_chars INTEGER, n_mcp_calls INTEGER);
CREATE TABLE durations (key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    version TEXT, entrypoint TEXT, is_sidechain INTEGER, duration_ms INTEGER, message_count INTEGER);
INSERT INTO meta VALUES ('schema_version', '1'), ('parser_version', '1'), ('source', '/somewhere');
INSERT INTO files VALUES (1, 'old.jsonl', 10, 10, 'old');
INSERT INTO responses (key, file_id, ts, model, is_sidechain, new_prompt, after_compaction, input_tokens,
    output_tokens, cache_creation, cache_read, cache_1h, cache_5m, signature_chars, visible_chars, n_mcp_calls)
    VALUES (7, 1, 1788000000000000, 'claude-opus-5', 0, 1, 0, 10, 20, 30, 40, 0, 0, 0, 0, 0);
"""


def test_a_store_from_ccdrift_0_2_is_upgraded_in_place_and_keeps_its_rows(tmp_path):
    db = sqlite3.connect(tmp_path / "history.sqlite")
    db.executescript(V1_SCHEMA)
    db.close()
    with History(tmp_path / "history.sqlite") as history:
        assert history.meta["schema_version"] == "2"
        columns = {row[1] for row in history.db.execute("PRAGMA table_info(responses)")}
        tables = {row[0] for row in history.db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert "agent_type" in columns
        assert {"hook_runs", "compactions"} <= tables
        assert history.responses()["model"].tolist() == ["claude-opus-5"]


def test_history_holds_the_same_hook_runs_compactions_and_agent_types_as_the_transcripts(tmp_path):
    write(tmp_path / "logs" / "s1.jsonl", [
        prompt(at(0)), line("m1", text(40), ts=at(0)),
        stop_hook_summary(at(10), 2, durations=(500, 700), uuid="h1"),
        stop_hook_summary(at(20), 1, errors=("boom",), uuid="h2"),
        compact_boundary(at(30), trigger="auto", pre_tokens=965_000),
    ])
    write(tmp_path / "logs" / "s1" / "subagents" / "agent-a.jsonl",
          [line("a1", text(40), ts=at(5), sidechain=True, agent_type="Plan")])
    stored = load_history(tmp_path / "logs", tmp_path / "state.json", claim=True)
    parsed = parse_all(tmp_path / "logs")
    pd.testing.assert_frame_equal(stored.responses, parsed.responses, check_like=True)
    pd.testing.assert_frame_equal(stored.hook_runs, parsed.hook_runs, check_like=True)
    pd.testing.assert_frame_equal(stored.compactions, parsed.compactions, check_like=True)
    agent_types = stored.responses["agent_type"].tolist()
    # pandas' inferred string dtype uses NaN, not None, once a text column mixes
    # missing and real values (pandas >= 3.0); pd.isna covers both.
    assert pd.isna(agent_types[0]) and agent_types[1] == "Plan"


def test_a_transcript_that_cant_be_read_is_tried_again_and_the_parser_version_still_recorded(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("root can read unreadable files")
    transcripts(tmp_path / "logs")
    blocked = tmp_path / "logs" / "p" / "s1.jsonl"
    with History(tmp_path / "history.sqlite") as history:
        history.update(tmp_path / "logs")
        with blocked.open("a") as fh:
            fh.write('{"type": "user"}\n')
        os.chmod(blocked, 0)
        try:
            assert history.update(tmp_path / "logs") == 0
            assert history.meta["parser_version"] == "2"
        finally:
            os.chmod(blocked, 0o644)
        assert history.update(tmp_path / "logs") == 1


def test_report_and_incident_list_leave_an_unclaimed_store_alone(tmp_path):
    transcripts(tmp_path / "logs")
    load_history(tmp_path / "logs", tmp_path / "state.json", claim=False)
    assert not (tmp_path / "history.sqlite").exists()
    History(tmp_path / "history.sqlite").close()
    assert len(load_history(tmp_path / "logs", tmp_path / "state.json", claim=False).responses) == 4
    with History(tmp_path / "history.sqlite") as history:
        assert history.built_from() is None


def test_a_sqlite_too_old_for_the_store_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(sqlite3, "sqlite_version_info", (3, 22, 0))
    with pytest.raises(HistoryError, match="needs SQLite 3.24"):
        History(tmp_path / "history.sqlite")


def test_a_folder_in_the_stores_place_says_to_move_it_aside(tmp_path):
    (tmp_path / "history.sqlite").mkdir()
    with pytest.raises(HistoryError, match="is a folder. Move it aside"):
        History(tmp_path / "history.sqlite")
