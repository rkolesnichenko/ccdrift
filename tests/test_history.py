"""ccdrift's history store: the responses it keeps after Claude Code deletes transcripts."""

import sqlite3

import pandas as pd
import pytest

import ccdrift.history
from ccdrift.history import History, HistoryError, load_turns
from ccdrift.logs import parse_durations, parse_source
from tests.helpers import at, line, prompt, response, text, thinking, tool_result, turn_duration, write


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
    monkeypatch.setattr(ccdrift.history, "PARSER_VERSION", 2)
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


def test_an_unusable_store_says_how_to_rebuild_it(tmp_path):
    transcripts(tmp_path / "logs")
    (tmp_path / "history.sqlite").write_text("not a database")
    with pytest.raises(HistoryError, match="Move it aside"):
        load_turns(tmp_path / "logs", tmp_path / "state.json")
