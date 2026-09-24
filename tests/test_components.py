"""What a session started with: the attachment records Claude Code writes before its first
response, read into one row per transcript and kept in the history store."""

import json
import sqlite3

import ccdrift.history
from ccdrift.history import History, load_history
from ccdrift.logs import COMPONENT_COLUMNS, parse_all, parse_file
from ccdrift.state import new_state, save_state
from tests.helpers import (PRIVATE_PATH, PRIVATE_TEXT, agent_listing, api_error, at, attachment, deferred_tools,
                           instructions, line, mcp_instructions, prompt, prompt_snapshot, skill_listing, text,
                           tool_result, write)

TOOLS = {"Bash": 100, "Read": 50}


def started(path, *records, after=(), version="2.1.267"):
    """A main-thread transcript: the attachment records `records` and a prompt before its
    first response, then the records `after` it."""
    write(path, [*(attachment(at(0), r, version=version) for r in records), prompt(at(1)),
                 line("m1", text(40), ts=at(2), version=version, entrypoint="cli"),
                 *(attachment(at(3), r, version=version) for r in after)])
    return parse_file(path, path.name).components


def test_a_session_start_keeps_the_names_and_sizes_of_what_it_started_with(tmp_path):
    row = started(tmp_path / "s1.jsonl",
                  skill_listing(["review", "deploy"], chars=2000),
                  deferred_tools(["Read", "mcp__gh__pr", "mcp__gh__issue"], line_chars=20),
                  agent_listing(["Plan", "Explore"], line_chars=30),
                  mcp_instructions(["gh"], block_chars=500),
                  instructions(1000, 300),
                  prompt_snapshot(4000, 500))
    assert row["skills"] == ["deploy", "review"] and row["skills_chars"] == 2000
    assert row["deferred"] == ["Read", "mcp__gh__issue", "mcp__gh__pr"] and row["deferred_chars"] == 60
    assert row["agents"] == ["Explore", "Plan"] and row["agents_chars"] == 60
    assert row["mcp"] == ["gh"] and row["mcp_chars"] == 500
    assert row["claude_md_files"] == 2 and row["claude_md_chars"] == 1300
    assert row["system_chars"] == 4500
    assert row["message_chars"] == len("next request")


def test_records_after_the_first_response_are_ignored(tmp_path):
    row = started(tmp_path / "s1.jsonl", skill_listing(["review"], chars=100),
                  after=[skill_listing(["review", "late"], chars=900), deferred_tools(["Late"]),
                         instructions(50), prompt_snapshot(70)])
    assert row["skills"] == ["review"] and row["skills_chars"] == 100
    assert row["deferred"] is None and row["claude_md_files"] is None and row["system_chars"] is None


def test_the_tool_definitions_come_from_the_first_snapshot_that_carries_them_even_after_the_first_response(
        tmp_path):
    snapshot = prompt_snapshot(4000, tools=TOOLS)
    row = started(tmp_path / "s1.jsonl", prompt_snapshot(4000),
                  after=[snapshot, prompt_snapshot(4000, tools={"Bash": 100, "Read": 50, "Later": 999})])
    assert row["tools"] == ["Bash", "Read"]
    assert row["tools_chars"] == len(json.dumps(snapshot["tools"], ensure_ascii=False))


def test_a_version_before_tool_definitions_claude_md_and_the_system_prompt_reads_them_as_not_logged(tmp_path):
    row = started(tmp_path / "s1.jsonl", skill_listing(["review"]), deferred_tools(["Read"]),
                  agent_listing(["Plan"]), version="2.1.250")
    assert row["tools"] is None and row["tools_chars"] is None
    assert row["claude_md_files"] is None and row["claude_md_chars"] is None and row["system_chars"] is None
    # No MCP instructions record either: the store keeps that as not logged, and the
    # comparison decides what it means.
    assert row["mcp"] is None and row["mcp_chars"] is None


def test_deltas_before_the_first_response_accumulate(tmp_path):
    row = started(tmp_path / "s1.jsonl", deferred_tools(["Read", "mcp__gh__pr"], line_chars=20),
                  deferred_tools(["mcp__gh__issue"], removed=["mcp__gh__pr"], line_chars=20))
    assert row["deferred"] == ["Read", "mcp__gh__issue"]
    assert row["deferred_chars"] == 60


def test_an_initial_listing_replaces_what_came_before_it(tmp_path):
    row = started(tmp_path / "s1.jsonl", agent_listing(["Plan", "Old"], line_chars=30),
                  agent_listing(["Plan", "Explore"], line_chars=30, initial=True))
    assert row["agents"] == ["Explore", "Plan"] and row["agents_chars"] == 60


def test_an_agent_delta_that_isnt_initial_adds_to_the_listing(tmp_path):
    row = started(tmp_path / "s1.jsonl", agent_listing(["Plan"], line_chars=30),
                  agent_listing(["Explore"], line_chars=30, initial=False))
    assert row["agents"] == ["Explore", "Plan"] and row["agents_chars"] == 60


def test_a_malformed_record_reads_as_not_logged(tmp_path):
    broken_skills = {**skill_listing(["review"]), "names": "review"}
    broken_tools = {**deferred_tools(["Read"]), "addedNames": None}
    broken_files = {**instructions(100), "files": [{"path": "x", "content": 7}]}
    broken_prompt = {**prompt_snapshot(100), "systemPrompt": {"text": "x"}}
    broken_defs = {**prompt_snapshot(100, tools=TOOLS), "tools": ["Bash"]}
    row = started(tmp_path / "s1.jsonl", broken_skills, broken_tools, broken_files, broken_prompt,
                  after=[broken_defs])
    assert all(row[column] is None for column in COMPONENT_COLUMNS
               if column not in ("source_file", "session_id", "message_chars"))


def test_a_record_written_in_a_subagent_is_ignored(tmp_path):
    path = tmp_path / "s1.jsonl"
    write(path, [attachment(at(0), skill_listing(["review"]), sidechain=True), prompt(at(1)),
                 line("m1", text(40), ts=at(2), entrypoint="cli")])
    assert parse_file(path, path.name).components["skills"] is None


def test_no_text_or_path_from_any_record_is_kept(tmp_path):
    row = started(tmp_path / "s1.jsonl", skill_listing(["review"]), deferred_tools(["Read"]),
                  agent_listing(["Plan"]), mcp_instructions(["gh"]), instructions(900), prompt_snapshot(900),
                  after=[prompt_snapshot(900, tools=TOOLS)])
    kept = json.dumps(row)
    assert PRIVATE_TEXT not in kept and PRIVATE_PATH not in kept and "CLAUDE" not in kept


def test_only_what_the_user_sent_before_the_first_response_counts_as_the_first_message(tmp_path):
    path = tmp_path / "s1.jsonl"
    write(path, [prompt(at(0)), prompt(at(1)), line("m1", text(40), ts=at(2), entrypoint="cli"),
                 tool_result(at(3)), prompt(at(4)), line("m2", text(40), ts=at(5), entrypoint="cli")])
    assert parse_file(path, path.name).components["message_chars"] == 2 * len("next request")


def test_an_error_banner_before_the_first_response_doesnt_end_the_session_start(tmp_path):
    path = tmp_path / "s1.jsonl"
    write(path, [prompt(at(0)), api_error(at(1)), attachment(at(2), skill_listing(["review"])),
                 line("m1", text(40), ts=at(3), entrypoint="cli")])
    assert parse_file(path, path.name).components["skills"] == ["review"]


def test_a_name_is_kept_without_its_control_characters(tmp_path):
    row = started(tmp_path / "s1.jsonl", agent_listing(["Plan\x1b]0;title\x07", "Explore"]))
    assert row["agents"] == ["Explore", "Plan]0;title"]


def test_every_transcript_gets_a_row_with_its_path_and_session(tmp_path):
    write(tmp_path / "p" / "s1.jsonl", [prompt(at(1), sid="abc"), line("m1", text(40), ts=at(2), sid="abc")])
    rows = parse_all(tmp_path).components
    assert rows[["source_file", "session_id"]].values.tolist() == [["p/s1.jsonl", "abc"]]
    assert rows.loc[0, "skills"] is None and rows.loc[0, "message_chars"] == len("next request")


def stored(tmp_path):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    return load_history(tmp_path / "logs", state, claim=True).components


def test_the_store_and_a_fresh_parse_return_the_same_components_table(tmp_path):
    started(tmp_path / "logs" / "p" / "s1.jsonl", skill_listing(["review"]), deferred_tools(["Read"]),
            agent_listing(["Plan"]), instructions(900), prompt_snapshot(900), after=[prompt_snapshot(900, tools=TOOLS)])
    started(tmp_path / "logs" / "p" / "s2.jsonl", skill_listing(["review"]), version="2.1.250")
    kept, parsed = stored(tmp_path), parse_all(tmp_path / "logs").components
    assert list(kept.columns) == list(parsed.columns) == list(COMPONENT_COLUMNS)
    assert kept.astype(str).values.tolist() == parsed.astype(str).values.tolist()
    assert kept.loc[1, "system_chars"] != kept.loc[1, "system_chars"]  # NaN: not logged, not 0


def test_a_transcript_that_grows_keeps_one_components_row(tmp_path):
    path = tmp_path / "logs" / "p" / "s1.jsonl"
    started(path, skill_listing(["review"]))
    stored(tmp_path)
    started(path, skill_listing(["review", "deploy"]))
    kept = stored(tmp_path)
    assert kept["skills"].tolist() == [["deploy", "review"]]


def test_the_store_since_a_day_keeps_the_components_of_transcripts_active_since_then(tmp_path):
    started(tmp_path / "logs" / "p" / "s1.jsonl", skill_listing(["review"]))
    state = tmp_path / "state.json"
    save_state(state, new_state())
    assert len(load_history(tmp_path / "logs", state, claim=True, since="2026-09-01").components) == 1
    assert load_history(tmp_path / "logs", state, claim=True, since="2026-09-02").components.empty


def test_a_store_from_before_components_gains_the_table_and_reads_every_transcript_again(tmp_path):
    started(tmp_path / "logs" / "s1.jsonl", skill_listing(["review"]))
    path = tmp_path / "history.sqlite"
    with History(path) as history:
        history.update(tmp_path / "logs")
    db = sqlite3.connect(path)
    db.execute("DROP TABLE components")
    db.execute("UPDATE meta SET value = '6' WHERE key = 'schema_version'")
    db.execute("UPDATE meta SET value = '9' WHERE key = 'parser_version'")
    db.commit()
    db.close()
    with History(path) as history:
        assert history.meta["schema_version"] == str(ccdrift.history.SCHEMA_VERSION) == "7"
        assert history.update(tmp_path / "logs") == 1
        assert history.components()["skills"].tolist() == [["review"]]
