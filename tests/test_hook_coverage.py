"""Hook coverage: which tool calls got a PreToolUse or PostToolUse hook record, read into
rows per transcript and kept in the history store."""

import json
import sqlite3

import ccdrift.history
from ccdrift.history import History, load_history
from ccdrift.logs import COVERAGE_COLUMNS, parse_all, parse_file
from ccdrift.state import new_state, save_state
from tests.helpers import PRIVATE_PATH, PRIVATE_TEXT, at, hook_record, line, prompt, tool_use, write


def coverage(path):
    """A parsed transcript's coverage rows as {(event, tool): (calls, hooked)}."""
    rows = parse_file(path, path.name).hook_coverage
    return {(key[4], key[5]): tuple(counts) for key, counts in rows.items()}


def called(path, *calls, hooks=(), sidechain=False, version="2.1.261"):
    """A transcript: a prompt, then one response per tool call in `calls` ((id, name)
    pairs), then the hook records `hooks`."""
    write(path, [prompt(at(0), sidechain=sidechain),
                 *(line(f"m-{tool_id}", tool_use(tool_id, name), ts=at(1 + i), sidechain=sidechain, version=version,
                        entrypoint="cli") for i, (tool_id, name) in enumerate(calls)),
                 *hooks])
    return path


def test_a_tool_call_with_a_hook_record_counts_as_hooked_for_that_event_only(tmp_path):
    path = called(tmp_path / "s1.jsonl", ("t1", "Bash"), ("t2", "Bash"),
                  hooks=[hook_record(at(2), "PreToolUse", "t1", "Bash")])
    assert coverage(path) == {("PreToolUse", "Bash"): (2, 1), ("PostToolUse", "Bash"): (2, 0)}


def test_a_hook_that_ran_and_failed_still_ran(tmp_path):
    path = called(tmp_path / "s1.jsonl", ("t1", "Bash"), ("t2", "Edit"),
                  hooks=[hook_record(at(2), "PreToolUse", "t1", "Bash", kind="hook_non_blocking_error"),
                         hook_record(at(3), "PostToolUse", "t2", "Edit", kind="hook_blocking_error")])
    rows = coverage(path)
    assert rows[("PreToolUse", "Bash")] == (1, 1) and rows[("PostToolUse", "Edit")] == (1, 1)


def test_other_hook_output_records_are_not_runs(tmp_path):
    extra = hook_record(at(2), "PostToolUse", "t1", "Bash")
    extra["attachment"] = {"type": "hook_additional_context", "hookEvent": "PostToolUse", "hookName": "PostToolUse:Bash",
                           "toolUseID": "t1", "content": [PRIVATE_TEXT]}
    path = called(tmp_path / "s1.jsonl", ("t1", "Bash"), hooks=[extra])
    assert coverage(path)[("PostToolUse", "Bash")] == (1, 0)


def test_a_hook_record_that_matches_no_tool_call_is_ignored(tmp_path):
    path = called(tmp_path / "s1.jsonl", ("t1", "Bash"), hooks=[hook_record(at(2), "PreToolUse", "t9", "Bash")])
    assert coverage(path) == {("PreToolUse", "Bash"): (1, 0), ("PostToolUse", "Bash"): (1, 0)}


def test_other_hook_events_are_not_counted(tmp_path):
    path = called(tmp_path / "s1.jsonl", ("t1", "Bash"), hooks=[hook_record(at(2), "UserPromptSubmit", "t1", "Bash")])
    assert set(coverage(path)) == {("PreToolUse", "Bash"), ("PostToolUse", "Bash")}
    assert coverage(path)[("PreToolUse", "Bash")] == (1, 0)


def test_a_hook_event_missing_is_read_from_the_hook_name(tmp_path):
    record = hook_record(at(2), "PreToolUse", "t1", "Bash")
    del record["attachment"]["hookEvent"]
    path = called(tmp_path / "s1.jsonl", ("t1", "Bash"), hooks=[record])
    assert coverage(path)[("PreToolUse", "Bash")] == (1, 1)


def test_a_malformed_hook_record_reads_as_no_record(tmp_path):
    record = hook_record(at(2), "PreToolUse", "t1", "Bash")
    record["attachment"]["toolUseID"] = ["t1"]
    path = called(tmp_path / "s1.jsonl", ("t1", "Bash"), hooks=[record])
    assert coverage(path)[("PreToolUse", "Bash")] == (1, 0)


def test_mcp_tools_are_counted_by_their_server(tmp_path):
    path = called(tmp_path / "s1.jsonl", ("t1", "mcp__gh__pr"), ("t2", "mcp__gh__issue"), ("t3", "mcp__jira__get"),
                  hooks=[hook_record(at(4), "PreToolUse", "t1", "mcp__gh__pr")])
    rows = coverage(path)
    assert rows[("PreToolUse", "mcp__gh")] == (2, 1) and rows[("PreToolUse", "mcp__jira")] == (1, 0)


def test_a_row_keeps_the_day_version_entrypoint_and_thread_of_the_response_carrying_the_call(tmp_path):
    path = called(tmp_path / "agent-a1.jsonl", ("t1", "Read"), sidechain=True, version="2.1.247",
                  hooks=[hook_record(at(2), "PreToolUse", "t1", "Read", sidechain=True, version="2.1.247")])
    # A subagent's hook records carry isSidechain too, and must still be read.
    assert parse_file(path, path.name).hook_coverage == {
        ("2026-09-01", "2.1.247", "cli", True, "PreToolUse", "Read"): [1, 1],
        ("2026-09-01", "2.1.247", "cli", True, "PostToolUse", "Read"): [1, 0]}


def test_no_text_or_id_from_a_hook_record_is_kept(tmp_path):
    path = called(tmp_path / "s1.jsonl", ("toolu_kept_nowhere", "Bash"),
                  hooks=[hook_record(at(2), "PreToolUse", "toolu_kept_nowhere", "Bash")])
    kept = json.dumps(parse_all(tmp_path).hook_coverage.astype(str).values.tolist())
    assert PRIVATE_TEXT not in kept and PRIVATE_PATH not in kept and "toolu_kept_nowhere" not in kept


def test_every_row_carries_its_transcript_and_session(tmp_path):
    called(tmp_path / "p" / "s1.jsonl", ("t1", "Bash"))
    rows = parse_all(tmp_path).hook_coverage
    assert list(rows.columns) == list(COVERAGE_COLUMNS)
    assert rows[["source_file", "session_id", "event", "calls", "hooked"]].values.tolist() == [
        ["p/s1.jsonl", "s1", "PostToolUse", 1, 0], ["p/s1.jsonl", "s1", "PreToolUse", 1, 0]]


def stored(tmp_path, since=None):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    return load_history(tmp_path / "logs", state, claim=True, since=since).hook_coverage


def test_the_store_and_a_fresh_parse_return_the_same_hook_coverage_table(tmp_path):
    called(tmp_path / "logs" / "p" / "s1.jsonl", ("t1", "Bash"), ("t2", "mcp__gh__pr"),
           hooks=[hook_record(at(3), "PreToolUse", "t1", "Bash")])
    called(tmp_path / "logs" / "p" / "s2.jsonl", ("t3", "Read"), sidechain=True)
    kept, parsed = stored(tmp_path), parse_all(tmp_path / "logs").hook_coverage
    assert list(kept.columns) == list(parsed.columns) == list(COVERAGE_COLUMNS)
    assert kept.astype(str).values.tolist() == parsed.astype(str).values.tolist()
    assert len(kept) == 6


def test_a_transcript_that_grows_replaces_its_own_coverage_rows(tmp_path):
    path = tmp_path / "logs" / "p" / "s1.jsonl"
    called(path, ("t1", "Bash"))
    stored(tmp_path)
    called(path, ("t1", "Bash"), ("t2", "Bash"), hooks=[hook_record(at(3), "PreToolUse", "t2", "Bash")])
    kept = stored(tmp_path)
    assert kept[kept["event"] == "PreToolUse"][["calls", "hooked"]].values.tolist() == [[2, 1]]


def test_the_store_since_a_day_keeps_the_coverage_of_that_day_on(tmp_path):
    called(tmp_path / "logs" / "p" / "s1.jsonl", ("t1", "Bash"))
    assert len(stored(tmp_path, since="2026-09-01")) == 2
    assert stored(tmp_path, since="2026-09-02").empty


def test_a_store_from_before_hook_coverage_gains_the_table_and_reads_every_transcript_again(tmp_path):
    called(tmp_path / "logs" / "s1.jsonl", ("t1", "Bash"), hooks=[hook_record(at(2), "PreToolUse", "t1", "Bash")])
    path = tmp_path / "history.sqlite"
    with History(path) as history:
        history.update(tmp_path / "logs")
    db = sqlite3.connect(path)
    db.execute("DROP TABLE hook_coverage")
    db.execute("UPDATE meta SET value = '7' WHERE key = 'schema_version'")
    db.execute("UPDATE meta SET value = '10' WHERE key = 'parser_version'")
    db.commit()
    db.close()
    with History(path) as history:
        assert history.meta["schema_version"] == str(ccdrift.history.SCHEMA_VERSION) == "8"
        assert history.update(tmp_path / "logs") == 1
        assert history.hook_coverage()[["event", "calls", "hooked"]].values.tolist() == [
            ["PostToolUse", 1, 0], ["PreToolUse", 1, 1]]
