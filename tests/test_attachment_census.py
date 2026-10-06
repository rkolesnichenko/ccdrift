"""The attachment census: which attachment types each transcript holds, by day, version,
entrypoint and thread, read into rows and kept in the history store."""

import sqlite3

import ccdrift.history
from ccdrift.history import History, load_history
from ccdrift.logs import ATTACHMENT_COLUMNS, READ_ATTACHMENTS, attachment_frame, coverage_frame, parse_all, parse_file
from ccdrift.state import new_state, save_state
from tests.helpers import PRIVATE_TEXT, at, attachment, line, prompt, text, write


def kind(name, **fields):
    return {"type": name, **fields}


def session(path, *records, version="2.1.267", entrypoint="cli", sidechain=False):
    """A transcript: a prompt, a response, then attachment records of the kinds `records`."""
    write(path, [prompt(at(0), sidechain=sidechain),
                 line("m1", text(20), ts=at(1), version=version, entrypoint=entrypoint, sidechain=sidechain),
                 *(attachment(at(2 + i), record, version=version, entrypoint=entrypoint, sidechain=sidechain)
                   for i, record in enumerate(records))])
    return path


def test_each_attachment_type_is_counted_by_day_version_entrypoint_and_thread(tmp_path):
    path = session(tmp_path / "s1.jsonl", kind("date", date="2026-09-01"), kind("date", date="2026-09-01"),
                   kind("model", model="claude-opus-5"))
    assert parse_file(path, path.name).attachment_census == {
        ("2026-09-01", "2.1.267", "cli", False, "date"): 2, ("2026-09-01", "2.1.267", "cli", False, "model"): 1}


def test_an_agent_sdk_session_and_a_subagent_are_kept_apart(tmp_path):
    sdk = session(tmp_path / "s1.jsonl", kind("date"), entrypoint="sdk-py")
    sub = session(tmp_path / "s2.jsonl", kind("date"), sidechain=True)
    assert list(parse_file(sdk, sdk.name).attachment_census) == [("2026-09-01", "2.1.267", "sdk-py", False, "date")]
    assert list(parse_file(sub, sub.name).attachment_census) == [("2026-09-01", "2.1.267", "cli", True, "date")]


def test_the_census_keeps_the_type_and_none_of_the_attachment(tmp_path):
    path = session(tmp_path / "s1.jsonl", kind("credential_org", org=PRIVATE_TEXT, content=PRIVATE_TEXT))
    assert PRIVATE_TEXT not in repr(parse_file(path, path.name).attachment_census)


def test_a_record_with_no_type_or_time_is_not_counted(tmp_path):
    path = tmp_path / "s1.jsonl"
    write(path, [prompt(at(0)), line("m1", text(20), ts=at(1), version="2.1.267", entrypoint="cli"),
                 {**attachment(at(2), kind("date")), "timestamp": None},
                 {**attachment(at(3), kind("date")), "attachment": {"content": "no type"}}])
    assert parse_file(path, path.name).attachment_census == {}


def stored(tmp_path, since=None):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    return load_history(tmp_path / "logs", state, claim=True, since=since).attachment_census


def test_the_store_and_a_fresh_parse_return_the_same_attachment_census(tmp_path):
    session(tmp_path / "logs" / "p" / "s1.jsonl", kind("date"), kind("model"))
    session(tmp_path / "logs" / "p" / "s2.jsonl", kind("environment"), entrypoint="sdk-py")
    kept, parsed = stored(tmp_path), parse_all(tmp_path / "logs").attachment_census
    assert list(kept.columns) == list(parsed.columns) == list(ATTACHMENT_COLUMNS)
    assert kept.astype(str).values.tolist() == parsed.astype(str).values.tolist()
    assert sorted(parsed["type"]) == ["date", "environment", "model"]


def test_a_transcript_that_grows_replaces_its_own_census_rows(tmp_path):
    path = tmp_path / "logs" / "p" / "s1.jsonl"
    session(path, kind("date"))
    stored(tmp_path)
    session(path, kind("date"), kind("date"))
    assert stored(tmp_path)[["type", "records"]].values.tolist() == [["date", 2]]


def test_the_store_since_a_day_keeps_the_census_of_that_day_on(tmp_path):
    session(tmp_path / "logs" / "p" / "s1.jsonl", kind("date"))
    assert len(stored(tmp_path, since="2026-09-01")) == 1
    assert stored(tmp_path, since="2026-09-02").empty


def test_a_store_from_before_the_census_gains_the_table_and_reads_every_transcript_again(tmp_path):
    session(tmp_path / "logs" / "s1.jsonl", kind("date"))
    path = tmp_path / "history.sqlite"
    with History(path) as history:
        history.update(tmp_path / "logs")
    db = sqlite3.connect(path)
    db.execute("DROP TABLE attachment_census")
    db.execute("UPDATE meta SET value = '8' WHERE key = 'schema_version'")
    db.execute("UPDATE meta SET value = '11' WHERE key = 'parser_version'")
    db.commit()
    db.close()
    with History(path) as history:
        assert history.meta["schema_version"] == str(ccdrift.history.SCHEMA_VERSION) == "10"
        assert history.update(tmp_path / "logs") == 1
        assert history.attachment_census()["type"].tolist() == ["date"]


def test_the_types_ccdrift_reads_are_the_listings_the_session_start_and_the_hook_runs():
    assert READ_ATTACHMENTS == {"skill_listing", "deferred_tools_delta", "agent_listing_delta", "mcp_instructions_delta",
                                "instructions", "prompt_snapshot", "hook_success", "hook_non_blocking_error",
                                "hook_blocking_error"}


def test_rows_that_differ_only_by_version_or_entrypoint_come_out_in_one_order():
    # Found in the 0.21.0 review: sorted without them, such rows kept whatever order they came in.
    base = {"source_file": "p/s1.jsonl", "session_id": "s1", "day": "2026-09-01", "is_sidechain": False}
    census = [{**base, "version": v, "entrypoint": e, "type": "date", "records": 1}
              for v, e in (("2.1.267", "cli"), ("2.1.266", "sdk-py"), ("2.1.266", "cli"))]
    coverage = [{**base, "version": v, "entrypoint": e, "event": "PreToolUse", "tool": "Bash", "calls": 1, "hooked": 0}
                for v, e in (("2.1.267", "cli"), ("2.1.266", "sdk-py"), ("2.1.266", "cli"))]
    for frame, rows in ((attachment_frame, census), (coverage_frame, coverage)):
        forward, backward = frame(rows), frame(rows[::-1])
        assert list(zip(forward["version"], forward["entrypoint"])) == [("2.1.266", "cli"), ("2.1.266", "sdk-py"),
                                                                        ("2.1.267", "cli")]
        assert forward.values.tolist() == backward.values.tolist()
