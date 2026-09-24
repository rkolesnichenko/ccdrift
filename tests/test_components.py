"""What a session started with: the attachment records Claude Code writes before its first
response, read into one row per transcript and kept in the history store."""

import json
import sqlite3

import ccdrift.history
from ccdrift.components import compare_components
from ccdrift.history import History, load_history
from ccdrift.logs import COMPONENT_COLUMNS, components_frame, new_components, parse_all, parse_file
from ccdrift.state import new_state, save_state
from ccdrift.texts import component_lines, context_message
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


def rows(prefix, count, **values):
    """`count` component rows of transcripts `<prefix>0.jsonl` onwards, each with `values`
    and everything else as a session start on 2.1.250 logs it: skills, deferred tools and
    agent types, no CLAUDE.md, system prompt or tool definitions."""
    base = {"skills": ["review"], "skills_chars": 1000, "deferred": ["Read"], "deferred_chars": 20,
            "agents": ["Plan"], "agents_chars": 30}
    return [{**new_components(f"{prefix}{i}.jsonl"), **base, **values} for i in range(count)]


def compared(window, baseline):
    table = components_frame(window + baseline)
    return compare_components(table, [row["source_file"] for row in window], [row["source_file"] for row in baseline])


def test_a_name_in_most_of_the_window_and_under_half_the_baseline_is_added():
    window = rows("w", 2, agents=["Plan", "Explore"]) + rows("x", 1)
    baseline = rows("b", 4, agents=["Plan", "Explore"]) + rows("c", 6)
    assert compared(window, baseline)["added"] == {"agents": ["Explore"]}


def test_a_name_in_one_window_session_is_a_one_off_not_an_addition():
    assert compared(rows("w", 1, agents=["Plan", "Explore"]) + rows("x", 2), rows("b", 10))["added"] == {}


def test_a_name_in_half_the_baseline_is_not_counted_either_way():
    window = rows("w", 3, skills=["review", "new"])
    assert compared(window, rows("b", 5, skills=["review", "new"]) + rows("c", 5))["added"] == {}


def test_a_name_in_most_of_the_baseline_and_under_half_the_window_is_removed():
    window = rows("w", 2, skills=[]) + rows("x", 1)
    assert compared(window, rows("b", 10))["removed"] == {"skills": ["review"]}


def test_mcp_tools_are_grouped_by_their_server_and_built_in_deferred_tools_kept_apart():
    window = rows("w", 3, deferred=["Read", "Monitor", "mcp__gh__pr", "mcp__gh__issue", "mcp__jira__get"])
    assert compared(window, rows("b", 10))["added"] == {
        "mcp_tools": {"gh": ["mcp__gh__issue", "mcp__gh__pr"], "jira": ["mcp__jira__get"]}, "deferred": ["Monitor"]}


def test_a_size_that_moved_is_compared_by_its_medians():
    window = rows("w", 2, skills_chars=1300) + rows("x", 1, skills_chars=900)
    assert compared(window, rows("b", 10))["sizes"] == {"skills": (1000.0, 1300.0)}


def test_a_part_logged_in_half_of_one_sides_sessions_or_fewer_is_unknown_and_left_out():
    window = rows("w", 3, claude_md_files=1, claude_md_chars=5000)
    baseline = rows("b", 5, claude_md_files=1, claude_md_chars=1000) + rows("c", 5)
    what = compared(window, baseline)
    assert "claude_md" in what["unknown"] and "claude_md" not in what["sizes"]
    assert what["unknown"] == ["claude_md", "system", "tools"]


def test_mcp_instructions_count_as_none_wherever_the_deferred_tools_are_logged():
    what = compared(rows("w", 3, mcp=["gh"], mcp_chars=500), rows("b", 10))
    assert what["added"] == {"mcp": ["gh"]} and what["sizes"] == {"mcp": (0.0, 500.0)}
    assert "mcp" not in what["unknown"]


def test_a_session_without_a_row_counts_as_unknown_on_every_part():
    window = rows("w", 3)
    what = compare_components(components_frame(window + rows("b", 3)), [r["source_file"] for r in window],
                              ["gone0.jsonl", "gone1.jsonl", "gone2.jsonl", "b0.jsonl", "b1.jsonl", "b2.jsonl"])
    assert what is None


def test_nothing_to_compare_when_no_part_is_logged_on_both_sides():
    empty = [new_components(f"w{i}.jsonl") for i in range(3)]
    assert compared(empty, rows("b", 10)) is None
    assert compare_components(components_frame([]), ["w0.jsonl"], ["b0.jsonl"]) is None


CHANGE = {"since": "2026-08-27", "from": 106_031.0, "to": 128_699.0, "days": ["2026-08-27", "2026-08-29"],
          "projects": ["-Users-me-app"], "of_projects": 1, "new_version": False}
STEP = {"added": {"agents": [f"agent-{i}" for i in range(11)], "skills": ["a", "b", "c", "d"],
                  "mcp_tools": {"srv-one": [f"mcp__srv-one__t{i}" for i in range(9)],
                                "srv-two": ["mcp__srv-two__a", "mcp__srv-two__b"]}},
        "removed": {}, "sizes": {"skills": (21_077.0, 22_976.0), "agents": (1_900.0, 3_100.0)},
        "unknown": ["claude_md", "system", "tools"]}


def test_the_alert_counts_what_changed_and_names_none_of_it():
    message = context_message(CHANGE, [], STEP)
    assert message.startswith(context_message(CHANGE, []))
    assert message.endswith(
        " Of what Claude Code logs about a session's start, 11 agent types, 4 skills and 11 MCP tools were added, "
        "about 3.1k more characters, though the logs can't say how many of the tokens that is. Claude Code didn't "
        "log CLAUDE.md files, the system prompt or tool definitions in every session compared, so ccdrift couldn't "
        "compare those parts.")
    assert not any(name in message for name in ("agent-0", "srv-one", "srv-two", "mcp__"))


def test_the_detail_lines_name_what_changed():
    assert component_lines(STEP) == [
        "agent types added: " + ", ".join(f"agent-{i}" for i in range(11)),
        "skills added: a, b, c, d",
        "MCP tools added: srv-one (9), srv-two (2)",
        "the skills listing: 21,077 -> 22,976 characters",
        "the agent types: 1,900 -> 3,100 characters",
    ]


def test_a_removal_and_a_single_addition_read_as_one_sentence():
    what = {"added": {"skills": ["a"]}, "removed": {"agents": ["x", "y"]}, "sizes": {"skills": (100.0, 90.0)},
            "unknown": []}
    assert context_message(CHANGE, [], what).endswith(
        " Of what Claude Code logs about a session's start, 1 skill was added and 2 agent types removed, "
        "about 10 fewer characters, though the logs can't say how many of the tokens that is.")
    assert component_lines(what)[:2] == ["skills added: a", "agent types removed: x, y"]


def test_a_size_that_moved_with_no_name_added_or_removed_is_still_said():
    what = {"added": {}, "removed": {}, "sizes": {"system": (4_500.0, 6_000.0)}, "unknown": []}
    assert context_message(CHANGE, [], what).endswith(
        " Of what Claude Code logs about a session's start, only the system prompt changed, about 1.5k more "
        "characters, though the logs can't say how many of the tokens that is.")


def test_nothing_logged_having_changed_points_away_from_skills_agents_and_claude_md():
    what = {"added": {}, "removed": {}, "sizes": {}, "unknown": ["tools"]}
    assert context_message(CHANGE, [], what).endswith(
        " Nothing Claude Code logs about a session's start changed, so the step is in what it doesn't log. Claude "
        "Code didn't log tool definitions in every session compared, so ccdrift couldn't compare that part.")


def test_the_alert_says_nothing_more_when_there_was_nothing_to_compare():
    assert context_message(CHANGE, [], None) == context_message(CHANGE, [])
    assert component_lines(None) == []
