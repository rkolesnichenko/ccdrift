"""Parsing Claude Code transcripts into responses and their metric columns."""

from datetime import date

import pandas as pd
import pytest

from ccdrift.detector import bin_metrics
from ccdrift.logs import frame, judged_turns, outside_sdk, parse_all, parse_durations, parse_source
from tests.helpers import (DAY, at, compact_boundary, line, prompt, response, stop_hook_summary, text,
                           thinking, tool_result, turn_duration, write)


def test_split_lines_of_one_response_become_one_turn(tmp_path):
    write(tmp_path / "s1.jsonl", response("m1", thinking(400), text(400), ts=at(0)))
    assert len(parse_source(tmp_path)) == 1


def test_turn_takes_the_final_output_token_count(tmp_path):
    write(tmp_path / "s1.jsonl", [
        line("m1", text(40), ts=at(0), out=3),
        line("m1", text(400), ts=at(1), out=250),
    ])
    assert parse_source(tmp_path).loc[0, "output_tokens"] == 250


def test_response_repeated_in_a_second_file_counts_once(tmp_path):
    rec = line("m1", text(40), ts=at(0))
    write(tmp_path / "a.jsonl", [rec])
    write(tmp_path / "b.jsonl", [rec])
    assert len(parse_source(tmp_path)) == 1


def test_synthetic_placeholder_messages_are_not_turns(tmp_path):
    write(tmp_path / "s1.jsonl", [
        line("m1", text(40), ts=at(0)),
        line("m2", text(40), ts=at(5), model="<synthetic>", out=0),
    ])
    assert list(parse_source(tmp_path)["model"]) == ["claude-opus-5"]


def test_lines_without_a_message_id_are_separate_turns(tmp_path):
    write(tmp_path / "s1.jsonl", [
        line(None, text(40), ts=at(0)),
        line(None, text(40), ts=at(9)),
    ])
    assert len(parse_source(tmp_path)) == 2


def test_thinking_tokens_are_estimated_from_the_signature_when_text_is_omitted(tmp_path):
    write(tmp_path / "s1.jsonl",
          response("m1", thinking(1000), text(400), ts=at(0))
          + response("m2", thinking(2000), text(400), ts=at(60)))
    tokens = parse_source(tmp_path)["thinking_tokens"]
    tokens = tokens[tokens > 0].tolist()
    assert len(tokens) == 2
    assert tokens[1] == pytest.approx(2 * tokens[0], rel=0.05)


def test_effort_metric_drops_when_responses_think_less(tmp_path):
    write(tmp_path / "s1.jsonl",
          response("m1", thinking(3000), text(400), ts=at(0))
          + response("m2", thinking(3000), text(400), ts=at(60)))
    write(tmp_path / "s2.jsonl",
          response("m3", thinking(500), text(400), ts=at(DAY), sid="s2")
          + response("m4", thinking(500), text(400), ts=at(DAY + 60), sid="s2"))
    effort = bin_metrics(parse_source(tmp_path))["effort_proxy"]
    assert effort[0] > effort[1] > 0


def test_effort_metric_counts_responses_that_skip_thinking(tmp_path):
    # Roughly half of real responses don't think at all, so a day where most
    # responses skip thinking must still register the ones that did.
    write(tmp_path / "s1.jsonl",
          response("m1", thinking(3000), text(400), ts=at(0))
          + response("m2", text(400), ts=at(60))
          + response("m3", text(400), ts=at(120)))
    assert bin_metrics(parse_source(tmp_path)).loc[0, "effort_proxy"] > 0


def test_cache_metric_uses_only_new_prompt_turns_within_the_cache_ttl(tmp_path):
    # A caching regression shows up on turns that open with a new prompt,
    # whatever the pause before them. Session start and a pause past the 1h TTL
    # are expected misses, and a tool-loop continuation carries no signal.
    write(tmp_path / "s1.jsonl", [
        prompt(at(0)),       line("m1", text(40), ts=at(0),    cache_read=0,   cache_creation=1000),
        tool_result(at(30)), line("m2", text(40), ts=at(30),   cache_read=0,   cache_creation=1000),
        prompt(at(1230)),    line("m3", text(40), ts=at(1230), cache_read=250, cache_creation=750),
        prompt(at(1290)),    line("m4", text(40), ts=at(1290), cache_read=750, cache_creation=250),
        prompt(at(9000)),    line("m5", text(40), ts=at(9000), cache_read=0,   cache_creation=1000),
    ])
    metrics = bin_metrics(parse_source(tmp_path))
    assert metrics.loc[0, "cache_ratio"] == pytest.approx(0.5)


def test_cache_metric_ignores_subagent_turns(tmp_path):
    # Subagents run on a short cache TTL (88% of their turns resumed after 5 min
    # miss), so including them would track the subagent share, not regressions.
    write(tmp_path / "s1.jsonl", [
        prompt(at(0)),    line("m1", text(40), ts=at(0),    cache_read=0,    cache_creation=1000),
        prompt(at(1200)), line("m2", text(40), ts=at(1200), cache_read=1000, cache_creation=0),
    ])
    write(tmp_path / "s1" / "subagents" / "agent-a.jsonl", [
        prompt(at(100), sidechain=True),  line("a1", text(40), ts=at(100),  cache_read=0, cache_creation=1000, sidechain=True),
        prompt(at(1300), sidechain=True), line("a2", text(40), ts=at(1300), cache_read=0, cache_creation=1000, sidechain=True),
    ])
    metrics = bin_metrics(parse_source(tmp_path))
    assert metrics.loc[0, "cache_ratio"] == pytest.approx(1.0)


def test_cache_metric_skips_the_turn_after_a_compaction(tmp_path):
    # Compaction rewrites the conversation, so the next turn misses by design.
    write(tmp_path / "s1.jsonl", [
        prompt(at(0)),   line("m1", text(40), ts=at(0),   cache_read=0,    cache_creation=1000),
        prompt(at(60)),  line("m2", text(40), ts=at(60),  cache_read=1000, cache_creation=0),
        compact_boundary(at(100)),
        prompt(at(120)), line("m3", text(40), ts=at(120), cache_read=100,  cache_creation=900),
    ])
    metrics = bin_metrics(parse_source(tmp_path))
    assert metrics.loc[0, "cache_ratio"] == pytest.approx(1.0)


def test_idle_gap_is_measured_within_one_transcript(tmp_path):
    # A subagent has its own transcript and cache prefix, so its activity must
    # not make a paused main thread look warm.
    write(tmp_path / "s1.jsonl", [line("m1", text(40), ts=at(0)),
                                  line("m3", text(40), ts=at(1200))])
    write(tmp_path / "s1" / "subagents" / "agent-a.jsonl",
          [line("m2", text(40), ts=at(600), sidechain=True)])
    df = parse_source(tmp_path)
    assert df.loc[~df["is_sidechain"], "gap_seconds"].dropna().tolist() == [1200]


def test_tool_loop_turns_are_judged_against_what_the_response_before_had_cached(tmp_path):
    # A tool-loop turn should read back everything the response before it cached.
    # Prompts, compactions, pauses past 5 minutes and responses without cache token
    # counts (on the turn or the one before) aren't judged.
    write(tmp_path / "s1.jsonl", [
        prompt(at(0)),        line("m1", text(40), ts=at(0),   cache_read=0,    cache_creation=1000),
        tool_result(at(10)),  line("m2", text(40), ts=at(10),  cache_read=1000, cache_creation=0),
        tool_result(at(20)),  line("m3", text(40), ts=at(20),  cache_read=500,  cache_creation=600),
        tool_result(at(30)),  line("m4", text(40), ts=at(30),  cache_read=549,  cache_creation=700),
        tool_result(at(330)), line("m5", text(40), ts=at(330), cache_read=1249, cache_creation=0),
        tool_result(at(631)), line("m6", text(40), ts=at(631), cache_read=0,    cache_creation=1300),
        prompt(at(640)),      line("m7", text(40), ts=at(640), cache_read=0,    cache_creation=1300),
        compact_boundary(at(645)),
        tool_result(at(650)), line("m8", text(40), ts=at(650), cache_read=0,    cache_creation=500),
        tool_result(at(660)), line("m9", text(40), ts=at(660)),
        tool_result(at(670)), line("m10", text(40), ts=at(670), cache_read=0,   cache_creation=500),
    ])
    df = parse_source(tmp_path)
    assert df["loop_turn"].tolist() == [False, True, True, True, True, False, False, False, False, False]
    # m3 reads exactly half of the 1,000 tokens cached before it; m4 reads 549 of 1,100.
    assert df["is_loop_miss"].tolist() == [False, False, False, True, False, False, False, False, False, False]


def test_a_tool_loop_turn_is_judged_against_the_response_before_it_in_its_own_transcript(tmp_path):
    # A subagent keeps its own cache prefix, so what the main thread should read back is
    # what its own previous response cached, not the subagent's that ran in between.
    write(tmp_path / "s1.jsonl", [
        prompt(at(0)),       line("m1", text(40), ts=at(0),  cache_read=0,    cache_creation=1000),
        tool_result(at(60)), line("m2", text(40), ts=at(60), cache_read=1000, cache_creation=50),
    ])
    write(tmp_path / "s1" / "subagents" / "agent-a.jsonl", [
        prompt(at(10), sidechain=True),
        line("a1", text(40), ts=at(10), cache_read=0, cache_creation=200, sidechain=True),
        tool_result(at(20), sidechain=True),
        line("a2", text(40), ts=at(20), cache_read=90, cache_creation=300, sidechain=True),
    ])
    df = parse_source(tmp_path).sort_values("timestamp")
    assert df["prev_cached"].fillna(-1).tolist() == [-1, -1, 200, 1000]
    assert df["is_loop_miss"].tolist() == [False, False, True, False]


def test_responses_carry_the_version_entrypoint_and_settings_claude_code_logged(tmp_path):
    write(tmp_path / "s1.jsonl", [
        line("m1", thinking(40), ts=at(0)),
        line("m1", text(40), ts=at(1), version="2.1.233", entrypoint="cli", effort="xhigh",
             speed="standard", service_tier="standard"),
    ])
    row = parse_source(tmp_path).iloc[0]
    assert (row["version"], row["entrypoint"], row["effort"], row["speed"], row["service_tier"]) == \
        ("2.1.233", "cli", "xhigh", "standard", "standard")


def test_control_characters_are_dropped_from_logged_text(tmp_path):
    # Agent names come from agent files in a repository or plugin, models from whatever
    # endpoint answered; the report, the log and notifications print them.
    write(tmp_path / "s1.jsonl", [line("m1", text(40), ts=at(0), model="claude-\x1b[31mopus",
                                       agent_type="auditor\x1b]0;title\x07", version="2.1.\x9b260\n")])
    row = parse_source(tmp_path).iloc[0]
    assert (row["model"], row["agent_type"], row["version"]) == ("claude-[31mopus", "auditor]0;title", "2.1.260")


def test_settings_missing_from_older_versions_are_left_empty(tmp_path):
    write(tmp_path / "s1.jsonl", [line("m1", text(40), ts=at(0))])
    row = parse_source(tmp_path).iloc[0]
    assert row["version"] is None and row["effort"] is None


def test_cache_writes_are_split_into_the_1_hour_and_5_minute_tiers(tmp_path):
    # Claude Code writes the main thread's cache for an hour and subagents' and
    # Agent SDK sessions' for 5 minutes.
    write(tmp_path / "s1.jsonl", [
        line("m1", text(40), ts=at(0), cache_creation=900, cache_1h=900, cache_5m=0),
        line("m2", text(40), ts=at(60), cache_creation=900, cache_1h=0, cache_5m=900),
        line("m3", text(40), ts=at(120), cache_read=900, cache_1h=0, cache_5m=0),
        line("m4", text(40), ts=at(180), cache_creation=900),
    ])
    assert parse_source(tmp_path)["cache_tier"].tolist() == ["1h", "5m", None, None]


def test_logged_thinking_tokens_are_read_from_the_final_line(tmp_path):
    write(tmp_path / "s1.jsonl", [
        line("m1", thinking(400), ts=at(0)),
        line("m1", text(40), ts=at(1), thinking_logged=130),
        line("m2", text(40), ts=at(60)),
    ])
    logged = parse_source(tmp_path)["thinking_logged"].tolist()
    assert logged[0] == 130.0
    assert pd.isna(logged[1])


def test_a_miss_is_a_prompt_turn_that_reads_under_half_its_input_from_cache(tmp_path):
    # Misses are all-or-nothing: in real logs cutoffs of 0.3, 0.5 and 0.8
    # selected 2.82%, 2.82% and 2.88% of prompt turns.
    write(tmp_path / "s1.jsonl", [
        prompt(at(0)),   line("m1", text(40), ts=at(0),   cache_creation=1000),
        prompt(at(60)),  line("m2", text(40), ts=at(60),  cache_read=400, cache_creation=600),
        prompt(at(120)), line("m3", text(40), ts=at(120), cache_read=990, cache_creation=10),
    ])
    assert parse_source(tmp_path)["is_miss"].tolist() == [False, True, False]


def test_a_response_in_two_transcripts_counts_from_the_path_that_sorts_first(tmp_path):
    # Paths compare as strings, so "a-b.jsonl" comes before "a/b.jsonl"; the
    # history store relies on the same order.
    rec = line("m1", text(40), ts=at(0))
    write(tmp_path / "a" / "b.jsonl", [rec])
    write(tmp_path / "a-b.jsonl", [rec])
    assert parse_source(tmp_path)["source_file"].tolist() == ["a-b.jsonl"]


def test_every_response_in_a_transcript_takes_its_first_session_id(tmp_path):
    write(tmp_path / "s1.jsonl", [line("m1", text(40), ts=at(0), sid="first"),
                                  line("m2", text(40), ts=at(60), sid="second")])
    assert parse_source(tmp_path)["session_id"].tolist() == ["first", "first"]


def test_turn_durations_are_read_with_their_day_and_version(tmp_path):
    write(tmp_path / "s1.jsonl", [
        prompt(at(0)), line("m1", text(40), ts=at(0)),
        turn_duration(at(90), 90000, 3, uuid="d1", version="2.1.233"),
        turn_duration(at(DAY), 30000, 1),
    ])
    durations = parse_durations(tmp_path)
    assert durations[["day", "version", "duration_ms", "message_count"]].values.tolist() == [
        ["2026-09-01", "2.1.233", 90000.0, 3], ["2026-09-02", "2.1.226", 30000.0, 1]]


def test_frame_coerces_integer_flags_from_the_history_store_to_bool():
    # The history store (next task) loads rows from SQLite, where boolean columns
    # come back as 0/1 integers. add_ratios combines new_prompt/after_compaction
    # with & and ~, which is silently wrong on raw ints, so frame() must coerce
    # them to bool the same way it already does for is_sidechain.
    base = {
        "key": "m1", "timestamp": at(0), "model": "claude-opus-5",
        "version": None, "entrypoint": None, "effort": None, "speed": None, "service_tier": None,
        "input_tokens": 10.0, "output_tokens": 40.0, "cache_creation": 0.0, "cache_read": 0.0,
        "cache_1h": 0.0, "cache_5m": 0.0, "thinking_logged": None,
        "signature_chars": 0, "visible_chars": 40, "n_mcp_calls": 0,
        "is_sidechain": 0, "new_prompt": 0, "after_compaction": 0,
        "source_file": "s1.jsonl", "session_id": "s1",
    }
    row1 = dict(base)
    row2 = {**base, "key": "m2", "timestamp": at(60), "cache_read": 900.0, "cache_creation": 100.0,
            "new_prompt": 1, "after_compaction": 0}
    row = frame([row1, row2]).iloc[1]
    assert row["prompt_within_ttl"]
    assert not row["is_miss"]


def test_judged_turns_leave_out_subagents_sdk_sessions_and_the_current_day(tmp_path):
    # Agent SDK sessions are the user's own scripts, on the 5-minute cache.
    write(tmp_path / "cli.jsonl", [line("m1", text(40), ts=at(0), entrypoint="cli"),
                                   line("m2", text(40), ts=at(DAY), entrypoint="cli")])
    write(tmp_path / "old.jsonl", [line("m3", text(40), ts=at(60))])
    write(tmp_path / "sdk.jsonl", [line("m4", text(40), ts=at(120), entrypoint="sdk-py")])
    write(tmp_path / "cli" / "subagents" / "agent-a.jsonl",
          [line("a1", text(40), ts=at(180), sidechain=True, entrypoint="cli")])
    turns = judged_turns(parse_source(tmp_path), date(2026, 9, 2))
    assert sorted(turns["source_file"]) == ["cli.jsonl", "old.jsonl"]


def test_stop_hook_summaries_become_hook_runs_without_their_commands(tmp_path):
    write(tmp_path / "s1.jsonl", [
        stop_hook_summary(at(0), 2, durations=(1200, 800), uuid="h1"),
        stop_hook_summary(at(60), 1, errors=("hook exited with 1",), uuid="h2"),
    ])
    runs = parse_all(tmp_path).hook_runs
    assert runs[["day", "hook_count", "error_count", "prevented"]].values.tolist() == [
        ["2026-09-01", 2, 0, False], ["2026-09-01", 1, 1, False]]
    assert runs["duration_ms"].tolist()[0] == 2000.0
    assert pd.isna(runs["duration_ms"].tolist()[1])
    assert "secret-hook" not in runs.to_string()


def test_infinite_or_nan_counts_read_as_0(tmp_path):
    # JSON parsers accept Infinity and NaN, and int() of them raised, failing every check
    # while such a transcript was on disk.
    hooks = stop_hook_summary(at(10), 1, uuid="h1")
    hooks["hookCount"] = float("inf")
    write(tmp_path / "s1.jsonl", [prompt(at(0)), line("m1", text(40), ts=at(0)), hooks,
                                  turn_duration(at(30), float("inf"), float("nan"), uuid="d1")])
    tables = parse_all(tmp_path)
    assert tables.hook_runs["hook_count"].tolist() == [0]
    assert tables.durations[["duration_ms", "message_count"]].values.tolist() == [[0.0, 0]]


def test_compaction_boundaries_become_compactions_and_still_mark_the_next_turn(tmp_path):
    write(tmp_path / "s1.jsonl", [
        prompt(at(0)), line("m1", text(40), ts=at(0)),
        compact_boundary(at(100), trigger="auto", pre_tokens=971_000),
        prompt(at(120)), line("m2", text(40), ts=at(120)),
    ])
    tables = parse_all(tmp_path)
    assert tables.compactions[["day", "trigger", "pre_tokens"]].values.tolist() == [["2026-09-01", "auto", 971_000.0]]
    assert tables.responses["after_compaction"].tolist() == [False, True]


def test_parse_all_reads_what_the_single_readers_read(tmp_path):
    write(tmp_path / "s1.jsonl", [prompt(at(0)), line("m1", text(40), ts=at(0)),
                                  turn_duration(at(30), 30000, 2, uuid="d1")])
    tables = parse_all(tmp_path)
    pd.testing.assert_frame_equal(tables.responses, parse_source(tmp_path))
    pd.testing.assert_frame_equal(tables.durations, parse_durations(tmp_path))


def test_agent_sdk_sessions_are_told_apart_by_their_entrypoint():
    # Transcripts from before Claude Code logged an entrypoint count as the CLI.
    rows = pd.DataFrame({"entrypoint": ["cli", "sdk-py", None, "SDK-ts"]})
    assert outside_sdk(rows).tolist() == [True, False, True, True]
    assert outside_sdk(pd.DataFrame({"day": ["2026-09-01"]})).tolist() == [True]
