"""Parsing Claude Code transcripts into responses and their metric columns."""

import pytest

from ccdrift.detector import bin_metrics
from ccdrift.logs import parse_source
from tests.helpers import (DAY, at, compact_boundary, line, prompt, response, text, thinking,
                           tool_result, write)


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
