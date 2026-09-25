"""PARSER_VERSION moves with what parse_file returns: the store skips a transcript it has
read unless the version changed, so output that changes without a bump leaves stale rows
that no check notices."""

import hashlib
import json
import math
import numbers
from datetime import date, datetime

from ccdrift.history import PARSER_VERSION
from ccdrift.logs import parse_file
from tests.helpers import (DAY, agent_listing, api_error, at, attachment, compact_boundary, cost_state,
                           deferred_tools, hook_record, instructions, line, mcp_instructions, no_response_stub,
                           prompt, prompt_snapshot, retry_record, skill_listing, stop_hook_summary, text, thinking,
                           tool_result, tool_use, turn_duration, write)

# The version, and a digest of parse_file on the transcripts below at that version. When a
# parser change moves the digest, bump PARSER_VERSION and extend its numbered comment
# (history.py), then put both new values here.
PINNED = (11, "b5a606f341910c0f42f9263e148c41764257e6b8a5733eda426f222ac6c4fbab")


def plain(value):
    """`value` as JSON can hold it, the same on every Python, numpy and pandas the tests run on."""
    if isinstance(value, dict):
        return sorted(([plain(k), plain(v)] for k, v in value.items()), key=repr)
    if isinstance(value, (set, frozenset)):
        return sorted((plain(v) for v in value), key=repr)
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return "nan" if math.isnan(value) else repr(float(value))
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"no plain form for {type(value).__name__}")


def parsed(path, rel):
    result = parse_file(path, rel)
    return {name: plain(getattr(result, name)) for name in sorted(vars(result))}


def transcripts(root):
    """A main-thread session and one of its subagents holding every record kind the tests
    build: responses with each optional field, prompts, tool calls and results, hooks,
    compactions, banners, retries, stop hooks, turn durations, cost and what the session
    started with."""
    main = [attachment(at(0), skill_listing(["review", "plan"])), attachment(at(0), deferred_tools(["mcp__s__t", "Read"])),
            attachment(at(0), agent_listing(["Explore"])), attachment(at(0), mcp_instructions(["s"])),
            attachment(at(0), instructions(300, 40)), attachment(at(0), prompt_snapshot(900, 100)),
            prompt(at(1)),
            line("m1", thinking(400), ts=at(2), cache_read=900, cache_creation=100, cache_1h=100, cache_5m=0,
                 version="2.1.281", entrypoint="cli", effort="xhigh", thinking_logged=250, speed="standard",
                 service_tier="standard", stop_reason="tool_use", miss_reason="messages_changed", skill="review",
                 plugin="p", mcp_server="s", branch="main", extra={"newField": 1}),
            line("m1", tool_use("t1", "Bash"), ts=at(2), cache_read=900, cache_creation=100, cache_1h=100, cache_5m=0,
                 version="2.1.281", entrypoint="cli"),
            hook_record(at(3), "PreToolUse", "t1", "Bash", version="2.1.281"),
            tool_result(at(4)), hook_record(at(4), "PostToolUse", "t1", "Bash", kind="hook_non_blocking_error",
                                            version="2.1.281"),
            attachment(at(4), prompt_snapshot(900, 100, tools={"Bash": 500, "Read": 300}), version="2.1.281"),
            line("m2", text(80), ts=at(5), cache_read=1000, cache_creation=0, version="2.1.281", entrypoint="cli",
                 stop_reason="end_turn"),
            stop_hook_summary(at(6), 2, errors=("exit 1",), durations=(40,), uuid="stop-1"),
            turn_duration(at(7), 5000, 4, uuid="dur-1"),
            api_error(at(8), version="2.1.281"), api_error(at(9), kind="slept", version="2.1.281"),
            retry_record(at(10), version="2.1.281"), retry_record(at(11), version="2.1.281", attempt=2),
            no_response_stub(at(12)),
            compact_boundary(at(13), trigger="auto", pre_tokens=150_000, version="2.1.281"),
            prompt(at(14)),
            line("m3", text(20), ts=at(15), model="claude-opus-5-5", cache_read=0, cache_creation=5000,
                 version="2.1.281", entrypoint="cli", stop_reason="max_tokens"),
            cost_state(at(16), {"claude-opus-5": {"input": 30, "output": 300, "cache_read": 2800,
                                                  "cache_creation": 200, "costUSD": 0.12},
                                "claude-opus-5-5[1m]": {"input": 10, "output": 100, "cache_creation": 5000,
                                                        "costUSD": 0.05}}),
            "not json {", ["not", "an", "object"]]
    sub = [prompt(at(DAY), sidechain=True),
           line("a1", tool_use("t2", "Read"), ts=at(DAY + 1), sidechain=True, model="claude-haiku-4-5",
                cache_read=500, cache_creation=50, cache_5m=50, version="2.1.281", entrypoint="cli",
                agent_type="Explore"),
           hook_record(at(DAY + 2), "PreToolUse", "t2", "Read", version="2.1.281", sidechain=True)]
    write(root / "p" / "s1.jsonl", main)
    write(root / "p" / "s1" / "subagents" / "agent-a1.jsonl", sub)
    # write() dumps each record, so the two lines that aren't JSON objects are put back raw.
    path = root / "p" / "s1.jsonl"
    path.write_text(path.read_text().replace('"not json {"', "not json {"))


def test_parse_file_output_changes_only_with_a_parser_version_bump(tmp_path):
    transcripts(tmp_path)
    output = {rel: parsed(tmp_path / rel, rel) for rel in ("p/s1.jsonl", "p/s1/subagents/agent-a1.jsonl")}
    digest = hashlib.sha256(json.dumps(output, sort_keys=True).encode()).hexdigest()
    assert (PARSER_VERSION, digest) == PINNED
