"""Session starts: the context a new Claude Code session sends with its first request."""

from datetime import date

import pandas as pd

from ccdrift.logs import parse_source
from ccdrift.sessions import (ContextChange, context_alerts, context_changes_in, context_message, first_of_each,
                              session_starts)
from ccdrift.state import new_state
from tests.helpers import DAY, at, line, nth_day, prompt, text, write


def test_a_session_start_is_the_first_main_thread_cli_response_of_each_transcript(tmp_path):
    write(tmp_path / "a.jsonl", [prompt(at(0)), line("a1", text(40), ts=at(0), cache_creation=90_000, version="2.1.261"),
                                 prompt(at(60)), line("a2", text(40), ts=at(60), cache_read=95_000)])
    write(tmp_path / "b.jsonl", [prompt(at(DAY)), line("b1", text(40), ts=at(DAY), cache_read=40_000,
                                                       cache_creation=14_000, version="2.1.267")])
    write(tmp_path / "a" / "subagents" / "agent-x.jsonl",
          [line("x1", text(40), ts=at(-60), sidechain=True, cache_creation=5_000)])
    write(tmp_path / "sdk.jsonl", [line("s1", text(40), ts=at(30), entrypoint="sdk-py", cache_creation=7_000)])
    starts = session_starts(parse_source(tmp_path))
    assert starts[["source_file", "day", "version", "prompt_tokens"]].values.tolist() == [
        ["a.jsonl", "2026-09-01", "2.1.261", 90_010.0], ["b.jsonl", "2026-09-02", "2.1.267", 54_010.0]]


def test_a_first_response_without_token_counts_is_not_a_session_start(tmp_path):
    # If Claude Code stops logging usage, every start reads as 0 tokens: not a change in context.
    unlogged = line("b1", text(40), ts=at(DAY), version="2.1.280")
    del unlogged["message"]["usage"]
    write(tmp_path / "a.jsonl", [prompt(at(0)), line("a1", text(40), ts=at(0), cache_creation=90_000)])
    write(tmp_path / "b.jsonl", [prompt(at(DAY)), unlogged])
    assert session_starts(parse_source(tmp_path))["source_file"].tolist() == ["a.jsonl"]


def starts_of(tokens, versions=None):
    return pd.DataFrame({"day": [nth_day(i) for i in range(len(tokens))],
                         "version": versions or ["2.1.261"] * len(tokens),
                         "prompt_tokens": [float(t) for t in tokens]})


def test_a_step_in_session_start_size_is_found_once():
    starts = starts_of([128_000] * 8 + [54_000] * 5)
    changes = context_changes_in(starts)
    assert [(c.since, c.before, c.after) for c in changes[:1]] == [("2026-09-09", 128_000.0, 54_000.0)]
    assert [(c.since, c.up) for c in first_of_each(changes)] == [("2026-09-09", False)]


def test_one_outlier_in_the_window_is_not_a_step():
    # Two small sessions and one normal one: the median moved, but not every session did.
    assert context_changes_in(starts_of([128_000] * 8 + [60_000, 60_000, 122_000])) == []


def test_a_step_already_recorded_is_not_found_again():
    changes = context_changes_in(starts_of([128_000] * 8 + [54_000] * 3))
    recorded = [{"since": "2026-09-05", "from": 130_000.0, "to": 50_000.0}]
    assert first_of_each(changes, recorded) == []


def test_a_stale_recorded_step_re_detected_by_sparse_sessions_is_not_found_again():
    # With under one session a day, the same step can still be showing up in windows
    # more than DEDUPE_DAYS after it was first recorded, as the baseline slowly refills.
    change = ContextChange("2026-09-30", "2026-10-02", 128_000.0, 54_000.0)
    recorded = [{"since": "2026-09-10", "from": 128_000.0, "to": 54_000.0}]
    assert first_of_each([change], recorded) == []


def test_a_later_genuine_step_in_the_same_direction_is_still_found():
    change = ContextChange("2026-09-30", "2026-10-02", 54_000.0, 30_000.0)
    recorded = [{"since": "2026-09-10", "from": 128_000.0, "to": 54_000.0}]
    assert first_of_each([change], recorded) == [change]


def test_a_step_recorded_from_or_to_0_tokens_is_compared_by_date_only():
    # A check that counted 0-token starts could record one; dividing by it failed every later check.
    down = ContextChange("2026-09-30", "2026-10-02", 54_000.0, 30_000.0)
    up = ContextChange("2026-09-30", "2026-10-02", 54_000.0, 130_000.0)
    recorded = [{"since": "2026-09-10", "from": 128_000.0, "to": 0.0},
                {"since": "2026-09-12", "from": 0.0, "to": 54_000.0}]
    assert first_of_each([down, up], recorded) == [down, up]


def test_a_session_start_step_is_alerted_once_with_its_sizes():
    state = new_state()
    starts = starts_of([128_000] * 8 + [54_000] * 3)
    changes = context_alerts(starts, state, date(2026, 9, 12))
    assert [(c["since"], c["from"], c["to"], c["days"]) for c in changes] == [
        ("2026-09-09", 128_000.0, 54_000.0, ["2026-09-09", "2026-09-11"])]
    assert context_alerts(starts, state, date(2026, 9, 12)) == []
    assert context_message(changes[0], ["2.1.267 (since 09-09)"]) == (
        "New sessions start with ~54k tokens of context from 2026-09-09, on Claude Code 2.1.267 (since 09-09), "
        "down from ~130k. Your MCP servers, plugins or CLAUDE.md can change this too.")


def test_steps_from_weeks_ago_and_on_the_current_day_are_not_alerted():
    starts = starts_of([128_000] * 8 + [54_000] * 3)
    assert context_alerts(starts, new_state(), date(2026, 10, 10)) == []
    assert context_alerts(starts, new_state(), date(2026, 9, 11)) == []
