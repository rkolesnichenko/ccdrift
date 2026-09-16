"""Session starts: the context a new Claude Code session sends with its first request."""

import pandas as pd

from ccdrift.logs import parse_source
from ccdrift.sessions import context_changes_in, first_of_each, session_starts
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
