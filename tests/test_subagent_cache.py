"""The subagent cache metric: what subagent tool-loop turns read back, judged daily."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ccdrift.detector import METRICS, SUBAGENT_METRICS, DetectorConfig, bin_metrics, detect
from ccdrift.logs import judged_subagent_loops, parse_source
from tests.helpers import at, line, prompt, text, tool_result, write


def subagent_turns(path, reads, sid="s1", entrypoint="cli", agent_type=None, day=0):
    """A subagent transcript: a prompt, a response writing 1000 tokens, then one tool-loop
    turn per entry of `reads`, each reading that many tokens and writing 100, a minute apart."""
    records = [prompt(at(day * 86400), sid=sid, sidechain=True),
               line(f"{sid}-0", text(40), ts=at(day * 86400), sid=sid, sidechain=True, cache_creation=1000,
                    entrypoint=entrypoint, agent_type=agent_type)]
    for k, read in enumerate(reads, start=1):
        ts = at(day * 86400 + 60 * k)
        records += [tool_result(ts, sid=sid, sidechain=True),
                    line(f"{sid}-{k}", text(40), ts=ts, sid=sid, sidechain=True, cache_read=read, cache_creation=100,
                         entrypoint=entrypoint, agent_type=agent_type)]
    write(path / sid / "subagents" / "agent-a.jsonl", records)


def test_a_loop_turn_reads_back_its_share_of_what_the_response_before_it_had_cached(tmp_path):
    # The first loop turn reads all 1000 the opening response wrote; the second reads 550 of the 1100 cached.
    subagent_turns(tmp_path, [1000, 550])
    df = parse_source(tmp_path).sort_values("timestamp", kind="stable")
    assert df["loop_readback"].tolist()[1:] == [1.0, 0.5]


def test_a_loop_turn_that_reads_more_than_was_cached_reads_back_all_of_it(tmp_path):
    subagent_turns(tmp_path, [1500])
    df = parse_source(tmp_path).sort_values("timestamp", kind="stable")
    assert df["cache_read"].iloc[1] > df["prev_cached"].iloc[1]
    assert df["loop_readback"].iloc[1] == 1.0


def test_a_turn_outside_the_tool_loop_has_no_read_back(tmp_path):
    # The second response opens with a prompt, so it is no loop turn, though it reads what the first had cached.
    write(tmp_path / "s1.jsonl", [prompt(at(0)), line("m0", text(40), ts=at(0), cache_creation=1000),
                                  prompt(at(60)), line("m1", text(40), ts=at(60), cache_read=1000)])
    df = parse_source(tmp_path).sort_values("timestamp", kind="stable")
    assert df["prev_cached"].iloc[1] == 1000 and not df["loop_turn"].iloc[1]
    assert df["loop_readback"].isna().all()


def test_the_subagent_loops_judged_are_those_of_complete_days_outside_the_sdk_with_or_without_an_agent_type(tmp_path):
    subagent_turns(tmp_path, [1000, 1000], sid="typed", agent_type="Explore")
    subagent_turns(tmp_path, [1000, 1000], sid="untyped")
    subagent_turns(tmp_path, [1000, 1000], sid="sdk", entrypoint="sdk-py")
    subagent_turns(tmp_path, [1000, 1000], sid="today", day=1)
    write(tmp_path / "main.jsonl", [prompt(at(0), sid="main"), line("m0", text(40), ts=at(0), sid="main", cache_creation=100),
                                    tool_result(at(60), sid="main"),
                                    line("m1", text(40), ts=at(60), sid="main", cache_read=100)])
    df = parse_source(tmp_path)
    assert df.loc[~df["is_sidechain"], "loop_turn"].any()
    judged = judged_subagent_loops(df, date(2026, 9, 2))
    assert sorted(judged["session_id"].unique()) == ["typed", "untyped"]
    assert len(judged) == 4


def test_a_table_without_loop_turns_has_no_subagent_loops_to_judge():
    assert judged_subagent_loops(pd.DataFrame(), date(2026, 9, 2)).empty
    no_loops = pd.DataFrame({"day": ["2026-09-01"], "is_sidechain": [True]})
    assert judged_subagent_loops(no_loops, date(2026, 9, 2)).empty


def test_the_subagent_metric_is_binned_over_its_own_days_only_when_asked_for():
    turns = pd.DataFrame({"day": ["2026-09-01", "2026-09-01", "2026-09-03"],
                          "loop_readback": [1.0, 0.5, 1.0]})
    binned = bin_metrics(turns, metrics=SUBAGENT_METRICS)
    assert binned["bin"].tolist() == ["2026-09-01", "2026-09-03"]
    assert binned["subagent_cache"].tolist() == [0.75, 1.0]
    assert binned["subagent_cache__n"].tolist() == [2, 1]
    assert not any(name in binned for name in METRICS)


def test_the_main_thread_bins_carry_no_subagent_metric_and_are_judged_without_one():
    turns = pd.DataFrame({"day": ["2026-09-01"], "thinking_fraction": [0.5], "prompt_cache_read_ratio": [1.0],
                          "is_haiku": [0.0], "loop_readback": [0.1]})
    detected = detect(bin_metrics(turns), DetectorConfig())
    assert not any(column.startswith("subagent_cache") for column in detected.columns)


def test_a_drop_in_subagent_read_back_flags_the_subagent_metric():
    rng = np.random.default_rng(0)
    days = [f"2026-09-{d:02d}" for d in range(1, 21)]
    reads = [np.where(rng.random(1000) < (0.002 if d < 15 else 0.05), 0.09, 1.0) for d in range(20)]
    turns = pd.DataFrame({"day": np.repeat(days, 1000), "loop_readback": np.concatenate(reads)})
    detected = detect(bin_metrics(turns, metrics=SUBAGENT_METRICS),
                      DetectorConfig(metric_z_thresholds={"subagent_cache": 3.0}))
    assert detected["subagent_cache__flag"].tolist() == [False] * 15 + [True] * 5
