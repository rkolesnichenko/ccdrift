"""The research harness: synthetic logs, sweeps and the streaming detector."""

import json
import random

import numpy as np
import pandas as pd
import pytest

from ccdrift.detector import DetectorConfig
from ccdrift.logs import add_ratios, parse_source
from lab.harness import cusum_latency_curve, generate_synthetic, main, replay_cusum, sweep
from tests.helpers import DAY, at, line, prompt, text, write


def test_stream_detects_a_drop_in_a_metric_that_is_mostly_zero():
    # About half of real responses don't think, so the per-turn effort stream is
    # mostly zeros; a baseline taken from its median sits at 0 and can never
    # see a drop.
    rng = random.Random(0)
    before = [0.0 if rng.random() < 0.55 else 0.5 for _ in range(400)]
    after = [0.0 if rng.random() < 0.55 else 0.15 for _ in range(400)]
    alarms = replay_cusum(np.array(before + after), "down", False, k=0.5, h=8, warmup=200)
    assert alarms and min(alarms) >= 400


def test_stream_ignores_isolated_misses_in_a_near_saturated_ratio():
    # Main-thread cache hits sit near 1.0 with a few percent of misses; a
    # baseline spread taken from the median absolute deviation is so narrow
    # that every single miss raises an alarm.
    rng = random.Random(0)
    values = [0.05 if rng.random() < 0.03 else rng.uniform(0.97, 1.0) for _ in range(800)]
    misses = sum(v < 0.5 for v in values[200:])
    alarms = replay_cusum(np.array(values), "down", False, k=0.5, h=8, warmup=200)
    assert len(alarms) < misses / 2


def test_stream_tolerates_rare_misses_after_a_warmup_without_any():
    # On real logs the first 150 new-prompt turns had no cache misses (spread
    # 0.0014), so a single later miss scored z = 692 and every miss alarmed.
    rng = random.Random(1)
    values = [rng.uniform(0.997, 1.0) for _ in range(150)]
    values += [0.05 if i in (100, 400) else rng.uniform(0.997, 1.0) for i in range(600)]
    alarms = replay_cusum(np.array(values), "down", False, k=0.5, h=15, warmup=150)
    assert alarms == []


def test_stream_still_alarms_on_a_run_of_misses_after_a_clean_warmup():
    rng = random.Random(1)
    values = [rng.uniform(0.997, 1.0) for _ in range(150)]
    values += [0.05 if rng.random() < 0.3 else rng.uniform(0.997, 1.0) for _ in range(200)]
    alarms = replay_cusum(np.array(values), "down", False, k=0.5, h=15, warmup=150)
    assert alarms


@pytest.fixture(scope="module")
def synthetic_project(tmp_path_factory):
    return generate_synthetic(tmp_path_factory.mktemp("syn"), days=40, seed=1)


@pytest.fixture(scope="module")
def synthetic_turns(synthetic_project):
    return parse_source(synthetic_project)


def synthetic_records(project, kind="assistant"):
    records = (json.loads(l) for fp in project.rglob("*.jsonl") for l in fp.open())
    return [r for r in records if r["type"] == kind]


def test_synthetic_logs_write_each_response_across_several_lines(synthetic_project, synthetic_turns):
    ids = [r["message"].get("id") for r in synthetic_records(synthetic_project)]
    assert None not in ids
    assert len(ids) > len(set(ids))
    assert len(synthetic_turns) == len(set(ids))


def test_synthetic_thinking_omits_text_but_keeps_a_signature(synthetic_project):
    blocks = [b for r in synthetic_records(synthetic_project)
              for b in r["message"]["content"] if b["type"] == "thinking"]
    assert blocks
    assert all(b["thinking"] == "" and b.get("signature") for b in blocks)


def test_synthetic_main_thread_mixes_new_prompts_and_tool_loop_continuations(synthetic_turns):
    main = synthetic_turns[synthetic_turns["main_thread"]]
    share = main["new_prompt"].mean() if "new_prompt" in main else 0.0
    assert 0.2 < share < 0.6


def test_synthetic_cache_misses_follow_idle_gaps(synthetic_turns):
    df = synthetic_turns
    within_ttl = df[(df["gap_seconds"] > 300) & (df["gap_seconds"] <= 3600)]
    past_ttl = df[df["gap_seconds"] > 3600]
    assert within_ttl["cache_read_ratio"].median() > 0.8
    assert past_ttl["cache_read_ratio"].median() < 0.2


@pytest.mark.parametrize("kind", ["effort", "haiku", "cache"])
def test_sweep_detects_a_large_regression_on_synthetic_logs(synthetic_turns, kind):
    res = sweep(synthetic_turns, kind, DetectorConfig(), grid=[0.7])
    assert res["detected"].all()


def test_sweep_ignores_flags_that_start_before_the_planted_change(synthetic_turns):
    # Real logs can hold an incident of their own before the planted change
    # (on the user's logs the cache metric flags Aug 18, the change starts Sep
    # 1); counting it made every size look caught.
    df = synthetic_turns.copy()
    days = sorted(df["day"].unique())
    early = df["day"].isin(days[6:12]) & df["prompt_within_ttl"]
    df.loc[early, "cache_creation"] += df.loc[early, "cache_read"]
    df.loc[early, "cache_read"] = 0
    df = add_ratios(df)
    res = sweep(df, "cache", DetectorConfig(), grid=[0.05])
    assert not res["detected"].any()


def test_sweep_plants_the_change_at_several_starting_days(synthetic_turns):
    # One planted start can mislead: on the user's logs the midpoint landed on
    # Sep 1, inside a real incident. Each start needs room for a flag run after
    # it, and enough days before it that the detector still has its minimum
    # baseline once the run's first planted days enter the trailing window: with
    # only 5 days, synthetic Haiku swung 0-19% and two planted days moved the
    # median enough to hide a 70% change.
    cfg = DetectorConfig()
    n_days = synthetic_turns["day"].nunique()
    res = sweep(synthetic_turns, "cache", cfg, grid=[0.7], n_starts=4)
    assert res["start"].nunique() == 4
    assert res["start_bin"].min() >= cfg.min_baseline + cfg.deviant_bins - 1
    assert res["start_bin"].max() <= n_days - cfg.deviant_bins
    assert res["detected"].all()


def test_sweep_leaves_out_known_incident_days(synthetic_turns):
    df = synthetic_turns.copy()
    days = sorted(df["day"].unique())
    early = df["day"].isin(days[6:12]) & df["prompt_within_ttl"]
    df.loc[early, "cache_creation"] += df.loc[early, "cache_read"]
    df.loc[early, "cache_read"] = 0
    df = add_ratios(df)
    res = sweep(df, "cache", DetectorConfig(), grid=[0.3], incidents=[(days[6], days[11])])
    assert res.attrs["clean_flag_onsets"] == []
    assert not res["start"].isin(days[6:12]).any()


def test_stream_latency_ignores_alarms_that_fire_without_the_planted_change(synthetic_turns):
    # Real logs can hold an incident after the planted change starts (on the
    # user's logs cache alarms fired Sep 1, the change started Sep 1 07:44); an
    # alarm that fires with or without the change doesn't show it was caught.
    df = synthetic_turns.copy()
    days = sorted(df["day"].unique())
    late = df["day"].isin(days[-8:]) & df["prompt_within_ttl"]
    df.loc[late, "cache_creation"] += df.loc[late, "cache_read"]
    df.loc[late, "cache_read"] = 0
    df = add_ratios(df)
    curve = cusum_latency_curve(df, "cache", magnitude=0.01, h_grid=[20], n_seeds=1)
    assert curve["detect_rate"].tolist() == [0.0]


def test_stream_latency_catches_a_large_planted_drop_on_synthetic_logs(synthetic_turns):
    curve = cusum_latency_curve(synthetic_turns, "cache", magnitude=0.5, h_grid=[20], n_seeds=1)
    assert curve["detect_rate"].tolist() == [1.0]


def test_stream_false_alarms_leave_out_known_incident_days(synthetic_turns):
    # Alarms during a real incident are correct, not false; on the user's logs
    # every cache alarm at h >= 10 fell inside the Aug 16 - Sep 4 regression.
    df = synthetic_turns.copy()
    days = sorted(df["day"].unique())
    late = df["day"].isin(days[-8:]) & df["prompt_within_ttl"]
    df.loc[late, "cache_creation"] += df.loc[late, "cache_read"]
    df.loc[late, "cache_read"] = 0
    df = add_ratios(df)
    curve = cusum_latency_curve(df, "cache", magnitude=0.3, h_grid=[20], n_seeds=1,
                                        incidents=[(days[-8], days[-1])])
    assert curve["false_alarms_clean"].tolist() == [0]


def test_since_and_until_limit_the_days_analyzed(tmp_path):
    write(tmp_path / "logs" / "s1.jsonl", [
        prompt(at(0)),       line("m1", text(40), ts=at(0)),
        prompt(at(DAY)),     line("m2", text(40), ts=at(DAY)),
        prompt(at(2 * DAY)), line("m3", text(40), ts=at(2 * DAY)),
    ])
    main(["--source", str(tmp_path / "logs"), "--out", str(tmp_path / "out"),
                  "--since", "2026-09-02", "--until", "2026-09-02"])
    assert pd.read_csv(tmp_path / "out" / "metrics.csv")["bin"].tolist() == ["2026-09-02"]
