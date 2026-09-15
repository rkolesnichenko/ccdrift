"""Tests for ccdrift.py. Run: python3 -m pytest -q test_ccdrift.py"""

import json
import random
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import ccdrift

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
DAY = 86400


def at(seconds: float) -> str:
    return (T0 + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def text(n: int) -> dict:
    return {"type": "text", "text": "x" * n}


def thinking(signature_chars: int) -> dict:
    # Claude Code stores thinking blocks without their text, only the signature.
    return {"type": "thinking", "thinking": "", "signature": "s" * signature_chars}


def line(mid, block, *, ts, sid="s1", out=100, cache_read=0, cache_creation=0,
         model="claude-opus-5", sidechain=False):
    """One JSONL line as Claude Code writes it: a single content block, with the
    response's message.id and usage repeated on every line of that response."""
    msg = {"role": "assistant", "model": model, "content": [block],
           "usage": {"input_tokens": 10, "output_tokens": out,
                     "cache_read_input_tokens": cache_read,
                     "cache_creation_input_tokens": cache_creation}}
    rec = {"type": "assistant", "timestamp": ts, "sessionId": sid,
           "isSidechain": sidechain, "message": msg}
    if mid is not None:
        msg["id"] = mid
        rec["requestId"] = f"req_{mid}"
    return rec


def response(mid, *blocks, ts, **kw):
    """All lines of one API response."""
    return [line(mid, b, ts=ts, **kw) for b in blocks]


def prompt(ts, sid="s1", sidechain=False):
    """A prompt typed by the user, which opens a new turn."""
    return {"type": "user", "timestamp": ts, "sessionId": sid, "isSidechain": sidechain,
            "message": {"role": "user", "content": "next request"}}


def tool_result(ts, sid="s1", sidechain=False):
    """A tool result, which continues the current turn."""
    return {"type": "user", "timestamp": ts, "sessionId": sid, "isSidechain": sidechain,
            "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}}


def compact_boundary(ts, sid="s1"):
    return {"type": "system", "subtype": "compact_boundary", "timestamp": ts, "sessionId": sid}


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


# --- parsing: one row per API response -------------------------------------

def test_split_lines_of_one_response_become_one_turn(tmp_path):
    write(tmp_path / "s1.jsonl", response("m1", thinking(400), text(400), ts=at(0)))
    assert len(ccdrift.parse_source(tmp_path)) == 1


def test_turn_takes_the_final_output_token_count(tmp_path):
    write(tmp_path / "s1.jsonl", [
        line("m1", text(40), ts=at(0), out=3),
        line("m1", text(400), ts=at(1), out=250),
    ])
    assert ccdrift.parse_source(tmp_path).loc[0, "output_tokens"] == 250


def test_response_repeated_in_a_second_file_counts_once(tmp_path):
    rec = line("m1", text(40), ts=at(0))
    write(tmp_path / "a.jsonl", [rec])
    write(tmp_path / "b.jsonl", [rec])
    assert len(ccdrift.parse_source(tmp_path)) == 1


def test_synthetic_placeholder_messages_are_not_turns(tmp_path):
    write(tmp_path / "s1.jsonl", [
        line("m1", text(40), ts=at(0)),
        line("m2", text(40), ts=at(5), model="<synthetic>", out=0),
    ])
    assert list(ccdrift.parse_source(tmp_path)["model"]) == ["claude-opus-5"]


def test_lines_without_a_message_id_are_separate_turns(tmp_path):
    write(tmp_path / "s1.jsonl", [
        line(None, text(40), ts=at(0)),
        line(None, text(40), ts=at(9)),
    ])
    assert len(ccdrift.parse_source(tmp_path)) == 2


# --- effort metric ----------------------------------------------------------

def test_thinking_tokens_are_estimated_from_the_signature_when_text_is_omitted(tmp_path):
    write(tmp_path / "s1.jsonl",
          response("m1", thinking(1000), text(400), ts=at(0))
          + response("m2", thinking(2000), text(400), ts=at(60)))
    tokens = ccdrift.parse_source(tmp_path)["thinking_tokens"]
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
    effort = ccdrift.bin_metrics(ccdrift.parse_source(tmp_path))["effort_proxy"]
    assert effort[0] > effort[1] > 0


def test_effort_metric_counts_responses_that_skip_thinking(tmp_path):
    # Roughly half of real responses don't think at all, so a day where most
    # responses skip thinking must still register the ones that did.
    write(tmp_path / "s1.jsonl",
          response("m1", thinking(3000), text(400), ts=at(0))
          + response("m2", text(400), ts=at(60))
          + response("m3", text(400), ts=at(120)))
    assert ccdrift.bin_metrics(ccdrift.parse_source(tmp_path)).loc[0, "effort_proxy"] > 0


# --- cache metric -----------------------------------------------------------

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
    metrics = ccdrift.bin_metrics(ccdrift.parse_source(tmp_path))
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
    metrics = ccdrift.bin_metrics(ccdrift.parse_source(tmp_path))
    assert metrics.loc[0, "cache_ratio"] == pytest.approx(1.0)


def test_cache_metric_skips_the_turn_after_a_compaction(tmp_path):
    # Compaction rewrites the conversation, so the next turn misses by design.
    write(tmp_path / "s1.jsonl", [
        prompt(at(0)),   line("m1", text(40), ts=at(0),   cache_read=0,    cache_creation=1000),
        prompt(at(60)),  line("m2", text(40), ts=at(60),  cache_read=1000, cache_creation=0),
        compact_boundary(at(100)),
        prompt(at(120)), line("m3", text(40), ts=at(120), cache_read=100,  cache_creation=900),
    ])
    metrics = ccdrift.bin_metrics(ccdrift.parse_source(tmp_path))
    assert metrics.loc[0, "cache_ratio"] == pytest.approx(1.0)


def test_idle_gap_is_measured_within_one_transcript(tmp_path):
    # A subagent has its own transcript and cache prefix, so its activity must
    # not make a paused main thread look warm.
    write(tmp_path / "s1.jsonl", [line("m1", text(40), ts=at(0)),
                                  line("m3", text(40), ts=at(1200))])
    write(tmp_path / "s1" / "subagents" / "agent-a.jsonl",
          [line("m2", text(40), ts=at(600), sidechain=True)])
    df = ccdrift.parse_source(tmp_path)
    assert df.loc[~df["is_sidechain"], "gap_seconds"].dropna().tolist() == [1200]


# --- batch detector ---------------------------------------------------------

def daily_turns(days):
    """Turn-level rows for bin_metrics, one dict per day mapping a per-turn
    column to its values; metric columns left out are 0."""
    df = pd.concat([pd.DataFrame({"day": f"2026-09-{i + 1:02d}", **cols})
                    for i, cols in enumerate(days)], ignore_index=True)
    for col in ("thinking_fraction", "prompt_cache_read_ratio", "is_haiku"):
        if col not in df:
            df[col] = 0.0
    return df


def prompt_turn_days(misses_per_day, rng):
    """40 new-prompt turns per day: hits just under 1.0, the rest cold."""
    return [{"prompt_cache_read_ratio": [0.05] * k + [rng.uniform(0.99, 1.0) for _ in range(40 - k)]}
            for k in misses_per_day]


# One cold turn on 4 of 14 days: 4 misses in 560 turns.
MOSTLY_CLEAN_CACHE = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1]


def test_detector_flags_haiku_appearing_after_days_without_any():
    # Main-thread Haiku share is exactly 0 on every real day, so past days show
    # no spread at all; that must not leave the detector blind.
    days = [{"is_haiku": [0.0] * 400}] * 14 + [{"is_haiku": [1.0] * 20 + [0.0] * 380}] * 6
    det = ccdrift.detect(ccdrift.bin_metrics(daily_turns(days)), ccdrift.DetectorConfig())
    assert ccdrift.first_flag_bin(det, "haiku_fraction") == 14


def test_detector_ignores_one_cache_miss_a_day_against_a_mostly_clean_baseline():
    # Days near 1.0 barely differ from each other, so judged against that
    # spread alone one miss in 40 turns looks like z = -40 (real logs: -12.9).
    days = prompt_turn_days(MOSTLY_CLEAN_CACHE + [1] * 6, random.Random(0))
    det = ccdrift.detect(ccdrift.bin_metrics(daily_turns(days)), ccdrift.DetectorConfig())
    assert ccdrift.first_flag_bin(det, "cache_ratio") is None


def test_detector_flags_a_sustained_rise_in_cache_misses():
    days = prompt_turn_days(MOSTLY_CLEAN_CACHE + [8] * 6, random.Random(0))
    det = ccdrift.detect(ccdrift.bin_metrics(daily_turns(days)), ccdrift.DetectorConfig())
    assert ccdrift.first_flag_bin(det, "cache_ratio") == 14


def test_cache_metric_flags_at_its_own_lower_threshold():
    # A confirmed caching regression in real logs scored z = -4.9, -6.4, -3.1,
    # -3.7 on its first days: 3.5 misses it and 3.0 catches it, but 3.0 on every
    # metric raised a false Haiku flag on clean synthetic logs. Here both
    # metrics shift by the same z = 3.37 for four days.
    ks = [[0, 1, 2][i % 3] for i in range(14)] + [6] * 4
    days = [{"prompt_cache_read_ratio": [0.0] * k + [1.0] * (100 - k),
             "is_haiku": [1.0] * k + [0.0] * (100 - k)} for k in ks]
    det = ccdrift.detect(ccdrift.bin_metrics(daily_turns(days)), ccdrift.DetectorConfig())
    assert ccdrift.first_flag_bin(det, "cache_ratio") == 14
    assert ccdrift.first_flag_bin(det, "haiku_fraction") is None


# --- streaming detector -----------------------------------------------------

def test_stream_detects_a_drop_in_a_metric_that_is_mostly_zero():
    # About half of real responses don't think, so the per-turn effort stream is
    # mostly zeros; a baseline taken from its median sits at 0 and can never
    # see a drop.
    rng = random.Random(0)
    before = [0.0 if rng.random() < 0.55 else 0.5 for _ in range(400)]
    after = [0.0 if rng.random() < 0.55 else 0.15 for _ in range(400)]
    alarms = ccdrift.replay_cusum(np.array(before + after), "down", False, k=0.5, h=8, warmup=200)
    assert alarms and min(alarms) >= 400


def test_stream_ignores_isolated_misses_in_a_near_saturated_ratio():
    # Main-thread cache hits sit near 1.0 with a few percent of misses; a
    # baseline spread taken from the median absolute deviation is so narrow
    # that every single miss raises an alarm.
    rng = random.Random(0)
    values = [0.05 if rng.random() < 0.03 else rng.uniform(0.97, 1.0) for _ in range(800)]
    misses = sum(v < 0.5 for v in values[200:])
    alarms = ccdrift.replay_cusum(np.array(values), "down", False, k=0.5, h=8, warmup=200)
    assert len(alarms) < misses / 2


def test_stream_tolerates_rare_misses_after_a_warmup_without_any():
    # On real logs the first 150 new-prompt turns had no cache misses (spread
    # 0.0014), so a single later miss scored z = 692 and every miss alarmed.
    rng = random.Random(1)
    values = [rng.uniform(0.997, 1.0) for _ in range(150)]
    values += [0.05 if i in (100, 400) else rng.uniform(0.997, 1.0) for i in range(600)]
    alarms = ccdrift.replay_cusum(np.array(values), "down", False, k=0.5, h=15, warmup=150)
    assert alarms == []


def test_stream_still_alarms_on_a_run_of_misses_after_a_clean_warmup():
    rng = random.Random(1)
    values = [rng.uniform(0.997, 1.0) for _ in range(150)]
    values += [0.05 if rng.random() < 0.3 else rng.uniform(0.997, 1.0) for _ in range(200)]
    alarms = ccdrift.replay_cusum(np.array(values), "down", False, k=0.5, h=15, warmup=150)
    assert alarms


# --- synthetic logs ---------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_project(tmp_path_factory):
    return ccdrift.generate_synthetic(tmp_path_factory.mktemp("syn"), days=40, seed=1)


@pytest.fixture(scope="module")
def synthetic_turns(synthetic_project):
    return ccdrift.parse_source(synthetic_project)


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
    res = ccdrift.sweep(synthetic_turns, kind, ccdrift.DetectorConfig(), grid=[0.7])
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
    df = ccdrift.add_ratios(df)
    res = ccdrift.sweep(df, "cache", ccdrift.DetectorConfig(), grid=[0.05])
    assert not res["detected"].any()


def test_stream_latency_ignores_alarms_that_fire_without_the_planted_change(synthetic_turns):
    # Real logs can hold an incident after the planted change starts (on the
    # user's logs cache alarms fired Sep 1, the change started Sep 1 07:44); an
    # alarm that fires with or without the change doesn't show it was caught.
    df = synthetic_turns.copy()
    days = sorted(df["day"].unique())
    late = df["day"].isin(days[-8:]) & df["prompt_within_ttl"]
    df.loc[late, "cache_creation"] += df.loc[late, "cache_read"]
    df.loc[late, "cache_read"] = 0
    df = ccdrift.add_ratios(df)
    curve = ccdrift.cusum_latency_curve(df, "cache", magnitude=0.01, h_grid=[20], n_seeds=1)
    assert curve["detect_rate"].tolist() == [0.0]


def test_stream_latency_catches_a_large_planted_drop_on_synthetic_logs(synthetic_turns):
    curve = ccdrift.cusum_latency_curve(synthetic_turns, "cache", magnitude=0.5, h_grid=[20], n_seeds=1)
    assert curve["detect_rate"].tolist() == [1.0]


def test_stream_false_alarms_leave_out_known_incident_days(synthetic_turns):
    # Alarms during a real incident are correct, not false; on the user's logs
    # every cache alarm at h >= 10 fell inside the Aug 16 - Sep 4 regression.
    df = synthetic_turns.copy()
    days = sorted(df["day"].unique())
    late = df["day"].isin(days[-8:]) & df["prompt_within_ttl"]
    df.loc[late, "cache_creation"] += df.loc[late, "cache_read"]
    df.loc[late, "cache_read"] = 0
    df = ccdrift.add_ratios(df)
    curve = ccdrift.cusum_latency_curve(df, "cache", magnitude=0.3, h_grid=[20], n_seeds=1,
                                        incidents=[(days[-8], days[-1])])
    assert curve["false_alarms_clean"].tolist() == [0]


def test_since_and_until_limit_the_days_analyzed(tmp_path):
    write(tmp_path / "logs" / "s1.jsonl", [
        prompt(at(0)),       line("m1", text(40), ts=at(0)),
        prompt(at(DAY)),     line("m2", text(40), ts=at(DAY)),
        prompt(at(2 * DAY)), line("m3", text(40), ts=at(2 * DAY)),
    ])
    ccdrift.main(["--source", str(tmp_path / "logs"), "--out", str(tmp_path / "out"),
                  "--since", "2026-09-02", "--until", "2026-09-02"])
    assert pd.read_csv(tmp_path / "out" / "metrics.csv")["bin"].tolist() == ["2026-09-02"]
