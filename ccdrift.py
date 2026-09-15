#!/usr/bin/env python3
"""
ccdrift.py — Week-one validation harness for silent-downgrade / cost-regression
detection in Claude Code session logs.

Purpose (per the agreed MVP test): answer ONE question — is the drift signal real
or is it noise? It does this three ways:

  1. Parses ~/.claude/projects/**/*.jsonl into a table with one row per API
     response (defensively; degrades gracefully when field names differ across
     Claude Code versions).
  2. Derives the three target metrics that map to the documented Mar/Apr 2026
     incidents: reasoning-effort proxy, cache-read ratio on turns that open with
     a new prompt, and Haiku-delegation fraction.
  3. Runs a robust (median/MAD, floored at sampling noise) sustained-deviation
     detector, and an INJECTION harness that plants controlled regime changes and
     sweeps their magnitude to find the detection floor — your product's headline
     number.

This is a throwaway experiment script, not the product. Outputs: CSVs + PNGs.

Findings so far (the user's own logs, Aug 6 - Sep 15, 2026):
  - Cache: a real signal. A Claude Code caching regression (versions
    2.1.233-2.1.258, in use Aug 16 - Sep 4) is flagged from Aug 18. Its first
    days score z = -3.8, -5.4, -2.8, -3.2 against a cutoff of 3.0, so a flag
    takes 3 deviant days among any 4: once Claude Code's 30-day transcript
    cleanup deleted part of Aug 15-16, requiring 3 in a row missed it. With the
    incident left out, a 5% drop is caught from all 3 starting days in the 12
    clean days that remain.
  - Haiku: a real signal on the main thread, which never uses Haiku, so a 5%
    shift is caught from every starting day. Subagent Haiku bursts make the
    all-turn share noisy: on 5 synthetic logs it was falsely flagged on 3 (on 1
    when the deviant days had to be in a row), the main thread on none.
  - Effort: not detectable from one user's logs. Over the 12 clean days its
    daily level swung between 0.33 and 0.67, more than a 70% cut in thinking
    tokens moves it, so no size of drop was caught, on all turns or the main
    thread. Detecting it points to pooled data from many users.

Usage:
  # Smoke-test with zero real data:
  python3 ccdrift.py --synthetic --out ./out

  # Against your real logs:
  python3 ccdrift.py --source ~/.claude/projects --out ./out

  # Find the detection floor for a silent Haiku-delegation regime change:
  python3 ccdrift.py --synthetic --sweep haiku --out ./out

  # Detection floors from several starting days, leaving out a known incident:
  python3 ccdrift.py --sweep cache --incident 2026-08-16..2026-09-04 --out ./out

  # Streaming detection time vs false alarms, not counting a known incident:
  python3 ccdrift.py --stream cache --incident 2026-08-16..2026-09-04 --out ./out

  # Limit any mode to a range of UTC dates, e.g. a stretch without incidents:
  python3 ccdrift.py --since 2026-09-05 --stream cache --out ./out

  # Inspect the raw schema of your own logs first (recommended Step 0):
  python3 ccdrift.py --source ~/.claude/projects --schema-peek

  # Tests:
  python3 -m pytest -q test_ccdrift.py

The cache metric flags at |z| >= 3.0 (--cache-z-threshold); the other metrics
use --z-threshold (3.5).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Defensive field access
# ---------------------------------------------------------------------------
# Claude Code's JSONL schema has shifted across versions. Rather than hard-code
# one path, every field is resolved by trying a list of candidate dotted paths
# and taking the first that exists. If you discover a new variant, add it here —
# this is the single place schema drift is absorbed.

CANDIDATES: dict[str, list[str]] = {
    "role":              ["type", "message.role", "role"],
    "model":             ["message.model", "model"],
    "input_tokens":      ["message.usage.input_tokens", "usage.input_tokens"],
    "output_tokens":     ["message.usage.output_tokens", "usage.output_tokens"],
    "cache_creation":    ["message.usage.cache_creation_input_tokens",
                          "usage.cache_creation_input_tokens"],
    "cache_read":        ["message.usage.cache_read_input_tokens",
                          "usage.cache_read_input_tokens"],
    "timestamp":         ["timestamp", "message.timestamp", "createdAt"],
    "session_id":        ["sessionId", "session_id", "message.sessionId"],
    "content":           ["message.content", "content"],
    "is_sidechain":      ["isSidechain", "message.isSidechain", "is_sidechain"],
    "message_id":        ["message.id"],
    "request_id":        ["requestId", "request_id"],
    "is_meta":           ["isMeta", "is_meta"],
    "subtype":           ["subtype"],
}


def _dig(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def field_get(obj: dict, logical: str, default: Any = None) -> Any:
    for path in CANDIDATES.get(logical, []):
        val = _dig(obj, path)
        if val is not None:
            return val
    return default


def is_assistant(obj: dict) -> bool:
    role = field_get(obj, "role")
    # "type" variant carries "assistant"; "message.role" variant also "assistant".
    return role == "assistant"


# Thinking tokens per signature character. Claude Code stores thinking blocks
# without their text (or with a short summary), but each keeps its encrypted
# signature. Fitted on 20,896 real responses with a final output count: output
# tokens minus visible chars/4 ≈ 0.30 × signature chars (r = 0.97; per-model
# slopes 0.27–0.31). The stored thinking text tracks it poorly (r = 0.20).
TOKENS_PER_SIGNATURE_CHAR = 0.30


def content_chars(content: Any) -> tuple[int, int]:
    """Return (signature_chars, visible_chars) for one line's content: signature
    length of thinking blocks, and length of text plus tool-call input."""
    if isinstance(content, str):
        return 0, len(content)
    signature = visible = 0
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype in ("thinking", "redacted_thinking", "reasoning"):
                sig = block.get("signature") or block.get("data") or ""
                if isinstance(sig, str):
                    signature += len(sig)
            elif btype == "text":
                txt = block.get("text") or ""
                if isinstance(txt, str):
                    visible += len(txt)
            elif btype == "tool_use":
                visible += len(json.dumps(block.get("input") or {}))
    return signature, visible


def mcp_tool_names(content: Any) -> list[str]:
    names: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name", "")
                if isinstance(name, str) and name.startswith("mcp__"):
                    names.append(name)
    return names


def has_tool_result(content: Any) -> bool:
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


def parse_ts(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        # epoch seconds or millis
        val = float(raw)
        if val > 1e12:
            val /= 1000.0
        return datetime.fromtimestamp(val, tz=timezone.utc)
    if isinstance(raw, str):
        s = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Parsing → turn-level feature table
# ---------------------------------------------------------------------------

def iter_jsonl_files(source: Path) -> Iterable[Path]:
    if source.is_file():
        yield source
        return
    yield from sorted(source.rglob("*.jsonl"))


def parse_source(source: Path, verbose: bool = False) -> pd.DataFrame:
    # One row per API response. Claude Code writes each content block of a
    # response on its own line with the response's usage repeated, so lines that
    # share a message.id are merged. A response copied into a second file (e.g.
    # when a session is resumed) is counted from the first file only.
    turns: dict[str, dict] = {}
    n_files = 0
    n_lines = 0
    n_bad = 0
    n_assistant = 0
    for fp in iter_jsonl_files(source):
        n_files += 1
        rel = fp.name if source.is_file() else str(fp.relative_to(source))
        # Set by lines between two responses: a prompt typed by the user opens
        # a new turn, and a compaction rewrites the conversation.
        prompt_pending = compact_pending = False
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as fh:
                for line_no, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    n_lines += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        n_bad += 1
                        continue
                    if not isinstance(obj, dict):
                        continue
                    role = field_get(obj, "role")
                    if role == "user":
                        if not field_get(obj, "is_meta") and not has_tool_result(field_get(obj, "content")):
                            prompt_pending = True
                        continue
                    if role == "system":
                        if field_get(obj, "subtype") == "compact_boundary":
                            compact_pending = True
                        continue
                    if role != "assistant":
                        continue
                    n_assistant += 1
                    model = field_get(obj, "model") or "unknown"
                    if model == "<synthetic>":  # Claude Code placeholder, no API call
                        continue
                    key = (field_get(obj, "message_id") or field_get(obj, "request_id")
                           or f"{rel}:{line_no}")
                    row = turns.get(key)
                    if row is None:
                        row = turns[key] = {
                            "session_id":      field_get(obj, "session_id", default=fp.stem),
                            "timestamp":       parse_ts(field_get(obj, "timestamp")),
                            "model":           model,
                            "input_tokens":    0.0,
                            "output_tokens":   0.0,
                            "cache_creation":  0.0,
                            "cache_read":      0.0,
                            "signature_chars": 0,
                            "visible_chars":   0,
                            "n_mcp_calls":     0,
                            "is_sidechain":    bool(field_get(obj, "is_sidechain", default=False)),
                            "new_prompt":      prompt_pending,
                            "after_compaction": compact_pending,
                            "source_file":     rel,
                        }
                    prompt_pending = compact_pending = False
                    if row["source_file"] != rel:
                        continue
                    # output_tokens grows while streaming, so the largest is the
                    # latest. About a third of real responses never log the final
                    # count (stop_reason stays null); no metric relies on it.
                    for col in ("input_tokens", "output_tokens", "cache_creation", "cache_read"):
                        row[col] = max(row[col], _num(field_get(obj, col)))
                    content = field_get(obj, "content")
                    signature, visible = content_chars(content)
                    row["signature_chars"] += signature
                    row["visible_chars"] += visible
                    row["n_mcp_calls"] += len(mcp_tool_names(content))
        except OSError as e:
            if verbose:
                print(f"  ! could not read {fp}: {e}", file=sys.stderr)

    if verbose:
        print(f"  files={n_files} lines={n_lines} bad_json={n_bad} "
              f"assistant_lines={n_assistant} responses={len(turns)}", file=sys.stderr)

    df = pd.DataFrame(list(turns.values()))
    if df.empty:
        return df

    df["thinking_tokens"] = (df["signature_chars"] * TOKENS_PER_SIGNATURE_CHAR).round()
    df["visible_tokens"] = (df["visible_chars"] / 4.0).round()
    df["is_sidechain"] = df["is_sidechain"].fillna(False).astype(bool)
    df["main_thread"] = ~df["is_sidechain"]

    # Gaps are measured within one transcript: subagents share the parent's
    # sessionId but write their own file and keep their own cache prefix.
    transcript = ["source_file", "is_sidechain"]
    df = df.sort_values(transcript + ["timestamp"], kind="stable").reset_index(drop=True)
    df["gap_seconds"] = df.groupby(transcript)["timestamp"].diff().dt.total_seconds()
    # day bucket (UTC) for binning
    df["day"] = df["timestamp"].dt.tz_convert("UTC").dt.date.astype("string")
    return add_ratios(df)


# Turns the cache metric uses. In real logs a caching regression showed up on
# main-thread turns that open with a new user prompt: on Claude Code
# 2.1.233-2.1.258 they missed 4.3% of the time (0.5% before and after), however
# long the pause, while tool-loop continuations almost never missed. A turn
# more than an hour after the previous one misses anyway (1 h TTL), as does the
# turn right after a compaction, so both are left out. Subagents are left out
# too: they keep a 5 min cache, and 88% of their turns after a longer pause miss.
CACHE_TTL_SECONDS = 3600
SUBAGENT_CACHE_TTL_SECONDS = 300


def add_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the per-turn metric columns from the raw ones. Runs after parsing
    and again after every injection."""
    df["thinking_fraction"] = df["thinking_tokens"] / (
        df["thinking_tokens"] + df["visible_tokens"]).clip(lower=1)
    denom = (df["cache_read"] + df["cache_creation"]).clip(lower=1)
    df["cache_read_ratio"] = df["cache_read"] / denom
    df["is_haiku"] = df["model"].str.lower().str.contains("haiku").astype(float)
    df["prompt_within_ttl"] = (df["main_thread"] & df["new_prompt"] & ~df["after_compaction"]
                               & (df["gap_seconds"] <= CACHE_TTL_SECONDS))
    df["prompt_cache_read_ratio"] = df["cache_read_ratio"].where(df["prompt_within_ttl"])
    return df


def _num(v: Any) -> float:
    try:
        if v is None:
            return 0.0
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Metric binning
# ---------------------------------------------------------------------------

METRICS = {
    # name -> (column, aggregation, harmful_direction)
    # harmful_direction = "down" means a DROP is the incident (effort, cache),
    #                      "up"   means a RISE is the incident (haiku share).
    # Means, not medians: about half of real responses don't think, so a median
    # thinking_fraction sits at 0; cache hits are all-or-nothing, so a median
    # ignores misses until half the turns miss. NaN (turns the cache metric
    # doesn't use) is skipped.
    "effort_proxy":   ("thinking_fraction",       "mean", "down"),
    "cache_ratio":    ("prompt_cache_read_ratio", "mean", "down"),
    "haiku_fraction": ("is_haiku",                "mean", "up"),
}


def bin_metrics(df: pd.DataFrame, by: str = "day") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    key = "day" if by == "day" else "session_id"
    g = df.groupby(key, sort=True)
    out = pd.DataFrame({"bin": list(g.groups.keys())})
    out = out.sort_values("bin").reset_index(drop=True)
    for name, (col, agg, _) in METRICS.items():
        series = g[col].median() if agg == "median" else g[col].mean()
        out[name] = out["bin"].map(series).astype(float)
        # turn count and within-bin variance set the detector's noise floor
        out[f"{name}__n"] = out["bin"].map(g[col].count()).astype(int)
        out[f"{name}__var"] = out["bin"].map(g[col].var()).fillna(0.0).astype(float)
    out["n_turns"] = out["bin"].map(g.size()).astype(int)
    return out


# ---------------------------------------------------------------------------
# Robust sustained-deviation detector (median + MAD)
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    baseline_window: int = 14   # trailing bins used as baseline
    z_threshold: float = 3.5    # robust-z magnitude to count a bin as deviant
    deviant_bins: int = 3       # deviant bins required to FLAG...
    flag_window: int = 4        # ...among this many bins in a row
    min_baseline: int = 5       # need at least this many baseline bins to judge
    # Per-metric overrides of z_threshold. A confirmed caching regression in real
    # logs (Claude Code 2.1.233-2.1.258, Aug 2026) scored z = -4.9, -6.4, -3.1,
    # -3.7 on its first days on the cache metric: 3.5 misses it, 3.0 catches it.
    # 3.0 on every metric raised a false Haiku flag on clean synthetic logs.
    metric_z_thresholds: dict[str, float] = field(default_factory=lambda: {"cache_ratio": 3.0})


def _robust_z(value: float, baseline: np.ndarray, counts: np.ndarray,
              variances: np.ndarray, n: float) -> float:
    """Robust z of `value`, a mean of a [0, 1] metric over n turns, against the
    baseline bins' means (with their turn counts and within-bin variances).

    The spread is the larger of the baseline MAD and the sampling noise of a
    mean over n turns. MAD alone collapses at a metric's limits: main-thread
    Haiku share is 0 every day (MAD 0, so nothing could ever flag), and cache
    ratios near 1.0 barely differ day to day (one miss scored z = -12.9).
    Per-turn variance is pooled from the baseline bins plus one pseudo-turn of
    Bernoulli variance at the smoothed mean, so a baseline that never varied
    still gets a small nonzero floor."""
    if len(baseline) == 0:
        return 0.0
    med = np.median(baseline)
    mad_spread = 1.4826 * np.median(np.abs(baseline - med))
    mean = (np.sum(baseline * counts) + 0.5) / (np.sum(counts) + 1)
    weights = np.maximum(counts - 1, 0)
    turn_var = (np.sum(weights * variances) + mean * (1 - mean)) / (np.sum(weights) + 1)
    spread = max(mad_spread, math.sqrt(turn_var / max(n, 1)))
    return (value - med) / spread


def detect(metrics: pd.DataFrame, cfg: DetectorConfig) -> pd.DataFrame:
    """Annotate each bin with robust-z and a sustained-flag per metric."""
    m = metrics.copy()
    for name, (_, _, direction) in METRICS.items():
        zs: list[float] = []
        deviant: list[bool] = []
        threshold = cfg.metric_z_thresholds.get(name, cfg.z_threshold)
        vals = m[name].to_numpy(dtype=float)
        counts = m[f"{name}__n"].to_numpy(dtype=float)
        variances = m[f"{name}__var"].to_numpy(dtype=float)
        for i in range(len(vals)):
            lo = max(0, i - cfg.baseline_window)
            keep = ~np.isnan(vals[lo:i])
            baseline = vals[lo:i][keep]
            if len(baseline) < cfg.min_baseline or math.isnan(vals[i]):
                zs.append(np.nan)
                deviant.append(False)
                continue
            z = _robust_z(vals[i], baseline, counts[lo:i][keep], variances[lo:i][keep], counts[i])
            zs.append(z)
            harmful = (z <= -threshold) if direction == "down" else (z >= threshold)
            deviant.append(bool(harmful))
        m[f"{name}__z"] = zs
        # sustained flag: `deviant_bins` deviant bins among any `flag_window` bins
        # in a row, so one bin just short of the cutoff doesn't restart the
        # count. Mark from the first to the last of those deviant bins so the
        # onset is visible.
        flags = [False] * len(deviant)
        for i in range(len(deviant)):
            hits = [j for j in range(max(0, i - cfg.flag_window + 1), i + 1) if deviant[j]]
            if len(hits) >= cfg.deviant_bins:
                for j in range(hits[0], hits[-1] + 1):
                    flags[j] = True
        m[f"{name}__flag"] = flags
    return m


def first_flag_bin(detected: pd.DataFrame, metric: str) -> Optional[int]:
    col = f"{metric}__flag"
    if col not in detected.columns:
        return None
    idx = detected.index[detected[col]]
    return int(idx[0]) if len(idx) else None


def flag_onsets(detected: pd.DataFrame, metric: str) -> list[int]:
    """Bins where a run of flagged bins starts."""
    col = f"{metric}__flag"
    if col not in detected.columns:
        return []
    flags = detected[col].to_numpy(dtype=bool)
    return [i for i in range(len(flags)) if flags[i] and (i == 0 or not flags[i - 1])]


# ---------------------------------------------------------------------------
# Injection harness (Pass B) — plant a controlled regime change, measure floor
# ---------------------------------------------------------------------------

def inject(df: pd.DataFrame, kind: str, magnitude: float, seed: int = 0,
           start: Optional[int] = None) -> pd.DataFrame:
    """Return a copy of the turn-level df with a regime change applied from turn
    `start` (by global time order; default: the midpoint turn) to the end.
    magnitude in [0,1]."""
    rng = random.Random(seed)
    d = df.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    cut = len(d) // 2 if start is None else start
    post = d.index >= cut

    if kind == "effort":
        # Reduce thinking by `magnitude` fraction after the cut.
        d.loc[post, "thinking_tokens"] = d.loc[post, "thinking_tokens"] * (1 - magnitude)

    elif kind == "haiku":
        # Flip `magnitude` fraction of post-cut turns to a haiku model.
        post_idx = list(d.index[post])
        k = int(round(len(post_idx) * magnitude))
        d.loc[rng.sample(post_idx, k), "model"] = "claude-haiku-4-5"

    elif kind == "cache":
        # Depress cache-read ratio on post-cut new-prompt turns by moving mass
        # from cache_read into cache_creation.
        mask = post & d["prompt_within_ttl"]
        moved = d.loc[mask, "cache_read"] * magnitude
        d.loc[mask, "cache_read"] = d.loc[mask, "cache_read"] - moved
        d.loc[mask, "cache_creation"] = d.loc[mask, "cache_creation"] + moved

    else:
        raise ValueError(f"unknown injection kind: {kind}")

    return add_ratios(d)


KIND_TO_METRIC = {"effort": "effort_proxy", "haiku": "haiku_fraction", "cache": "cache_ratio"}


def in_date_ranges(days: Iterable[str], ranges: Optional[list[tuple[str, str]]]) -> np.ndarray:
    """Mask of days (ISO date strings) inside any inclusive START..END range."""
    return np.array([any(lo <= d <= hi for lo, hi in ranges or []) for d in days], dtype=bool)


def sweep(df: pd.DataFrame, kind: str, cfg: DetectorConfig,
          grid: Optional[list[float]] = None, by: str = "day", n_starts: int = 10,
          incidents: Optional[list[tuple[str, str]]] = None) -> pd.DataFrame:
    """Plant a regime change of each magnitude from each of up to n_starts
    starting days and report whether, and how many bins later, it is flagged.
    One start can mislead: on the user's logs the midpoint landed on Sep 1,
    inside a real incident. Known incident days are left out first, and each
    start keeps enough days before it for a clean baseline and room for a flag
    run after.
    The detection floor at a start is the smallest magnitude flagged there;
    attrs["detection_floor"] is the median across starts.

    Only flags the planted change adds, from its starting bin on, count: real
    logs can hold an incident of their own, and borderline days just before the
    change can open the run that the change completes."""
    if grid is None:
        grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70]
    metric = KIND_TO_METRIC[kind]
    df = df[~in_date_ranges(df["day"].astype(str), incidents)].reset_index(drop=True)
    columns = ["start", "start_bin", "magnitude", "detected", "lag_bins"]
    if df.empty:
        res = pd.DataFrame(columns=columns)
        res.attrs.update(metric=metric, floors={}, detection_floor=None, clean_flag_onsets=[])
        return res
    base = df.sort_values("timestamp", kind="stable").reset_index(drop=True)

    clean = detect(bin_metrics(df, by=by), cfg)
    clean_flags = clean[f"{metric}__flag"].to_numpy(dtype=bool)
    labels = clean["bin"].tolist()

    if by == "day":
        # A start needs room for a flag run after it, and enough days before it
        # that the detector keeps its minimum baseline once the run's first
        # planted days enter the trailing window: from a start with only
        # min_baseline days, synthetic Haiku (0-19% a day) hid a 70% change.
        first = cfg.min_baseline + cfg.deviant_bins - 1
        last = len(labels) - cfg.deviant_bins
        count = min(n_starts, last - first + 1)
        start_bins = sorted({int(round(x)) for x in np.linspace(first, last, count)}) if count > 0 else []
        first_turn = base.reset_index().groupby("day")["index"].min()
        starts = [(b, int(first_turn[labels[b]])) for b in start_bins]
    else:
        # session bins aren't in time order: plant once, at the midpoint turn
        mid = len(base) // 2
        start_bins = [i for i, label in enumerate(labels) if label == base.loc[mid, "session_id"]]
        starts = [(start_bins[0], mid)] if start_bins else []

    rows = []
    for start_bin, start_turn in starts:
        for mag in grid:
            det = detect(bin_metrics(inject(df, kind, mag, seed=7, start=start_turn), by=by), cfg)
            flags = det[f"{metric}__flag"].to_numpy(dtype=bool)
            # first bin from the start on flagged only because of the planted change
            onset = next((i for i in range(start_bin, len(flags)) if flags[i] and not clean_flags[i]), None)
            rows.append({"start": labels[start_bin], "start_bin": start_bin, "magnitude": mag,
                         "detected": onset is not None,
                         "lag_bins": None if onset is None else onset - start_bin})
    res = pd.DataFrame(rows, columns=columns)
    floors = {}
    for label, g in res.groupby("start", sort=False):
        caught = g.loc[g["detected"].astype(bool), "magnitude"]
        floors[label] = float(caught.min()) if not caught.empty else None
    median = (statistics.median_low(sorted(math.inf if f is None else f for f in floors.values()))
              if floors else math.inf)
    res.attrs.update(metric=metric, floors=floors,
                     detection_floor=None if median == math.inf else median,
                     clean_flag_onsets=[labels[i] for i in flag_onsets(clean, metric)])
    return res


# ---------------------------------------------------------------------------
# Streaming CUSUM detector + latency-vs-false-alarm probe (the real question)
# ---------------------------------------------------------------------------
# The batch detector is retrospective by construction: a day must aggregate
# before it can speak. Real-time onset detection needs a per-turn sequential
# test. CUSUM is the right tool — it accumulates standardized evidence and
# fires as soon as the cumulative drift crosses a threshold h, with a
# controllable false-alarm budget (ARL0). This section measures the only number
# that matters for v2: detection latency (turns after a shift) at a fixed
# false-alarm rate — and whether single-user turn volume is enough to make that
# latency tolerable, or whether the signal genuinely requires the fleet.

# Per-metric streaming config: (turn-level column, direction, is_binary)
STREAM_METRICS = {
    "effort":  ("thinking_fraction", "down", False),
    "haiku":   ("is_haiku",          "up",   True),
    "cache":   ("cache_read_ratio",  "down", False),
}


@dataclass
class CusumState:
    mu0: float
    sigma: float
    k: float          # slack, in std units (≈ half the target shift you tune for)
    h: float          # decision threshold, in std units
    direction: str    # "up" or "down"
    S: float = 0.0

    def update(self, x: float) -> bool:
        z = (x - self.mu0) / self.sigma
        if self.direction == "down":
            z = -z                       # detect a drop as a rise in -z
        self.S = max(0.0, self.S + z - self.k)
        if self.S > self.h:
            self.S = 0.0                 # reset after firing
            return True
        return False


# Smallest event rate a streaming baseline assumes, so a warmup that happens to
# contain no events still gets the spread of a 1% yes/no event. On real logs
# the first 150 new-prompt turns had no cache misses (spread 0.0014), so a
# single later miss scored z = 692 and every miss alarmed.
MIN_EVENT_RATE = 0.01


def _baseline(vals: np.ndarray, is_binary: bool) -> tuple[float, float]:
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return 0.0, 1.0
    if is_binary:
        p = float(np.clip(vals.mean(), MIN_EVENT_RATE, 1 - MIN_EVENT_RATE))
        return p, math.sqrt(p * (1 - p))
    # Mean and std, not median and MAD: per-turn values are bounded in [0, 1]
    # and lumpy. Effort is 0 on about half of turns, so its median is 0 and a
    # drop can never register; cache hits cluster at 1.0, so the MAD is ~0.001
    # and every single miss would alarm.
    return float(vals.mean()), max(float(vals.std()), math.sqrt(MIN_EVENT_RATE * (1 - MIN_EVENT_RATE)))


def _stream_values(df: pd.DataFrame, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, original_indices) for the streaming metric, in global time
    order. For 'cache' we stream only main-thread turns that open with a new
    prompt within the cache TTL, where caching regressions show (see
    CACHE_TTL_SECONDS)."""
    col, _, _ = STREAM_METRICS[kind]
    d = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if kind == "cache":
        d = d[d["prompt_within_ttl"]].reset_index()  # keeps original idx in 'index'
        return d[col].to_numpy(float), d["index"].to_numpy()
    return d[col].to_numpy(float), np.arange(len(d))


def replay_cusum(values: np.ndarray, direction: str, is_binary: bool,
                 k: float, h: float, warmup: int) -> list[int]:
    """Feed values one at a time; return stream-positions where alarms fired.
    First `warmup` samples set the baseline and produce no alarms."""
    if len(values) <= warmup + 1:
        return []
    mu0, sigma = _baseline(values[:warmup], is_binary)
    det = CusumState(mu0=mu0, sigma=sigma, k=k, h=h, direction=direction)
    alarms = []
    for i in range(warmup, len(values)):
        if math.isnan(values[i]):
            continue
        if det.update(values[i]):
            alarms.append(i)
    return alarms


def cusum_latency_curve(df: pd.DataFrame, kind: str, magnitude: float,
                        k: float = 0.5, warmup: int = 150,
                        h_grid: Optional[list[float]] = None,
                        n_seeds: int = 25,
                        incidents: Optional[list[tuple[str, str]]] = None) -> pd.DataFrame:
    """For each threshold h: ARL0 (false-alarm budget on the CLEAN stream) and
    median detection latency in turns (on the INJECTED stream, over n_seeds).
    An alarm counts as catching the planted change only if it fires before the
    clean stream's own first alarm after the cut: later ones would fire anyway
    (on the user's logs a real incident was still alarming when it started).
    Alarms on known incident days (START, END ISO dates, inclusive) are real, so
    they don't count as false alarms, and those days don't count toward the
    turns the false-alarm rate is measured over."""
    _, direction, is_binary = STREAM_METRICS[kind]
    if h_grid is None:
        h_grid = [3, 4, 5, 6, 8, 10, 12, 15, 20]

    # --- clean stream: false-alarm behaviour (deterministic) ---
    clean_vals, clean_idx = _stream_values(df, kind)

    # --- injected streams: detection latency, in ORIGINAL turn coordinates ---
    base = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cut_turn = len(base) // 2
    sample_days = base["day"].astype(str).to_numpy()[clean_idx]
    in_incident = in_date_ranges(sample_days, incidents)

    rows = []
    h_clean_alarm_after_cut = []
    for h in h_grid:
        fa = replay_cusum(clean_vals, direction, is_binary, k, h, warmup)
        clean_after_cut = [clean_idx[a] for a in fa if clean_idx[a] >= cut_turn]
        attributable_until = clean_after_cut[0] if clean_after_cut else float("inf")
        if clean_after_cut:
            h_clean_alarm_after_cut.append(h)
        false_alarms = [a for a in fa if not in_incident[a]]
        n_eval = max(1, int((~in_incident[warmup:]).sum()))
        arl0 = (n_eval / len(false_alarms)) if false_alarms else float("inf")
        fa_per_1k = 1000.0 * len(false_alarms) / n_eval

        latencies = []
        for s in range(n_seeds):
            inj = inject(df, kind, magnitude, seed=100 + s)
            vals, orig_idx = _stream_values(inj, kind)
            alarms = replay_cusum(vals, direction, is_binary, k, h, warmup)
            # first alarm at/after the cut, in ORIGINAL turn coordinates, that
            # fires before the clean stream alarms on its own
            post = [orig_idx[a] for a in alarms if cut_turn <= orig_idx[a] < attributable_until]
            if post:
                latencies.append(int(post[0] - cut_turn))
        det_rate = len(latencies) / n_seeds
        med_lat = float(np.median(latencies)) if latencies else float("nan")
        p90_lat = float(np.percentile(latencies, 90)) if latencies else float("nan")
        rows.append({
            "h": h,
            "false_alarms_clean": len(false_alarms),
            "ARL0_turns": round(arl0, 1) if math.isfinite(arl0) else np.inf,
            "false_alarms_per_1k_turns": round(fa_per_1k, 2),
            "detect_rate": round(det_rate, 2),
            "median_latency_turns": med_lat,
            "p90_latency_turns": p90_lat,
        })
    out = pd.DataFrame(rows)
    out.attrs["kind"] = kind
    out.attrs["magnitude"] = magnitude
    out.attrs["h_clean_alarm_after_cut"] = h_clean_alarm_after_cut
    # turns/day for wall-clock translation
    if "day" in df.columns:
        per_day = df.groupby("day").size()
        out.attrs["turns_per_day"] = float(per_day.median())
    return out


def plot_cusum_curve(curve: pd.DataFrame, out_dir: Path) -> Path:
    kind = curve.attrs.get("kind")
    tpd = curve.attrs.get("turns_per_day")
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = curve["false_alarms_per_1k_turns"].to_numpy()
    y = curve["median_latency_turns"].to_numpy()
    ax.plot(x, y, marker="o", color="#3b6")
    for _, r in curve.iterrows():
        ax.annotate(f"h={r['h']:g}", (r["false_alarms_per_1k_turns"],
                    r["median_latency_turns"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("false alarms per 1,000 turns  (lower = stricter)")
    ax.set_ylabel("median detection latency (turns after shift)")
    title = f"Streaming CUSUM tradeoff — {kind} @ magnitude {curve.attrs.get('magnitude')}"
    if tpd:
        title += f"   (~{tpd:.0f} turns/day → secondary axis in days)"
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)
    if tpd:
        sec = ax.secondary_yaxis("right", functions=(lambda t: t / tpd, lambda d: d * tpd))
        sec.set_ylabel("≈ days")
    fig.tight_layout()
    p = out_dir / f"cusum_{kind}.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Synthetic log generator — makes the harness runnable with zero real data
# ---------------------------------------------------------------------------

def generate_synthetic(out_dir: Path, days: int = 40, seed: int = 1) -> Path:
    """Write realistic-ish JSONL mimicking ~/.claude/projects/<proj>/<sess>.jsonl,
    shaped like real logs: one line per content block sharing the response's
    message.id and usage, thinking stored as a bare signature, user prompts and
    tool results between main-thread responses, and subagents in
    <sess>/subagents/agent-*.jsonl.
    Clean baseline only (no planted incident) — use --sweep to plant one."""
    rng = random.Random(seed)
    proj = out_dir / "synthetic-project"
    proj.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc) - timedelta(days=days)

    for d in range(days):
        day0 = start + timedelta(days=d)
        n_sessions = rng.randint(3, 8)
        for s in range(n_sessions):
            sid = f"sess-{d:02d}-{s}"
            t = day0 + timedelta(hours=rng.randint(8, 20), minutes=rng.randint(0, 59))
            n_turns = rng.randint(20, 60)
            sidechain_left = 0
            agent = 0
            turn_open = False          # main thread is mid tool loop
            last_tool_use = ""
            transcripts: dict[Path, list[dict]] = {}
            last_seen: dict[Path, datetime] = {}
            for turn in range(n_turns):
                # Subagent bursts: mirror reality — Haiku concentrates in
                # dense sidechain runs, main thread stays ~Haiku-free.
                if sidechain_left == 0 and rng.random() < 0.03:
                    sidechain_left = rng.randint(4, 40)
                    agent += 1
                    # about 1 in 10 subagents runs on Haiku (6% of subagent
                    # transcripts in real logs)
                    agent_model = "claude-haiku-4-5" if rng.random() < 0.10 else "claude-sonnet-4-6"
                sidechain = sidechain_left > 0
                if sidechain:
                    sidechain_left -= 1
                    fp = proj / sid / "subagents" / f"agent-{agent}.jsonl"
                    t += timedelta(seconds=rng.randint(5, 60))
                    model = agent_model
                    visible = rng.randint(2, 40)     # cheap, mechanical
                    ttl = SUBAGENT_CACHE_TTL_SECONDS
                    new_prompt = False
                else:
                    fp = proj / f"{sid}.jsonl"
                    new_prompt = not turn_open
                    if new_prompt:
                        # The user types the next prompt after a pause: mostly
                        # quick, sometimes a break a 1 h cache survives,
                        # occasionally one it doesn't.
                        r = rng.random()
                        if r < 0.05:
                            t += timedelta(minutes=rng.randint(70, 180))
                        elif r < 0.15:
                            t += timedelta(minutes=rng.randint(6, 55))
                        else:
                            t += timedelta(seconds=rng.randint(20, 240))
                        user_content = "next request"
                    else:
                        t += timedelta(seconds=rng.randint(5, 90))   # tool result
                        user_content = [{"type": "tool_result", "tool_use_id": last_tool_use, "content": "ok"}]
                    transcripts.setdefault(fp, []).append({
                        "type": "user",
                        "timestamp": t.isoformat().replace("+00:00", "Z"),
                        "sessionId": sid,
                        "isSidechain": False,
                        "message": {"role": "user", "content": user_content},
                    })
                    model = "claude-sonnet-4-6" if rng.random() < 0.82 else "claude-opus-4-8"
                    visible = rng.randint(40, 400)
                    ttl = CACHE_TTL_SECONDS
                # About half of responses think, subagents as often as the main
                # thread (54% vs 56% in real logs); healthy effort when they do.
                think = int(visible * rng.uniform(0.5, 1.5)) if rng.random() < 0.55 else 0

                # cache: cold at the start of a transcript or past its TTL
                gap = (t - last_seen[fp]).total_seconds() if fp in last_seen else None
                last_seen[fp] = t
                # rare misses otherwise, likelier when a prompt opens a turn
                if gap is None or gap > ttl or rng.random() < (0.02 if new_prompt else 0.002):
                    cache_read = rng.randint(0, 3000)
                    cache_creation = rng.randint(8000, 30000)
                else:
                    cache_read = rng.randint(15000, 60000)
                    cache_creation = rng.randint(0, 4000)

                # Thinking is stored as a signature without text, as in real logs.
                blocks = []
                if think:
                    blocks.append({"type": "thinking", "thinking": "",
                                   "signature": "s" * int(think / TOKENS_PER_SIGNATURE_CHAR)})
                blocks.append({"type": "text", "text": "x" * (visible * 4)})
                # Main-thread responses call a tool, so the loop continues, about
                # two times in three; subagent responses less often.
                calls_tool = rng.random() < (0.3 if sidechain else 0.65)
                if calls_tool:
                    blocks.append({
                        "type": "tool_use",
                        "id": f"toolu_{sid}_{turn}",
                        "name": rng.choice(["mcp__github__search", "mcp__gmail__create_draft"]),
                        "input": {},
                    })
                if not sidechain:
                    turn_open, last_tool_use = calls_tool, f"toolu_{sid}_{turn}"
                usage = {
                    "input_tokens": rng.randint(2000, 20000),
                    "cache_creation_input_tokens": cache_creation,
                    "cache_read_input_tokens": cache_read,
                }
                # One line per content block, usage repeated on each; the output
                # count is only final on the last line.
                for i, block in enumerate(blocks):
                    last = i == len(blocks) - 1
                    transcripts.setdefault(fp, []).append({
                        "type": "assistant",
                        "timestamp": t.isoformat().replace("+00:00", "Z"),
                        "sessionId": sid,
                        "isSidechain": sidechain,
                        "requestId": f"req_{sid}_{turn}",
                        "message": {
                            "id": f"msg_{sid}_{turn}",
                            "role": "assistant",
                            "model": model,
                            "content": [block],
                            "stop_reason": ("tool_use" if block["type"] == "tool_use"
                                            else "end_turn") if last else None,
                            "usage": {**usage, "output_tokens":
                                      think + visible if last else rng.randint(1, 30)},
                        },
                    })
            for fp, recs in transcripts.items():
                fp.parent.mkdir(parents=True, exist_ok=True)
                with fp.open("w", encoding="utf-8") as fh:
                    fh.writelines(json.dumps(rec) + "\n" for rec in recs)
    return proj


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metrics(detected: pd.DataFrame, out_dir: Path, cfg: DetectorConfig) -> list[Path]:
    paths = []
    bins = list(range(len(detected)))
    for name, (_, _, direction) in METRICS.items():
        fig, ax = plt.subplots(figsize=(11, 3.6))
        vals = detected[name].to_numpy()
        ax.plot(bins, vals, marker="o", ms=3, lw=1.2, color="#3b6", label=name)
        # baseline band: rolling median +/- z*MAD, drawn for context
        med = pd.Series(vals).rolling(cfg.baseline_window, min_periods=cfg.min_baseline).median()
        ax.plot(bins, med, lw=1, ls="--", color="#888", label="rolling median")
        flag_col = f"{name}__flag"
        if flag_col in detected:
            flagged = [i for i in bins if detected[flag_col].iloc[i]]
            if flagged:
                ax.scatter(flagged, [vals[i] for i in flagged], color="#d33",
                           zorder=5, s=40, label="FLAGGED")
        ax.set_title(f"{name}  (harmful direction: {direction})", fontsize=10)
        ax.set_xlabel("bin index (chronological)")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        p = out_dir / f"metric_{name}.png"
        fig.savefig(p, dpi=120)
        plt.close(fig)
        paths.append(p)
    return paths


def plot_sweep(res: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    rate = res["detected"].astype(float).groupby(res["magnitude"]).mean()
    colors = ["#3b6" if r >= 0.5 else "#d33" for r in rate]
    ax.bar(rate.index.astype(str), rate.to_numpy(), color=colors)
    floor = res.attrs.get("detection_floor")
    ax.set_title(f"Detection by injected magnitude — {res.attrs.get('metric')}  "
                 f"(median floor={floor}, {res['start'].nunique()} starts)", fontsize=10)
    ax.set_xlabel("injected regime-change magnitude")
    ax.set_ylabel("share of starts caught")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    p = out_dir / f"sweep_{res.attrs.get('metric')}.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# Schema peek (Step 0 helper)
# ---------------------------------------------------------------------------

def schema_peek(source: Path) -> None:
    for fp in iter_jsonl_files(source):
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict) and is_assistant(obj):
                        print(f"# first assistant line from {fp}")
                        print(json.dumps(obj, indent=2)[:4000])
                        print("\n# resolved fields:")
                        for logical in CANDIDATES:
                            print(f"  {logical:16s} -> {field_get(obj, logical)!r}"[:120])
                        return
        except OSError:
            continue
    print("No assistant lines found.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def date_range(text: str) -> tuple[str, str]:
    """Parse START..END (UTC dates) into ISO date strings, for --incident."""
    start, sep, end = text.partition("..")
    if not sep:
        raise argparse.ArgumentTypeError("expected START..END, e.g. 2026-08-16..2026-09-04")
    return date.fromisoformat(start).isoformat(), date.fromisoformat(end).isoformat()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Claude Code drift-detection validation harness")
    ap.add_argument("--source", type=str, default=str(Path.home() / ".claude" / "projects"),
                    help="dir or file of Claude Code JSONL logs")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate synthetic logs and use them instead of --source")
    ap.add_argument("--synthetic-days", type=int, default=40)
    ap.add_argument("--since", type=date.fromisoformat, default=None,
                    help="only analyze days on or after this UTC date (YYYY-MM-DD)")
    ap.add_argument("--until", type=date.fromisoformat, default=None,
                    help="only analyze days on or before this UTC date (YYYY-MM-DD)")
    ap.add_argument("--incident", type=date_range, action="append", default=[],
                    help="a known real incident, START..END (UTC dates, inclusive); --sweep leaves "
                         "its days out and --stream doesn't count its alarms as false alarms (repeatable)")
    ap.add_argument("--out", type=str, default="./ccdrift_out")
    ap.add_argument("--bin", choices=["day", "session"], default="day")
    ap.add_argument("--sweep", choices=["effort", "haiku", "cache"], default=None,
                    help="run Pass-B injection sweep for this incident type")
    ap.add_argument("--sweep-starts", type=int, default=10,
                    help="number of starting days to plant the change at in --sweep")
    ap.add_argument("--stream", choices=["effort", "haiku", "cache"], default=None,
                    help="run the streaming-CUSUM latency-vs-false-alarm probe")
    ap.add_argument("--stream-magnitude", type=float, default=0.30,
                    help="injected shift magnitude for the --stream probe")
    ap.add_argument("--cusum-k", type=float, default=0.5,
                    help="CUSUM slack in std units (~half the target shift)")
    ap.add_argument("--warmup", type=int, default=150,
                    help="turns used to set the streaming baseline before detection")
    ap.add_argument("--main-thread-only", action="store_true",
                    help="drop subagent/sidechain turns before analysis — the fix for the Haiku-burst confound (requires isSidechain in logs)")
    ap.add_argument("--baseline-window", type=int, default=14)
    ap.add_argument("--z-threshold", type=float, default=3.5)
    ap.add_argument("--cache-z-threshold", type=float, default=3.0,
                    help="z threshold for the cache metric; the other metrics use --z-threshold")
    ap.add_argument("--deviant-bins", type=int, default=3,
                    help="deviant bins that flag a metric when they fall within --flag-window bins in a row")
    ap.add_argument("--flag-window", type=int, default=4,
                    help="bins in a row that must hold --deviant-bins deviant bins "
                         "(set it equal to --deviant-bins to require them in a row)")
    ap.add_argument("--schema-peek", action="store_true",
                    help="print the first assistant record + resolved fields, then exit")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DetectorConfig(baseline_window=args.baseline_window,
                         z_threshold=args.z_threshold,
                         deviant_bins=args.deviant_bins,
                         flag_window=args.flag_window,
                         metric_z_thresholds={"cache_ratio": args.cache_z_threshold})

    if args.synthetic:
        tmp = Path(tempfile.mkdtemp(prefix="ccdrift_syn_"))
        source = generate_synthetic(tmp, days=args.synthetic_days)
        print(f"[synthetic] generated {args.synthetic_days} days of logs at {source}")
    else:
        source = Path(args.source).expanduser()

    if args.schema_peek:
        schema_peek(source)
        return 0

    print(f"[parse] reading {source}")
    df = parse_source(source, verbose=True)
    if df.empty:
        print("No assistant turns parsed. Run with --schema-peek to inspect your "
              "log format, or --synthetic to smoke-test.", file=sys.stderr)
        return 2

    if args.since or args.until:
        days = df["day"].astype(str)
        keep = pd.Series(True, index=df.index)
        if args.since:
            keep &= days >= args.since.isoformat()
        if args.until:
            keep &= days <= args.until.isoformat()
        df = df[keep].reset_index(drop=True)
        print(f"[filter] days {args.since or 'start'} .. {args.until or 'end'}: {len(df)} responses remain")
        if df.empty:
            print("No responses in that date range.", file=sys.stderr)
            return 2

    df.to_csv(out_dir / "features.csv", index=False)
    print(f"[parse] {len(df)} responses -> {out_dir/'features.csv'}")

    if args.main_thread_only:
        before = len(df)
        df = df[df["main_thread"]].reset_index(drop=True)
        sc = before - len(df)
        print(f"[filter] main-thread-only: dropped {sc} sidechain turns "
              f"({sc/max(before,1):.1%}); {len(df)} remain. "
              f"main-thread haiku rate now {df['is_haiku'].mean():.4f}")

    if args.stream:
        curve = cusum_latency_curve(df, args.stream, magnitude=args.stream_magnitude,
                                    k=args.cusum_k, warmup=args.warmup, incidents=args.incident)
        curve.to_csv(out_dir / f"cusum_{args.stream}.csv", index=False)
        p = plot_cusum_curve(curve, out_dir)
        tpd = curve.attrs.get("turns_per_day")
        print(f"\n=== STREAMING CUSUM: {args.stream} @ magnitude {args.stream_magnitude} ===")
        if tpd:
            print(f"(your data: ~{tpd:.0f} turns/day)")
        print(curve.to_string(index=False))
        if args.incident:
            print("(false alarms leave out known incident days: "
                  f"{', '.join(f'{lo}..{hi}' for lo, hi in args.incident)})")
        if curve.attrs["h_clean_alarm_after_cut"]:
            print("Note: without a planted change the stream already alarms after the change starts, at h = "
                  f"{', '.join(f'{h:g}' for h in curve.attrs['h_clean_alarm_after_cut'])}. Alarms from then on "
                  "don't count as catches; use --since/--until to leave that stretch out.")
        # headline: strictest h that still detects in >=90% of runs
        ok = curve[curve["detect_rate"] >= 0.9]
        print()
        if ok.empty:
            print("No threshold detects this shift in >=90% of runs at single-user "
                  "volume. This is the 'needs the fleet' verdict for this signal.")
        else:
            best = ok.sort_values("false_alarms_per_1k_turns").iloc[0]
            lat = best["median_latency_turns"]
            wall = f" (~{lat/tpd:.1f} days)" if tpd and not math.isnan(lat) else ""
            print(f"Tightest reliable setting: h={best['h']:g} → "
                  f"{best['false_alarms_per_1k_turns']:.2f} false alarms / 1k turns, "
                  f"median latency {lat:.0f} turns{wall}.")
            print("Decision: viable on single-user volume if that latency is "
                  "tolerable; if it's days, the fleet shortens it.")
        print(f"[plot] {p}")
        return 0

    if args.sweep:
        res = sweep(df, args.sweep, cfg, by=args.bin, n_starts=args.sweep_starts, incidents=args.incident)
        res.to_csv(out_dir / f"sweep_{args.sweep}.csv", index=False)
        print(f"\n=== SWEEP: {args.sweep} ({KIND_TO_METRIC[args.sweep]}) ===")
        if args.incident:
            print("(known incident days left out: "
                  f"{', '.join(f'{lo}..{hi}' for lo, hi in args.incident)})")
        floors = res.attrs["floors"]
        if res.empty:
            print(f"No starting point has {cfg.min_baseline + cfg.deviant_bins - 1} bins before it "
                  f"and {cfg.deviant_bins} after it; widen the date window.")
        else:
            short = (lambda s: str(s)[5:]) if args.bin == "day" else str
            starts = list(floors)
            print("bins from each start of the planted change to its first flag ('-' = not caught):")
            print(f"{'magnitude':>9}  " + " ".join(f"{short(s):>6}" for s in starts) + "  caught")
            for mag, g in res.groupby("magnitude", sort=True):
                lags = dict(zip(g["start"], g["lag_bins"]))
                cells = " ".join(f"{'-' if pd.isna(lags[s]) else int(lags[s]):>6}" for s in starts)
                print(f"{mag:>9g}  {cells}  {int(g['detected'].sum())}/{len(g)}")
            print("\nDetection floor by start: "
                  + ", ".join(f"{short(s)} {'-' if f is None else f'{f:g}'}" for s, f in floors.items()))
            floor = res.attrs["detection_floor"]
            caught = [f for f in floors.values() if f is not None]
            if floor is None:
                print("Detection floor: NOT DETECTED at any level from most starts")
            else:
                missed = len(floors) - len(caught)
                print(f"Detection floor: {floor:g} (median of {len(floors)} starts; range "
                      f"{min(caught):g}-{max(caught):g}"
                      + (f"; not caught at any size from {missed}" if missed else "") + ")")
        if res.attrs["clean_flag_onsets"]:
            print("Note: without a planted change your logs already flag this metric, starting "
                  f"{', '.join(map(str, res.attrs['clean_flag_onsets']))}. That lowers sensitivity "
                  "around it; pass its dates with --incident to leave it out.")
        if not res.empty:
            print(f"[plot] {plot_sweep(res, out_dir)}")
        print("\nPass/fail reminder: floor <= 0.20 for haiku & cache is the bar; "
              "effort proxy may be coarser.")
    else:
        metrics = bin_metrics(df, by=args.bin)
        detected = detect(metrics, cfg)
        detected.to_csv(out_dir / "metrics.csv", index=False)
        plots = plot_metrics(detected, out_dir, cfg)
        print(f"[detect] {len(detected)} bins -> {out_dir/'metrics.csv'}")
        any_flag = False
        for name in METRICS:
            fb = first_flag_bin(detected, name)
            if fb is not None:
                any_flag = True
                print(f"  FLAG  {name}: onset at bin {fb} (bin='{detected['bin'].iloc[fb]}')")
            else:
                print(f"  ok    {name}: no sustained deviation")
        if not any_flag:
            print("  (clean baseline — expected when no incident is present. "
                  "Use --sweep to plant one and measure the floor.)")
        for p in plots:
            print(f"[plot] {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
