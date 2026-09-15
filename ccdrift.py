#!/usr/bin/env python3
"""
ccdrift.py — Week-one validation harness for silent-downgrade / cost-regression
detection in Claude Code session logs.

Purpose (per the agreed MVP test): answer ONE question — is the drift signal real
or is it noise? It does this three ways:

  1. Parses ~/.claude/projects/*.jsonl into a turn-level feature table (defensively;
     degrades gracefully when field names differ across Claude Code versions).
  2. Derives the three target metrics that map to the documented Mar/Apr 2026
     incidents: reasoning-effort proxy, cache-read ratio, Haiku-delegation fraction.
  3. Runs a robust (median/MAD) sustained-deviation detector, and an INJECTION
     harness that plants controlled regime changes and sweeps their magnitude to
     find the detection floor — your product's headline number.

This is a throwaway experiment script, not the product. Outputs: CSVs + PNGs.

Usage:
  # Smoke-test with zero real data:
  python3 ccdrift.py --synthetic --out ./out

  # Against your real logs:
  python3 ccdrift.py --source ~/.claude/projects --out ./out

  # Find the detection floor for a silent Haiku-delegation regime change:
  python3 ccdrift.py --synthetic --sweep haiku --out ./out

  # Inspect the raw schema of your own logs first (recommended Step 0):
  python3 ccdrift.py --source ~/.claude/projects --schema-peek
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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


def thinking_chars(content: Any) -> int:
    """Sum character length of thinking/reasoning blocks. Handles content as a
    plain string (no thinking), a list of blocks, or absent."""
    if content is None:
        return 0
    if isinstance(content, str):
        return 0
    total = 0
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype in ("thinking", "redacted_thinking", "reasoning"):
                txt = block.get("thinking") or block.get("text") or block.get("reasoning") or ""
                if isinstance(txt, str):
                    total += len(txt)
    return total


def mcp_tool_names(content: Any) -> list[str]:
    names: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                name = block.get("name", "")
                if isinstance(name, str) and name.startswith("mcp__"):
                    names.append(name)
    return names


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
    rows: list[dict] = []
    n_files = 0
    n_lines = 0
    n_bad = 0
    n_assistant = 0
    for fp in iter_jsonl_files(source):
        n_files += 1
        try:
            with fp.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    n_lines += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        n_bad += 1
                        continue
                    if not isinstance(obj, dict) or not is_assistant(obj):
                        continue
                    n_assistant += 1
                    content = field_get(obj, "content")
                    rows.append({
                        "session_id":     field_get(obj, "session_id", default=fp.stem),
                        "timestamp":      parse_ts(field_get(obj, "timestamp")),
                        "model":          (field_get(obj, "model") or "unknown"),
                        "input_tokens":   _num(field_get(obj, "input_tokens")),
                        "output_tokens":  _num(field_get(obj, "output_tokens")),
                        "cache_creation": _num(field_get(obj, "cache_creation")),
                        "cache_read":     _num(field_get(obj, "cache_read")),
                        "thinking_chars": thinking_chars(content),
                        "n_mcp_calls":    len(mcp_tool_names(content)),
                        "is_sidechain":   bool(field_get(obj, "is_sidechain", default=False)),
                        "source_file":    fp.name,
                    })
        except OSError as e:
            if verbose:
                print(f"  ! could not read {fp}: {e}", file=sys.stderr)

    if verbose:
        print(f"  files={n_files} lines={n_lines} bad_json={n_bad} "
              f"assistant_turns={n_assistant}", file=sys.stderr)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Derived: thinking-token estimate (~4 chars/token), thinking fraction,
    # cache-read ratio, is_haiku, inter-turn gap within session.
    df["thinking_tokens"] = (df["thinking_chars"] / 4.0).round()
    df["thinking_fraction"] = df["thinking_tokens"] / df["output_tokens"].clip(lower=1)
    denom = (df["cache_read"] + df["cache_creation"]).clip(lower=1)
    df["cache_read_ratio"] = df["cache_read"] / denom
    df["is_haiku"] = df["model"].str.lower().str.contains("haiku").astype(float)
    if "is_sidechain" not in df.columns:
        df["is_sidechain"] = False
    df["is_sidechain"] = df["is_sidechain"].fillna(False).astype(bool)
    df["main_thread"] = ~df["is_sidechain"]

    df = df.sort_values(["session_id", "timestamp"], kind="stable").reset_index(drop=True)
    df["gap_seconds"] = (
        df.groupby("session_id")["timestamp"].diff().dt.total_seconds()
    )
    df["post_idle"] = (df["gap_seconds"] > 3600).fillna(False)
    # day bucket (UTC) for binning
    df["day"] = df["timestamp"].dt.tz_convert("UTC").dt.date.astype("string")
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
    "effort_proxy":   ("thinking_fraction", "median", "down"),
    "cache_ratio":    ("cache_read_ratio",  "median", "down"),
    "haiku_fraction": ("is_haiku",          "mean",   "up"),
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
    out["n_turns"] = out["bin"].map(g.size()).astype(int)
    return out


# ---------------------------------------------------------------------------
# Robust sustained-deviation detector (median + MAD)
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    baseline_window: int = 14   # trailing bins used as baseline
    z_threshold: float = 3.5    # robust-z magnitude to count a bin as deviant
    consecutive: int = 3        # deviant bins in a row required to FLAG
    min_baseline: int = 5       # need at least this many baseline bins to judge


def _robust_z(value: float, baseline: np.ndarray) -> float:
    if len(baseline) == 0:
        return 0.0
    med = np.median(baseline)
    mad = np.median(np.abs(baseline - med))
    if mad == 0:
        # constant baseline: fall back to std; if that's 0 too, no signal.
        sd = np.std(baseline)
        if sd == 0:
            return 0.0
        return (value - med) / sd
    return (value - med) / (1.4826 * mad)


def detect(metrics: pd.DataFrame, cfg: DetectorConfig) -> pd.DataFrame:
    """Annotate each bin with robust-z and a sustained-flag per metric."""
    m = metrics.copy()
    for name, (_, _, direction) in METRICS.items():
        zs: list[float] = []
        deviant: list[bool] = []
        vals = m[name].to_numpy()
        for i in range(len(vals)):
            lo = max(0, i - cfg.baseline_window)
            baseline = vals[lo:i]
            baseline = baseline[~np.isnan(baseline)]
            if len(baseline) < cfg.min_baseline or math.isnan(vals[i]):
                zs.append(np.nan)
                deviant.append(False)
                continue
            z = _robust_z(vals[i], baseline)
            zs.append(z)
            harmful = (z <= -cfg.z_threshold) if direction == "down" else (z >= cfg.z_threshold)
            deviant.append(bool(harmful))
        m[f"{name}__z"] = zs
        # sustained flag: True at bin i if this and the prior (consecutive-1)
        # bins are all deviant.
        flags = [False] * len(deviant)
        run = 0
        for i, d in enumerate(deviant):
            run = run + 1 if d else 0
            if run >= cfg.consecutive:
                # mark the whole run so the onset is visible
                for j in range(i - cfg.consecutive + 1, i + 1):
                    flags[j] = True
        m[f"{name}__flag"] = flags
    return m


def first_flag_bin(detected: pd.DataFrame, metric: str) -> Optional[int]:
    col = f"{metric}__flag"
    if col not in detected.columns:
        return None
    idx = detected.index[detected[col]]
    return int(idx[0]) if len(idx) else None


# ---------------------------------------------------------------------------
# Injection harness (Pass B) — plant a controlled regime change, measure floor
# ---------------------------------------------------------------------------

def inject(df: pd.DataFrame, kind: str, magnitude: float, seed: int = 0) -> pd.DataFrame:
    """Return a copy of the turn-level df with a regime change applied to the
    SECOND HALF (by global time order). magnitude in [0,1]."""
    rng = random.Random(seed)
    d = df.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    cut = len(d) // 2
    post = d.index >= cut

    if kind == "effort":
        # Reduce thinking by `magnitude` fraction after the cut.
        d.loc[post, "thinking_tokens"] = d.loc[post, "thinking_tokens"] * (1 - magnitude)
        d["thinking_fraction"] = d["thinking_tokens"] / d["output_tokens"].clip(lower=1)

    elif kind == "haiku":
        # Flip `magnitude` fraction of post-cut turns to a haiku model.
        post_idx = list(d.index[post])
        k = int(round(len(post_idx) * magnitude))
        for i in rng.sample(post_idx, k) if k else []:
            d.loc[i, "model"] = "claude-haiku-4-5"
        d["is_haiku"] = d["model"].str.lower().str.contains("haiku").astype(float)

    elif kind == "cache":
        # Depress cache-read ratio on post-cut, post-idle turns by moving mass
        # from cache_read into cache_creation.
        mask = post & d["post_idle"].fillna(False)
        moved = d.loc[mask, "cache_read"] * magnitude
        d.loc[mask, "cache_read"] = d.loc[mask, "cache_read"] - moved
        d.loc[mask, "cache_creation"] = d.loc[mask, "cache_creation"] + moved
        denom = (d["cache_read"] + d["cache_creation"]).clip(lower=1)
        d["cache_read_ratio"] = d["cache_read"] / denom

    else:
        raise ValueError(f"unknown injection kind: {kind}")

    return d


KIND_TO_METRIC = {"effort": "effort_proxy", "haiku": "haiku_fraction", "cache": "cache_ratio"}


def sweep(df: pd.DataFrame, kind: str, cfg: DetectorConfig,
          grid: Optional[list[float]] = None, by: str = "day") -> pd.DataFrame:
    """Sweep injection magnitude; report detection + onset lag at each level.
    The smallest detected magnitude is the detection floor."""
    if grid is None:
        grid = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70]
    metric = KIND_TO_METRIC[kind]
    # cut bin index = first bin whose turns are mostly post-cut. Approx via the
    # day that contains the global midpoint turn.
    base = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cut_day = base.loc[len(base) // 2, "day"]

    results = []
    for mag in grid:
        injected = inject(df, kind, mag, seed=7)
        m = bin_metrics(injected, by=by)
        det = detect(m, cfg)
        flag_bin = first_flag_bin(det, metric)
        cut_bin = int(det.index[det["bin"] == cut_day][0]) if (det["bin"] == cut_day).any() else None
        lag = (flag_bin - cut_bin) if (flag_bin is not None and cut_bin is not None) else None
        results.append({
            "magnitude": mag,
            "detected": flag_bin is not None,
            "onset_bin": flag_bin,
            "cut_bin": cut_bin,
            "lag_bins": lag,
        })
    res = pd.DataFrame(results)
    detected = res[res["detected"]]
    floor = detected["magnitude"].min() if not detected.empty else None
    res.attrs["detection_floor"] = floor
    res.attrs["metric"] = metric
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


def _robust_baseline(vals: np.ndarray, is_binary: bool) -> tuple[float, float]:
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return 0.0, 1.0
    if is_binary:
        p = float(np.clip(vals.mean(), 0.01, 0.99))   # floor so sigma>0
        return p, math.sqrt(p * (1 - p))
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    sigma = 1.4826 * mad if mad > 0 else (float(np.std(vals)) or 1.0)
    return med, sigma


def _stream_values(df: pd.DataFrame, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, original_indices) for the streaming metric, in global time
    order. For 'cache' we stream only post-idle turns (the batch run showed the
    all-turn median dilutes the signal to nothing)."""
    col, _, _ = STREAM_METRICS[kind]
    d = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if kind == "cache":
        d = d[d["post_idle"].fillna(False)].reset_index()  # keeps original idx in 'index'
        return d[col].to_numpy(float), d["index"].to_numpy()
    return d[col].to_numpy(float), np.arange(len(d))


def replay_cusum(values: np.ndarray, direction: str, is_binary: bool,
                 k: float, h: float, warmup: int) -> list[int]:
    """Feed values one at a time; return stream-positions where alarms fired.
    First `warmup` samples set the baseline and produce no alarms."""
    if len(values) <= warmup + 1:
        return []
    mu0, sigma = _robust_baseline(values[:warmup], is_binary)
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
                        n_seeds: int = 25) -> pd.DataFrame:
    """For each threshold h: ARL0 (false-alarm budget on the CLEAN stream) and
    median detection latency in turns (on the INJECTED stream, over n_seeds)."""
    _, direction, is_binary = STREAM_METRICS[kind]
    if h_grid is None:
        h_grid = [3, 4, 5, 6, 8, 10, 12, 15, 20]

    # --- clean stream: false-alarm behaviour (deterministic) ---
    clean_vals, _ = _stream_values(df, kind)

    # --- injected streams: detection latency, in ORIGINAL turn coordinates ---
    base = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    cut_turn = len(base) // 2

    rows = []
    for h in h_grid:
        fa = replay_cusum(clean_vals, direction, is_binary, k, h, warmup)
        n_eval = max(1, len(clean_vals) - warmup)
        arl0 = (n_eval / len(fa)) if fa else float("inf")
        fa_per_1k = 1000.0 * len(fa) / n_eval

        latencies = []
        for s in range(n_seeds):
            inj = inject(df, kind, magnitude, seed=100 + s)
            vals, orig_idx = _stream_values(inj, kind)
            alarms = replay_cusum(vals, direction, is_binary, k, h, warmup)
            # first alarm whose ORIGINAL turn index is at/after the cut
            post = [orig_idx[a] for a in alarms if orig_idx[a] >= cut_turn]
            if post:
                latencies.append(int(post[0] - cut_turn))
        det_rate = len(latencies) / n_seeds
        med_lat = float(np.median(latencies)) if latencies else float("nan")
        p90_lat = float(np.percentile(latencies, 90)) if latencies else float("nan")
        rows.append({
            "h": h,
            "false_alarms_clean": len(fa),
            "ARL0_turns": round(arl0, 1) if math.isfinite(arl0) else np.inf,
            "false_alarms_per_1k_turns": round(fa_per_1k, 2),
            "detect_rate": round(det_rate, 2),
            "median_latency_turns": med_lat,
            "p90_latency_turns": p90_lat,
        })
    out = pd.DataFrame(rows)
    out.attrs["kind"] = kind
    out.attrs["magnitude"] = magnitude
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
    """Write realistic-ish JSONL mimicking ~/.claude/projects/<proj>/<sess>.jsonl.
    Clean baseline only (no planted incident) — use --sweep to plant one."""
    rng = random.Random(seed)
    proj = out_dir / "synthetic-project"
    proj.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc) - timedelta(days=days)

    for d in range(days):
        day0 = start + timedelta(days=d)
        n_sessions = rng.randint(1, 3)
        for s in range(n_sessions):
            sid = f"sess-{d:02d}-{s}"
            fp = proj / f"{sid}.jsonl"
            t = day0 + timedelta(hours=rng.randint(8, 20), minutes=rng.randint(0, 59))
            n_turns = rng.randint(8, 40)
            sidechain_left = 0
            with fp.open("w", encoding="utf-8") as fh:
                for turn in range(n_turns):
                    # occasional idle gap to exercise the cache signal
                    if rng.random() < 0.15:
                        t += timedelta(minutes=rng.randint(70, 180))
                    else:
                        t += timedelta(seconds=rng.randint(20, 400))

                    # Subagent bursts: mirror reality — Haiku concentrates in
                    # dense sidechain runs, main thread stays ~Haiku-free.
                    if sidechain_left == 0 and rng.random() < 0.03:
                        sidechain_left = rng.randint(4, 40)
                    sidechain = sidechain_left > 0
                    if sidechain:
                        sidechain_left -= 1
                        model = "claude-haiku-4-5"       # subagent → Haiku
                        out_tok = rng.randint(2, 40)     # cheap, mechanical
                        think = 0
                    else:
                        model = "claude-sonnet-4-6" if rng.random() < 0.82 else "claude-opus-4-8"
                        out_tok = rng.randint(300, 2500)
                        think = int(out_tok * rng.uniform(0.25, 0.6))  # healthy effort

                    # cache: first turn of a session, or post-idle, misses more
                    post_idle = turn > 0 and rng.random() < 0.15
                    if turn == 0 or post_idle:
                        cache_read = rng.randint(0, 3000)
                        cache_creation = rng.randint(8000, 30000)
                    else:
                        cache_read = rng.randint(15000, 60000)
                        cache_creation = rng.randint(0, 4000)
                    content = [
                        {"type": "thinking", "thinking": "x" * (think * 4)},
                        {"type": "text", "text": "ok"},
                    ]
                    if rng.random() < 0.3:
                        content.append({
                            "type": "tool_use",
                            "name": rng.choice(["mcp__github__search", "mcp__gmail__create_draft"]),
                            "input": {},
                        })
                    rec = {
                        "type": "assistant",
                        "timestamp": t.isoformat().replace("+00:00", "Z"),
                        "sessionId": sid,
                        "isSidechain": sidechain,
                        "message": {
                            "role": "assistant",
                            "model": model,
                            "content": content,
                            "usage": {
                                "input_tokens": rng.randint(2000, 20000),
                                "output_tokens": out_tok,
                                "cache_creation_input_tokens": cache_creation,
                                "cache_read_input_tokens": cache_read,
                            },
                        },
                    }
                    fh.write(json.dumps(rec) + "\n")
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
    colors = ["#d33" if not d else "#3b6" for d in res["detected"]]
    ax.bar(res["magnitude"].astype(str), res["detected"].astype(int), color=colors)
    floor = res.attrs.get("detection_floor")
    ax.set_title(f"Detection by injected magnitude — {res.attrs.get('metric')}  "
                 f"(floor={floor})", fontsize=10)
    ax.set_xlabel("injected regime-change magnitude")
    ax.set_ylabel("detected (1) / missed (0)")
    ax.set_ylim(0, 1.2)
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

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Claude Code drift-detection validation harness")
    ap.add_argument("--source", type=str, default=str(Path.home() / ".claude" / "projects"),
                    help="dir or file of Claude Code JSONL logs")
    ap.add_argument("--synthetic", action="store_true",
                    help="generate synthetic logs and use them instead of --source")
    ap.add_argument("--synthetic-days", type=int, default=40)
    ap.add_argument("--out", type=str, default="./ccdrift_out")
    ap.add_argument("--bin", choices=["day", "session"], default="day")
    ap.add_argument("--sweep", choices=["effort", "haiku", "cache"], default=None,
                    help="run Pass-B injection sweep for this incident type")
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
    ap.add_argument("--consecutive", type=int, default=3)
    ap.add_argument("--schema-peek", action="store_true",
                    help="print the first assistant record + resolved fields, then exit")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DetectorConfig(baseline_window=args.baseline_window,
                         z_threshold=args.z_threshold,
                         consecutive=args.consecutive)

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

    df.to_csv(out_dir / "features.csv", index=False)
    print(f"[parse] {len(df)} assistant turns -> {out_dir/'features.csv'}")

    if args.main_thread_only:
        before = len(df)
        df = df[df["main_thread"]].reset_index(drop=True)
        sc = before - len(df)
        print(f"[filter] main-thread-only: dropped {sc} sidechain turns "
              f"({sc/max(before,1):.1%}); {len(df)} remain. "
              f"main-thread haiku rate now {df['is_haiku'].mean():.4f}")

    if args.stream:
        curve = cusum_latency_curve(df, args.stream, magnitude=args.stream_magnitude,
                                    k=args.cusum_k, warmup=args.warmup)
        curve.to_csv(out_dir / f"cusum_{args.stream}.csv", index=False)
        p = plot_cusum_curve(curve, out_dir)
        tpd = curve.attrs.get("turns_per_day")
        print(f"\n=== STREAMING CUSUM: {args.stream} @ magnitude {args.stream_magnitude} ===")
        if tpd:
            print(f"(your data: ~{tpd:.0f} turns/day)")
        print(curve.to_string(index=False))
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
        res = sweep(df, args.sweep, cfg, by=args.bin)
        res.to_csv(out_dir / f"sweep_{args.sweep}.csv", index=False)
        p = plot_sweep(res, out_dir)
        print(f"\n=== SWEEP: {args.sweep} ({KIND_TO_METRIC[args.sweep]}) ===")
        print(res.to_string(index=False))
        floor = res.attrs.get("detection_floor")
        print(f"\nDetection floor: {floor if floor is not None else 'NOT DETECTED at any level'}")
        print(f"[plot] {p}")
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
