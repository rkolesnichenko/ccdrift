"""Daily metrics and the robust sustained-deviation detector."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


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
