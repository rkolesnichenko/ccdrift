"""`ccdrift calibrate`: whether the daily verdict works on this history as it did on the logs
it was tuned on. Every cutoff ccdrift ships was measured on one person's logs; this runs the
measurement behind three of them on anyone's.

The plants are the ones the lab gates measured those settings with. lab/harness.py and
lab/subagent_cache.py import them from here, so a gate and this command can't plant two
different things under one name."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import pandas as pd

from ccdrift.logs import add_ratios

# Starting days for a plant, and the judged days each needs around it: the detector needs
# min_baseline (5) days to judge one, and a flag needs deviant_bins (3) deviant days of
# flag_window (4), so a start has 7 judged days before it and 3 after.
STARTS = 10
BEFORE = 7
AFTER = 3
# Seeds per start for the subagent plant, as G17 (lab/subagent_cache.py) measured it.
SEEDS = 5


def plant(df: pd.DataFrame, kind: str, magnitude: float, seed: int = 0, start: Optional[int] = None) -> pd.DataFrame:
    """A copy of the turns `df` in time order with a regime change of `magnitude` (0 to 1)
    from turn `start` (default: the midpoint turn) to the end, the per-turn columns
    recomputed by add_ratios. "cache" moves that share of each new-prompt turn's cache
    read within the cache's TTL into cache writes; "haiku" moves that share of the turns
    to a Haiku model, chosen with `seed`."""
    rng = random.Random(seed)
    d = df.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    cut = len(d) // 2 if start is None else start
    post = d.index >= cut
    if kind == "haiku":
        post_idx = list(d.index[post])
        k = int(round(len(post_idx) * magnitude))
        d.loc[rng.sample(post_idx, k), "model"] = "claude-haiku-4-5"
    elif kind == "cache":
        mask = post & d["prompt_within_ttl"]
        moved = d.loc[mask, "cache_read"] * magnitude
        d.loc[mask, "cache_read"] = d.loc[mask, "cache_read"] - moved
        d.loc[mask, "cache_creation"] = d.loc[mask, "cache_creation"] + moved
    else:
        raise ValueError(kind)
    return add_ratios(d)


def subagent_openings(responses: pd.DataFrame) -> pd.Series:
    """What each subagent transcript's first response read from the cache, by source file."""
    sub = responses[responses["is_sidechain"].astype(bool)].sort_values("timestamp", kind="stable")
    return sub.groupby("source_file", sort=True)["cache_read"].first()


def plant_subagent(loops: pd.DataFrame, opening: pd.Series, days: list[str], share: float,
                   seed: int) -> pd.DataFrame:
    """`loops` with an extra `share` of each of `days`' turns that read back at least half
    reading only what their transcript's opening turn read, at most what was cached, and
    writing the rest again, the per-turn columns recomputed by add_ratios."""
    planted = loops.copy()
    rng = np.random.default_rng(seed)
    for day in days:
        hits = planted.index[(planted["day"].astype(str) == day) & ~planted["is_loop_miss"].astype(bool)]
        count = int(round(share * int((planted["day"].astype(str) == day).sum())))
        prefix_reads(planted, opening, rng.choice(hits.to_numpy(), size=min(count, len(hits)), replace=False))
    return add_ratios(planted)


def prefix_reads(planted: pd.DataFrame, opening: pd.Series, chosen: np.ndarray) -> None:
    """The `chosen` turns of `planted` read only what their transcript's opening turn read,
    at most what was cached, and write the rest again."""
    before = planted.loc[chosen, "cache_read"]
    read = np.minimum(planted.loc[chosen, "source_file"].map(opening).fillna(0.0).to_numpy(),
                      planted.loc[chosen, "prev_cached"].to_numpy())
    planted.loc[chosen, "cache_read"] = read
    planted.loc[chosen, "cache_creation"] = planted.loc[chosen, "cache_creation"] + before - read


def plant_starts(days: list[str], starts: int = STARTS) -> list[str]:
    """Up to `starts` days, evenly spread, with BEFORE judged days before and AFTER after."""
    eligible = days[BEFORE:len(days) - AFTER]
    if len(eligible) <= starts:
        return eligible
    return [eligible[int(i)] for i in np.linspace(0, len(eligible) - 1, starts).round()]
