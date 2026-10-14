"""`ccdrift calibrate`: whether the daily verdict works on this history as it did on the logs
it was tuned on. Every cutoff ccdrift ships was measured on one person's logs; this runs the
measurement behind three of them on anyone's: the daily detector's cache ratio, main-thread
Haiku share and subagent read-back, judged on the turns the check judges with the shipped
DetectorConfig.

For each metric the days of incidents ccdrift follows (every status but dismissed) are left
out, as lab's --incident leaves out a known one: a plant on top of a real regression isn't a
plant on a clean history. Then the quiet half: whatever the detector flags on the rest is a
false alarm, dismissed incidents included, since dismissing one is the user saying so. And the
reach half: a regression is planted from up to STARTS days and counts as caught only by a flag
the unplanted history doesn't raise. Nothing is saved.

The plants are the ones the lab gates measured those settings with. lab/harness.py and
lab/subagent_cache.py import them from here, so a gate and this command can't plant two
different things under one name."""

from __future__ import annotations

import random
from datetime import date
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ccdrift.detector import SUBAGENT_METRICS, DetectorConfig, bin_metrics, detect
from ccdrift.incidents import OPEN_END
from ccdrift.logs import add_ratios, judged_subagent_loops, judged_turns

# Starting days for a plant, and the judged days each needs around it: the detector needs
# min_baseline (5) days to judge one, and a flag needs deviant_bins (3) deviant days of
# flag_window (4), so a start has 7 judged days before it and 3 after.
STARTS = 10
BEFORE = 7
AFTER = 3
# Seeds per start for the subagent plant, as G17 (lab/subagent_cache.py) measured it.
SEEDS = 5
# The bar the shipped settings cleared on the owner's logs, judged on the check's own turns
# with the recorded incident (cache, 2026-08-18..09-03) left out: a planted 5% caught from 10
# of 10 starts for the cache ratio and for Haiku share (2026-10-14, 51 days), and for subagent
# read-back in 50 of 50 runs against G17's bar of 90% (2026-09-29).
BAR = 0.05
SUBAGENT_SHARE = 0.9
# The plant sizes: the harness's sweep grid for the main thread, G17's for subagents. Each
# start's floor is the smallest size caught there.
SIZES = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70)
SUBAGENT_SIZES = (0.02, BAR)
# Seed of the Haiku plant, the harness sweep's.
HAIKU_SEED = 7
# The plant behind each main-thread metric.
KINDS = {"cache_ratio": "cache", "haiku_fraction": "haiku"}


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


def left_out(incidents: Sequence[dict], metric: str) -> list[tuple[str, Optional[str]]]:
    """The spans of `metric`'s incidents not dismissed, in order: (start, end), with no end
    while one is open."""
    return sorted((i["start"], i["end"]) for i in incidents if i["metric"] == metric and i["status"] != "dismissed")


def _outside(frame: pd.DataFrame, spans: Sequence[tuple[str, Optional[str]]]) -> pd.DataFrame:
    """The rows of `frame` on no day of `spans`."""
    days = frame["day"].astype(str)
    inside = pd.Series(False, index=frame.index)
    for start, end in spans:
        inside |= (days >= start) & (days <= (end or OPEN_END))
    return frame[~inside]


def _judged(frame: pd.DataFrame, metric: str) -> tuple[list[str], np.ndarray]:
    """The days the shipped detector judges `metric` on in `frame`, and which it flags."""
    binned = bin_metrics(frame, metrics=SUBAGENT_METRICS) if metric in SUBAGENT_METRICS else bin_metrics(frame)
    if binned.empty:
        return [], np.zeros(0, dtype=bool)
    detected = detect(binned, DetectorConfig(), only=[metric])
    return detected["bin"].astype(str).tolist(), detected[f"{metric}__flag"].to_numpy(dtype=bool)


def _onsets(days: list[str], flags: np.ndarray) -> list[str]:
    """The first day of each run of flagged days."""
    return [days[i] for i in range(len(flags)) if flags[i] and (i == 0 or not flags[i - 1])]


def reach_main(frame: pd.DataFrame, metric: str, days: list[str], clean: np.ndarray) -> dict[str, Any]:
    """Plant KINDS[metric] at each of SIZES from each start's first turn to the end, as the
    harness sweep does: caught counts by size and the floor per start."""
    base = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    first_turn = base.reset_index().groupby(base["day"].astype(str))["index"].min()
    starts = plant_starts(days)
    caught = {size: 0 for size in SIZES}
    floors: dict[str, Optional[float]] = {}
    for start in starts:
        at = days.index(start)
        floors[start] = None
        for size in SIZES:
            planted = plant(base, KINDS[metric], size, seed=HAIKU_SEED, start=int(first_turn[start]))
            _, flags = _judged(planted, metric)
            if any(flags[i] and not clean[i] for i in range(at, len(flags))):
                caught[size] += 1
                floors[start] = size if floors[start] is None else floors[start]
    return {"starts": starts, "runs": len(starts), "caught": caught, "floors": floors}


def reach_subagent(frame: pd.DataFrame, opening: pd.Series, days: list[str], clean: np.ndarray) -> dict[str, Any]:
    """Plant G17's prefix reads at each of SUBAGENT_SIZES over each start and the AFTER days
    after it, with SEEDS seeds, as G17 does: caught counts by size over every run, and the
    floor per start, the smallest size every seed caught there."""
    flagged = {day for day, flag in zip(days, clean) if flag}
    starts = plant_starts(days)
    caught = {size: 0 for size in SUBAGENT_SIZES}
    floors: dict[str, Optional[float]] = {start: None for start in starts}
    for start in starts:
        at = days.index(start)
        window = days[at:at + AFTER + 1]
        for size in SUBAGENT_SIZES:
            hits = 0
            for seed in range(SEEDS):
                planted = plant_subagent(frame, opening, window, size, seed + 1000 * at)
                planted_days, flags = _judged(planted, "subagent_cache")
                new = {day for day, flag in zip(planted_days, flags) if flag} - flagged
                hits += bool(new.intersection(window))
            caught[size] += hits
            if hits == SEEDS and floors[start] is None:
                floors[start] = size
    return {"starts": starts, "runs": len(starts) * SEEDS, "caught": caught, "floors": floors}


def measure(frame: pd.DataFrame, metric: str, incidents: Sequence[dict],
            opening: Optional[pd.Series] = None) -> dict[str, Any]:
    """One metric's calibration on `frame`, the turns the check judges it on: the days judged
    once its incidents are left out, the flags on them, the reach, and the verdict ("pass",
    "fail" or "not_enough_data" when no start fits) with the halves that failed."""
    spans = left_out(incidents, metric)
    kept = _outside(frame, spans)
    days, clean = _judged(kept, metric)
    flagged = _onsets(days, clean)
    if metric in SUBAGENT_METRICS:
        reach = reach_subagent(kept, opening if opening is not None else pd.Series(dtype=float), days, clean)
        reached = reach["caught"][BAR] >= SUBAGENT_SHARE * reach["runs"]
    else:
        reach = reach_main(kept, metric, days, clean)
        reached = reach["caught"][BAR] == reach["runs"]
    failed = (["quiet"] if flagged else []) + (["reach"] if reach["runs"] and not reached else [])
    verdict = "not_enough_data" if not reach["runs"] else "fail" if failed else "pass"
    return {"metric": metric, "verdict": verdict, "failed": failed, "days": len(days),
            "first": days[0] if days else None, "last": days[-1] if days else None,
            "left_out": spans, "flagged": flagged, **reach}


def calibration(responses: pd.DataFrame, incidents: Sequence[dict], today: date) -> dict[str, Any]:
    """Every metric's calibration on `responses` as the check on `today` would judge them,
    with the span of days judged and how many on the main thread and in subagent loops."""
    turns = judged_turns(responses, today)
    loops = judged_subagent_loops(responses, today)
    opening = subagent_openings(responses) if not loops.empty else pd.Series(dtype=float)
    results = [measure(turns, metric, incidents) for metric in KINDS]
    results.append(measure(loops, "subagent_cache", incidents, opening))
    days = sorted(set(turns["day"].astype(str)) | set(loops["day"].astype(str)))
    return {"first": days[0] if days else None, "last": days[-1] if days else None,
            "main_days": int(turns["day"].nunique()), "subagent_days": int(loops["day"].nunique()),
            "metrics": results}
