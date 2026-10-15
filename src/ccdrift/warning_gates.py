"""The early warnings' gates, measured on any history: how often the cache-miss warning
(G3) would warn on nothing and how many turns it takes to catch a planted miss rate, and
the same for the tool-loop warning (G8). `ccdrift calibrate` judges a user's history with
them; lab/early_warning.py and lab/loop_cache.py judge the owner's and sweep the settings."""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from ccdrift import early, loops
from ccdrift.early import MIN_P0, miss_cusum
from ccdrift.loops import LoopSetting, qualifying_alarms

Span = tuple[str, str]

STARTS = 10
SEEDS = 5

# G3: a planted 5% miss rate, the August 2026 regression's, caught within a median of 150
# new-prompt turns.
PROMPT_PLANT = 0.05
PROMPT_MAX_TURNS = 150

# A judgement, not a measurement, and it was made knowing what h = 4 costs: one spurious early
# warning every 6 weeks of turns is tolerable for a notice whose advice is "nothing yet, the daily
# verdict follows". h = 4 measures 0.063 to 0.144 a week depending on the usual rate, so it clears
# this with margin; h = 3 measures 0.230 to 0.736 and fails it by 1.4x to 4.4x.
MAX_FALSE_PER_WEEK = 1 / 6

RATE_STREAM = 1000         # turns per simulated stream in alarms_per_turn
RATE_STREAMS = 200
RATE_SEED = 20260921

# G8: a planted 2% tool-loop miss rate caught in 90% of runs within a median of 300 turns.
LOOP_PLANT = 0.02
LOOP_MAX_TURNS = 300
CAUGHT_SHARE = 0.9


def shift(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def within(days: pd.Series, spans: Sequence[Span]) -> pd.Series:
    """Whether each day falls inside one of `spans`, (START, END) both included."""
    inside = pd.Series(False, index=days.index)
    for first, last in spans:
        inside |= days.between(first, last)
    return inside


def prompt_base_rate(turns: pd.DataFrame, first_day: str, spans: Sequence[Span] = ()) -> Optional[float]:
    """The miss rate of the BASE_DAYS days before `first_day`, leaving out `spans`; None
    with fewer than MIN_BASE_TURNS turns."""
    days = turns["day"]
    base = (days < first_day) & (days >= shift(first_day, -early.BASE_DAYS)) & ~within(days, spans)
    if base.sum() < early.MIN_BASE_TURNS:
        return None
    return float(turns.loc[base, "is_miss"].mean())


def prompt_alarm_days(turns: pd.DataFrame, spans: Sequence[Span], h: float) -> list[str]:
    """The day of each alarm outside `spans`, each day judged as the hourly check would:
    over the WINDOW_DAYS days ending that day, counting only alarms that fall on it."""
    clean = turns[~within(turns["day"], spans)]
    found = []
    for day in sorted(clean["day"].unique()):
        first = shift(day, -(early.WINDOW_DAYS - 1))
        p0 = prompt_base_rate(turns, first, spans)
        if p0 is None:
            continue
        stretch = clean[clean["day"].between(first, day)]
        on_day = stretch["day"].to_numpy() == day
        found += [day for i in miss_cusum(stretch["is_miss"].tolist(), p0, h) if on_day[i]]
    return found


def prompt_plants(turns: pd.DataFrame, spans: Sequence[Span], h: float) -> tuple[list[str], list[Optional[int]]]:
    """The starting days, and the turns to the first alarm after planting PROMPT_PLANT
    misses from each of them, for up to STARTS starts and SEEDS seeds; None when never
    caught."""
    clean = turns[~within(turns["day"], spans)]
    candidates = [d for d in sorted(clean["day"].unique()) if prompt_base_rate(turns, d, spans) is not None]
    if not candidates:
        return [], []
    picks = sorted({candidates[int(round(x))] for x in np.linspace(0, len(candidates) - 1, min(STARTS, len(candidates)))})
    results = []
    for start in picks:
        stretch = clean[clean["day"].between(start, shift(start, early.WINDOW_DAYS - 1))]
        p0 = prompt_base_rate(turns, start, spans)
        for seed in range(SEEDS):
            rng = random.Random(seed)
            alarms = miss_cusum([rng.random() < PROMPT_PLANT for _ in range(len(stretch))], p0, h)
            results.append(alarms[0] + 1 if alarms else None)
    return picks, results


def alarms_per_turn(h: float, p0: float, streams: int = RATE_STREAMS, seed: int = RATE_SEED) -> float:
    """False alarms per turn, from streams generated at `p0` with nothing planted and
    judged against `p0` as the check judges it, clamped. Every alarm in them is false by
    construction, which real clean days cannot promise."""
    rng = np.random.default_rng(seed)
    alarms = sum(len(miss_cusum(list(rng.random(RATE_STREAM) < p0), p0, float(h))) for _ in range(streams))
    return alarms / (streams * RATE_STREAM)


def loop_base_rate(turns: pd.DataFrame, first_day: str) -> Optional[float]:
    """The tool-loop miss rate of the BASE_DAYS days before `first_day`; None with fewer
    than MIN_BASE_TURNS turns. No days are left out: a new-prompt incident isn't one here."""
    days = turns["day"]
    base = (days < first_day) & (days >= shift(first_day, -loops.BASE_DAYS))
    if base.sum() < loops.MIN_BASE_TURNS:
        return None
    return float(turns.loc[base, "is_loop_miss"].mean())


def loop_alarm_days(turns: pd.DataFrame, usual: Callable[[str], Optional[float]],
                    setting: LoopSetting, own: bool = False) -> tuple[list[str], list[str]]:
    """The days judged, those with a usual rate, and the day of each qualifying alarm on
    them, each day judged as the hourly check would: over the WINDOW_DAYS days ending on
    it, counting only alarms that fall on it. The usual rate is held at its floor, MIN_P0,
    the most sensitive rate the check can run at, or with `own` is the history's own, as
    the check runs it."""
    judged, found = [], []
    for day in sorted(turns["day"].unique()):
        first = shift(day, -(loops.WINDOW_DAYS - 1))
        rate = usual(first)
        if rate is None:
            continue
        judged.append(day)
        stretch = turns[turns["day"].between(first, day)].reset_index(drop=True)
        on_day = stretch["day"].to_numpy() == day
        found += [day for _, alarm in qualifying_alarms(stretch, rate if own else MIN_P0, setting) if on_day[alarm]]
    return judged, found


def loop_plants(turns: pd.DataFrame,
                usual: Callable[[str], Optional[float]]) -> tuple[list[str], list[tuple[pd.DataFrame, float]]]:
    """The starting days, up to STARTS with a usual rate spread evenly over them, and for
    SEEDS seeds each, seeded per starting day: the turns of the WINDOW_DAYS days from that
    day, each missing with probability LOOP_PLANT and keeping its session, with that day's
    usual rate."""
    candidates = [day for day in sorted(turns["day"].unique()) if usual(day) is not None]
    if not candidates:
        return [], []
    picks = sorted({candidates[int(round(x))]
                    for x in np.linspace(0, len(candidates) - 1, min(STARTS, len(candidates)))})
    stretches = []
    for start in picks:
        stretch = turns[turns["day"].between(start, shift(start, loops.WINDOW_DAYS - 1))].reset_index(drop=True)
        for seed in range(SEEDS):
            rng = random.Random(f"{start}/{seed}")
            planted = stretch.assign(is_loop_miss=[rng.random() < LOOP_PLANT for _ in range(len(stretch))])
            stretches.append((planted, usual(start)))
    return picks, stretches


def turns_to_catch(stretch: pd.DataFrame, p0: float, setting: LoopSetting) -> Optional[int]:
    alarms = qualifying_alarms(stretch, p0, setting)
    return alarms[0][1] + 1 if alarms else None
