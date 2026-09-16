"""Early warning: a turn-by-turn CUSUM on new-prompt cache misses, so a caching
regression can show within hours instead of after 3 of 4 bad days.

It is a Bernoulli likelihood-ratio CUSUM testing the usual miss rate against P1, the
August 2026 regression's rate. A z-score CUSUM with k = 0.5 can't accumulate on a
rare event: at a 5% miss rate its sum drains between misses. lab/early_warning.py
measured it on the owner's logs and set THRESHOLD; None means the warning isn't
built."""

from __future__ import annotations

import math
from typing import Optional, Sequence

P1 = 0.05
MIN_P0 = 0.002
MAX_P0 = 0.025
THRESHOLD: Optional[float] = None


def clamp_rate(rate: float) -> float:
    return min(max(rate, MIN_P0), MAX_P0)


def miss_cusum(misses: Sequence[bool], base_rate: float, h: float, p1: float = P1) -> list[int]:
    """Positions at which the CUSUM for misses at `p1` against `base_rate` passes h;
    the sum restarts from 0 after each alarm."""
    p0 = clamp_rate(base_rate)
    on_miss, on_hit = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
    total = 0.0
    alarms = []
    for position, missed in enumerate(misses):
        total = max(0.0, total + (on_miss if missed else on_hit))
        if total > h:
            alarms.append(position)
            total = 0.0
    return alarms
