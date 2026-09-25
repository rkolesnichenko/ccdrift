"""Does a turn-by-turn CUSUM catch a caching regression within hours? (G3)

For each threshold h: false alarms on clean days (known incidents left out), how many
new-prompt turns it takes to catch a planted 5% miss rate, and when it would have
alarmed on the real regression. The gate passes when some h raises false alarms no
faster than MAX_FALSE_PER_WEEK, catches the planted change within a median of 150
turns, and alarmed on the real regression within 5 days of its start (for August 2026:
before Aug 21 00:00 UTC, a day before the daily check's alert). The verdict is on the h
ccdrift ships, early.THRESHOLD, with the usual rate taken over the days and turns the
check takes it over; the other thresholds are measured beside it.

The false alarm condition used to be a count: no false alarm at all on the clean days.
No threshold has a zero rate, so that asked whether the corpus happened to be short
enough to hold none, and it flipped on 2026-09-21 when one day carrying two missed
turns put a single alarm on h = 4.

The clean days hold about 2.5 weeks of turns. At h = 4's measured 0.063 to 0.144 a week
that is 0.16 to 0.36 expected, so a single alarm turns up in roughly 30% of corpora this
size. Any bar phrased as a count, including an allowance of rate times weeks, rejects
that one alarm and so fails 30% of the time on nothing. The rate is what can be judged,
and it is judged where there is power to measure it: `false_alarm_rate` runs 200,000
simulated turns per setting with nothing planted, where every alarm is false by
construction. The corpus's own count is reported beside it rather than deciding it.

Run from the repo root:

  uv run --group lab python -m lab.early_warning --incident 2026-08-16..2026-09-04
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ccdrift.early import BASE_DAYS, MIN_BASE_TURNS, THRESHOLD, WINDOW_DAYS, miss_cusum, prompt_turns
from ccdrift.logs import default_source, parse_source
from lab.harness import date_range

H_GRID = (2, 3, 4, 5, 6, 8)
PLANT_RATE = 0.05
STARTS = 10
SEEDS = 5
MAX_TURNS = 150
DEADLINE_DAYS = 5

# A judgement, not a measurement, and it was made knowing what h = 4 costs: one spurious early
# warning every 6 weeks of turns is tolerable for a notice whose advice is "nothing yet, the daily
# verdict follows". h = 4 measures 0.063 to 0.144 a week depending on the usual rate, so it clears
# this with margin; h = 3 measures 0.230 to 0.736 and fails it by 1.4x to 4.4x.
MAX_FALSE_PER_WEEK = 1 / 6

TURNS_A_WEEK = 343         # new-prompt turns in 7 days, the median of the clean days on the owner's logs
RATE_STREAM = 1000         # turns per simulated stream in false_alarm_rate
RATE_STREAMS = 200
RATE_SEED = 20260921
RATE_P0S = (0.002, 0.0043, 0.010)   # the floor the check clamps to, the observed clean rate, and above it


def _shift(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def base_rate(turns: pd.DataFrame, first_day: str, excluded: Optional[tuple[str, str]] = None) -> Optional[float]:
    """The miss rate of the BASE_DAYS days before `first_day`, leaving out `excluded`
    (START, END); None with fewer than MIN_BASE_TURNS turns."""
    days = turns["day"]
    base = (days < first_day) & (days >= _shift(first_day, -BASE_DAYS))
    if excluded:
        base &= ~days.between(*excluded)
    if base.sum() < MIN_BASE_TURNS:
        return None
    return float(turns.loc[base, "is_miss"].mean())


def false_alarms(turns: pd.DataFrame, incident: tuple[str, str], h: float) -> int:
    """Alarms on clean days, each day judged as the hourly check would: over the 7
    days ending that day, counting only alarms that fall on it."""
    clean = turns[~turns["day"].between(*incident)]
    count = 0
    for day in sorted(clean["day"].unique()):
        first = _shift(day, -(WINDOW_DAYS - 1))
        p0 = base_rate(turns, first, incident)
        if p0 is None:
            continue
        stretch = clean[clean["day"].between(first, day)]
        on_day = stretch["day"].to_numpy() == day
        count += sum(1 for i in miss_cusum(stretch["is_miss"].tolist(), p0, h) if on_day[i])
    return count


def planted(turns: pd.DataFrame, incident: tuple[str, str], h: float) -> list[Optional[int]]:
    """Turns to the first alarm after planting PLANT_RATE misses from each starting
    day, for up to STARTS starts and SEEDS seeds; None when never caught."""
    clean = turns[~turns["day"].between(*incident)]
    candidates = [d for d in sorted(clean["day"].unique()) if base_rate(turns, d, incident) is not None]
    if not candidates:
        return []
    picks = sorted({candidates[int(round(x))] for x in np.linspace(0, len(candidates) - 1, min(STARTS, len(candidates)))})
    results = []
    for start in picks:
        stretch = clean[clean["day"].between(start, _shift(start, WINDOW_DAYS - 1))]
        p0 = base_rate(turns, start, incident)
        for seed in range(SEEDS):
            rng = random.Random(seed)
            alarms = miss_cusum([rng.random() < PLANT_RATE for _ in range(len(stretch))], p0, h)
            results.append(alarms[0] + 1 if alarms else None)
    return results


def real_alarm(turns: pd.DataFrame, incident: tuple[str, str], h: float) -> Optional[pd.Timestamp]:
    """When the CUSUM first alarms over the first WINDOW_DAYS days of the real incident."""
    p0 = base_rate(turns, incident[0])
    if p0 is None:
        before = turns[turns["day"] < incident[0]]
        p0 = float(before["is_miss"].mean()) if len(before) else 0.0
    stretch = turns[turns["day"].between(incident[0], _shift(incident[0], WINDOW_DAYS - 1))]
    alarms = miss_cusum(stretch["is_miss"].tolist(), p0, h)
    return stretch["timestamp"].iloc[alarms[0]] if alarms else None


def judged_weeks(turns: pd.DataFrame, incident: tuple[str, str]) -> float:
    """Weeks of turns the false alarm count covers: the clean days, at TURNS_A_WEEK each."""
    clean = turns[~turns["day"].between(*incident)]
    return len(clean) / TURNS_A_WEEK


def false_alarm_rate(h: float, p0: float, streams: int = RATE_STREAMS, seed: int = RATE_SEED) -> float:
    """False alarms a week of turns, from streams generated at `p0` with nothing planted.
    Every alarm in them is false by construction, which the real clean days cannot promise."""
    rng = np.random.default_rng(seed)
    alarms = sum(len(miss_cusum(list(rng.random(RATE_STREAM) < p0), p0, float(h))) for _ in range(streams))
    return alarms / (streams * RATE_STREAM) * TURNS_A_WEEK


def rate_sweep() -> pd.DataFrame:
    """The false alarm rate of each h at each usual rate the check runs at."""
    return pd.DataFrame([{"h": h, **{f"p0={p0}": false_alarm_rate(h, p0) for p0 in RATE_P0S}}
                         for h in H_GRID])


def evaluate(df: pd.DataFrame, incident: tuple[str, str]) -> pd.DataFrame:
    turns = prompt_turns(df)
    deadline = pd.Timestamp(incident[0], tz="UTC") + pd.Timedelta(days=DEADLINE_DAYS)
    rows = []
    for h in H_GRID:
        runs = planted(turns, incident, h)
        caught = [r for r in runs if r is not None]
        median = statistics.median(r if r is not None else math.inf for r in runs) if runs else math.inf
        alarm = real_alarm(turns, incident, h)
        wrong = false_alarms(turns, incident, h)
        weeks = judged_weeks(turns, incident)
        rate = max(false_alarm_rate(h, p0) for p0 in RATE_P0S)
        rows.append({"h": h, "false_alarms": wrong, "weeks": weeks, "rate": rate,
                     "planted_median": median, "planted_caught": len(caught),
                     "planted_runs": len(runs), "real_alarm": alarm,
                     "passes": rate <= MAX_FALSE_PER_WEEK and median <= MAX_TURNS
                     and alarm is not None and alarm < deadline})
    return pd.DataFrame(rows, columns=["h", "false_alarms", "weeks", "rate", "planted_median",
                                       "planted_caught", "planted_runs", "real_alarm", "passes"])


def verdict(table: pd.DataFrame, h: Optional[float]) -> str:
    """The gate's verdict on the threshold ccdrift ships, `h`: a pass only when its own row
    passes. Which others pass is reported beside it, since that is what a change would pick
    from; a threshold the grid didn't measure fails."""
    passing = ", ".join(f"h={row:g}" for row in table.loc[table["passes"], "h"]) or "none"
    if h is None:
        return f"G3: none ships (passing: {passing})"
    row = table[table["h"] == h]
    if len(row) and bool(row["passes"].iloc[0]):
        return f"G3: PASS h={h:g}"
    return f"G3: FAIL h={h:g} (passing: {passing})"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="G3: an early warning on cache misses, measured")
    ap.add_argument("--source", default=None, help="folder of Claude Code transcripts (default: as ccdrift)")
    ap.add_argument("--incident", type=date_range, required=True, help="the known incident, START..END")
    args = ap.parse_args(argv)
    source = Path(args.source).expanduser() if args.source else default_source()
    table = evaluate(parse_source(source), args.incident)
    for row in table.itertuples(index=False):
        alarm = row.real_alarm.strftime("%Y-%m-%d %H:%M UTC") if row.real_alarm is not None else "none"
        print(f"h={row.h:<2} rate {row.rate:.3f}/week (bar {MAX_FALSE_PER_WEEK:.3f}), "
              f"{row.false_alarms} on the {row.weeks:.1f} weeks judged  "
              f"planted: median {row.planted_median:>6} turns, "
              f"caught {row.planted_caught}/{row.planted_runs}  real regression: {alarm}  "
              f"{'pass' if row.passes else 'fail'}")
    print(verdict(table, THRESHOLD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
