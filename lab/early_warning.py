"""Does a turn-by-turn CUSUM catch a caching regression within hours? (G3)

For each threshold h: false alarms on clean days (known incidents left out), how many
new-prompt turns it takes to catch a planted 5% miss rate, and when it would have
alarmed on the real regression. The gate passes when some h has no false alarms,
catches the planted change within a median of 150 turns, and alarmed on the real
regression within 5 days of its start (for August 2026: before Aug 21 00:00 UTC, a
day before the daily check's alert).

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

from ccdrift.early import miss_cusum
from ccdrift.logs import default_source, parse_source
from lab.harness import date_range

H_GRID = (2, 3, 4, 5, 6, 8)
BASE_DAYS = 14
WINDOW_DAYS = 7
MIN_BASE_TURNS = 100
PLANT_RATE = 0.05
STARTS = 10
SEEDS = 5
MAX_TURNS = 150
DEADLINE_DAYS = 5


def prompt_turns(df: pd.DataFrame) -> pd.DataFrame:
    """CLI main-thread new-prompt turns, in time order."""
    keep = df["main_thread"].astype(bool) & df["prompt_within_ttl"].astype(bool)
    if "entrypoint" in df:
        keep &= ~df["entrypoint"].fillna("").astype(str).str.startswith("sdk-")
    turns = df.loc[keep, ["timestamp", "day", "is_miss"]].copy()
    turns["day"] = turns["day"].astype(str)
    turns["is_miss"] = turns["is_miss"].astype(bool)
    return turns.sort_values("timestamp", kind="stable").reset_index(drop=True)


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
        rows.append({"h": h, "false_alarms": wrong, "planted_median": median, "planted_caught": len(caught),
                     "planted_runs": len(runs), "real_alarm": alarm,
                     "passes": wrong == 0 and median <= MAX_TURNS and alarm is not None and alarm < deadline})
    return pd.DataFrame(rows, columns=["h", "false_alarms", "planted_median", "planted_caught", "planted_runs",
                                       "real_alarm", "passes"])


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="G3: an early warning on cache misses, measured")
    ap.add_argument("--source", default=None, help="folder of Claude Code transcripts (default: as ccdrift)")
    ap.add_argument("--incident", type=date_range, required=True, help="the known incident, START..END")
    args = ap.parse_args(argv)
    source = Path(args.source).expanduser() if args.source else default_source()
    table = evaluate(parse_source(source), args.incident)
    for row in table.itertuples(index=False):
        alarm = row.real_alarm.strftime("%Y-%m-%d %H:%M UTC") if row.real_alarm is not None else "none"
        print(f"h={row.h:<2} false alarms {row.false_alarms:>3}  planted: median {row.planted_median:>6} turns, "
              f"caught {row.planted_caught}/{row.planted_runs}  real regression: {alarm}  "
              f"{'pass' if row.passes else 'fail'}")
    passing = table[table["passes"]]
    if len(passing):
        print(f"G3: PASS h={passing['h'].iloc[0]}")
    else:
        print("G3: FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
