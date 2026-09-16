"""Is Claude Code's turn latency steady enough to watch?

Claude Code logs how long each main-thread turn took (`turn_duration`). This spike
measures how much the daily median swings between clean days and how large a
planted slowdown the daily flag rule catches, the way harness.py measures the cache
and Haiku metrics. It changes nothing in the daily check.

Run from the repo root:

  uv run --group lab python -m lab.latency --incident 2026-08-16..2026-09-04
  uv run --group lab python -m lab.latency --synthetic
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ccdrift.logs import default_source, parse_durations
from lab.harness import date_range, generate_synthetic, in_date_ranges

LATENCY_COLUMNS = {"duration_s": "median turn duration (s)",
                   "per_message_s": "median milliseconds per message (messageCount: the session's messages so far)"}
# Printed values per stored unit. messageCount counts every message of the session up
# to the turn, not the turn's own: often thousands, so seconds per message print as 0.0.
PRINT_SCALE = {"duration_s": 1, "per_message_s": 1000}
BASELINE_DAYS = 14
MIN_BASELINE = 5
Z_THRESHOLD = 3.5
DEVIANT_DAYS = 3
FLAG_WINDOW = 4
FACTORS = (1.1, 1.2, 1.3, 1.5, 2.0)


def daily_latency(durations: pd.DataFrame) -> pd.DataFrame:
    """Per UTC day of main-thread CLI turns: how many, the median duration and the
    median seconds per message."""
    cli = ~durations["entrypoint"].fillna("").astype(str).str.startswith("sdk-")
    d = durations[~durations["is_sidechain"].astype(bool) & cli & (durations["message_count"] > 0)]
    d = d.assign(duration_s=d["duration_ms"] / 1000, per_message_s=d["duration_ms"] / 1000 / d["message_count"])
    g = d.groupby(d["day"].astype(str), sort=True)
    return pd.DataFrame({"turns": g.size(), "duration_s": g["duration_s"].median(),
                         "per_message_s": g["per_message_s"].median()}).rename_axis("day").reset_index()


def mad_z(values: np.ndarray, i: int) -> float:
    """z of day i against the median and MAD of up to 14 days before it; NaN with
    fewer than 5. The MAD is floored at 1% of the median, so days that never varied
    still have a spread."""
    base = values[max(0, i - BASELINE_DAYS):i]
    base = base[~np.isnan(base)]
    if len(base) < MIN_BASELINE or math.isnan(values[i]):
        return math.nan
    med = float(np.median(base))
    spread = max(1.4826 * float(np.median(np.abs(base - med))), 0.01 * abs(med), 1e-9)
    return (values[i] - med) / spread


def flag_days(values: np.ndarray) -> list[bool]:
    """Days in a slowdown flag: 3 of any 4 days in a row at z >= 3.5, marked from the
    first to the last deviant day, as the daily check flags its metrics."""
    deviant = [not math.isnan(z) and z >= Z_THRESHOLD for z in (mad_z(values, i) for i in range(len(values)))]
    flags = [False] * len(values)
    for i in range(len(values)):
        hits = [j for j in range(max(0, i - FLAG_WINDOW + 1), i + 1) if deviant[j]]
        if len(hits) >= DEVIANT_DAYS:
            for j in range(hits[0], hits[-1] + 1):
                flags[j] = True
    return flags


def sweep_latency(durations: pd.DataFrame, column: str, factors=FACTORS, n_starts: int = 10,
                  incidents: Optional[list[tuple[str, str]]] = None) -> pd.DataFrame:
    """Multiply turn durations from each of up to n_starts starting days by each
    factor, and record whether the flag rule catches it: a flagged day from the
    start on that the unchanged data doesn't flag. Starts keep 7 days before them
    and 3 after, as harness.sweep does."""
    d = durations[~in_date_ranges(durations["day"].astype(str), incidents)]
    clean = daily_latency(d)
    days = clean["day"].tolist()
    clean_flags = flag_days(clean[column].to_numpy(dtype=float))
    first, last = MIN_BASELINE + DEVIANT_DAYS - 1, len(days) - DEVIANT_DAYS
    count = min(n_starts, last - first + 1)
    starts = sorted({int(round(x)) for x in np.linspace(first, last, count)}) if count > 0 else []
    rows = []
    for start in starts:
        after = d["day"].astype(str) >= days[start]
        for factor in factors:
            slowed = d.assign(duration_ms=np.where(after, d["duration_ms"] * factor, d["duration_ms"]))
            flags = flag_days(daily_latency(slowed)[column].to_numpy(dtype=float))
            rows.append({"start": days[start], "factor": factor,
                         "caught": any(flags[i] and not clean_flags[i] for i in range(start, len(flags)))})
    res = pd.DataFrame(rows, columns=["start", "factor", "caught"])
    res.attrs["clean_flag_days"] = [day for day, flagged in zip(days, clean_flags) if flagged]
    return res


def smallest_caught_everywhere(res: pd.DataFrame) -> Optional[float]:
    caught = [factor for factor, all_caught in res.groupby("factor")["caught"].all().items() if all_caught]
    return min(caught) if caught else None


def spread(daily: pd.DataFrame, column: str) -> float:
    """Day-to-day spread: 1.4826 x MAD over the median."""
    values = daily[column].dropna().to_numpy(dtype=float)
    med = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - med))) / med if med else math.nan


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Turn latency spike: day-to-day spread and the smallest "
                                             "slowdown the daily flag rule catches")
    ap.add_argument("--source", default=None, help="folder of Claude Code transcripts (default: as ccdrift)")
    ap.add_argument("--synthetic", action="store_true", help="generate synthetic logs and use them instead")
    ap.add_argument("--incident", type=date_range, action="append", default=[],
                    help="a known incident, START..END (UTC dates, inclusive), left out (repeatable)")
    ap.add_argument("--starts", type=int, default=10, help="starting days to plant a slowdown at")
    args = ap.parse_args(argv)
    if args.synthetic:
        source = generate_synthetic(Path(tempfile.mkdtemp(prefix="ccdrift_latency_")), days=40)
    else:
        source = Path(args.source).expanduser() if args.source else default_source()
    durations = parse_durations(source)
    if durations.empty:
        print(f"No turn_duration records in {source}.", file=sys.stderr)
        return 2
    daily = daily_latency(durations[~in_date_ranges(durations["day"].astype(str), args.incident)])
    print(f"{int(daily['turns'].sum())} main-thread turn durations over {len(daily)} clean days")
    for column, label in LATENCY_COLUMNS.items():
        res = sweep_latency(durations, column, n_starts=args.starts, incidents=args.incident)
        floor = smallest_caught_everywhere(res)
        scaled = daily[column] * PRINT_SCALE[column]
        print(f"\n=== {label} ===")
        print(f"daily median {scaled.median():.1f}, range {scaled.min():.1f}-"
              f"{scaled.max():.1f}, spread (MAD/median) {spread(daily, column):.2f}")
        print("slowdown caught from each start:")
        for factor, group in res.groupby("factor"):
            print(f"  x{factor:g}: {int(group['caught'].sum())}/{len(group)}")
        print(f"smallest slowdown caught from every start: {'none' if floor is None else f'x{floor:g}'}")
        print(f"clean days flagged without a planted slowdown: {', '.join(res.attrs['clean_flag_days']) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
