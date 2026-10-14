"""Does a turn-by-turn CUSUM catch tool-loop cache misses within hours? (G8, G9)

For the main thread (G8) and subagents (G9), and for every p1, h and number of
sessions an alarm's misses must come from: false alarms on the days of the logs, each
judged as the hourly check would but with the usual rate held at its 0.2% floor, the
most sensitive rate the check can run at; and how many tool-loop turns it takes to
catch a planted 2% miss rate against the measured usual rate, which catches it more
slowly. A stream passes when some combination has no false alarms over the days judged
and catches the planted rate in 90% of runs within a median of 300 turns. No real
tool-loop regression is known, so nothing checks an alarm against one. The verdict is on
the setting ccdrift ships for each stream, loops.LOOP_SETTINGS; the fastest passing
combination is reported beside it.

Run from the repo root:

  uv run --group lab python -m lab.loop_cache --incident 2026-08-16..2026-09-04
"""

from __future__ import annotations

import argparse
import functools
import itertools
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from ccdrift.logs import default_source, parse_source
from ccdrift.loops import LOOP_SETTINGS, STREAMS, LoopSetting, loop_turns
from ccdrift.warning_gates import (CAUGHT_SHARE, LOOP_MAX_TURNS as MAX_TURNS, loop_alarm_days, loop_base_rate as base_rate,
                                   loop_plants, turns_to_catch)
from lab.harness import date_range

P1_GRID = (0.01, 0.02, 0.05)
H_GRID = (2, 3, 4, 5, 6, 8)
SESSIONS_GRID = (1, 2)
GATES = {"main": "G8", "subagent": "G9"}
COLUMNS = ["stream", "p1", "h", "min_sessions", "false_alarms", "incident_alarms", "days_judged",
           "planted_median", "planted_caught", "planted_runs", "passes"]


def false_alarms(turns: pd.DataFrame, usual: Callable[[str], Optional[float]], setting: LoopSetting,
                 incident: Optional[tuple[str, str]] = None) -> tuple[int, int, int]:
    """Qualifying alarms on each day with a usual rate, as warning_gates.loop_alarm_days
    judges them. Returns (alarms outside `incident`, alarms inside it, days judged outside
    it)."""
    judged, alarms = loop_alarm_days(turns, usual, setting)
    inside = (lambda day: incident[0] <= day <= incident[1]) if incident is not None else (lambda day: False)
    return (sum(1 for day in alarms if not inside(day)), sum(1 for day in alarms if inside(day)),
            sum(1 for day in judged if not inside(day)))


def planted_stretches(turns: pd.DataFrame, usual: Callable[[str], Optional[float]]) -> list[tuple[pd.DataFrame, float]]:
    """The planted stretches of warning_gates.loop_plants, without their starting days."""
    return loop_plants(turns, usual)[1]


def evaluate(df: pd.DataFrame, incident: Optional[tuple[str, str]] = None,
             today: Optional[str] = None) -> pd.DataFrame:
    """One row per stream and combination of P1_GRID, H_GRID and SESSIONS_GRID, on the
    loop turns of the days before `today` (all days when None)."""
    rows = []
    for stream in STREAMS:
        turns = loop_turns(df, stream)
        if today is not None:
            turns = turns[turns["day"] < today].reset_index(drop=True)
        usual = functools.lru_cache(maxsize=None)(functools.partial(base_rate, turns))
        plants = planted_stretches(turns, usual)
        for p1, h, sessions in itertools.product(P1_GRID, H_GRID, SESSIONS_GRID):
            setting = LoopSetting(p1, h, sessions)
            counted, inside, judged = false_alarms(turns, usual, setting, incident)
            runs = [turns_to_catch(stretch, p0, setting) for stretch, p0 in plants]
            caught = [r for r in runs if r is not None]
            median = statistics.median(math.inf if r is None else r for r in runs) if runs else math.inf
            rows.append({"stream": stream, "p1": p1, "h": h, "min_sessions": sessions, "false_alarms": counted,
                         "incident_alarms": inside, "days_judged": judged, "planted_median": median,
                         "planted_caught": len(caught), "planted_runs": len(runs),
                         "passes": bool(judged > 0 and runs and counted == 0
                                        and len(caught) >= CAUGHT_SHARE * len(runs) and median <= MAX_TURNS)})
    return pd.DataFrame(rows, columns=COLUMNS)


def choose(table: pd.DataFrame, stream: str) -> Optional[LoopSetting]:
    """The passing combination for `stream` that catches the planted rate fastest; ties
    go to 1 session, then the larger h, then the larger p1. None when none passes."""
    passing = table[(table["stream"] == stream) & table["passes"]]
    if passing.empty:
        return None
    best = passing.sort_values(["planted_median", "min_sessions", "h", "p1"], ascending=[True, True, False, False],
                               kind="stable").iloc[0]
    return LoopSetting(float(best["p1"]), float(best["h"]), int(best["min_sessions"]))


def _named(setting: Optional[LoopSetting]) -> str:
    return "none" if setting is None else f"p1={setting.p1:g} h={setting.h:g} sessions={setting.min_sessions}"


def verdict(table: pd.DataFrame, stream: str, setting: Optional[LoopSetting]) -> str:
    """The gate's verdict on the setting ccdrift ships for `stream`: a pass only when its
    own row passes, with the combination choose() would pick beside it. A setting the
    grid didn't measure fails."""
    gate, best = GATES[stream], _named(choose(table, stream))
    if setting is None:
        return f"{gate}: none ships (fastest passing: {best})"
    row = table[(table["stream"] == stream) & (table["p1"] == setting.p1) & (table["h"] == setting.h)
                & (table["min_sessions"] == setting.min_sessions)]
    if len(row) and bool(row["passes"].iloc[0]):
        return f"{gate}: PASS {_named(setting)}"
    return f"{gate}: FAIL {_named(setting)} (fastest passing: {best})"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="G8 and G9: an early warning on tool-loop cache misses, measured")
    ap.add_argument("--source", default=None, help="folder of Claude Code transcripts (default: as ccdrift)")
    ap.add_argument("--incident", type=date_range, default=None,
                    help="a known incident, START..END, whose days' alarms are reported apart")
    args = ap.parse_args(argv)
    source = Path(args.source).expanduser() if args.source else default_source()
    df = parse_source(source)
    today = datetime.now(timezone.utc).date().isoformat()
    for stream in STREAMS:
        turns = loop_turns(df, stream)
        turns = turns[turns["day"] < today]
        span = f"{turns['day'].min()}..{turns['day'].max()}" if len(turns) else "no days"
        print(f"{GATES[stream]} {stream}: {len(turns)} loop turns, {int(turns['is_loop_miss'].sum())} misses, "
              f"{turns['day'].nunique()} days, {span}")
    table = evaluate(df, args.incident, today)
    for row in table.itertuples(index=False):
        print(f"{GATES[row.stream]} p1={row.p1:<4} h={row.h:<2} sessions={row.min_sessions}  "
              f"false alarms {row.false_alarms:>3} (+{row.incident_alarms} in the incident) over "
              f"{row.days_judged} days  planted: median {row.planted_median:>6} turns, "
              f"caught {row.planted_caught}/{row.planted_runs}  {'pass' if row.passes else 'fail'}")
    for stream in STREAMS:
        print(verdict(table, stream, LOOP_SETTINGS[stream]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
