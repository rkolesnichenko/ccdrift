"""Does thinking that rises get caught? (G16)

From 2026-09-10 to 09-15, on Claude Code 2.1.267 to 2.1.272, the owner's main thread thought
6 to 11 times as much per response as the median of the days before, then 2.9 and 2.2 times
it on 09-16 and 09-17 as 2.1.273 took over, and was back to usual on 09-18.
Thinking is billed as output, so it cost money and usage limits, and nothing noticed.
docs/findings.md rules effort out as undetectable, but that measured a drop in a normalised
level, not a multifold rise in raw tokens.

The metric is each model's mean logged thinking tokens per main-thread CLI response on a
complete UTC day, on days with enough responses that nearly all logged a count. The rule
raises a day at `ratio` times the median of the model's previous counted days, and reports a
rise once it has lasted `days` counted days in a row. The rule ships in ccdrift.thinking;
this module replays it on the owner's logs across a grid of settings.

The gate replays the rule day by day over a clean history, the counts with the known
episode's days taken out, as the check would have run it the morning after each day, with
one state carried along: every rise it reports there is a false alarm. Then, from every
counted day of the most used model with a baseline before it and room after it, once each,
it multiplies that day's and the next two counted days' thinking by m and asks whether a
rise is reported first on one of those three days that the clean replay did not report.
Last, it replays the full history and asks whether the episode is reported by its third
counted day.

A setting passes with no false alarm, every plant of BAR times or more caught, and the
episode caught. The verdict judges the setting ccdrift ships, RATIO over DAYS, as G3, G8 and
G9 do, and names the settings that pass when it fails.
Output is aggregate: models, days, levels and counts.

Run from the repo root:

  uv run --group lab python -m lab.thinking
"""

from __future__ import annotations

import argparse
import statistics
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ccdrift.logs import default_source, judged_turns, parse_all
from ccdrift.state import new_state
from ccdrift.thinking import (BASELINE_DAYS, DAYS, MIN_BASELINE_DAYS, MIN_MEDIAN, RATIO, counted, thinking_counts,
                              thinking_rises)

RATIO_GRID = (2, 2.5, 3, 4)
DAYS_GRID = (1, 2, 3)
PLANT_GRID = (1.5, 2, 3, 4, 6)
PLANT_RUN = 3           # counted days a planted rise lasts; the real one lasted four
BAR = 3                 # every plant this many times the thinking or more must be caught
PLANT_MODEL = "claude-opus-5"  # the owner's main-thread model for all but the last week
EPISODE = "2026-09-10..2026-09-17"  # 2.1.267 to 2.1.273: 6 to 11 times the usual thinking, then 2.9 and
                                    # 2.2 times as 2.1.273 took over; the owner's cut, see docs/findings.md


def level_lines(counts: pd.DataFrame, model: str) -> list[str]:
    """Every counted day of `model` with its responses, level, and ratio to the median of the up
    to BASELINE_DAYS counted days before it, the yardstick the rule uses: the lines a findings
    entry is written from."""
    days = counted(counts[counts["model"] == model])
    levels = days["level"].astype(float).tolist()
    lines = []
    for i, (day, responses, level) in enumerate(zip(days["day"].astype(str), days["responses"], levels)):
        before = levels[max(0, i - BASELINE_DAYS):i]
        if len(before) < MIN_BASELINE_DAYS:
            judged = f"not judged: {len(before)} counted days before"
        elif statistics.median(before) < MIN_MEDIAN:
            judged = f"not judged: a median of {statistics.median(before):.0f} is under {MIN_MEDIAN}"
        else:
            judged = f"{level / statistics.median(before):.2f}x the median of the {len(before)} counted days before"
        lines.append(f"  {day} {int(responses)} responses, level {level:.0f}, {judged}")
    return lines


def plant_run(counts: pd.DataFrame, model: str, first: str, run: int = PLANT_RUN) -> list[str]:
    """`first` and the counted days of `model` after it, `run` in all: the days a plant covers."""
    days = [str(day) for day in counted(counts[counts["model"] == model])["day"]]
    start = days.index(first)
    return days[start:start + run]


def plant(counts: pd.DataFrame, model: str, first: str, times: float) -> pd.DataFrame:
    """`counts` with `model`'s thinking multiplied by `times` on the days of plant_run from `first`."""
    planted = counts.copy()
    hit = (planted["model"] == model) & planted["day"].astype(str).isin(plant_run(counts, model, first))
    planted.loc[hit, "thinking"] = planted.loc[hit, "thinking"] * times
    return planted


def plant_starts(counts: pd.DataFrame, model: str) -> list[str]:
    """The counted days of `model` a plant can start on: MIN_BASELINE_DAYS counted days before
    it, so the rule judges it, and PLANT_RUN - 1 after it, so the whole plant fits."""
    days = [str(day) for day in counted(counts[counts["model"] == model])["day"]]
    return days[MIN_BASELINE_DAYS:len(days) - (PLANT_RUN - 1)]


def replay(counts: pd.DataFrame, ratio: float, days: int) -> list[dict[str, Any]]:
    """Every rise the rule reports, judging each day of `counts` the morning after it, as the
    check would, with one state carried along."""
    state = new_state()
    found = []
    for day in sorted(set(counts["day"].astype(str))):
        found += thinking_rises(counts, state, date.fromisoformat(day) + timedelta(days=1), ratio, days)
    return found


def rows(clean: pd.DataFrame, full: pd.DataFrame, episode: list[str], ratios=RATIO_GRID, days_grid=DAYS_GRID,
         plants=PLANT_GRID, model: str = PLANT_MODEL) -> list[dict[str, Any]]:
    """One row per setting: its false alarms on the clean history, the plants of each size it
    catches out of how many and the starts it missed, the most counted days a catch took, and
    the day it reports the episode (the counted days of `model` in `full` it covers) by its
    PLANT_RUN-th, or None. A plant that lands inside a run the clean history already reported
    is missed by definition, so the missed starts say which misses are that and which the rule
    didn't see."""
    starts = plant_starts(clean, model)
    out = []
    for ratio in ratios:
        for days in days_grid:
            alarms = replay(clean, ratio, days)
            seen = {(r["model"], r["since"]) for r in alarms}
            caught: dict[float, int] = {}
            slowest: dict[float, int] = {}
            missed: dict[float, list[str]] = {}
            for times in plants:
                caught[times] = slowest[times] = 0
                missed[times] = []
                for first in starts:
                    within = plant_run(clean, model, first)
                    hits = [r for r in replay(plant(clean, model, first, times), ratio, days)
                            if r["model"] == model and r["since"] in within and r["on"] in within
                            and (r["model"], r["since"]) not in seen]
                    if hits:
                        caught[times] += 1
                        slowest[times] = max(slowest[times], within.index(hits[0]["on"]) + 1)
                    else:
                        missed[times].append(first)
            found = [r["on"] for r in replay(full, ratio, days)
                     if r["model"] == model and r["since"] in episode and r["on"] in episode[:PLANT_RUN]]
            row = {"ratio": ratio, "days": days, "alarms": [f"{r['model']} {r['since']}" for r in alarms],
                   "caught": caught, "slowest": slowest, "missed": missed, "starts": len(starts),
                   "episode": found[0] if found else None}
            row["passes"] = (not row["alarms"] and bool(starts) and row["episode"] is not None
                             and all(caught[times] == len(starts) for times in plants if times >= BAR))
            out.append(row)
    return out


def _setting(row: dict[str, Any]) -> str:
    return f"ratio={row['ratio']:g} days={row['days']}"


def _big(row: dict[str, Any]) -> int:
    return sum(n for times, n in row["caught"].items() if times >= BAR)


def gate(rows: list[dict[str, Any]], chosen: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    """Whether the setting ccdrift ships (RATIO over DAYS, or `chosen`) passes, what it did,
    and, when it fails, the settings that pass instead."""
    chosen = chosen or {"ratio": RATIO, "days": DAYS}
    row = next((r for r in rows if r["ratio"] == chosen["ratio"] and r["days"] == chosen["days"]), None)
    if row is None:
        return False, ["the setting ccdrift ships isn't in the grid"]
    plants = row["starts"] * sum(1 for times in row["caught"] if times >= BAR)
    notes = [f"ships {_setting(row)}: {len(row['alarms'])} false alarm(s)"
             + (f" ({', '.join(row['alarms'])})" if row["alarms"] else "")
             + f", plants of x{BAR:g} or more caught {_big(row)} of {plants}"
             + f", episode {'on ' + row['episode'] if row['episode'] else 'missed'}"]
    if not row["passes"]:
        passing = [_setting(r) for r in rows if r["passes"]]
        notes.append(f"passing settings: {'; '.join(passing)}" if passing else "no setting in the grid passes")
    return bool(row["passes"]), notes


def counts_of(source: Path, today: date) -> pd.DataFrame:
    return thinking_counts(judged_turns(parse_all(source).responses, today))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    parser.add_argument("--episode", default=EPISODE, help="START..END, the known rise to leave out and catch")
    args = parser.parse_args(argv)

    start, end = args.episode.split("..")
    full = counts_of(args.source, args.today)
    in_episode = full["day"].astype(str).between(start, end)
    clean = full[~in_episode].reset_index(drop=True)
    judged = counted(full)
    if judged.empty:
        print("no counted days")
        return 1
    for model, group in judged.groupby("model", sort=True):
        print(f"{model}: {len(group)} counted days {group['day'].min()}..{group['day'].max()}, "
              f"level {group['level'].min():.0f}-{group['level'].max():.0f}")
    episode = [str(day) for day in judged[(judged["model"] == PLANT_MODEL)
                                          & judged["day"].astype(str).between(start, end)]["day"]]
    levels = judged.set_index(["model", "day"])["level"]
    print(f"episode {start}..{end}: " + ", ".join(f"{day} {levels[(PLANT_MODEL, day)]:.0f}" for day in episode))
    print(f"{PLANT_MODEL} by counted day:")
    for line in level_lines(full, PLANT_MODEL):
        print(line)
    table = rows(clean, full, episode)
    for row in table:
        print(f"G16 {_setting(row):<16} false-alarms={len(row['alarms'])} "
              + " ".join(f"x{times:g}={n}/{row['starts']}" for times, n in row["caught"].items())
              + f" slowest(x>={BAR:g})={max(d for t, d in row['slowest'].items() if t >= BAR)}"
              + f" episode={row['episode'] or 'missed'}"
              + (f"  alarms: {', '.join(row['alarms'])}" if row["alarms"] else ""))
        for times, starts in row["missed"].items():
            if times >= BAR and starts:
                print(f"    x{times:g} missed from {', '.join(starts)}")
    ok, notes = gate(table)
    print(f"G16: {'PASS' if ok else 'FAIL'}")
    for note in notes:
        print(f"  {note}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
