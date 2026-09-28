"""Does thinking that rises get caught? (G16)

From 2026-09-10 to 09-15, on Claude Code 2.1.267 to 2.1.272, the owner's main thread thought
6 to 9 times as much per response as on any other day, and came back down on 2.1.273.
Thinking is billed as output, so it cost money and usage limits, and nothing noticed.
docs/findings.md rules effort out as undetectable, but that measured a drop in a normalised
level, not a multifold rise in raw tokens.

The metric is each model's mean logged thinking tokens per main-thread CLI response on a
complete UTC day, on days with enough responses that nearly all logged a count. The rule
raises a day at `ratio` times the median of the model's previous counted days, and reports a
rise once it has lasted `days` counted days in a row. It lives here until an alert ships, in
the shape the check would call it with.

The gate replays the rule day by day over a clean history, the counts with the known
episode's days taken out, as the check would have run it the morning after each day, with
one state carried along: every rise it reports there is a false alarm. Then, from every
counted day of the most used model with a baseline before it and room after it, once each,
it multiplies that day's and the next two counted days' thinking by m and asks whether a
rise is reported first on one of those three days that the clean replay did not report.
Last, it replays the full history and asks whether the episode is reported by its third
counted day.

A setting passes with no false alarm, every plant of BAR times or more caught, and the
episode caught. Nothing ships yet, so the verdict names the settings that pass, or the one
that came closest; once an alert ships it will judge that setting, as G3, G8 and G9 do.
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

COUNT_COLUMNS = ["model", "day", "responses", "logged", "thinking"]
MIN_RESPONSES = 50      # a model's judged turns on a day, for the day to count: claude-opus-5's
                        # quietest counted day on the owner's logs had 78, and a day under 50 is a
                        # half-used one whose mean a handful of long answers can move
LOGGED_SHARE = 0.9      # of them with a logged thinking count: Claude Code logs one on every
                        # response from 2026-08-17, and on 35% of 08-16's, the first day it did
BASELINE_DAYS = 14      # counted days a day is judged against, at most and at least: the same
MIN_BASELINE_DAYS = 5   # as the setting alert, the other rule that judges a model's own days
RATIO_GRID = (2, 2.5, 3, 4)
DAYS_GRID = (1, 2, 3)
PLANT_GRID = (1.5, 2, 3, 4, 6)
PLANT_RUN = 3           # counted days a planted rise lasts; the real one lasted four
BAR = 3                 # every plant this many times the thinking or more must be caught
PLANT_MODEL = "claude-opus-5"  # the owner's main-thread model for all but the last week
EPISODE = "2026-09-10..2026-09-15"  # 2.1.267 to 2.1.272, 6 to 9 times the usual thinking


def thinking_counts(turns: pd.DataFrame) -> pd.DataFrame:
    """Per model and UTC day of `turns` (judged turns): its responses, how many logged a
    thinking count, and the sum of those counts."""
    if turns.empty:
        return pd.DataFrame(columns=COUNT_COLUMNS)
    frame = turns.assign(day=turns["day"].astype(str), logged=turns["thinking_logged"].notna(),
                         thinking=turns["thinking_logged"].fillna(0.0).astype(float))
    by_day = frame.groupby(["model", "day"], sort=True)
    counts = pd.DataFrame({"responses": by_day.size(), "logged": by_day["logged"].sum(),
                           "thinking": by_day["thinking"].sum()}).reset_index()
    return counts.sort_values(["model", "day"], kind="stable").reset_index(drop=True)[COUNT_COLUMNS]


def counted(counts: pd.DataFrame) -> pd.DataFrame:
    """The days that count, with their `level`: MIN_RESPONSES responses or more, LOGGED_SHARE
    of them with a logged count, the level being the mean over those."""
    keep = ((counts["responses"] >= MIN_RESPONSES) & (counts["logged"] >= LOGGED_SHARE * counts["responses"])
            & (counts["logged"] > 0))
    days = counts[keep].copy()
    days["level"] = days["thinking"] / days["logged"]
    return days.sort_values(["model", "day"], kind="stable").reset_index(drop=True)


def thinking_rises(counts: pd.DataFrame, state: dict[str, Any], today: date, ratio: float,
                   days: int) -> list[dict[str, Any]]:
    """New rises on the days of `counts` before `today`, each recorded in
    state["thinking_rises"]. Per model, each counted day with MIN_BASELINE_DAYS counted days
    before it is raised at `ratio` times the median level of up to BASELINE_DAYS of them; a
    rise is reported on the day it has lasted `days` counted days in a row, once. A counted
    day that isn't raised ends the run; a day that doesn't count is skipped. Raised days stay
    in the baseline, so a rise that lasts becomes the level. A baseline of no thinking at all
    raises nothing: any thinking would be infinitely more."""
    rises = state.setdefault("thinking_rises", [])
    judged = counted(counts[counts["day"].astype(str) < today.isoformat()])
    new = []
    for model, group in judged.groupby("model", sort=True):
        levels = list(zip(group["day"].astype(str), group["level"].astype(float)))
        run: list[tuple[str, float, float]] = []
        for i, (day, level) in enumerate(levels):
            before = [value for _, value in levels[max(0, i - BASELINE_DAYS):i]]
            if len(before) < MIN_BASELINE_DAYS:
                continue
            median = statistics.median(before)
            if median <= 0 or level < ratio * median:
                run = []
                continue
            run.append((day, level, median))
            if len(run) != days:
                continue
            rise = {"model": str(model), "since": run[0][0], "on": day, "median": round(run[0][2], 1),
                    "levels": [round(value, 1) for _, value, _ in run], "reported_on": today.isoformat()}
            if not any(r["model"] == rise["model"] and r["since"] == rise["since"] for r in rises):
                rises.append(rise)
                new.append(rise)
    return new


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
    catches out of how many, the most counted days a catch took, and the day it reports the
    episode (the counted days of `model` in `full` it covers) by its PLANT_RUN-th, or None."""
    starts = plant_starts(clean, model)
    out = []
    for ratio in ratios:
        for days in days_grid:
            alarms = replay(clean, ratio, days)
            seen = {(r["model"], r["since"]) for r in alarms}
            caught: dict[float, int] = {}
            slowest: dict[float, int] = {}
            for times in plants:
                caught[times] = slowest[times] = 0
                for first in starts:
                    within = plant_run(clean, model, first)
                    hits = [r for r in replay(plant(clean, model, first, times), ratio, days)
                            if r["model"] == model and r["since"] in within and r["on"] in within
                            and (r["model"], r["since"]) not in seen]
                    if hits:
                        caught[times] += 1
                        slowest[times] = max(slowest[times], within.index(hits[0]["on"]) + 1)
            found = [r["on"] for r in replay(full, ratio, days)
                     if r["model"] == model and r["since"] in episode and r["on"] in episode[:PLANT_RUN]]
            row = {"ratio": ratio, "days": days, "alarms": [f"{r['model']} {r['since']}" for r in alarms],
                   "caught": caught, "slowest": slowest, "starts": len(starts), "episode": found[0] if found else None}
            row["passes"] = (not row["alarms"] and bool(starts) and row["episode"] is not None
                             and all(caught[times] == len(starts) for times in plants if times >= BAR))
            out.append(row)
    return out


def _setting(row: dict[str, Any]) -> str:
    return f"ratio={row['ratio']:g} days={row['days']}"


def gate(rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Whether any setting passes, and which, or the one that came closest: fewest false
    alarms, then most plants of BAR times or more caught, then the episode caught."""
    passing = [row for row in rows if row["passes"]]
    if passing:
        return True, [f"passing settings: {'; '.join(_setting(row) for row in passing)}"]
    def big(row: dict[str, Any]) -> int:
        return sum(n for times, n in row["caught"].items() if times >= BAR)
    closest = min(rows, key=lambda row: (len(row["alarms"]), -big(row), row["episode"] is None))
    plants = closest["starts"] * sum(1 for times in closest["caught"] if times >= BAR)
    return False, [f"no setting passes; closest: {_setting(closest)}: {len(closest['alarms'])} false alarm(s)"
                   + (f" ({', '.join(closest['alarms'])})" if closest["alarms"] else "")
                   + f", plants of x{BAR:g} or more caught {big(closest)} of {plants}"
                   + f", episode {'on ' + closest['episode'] if closest['episode'] else 'missed'}"]


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
    table = rows(clean, full, episode)
    for row in table:
        print(f"G16 {_setting(row):<16} false-alarms={len(row['alarms'])} "
              + " ".join(f"x{times:g}={n}/{row['starts']}" for times, n in row["caught"].items())
              + f" slowest(x>={BAR:g})={max(d for t, d in row['slowest'].items() if t >= BAR)}"
              + f" episode={row['episode'] or 'missed'}"
              + (f"  alarms: {', '.join(row['alarms'])}" if row["alarms"] else ""))
    ok, notes = gate(table)
    print(f"G16: {'PASS' if ok else 'FAIL'}")
    for note in notes:
        print(f"  {note}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
