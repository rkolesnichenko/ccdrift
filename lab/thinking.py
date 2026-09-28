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
"""

from __future__ import annotations

import statistics
from datetime import date
from typing import Any

import pandas as pd

COUNT_COLUMNS = ["model", "day", "responses", "logged", "thinking"]
MIN_RESPONSES = 50      # a model's judged turns on a day, for the day to count: claude-opus-5's
                        # quietest counted day on the owner's logs had 78, and a day under 50 is a
                        # half-used one whose mean a handful of long answers can move
LOGGED_SHARE = 0.9      # of them with a logged thinking count: Claude Code logs one on every
                        # response from 2026-08-17, and on 35% of 08-16's, the first day it did
BASELINE_DAYS = 14      # counted days a day is judged against, at most and at least: the same
MIN_BASELINE_DAYS = 5   # as the setting alert, the other rule that judges a model's own days


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
