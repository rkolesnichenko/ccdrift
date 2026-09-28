"""Thinking that rises. Claude Code logs how many thinking tokens each response used, and
thinking is billed as output, so a release that makes a model think far more per response
costs money and usage limits without a word. From 2026-09-10 to 09-17 the owner's main thread
thought 2 to 11 times as much per response as the median of the days before, on Claude Code
2.1.267 to 2.1.273, and nothing noticed.

The metric is each model's mean logged thinking tokens per main-thread CLI response on a
complete UTC day, on days with enough responses that nearly all logged a count. A day is
raised at RATIO times the median of the model's previous counted days, and a rise is reported
once it lasts DAYS counted days. lab/thinking.py (G16) measures the setting on the owner's
logs."""

from __future__ import annotations

import statistics
from datetime import date, timedelta
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
# G16 on the owner's logs with the episode ending 2026-09-17: ratio 2 over 1 day is the one
# setting of 12 with no false alarm that catches every plant of 3, 4 and 6 times from all 19
# starts, within 2 counted days, and reports the episode on its first counted day.
RATIO = 2
DAYS = 1
MIN_MEDIAN = 100        # tokens per response a baseline needs to be judged: under it a model barely
                        # thinks and a doubling is a handful of tokens. The owner's lowest counted
                        # day is 140, on claude-opus-5-5, so the floor silences nothing real
RECENT_DAYS = 14        # a rise finished longer ago isn't news, as for the setting alert; the first
                        # check after an upgrade reads 90 days and would announce old ones


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


def thinking_rises(counts: pd.DataFrame, state: dict[str, Any], today: date, ratio: float = RATIO,
                   days: int = DAYS) -> list[dict[str, Any]]:
    """New rises on the days of `counts` before `today`, each recorded in
    state["thinking_rises"]. Per model, each counted day with MIN_BASELINE_DAYS counted days
    before it is raised at `ratio` times the median level of up to BASELINE_DAYS of them, when
    that median is MIN_MEDIAN or more; a rise is reported on the day it has lasted `days`
    counted days in a row, once, if that day is within RECENT_DAYS of today. A rise that began
    earlier and is still going on the model's latest counted day, within RECENT_DAYS, is
    reported then, for a check that first runs long after it began; one that has ended, or whose
    model hasn't been used since RECENT_DAYS ago, stays quiet. A counted day that isn't raised ends the run; a day that doesn't count is skipped.
    Raised days stay in the baseline, so a rise that lasts becomes the level, and the median
    never falls during a run. A run that starts within RECENT_DAYS of the model's last raised day
    is the same rise going on unless it starts at `ratio` times that day's level: a heavy day
    that recurs every week would otherwise be a new rise every week. A rise's `extra` is the
    thinking over each day's median on the days it covers when reported."""
    rises = state["thinking_rises"]
    judged = counted(counts[counts["day"].astype(str) < today.isoformat()])
    recent = (today - timedelta(days=RECENT_DAYS)).isoformat()
    new = []
    for model, group in judged.groupby("model", sort=True):
        levels = list(zip(group["day"].astype(str), group["level"].astype(float), group["logged"].astype(int)))
        run: list[tuple[str, float, float, int]] = []
        last: tuple[str, float] | None = None  # the model's latest raised day before the current run

        def report(covered: list[tuple[str, float, float, int]]) -> None:
            start = date.fromisoformat(covered[0][0])
            if (last is not None and last[0] >= (start - timedelta(days=RECENT_DAYS)).isoformat()
                    and covered[0][1] < ratio * last[1]):
                return
            rise = {"model": str(model), "since": covered[0][0], "on": covered[-1][0],
                    "days": [one[0] for one in covered], "median": round(covered[0][2], 1),
                    "levels": [round(one[1], 1) for one in covered],
                    "extra": round(sum((one[1] - one[2]) * one[3] for one in covered)),
                    "reported_on": today.isoformat()}
            if not any(r["model"] == rise["model"] and r["since"] == rise["since"] for r in rises):
                rises.append(rise)
                new.append(rise)

        for i, (day, level, logged) in enumerate(levels):
            before = [value for _, value, _ in levels[max(0, i - BASELINE_DAYS):i]]
            if len(before) < MIN_BASELINE_DAYS:
                continue
            median = statistics.median(before)
            if median < MIN_MEDIAN or level < ratio * median:
                last = (run[-1][0], run[-1][1]) if run else last
                run = []
                continue
            run.append((day, level, median, logged))
            if len(run) == days and day >= recent:
                report(run)
        if len(run) > days and run[days - 1][0] < recent <= run[-1][0]:
            report(run)
    return new
