"""The weekly digest: one alert on the first run after Monday 09:00 that sums up the
week before, so a quiet week reads as checked rather than as a check that stopped."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from ccdrift.failures import week_failures
from ccdrift.texts import digest_text, version_key

DIGEST_HOUR = 9


def digest_week(now: datetime) -> str:
    year, week, _ = now.isocalendar()
    return f"{year}-W{week:02d}"


def digest_due(state: dict[str, Any], now: datetime) -> Optional[date]:
    """The Monday starting the week to sum up, when the digest is due: at or after
    Monday DIGEST_HOUR:00 local of this ISO week and once that week's Sunday has ended
    in UTC, not sent this week, and some earlier run happened before this Monday. The
    week is summed by UTC day, and east of UTC+9 Monday 09:00 falls on Sunday there,
    which the check still counts as in progress."""
    monday = now.date() - timedelta(days=now.weekday())
    if now < datetime.combine(monday, time(DIGEST_HOUR), tzinfo=now.tzinfo):
        return None
    if now.astimezone(timezone.utc).date() < monday:
        return None
    if state.get("digest_week") == digest_week(now):
        return None
    if not any(day < monday.isoformat() for day in state.get("runs", [])):
        return None
    return monday - timedelta(days=7)


def week_summary(turns: pd.DataFrame, state: dict[str, Any], week_start: date,
                 loops: Optional[pd.DataFrame] = None, failures: Optional[pd.DataFrame] = None) -> dict[str, Any]:
    """The numbers the weekly summary states (texts.digest_text words them): the judged
    turns of the week from `week_start`, their versions, cache ratio, misses and Haiku
    share, its tool-loop turns and misses from `loops` (loops.loop_counts by day), its
    failed requests and responses cut short from `failures` (failures.failure_counts),
    the open incidents, the setting changes and new fields reported that week, and the
    days the check ran."""
    days = [(week_start + timedelta(days=i)).isoformat() for i in range(7)]
    week = turns[turns["day"].astype(str).isin(days)] if not turns.empty else turns
    summary: dict[str, Any] = {"week_start": days[0], "responses": len(week), "versions": [], "prompts": None,
                               "haiku": 0.0, "loops": {}, "failures": None}
    if not week.empty:
        if "version" in week:
            summary["versions"] = sorted(week["version"].dropna().astype(str).unique(), key=version_key)
        prompts = week[week["prompt_within_ttl"].astype(bool)]
        if not prompts.empty:
            summary["prompts"] = (float(prompts["cache_read_ratio"].mean()),
                                  float(prompts["is_miss"].astype(bool).mean()))
        summary["haiku"] = float(week["is_haiku"].mean())
        counted = loops[loops.index.isin(days)].sum() if loops is not None and not loops.empty else {}
        summary["loops"] = {prefix: (int(counted.get(f"{prefix}_turns", 0)), int(counted.get(f"{prefix}_misses", 0)))
                            for prefix in ("loop", "subagent_loop")}
        if failures is not None:
            summary["failures"] = week_failures(failures, days)
    summary["open_incidents"] = sum(1 for i in state["incidents"] if i["status"] == "open")
    summary["setting_changes"] = sum(1 for c in state["settings"] if c["reported_on"] in days)
    summary["new_fields"] = sum(len(record["paths"]) for record in state.get("new_fields", [])
                                if record["reported_on"] in days)
    summary["ran"] = len(set(state.get("runs", [])) & set(days))
    return summary


def weekly_digest(turns: pd.DataFrame, state: dict[str, Any], week_start: date,
                  loops: Optional[pd.DataFrame] = None, failures: Optional[pd.DataFrame] = None) -> str:
    """The weekly summary of the week from `week_start` as one line (see week_summary)."""
    return digest_text(week_summary(turns, state, week_start, loops, failures))
