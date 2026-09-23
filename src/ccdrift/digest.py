"""The weekly digest: one alert on the first run after Monday 09:00 that sums up the
week before, so a quiet week reads as checked rather than as a check that stopped."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional

import pandas as pd

from ccdrift.failures import digest_part
from ccdrift.texts import version_key

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


def _count(n: int, noun: str) -> str:
    return f"no {noun}s" if n == 0 else f"{n} {noun}{'' if n == 1 else 's'}"


def _loop_part(loops: Optional[pd.DataFrame], days: list[str]) -> str:
    """"tool-loop misses 2 of 1,880, subagent 36 of 13,400" over `days` of `loops`
    (loops.loop_counts by day)."""
    week = loops[loops.index.isin(days)].sum() if loops is not None and not loops.empty else {}
    parts = []
    for prefix, found, missing in (("loop", "tool-loop misses", "no tool-loop turns"),
                                   ("subagent_loop", "subagent", "no subagent loop turns")):
        turns, misses = int(week.get(f"{prefix}_turns", 0)), int(week.get(f"{prefix}_misses", 0))
        parts.append(f"{found} {misses:,} of {turns:,}" if turns else missing)
    return ", ".join(parts)


def weekly_digest(turns: pd.DataFrame, state: dict[str, Any], week_start: date,
                  loops: Optional[pd.DataFrame] = None, failures: Optional[pd.DataFrame] = None) -> str:
    """One line on the judged turns of the week from `week_start`, its tool-loop misses
    from `loops` (loops.loop_counts by day), its failed requests from `failures`
    (failures.failure_counts), the open incidents, the setting changes reported that
    week, and the days the check ran."""
    days = [(week_start + timedelta(days=i)).isoformat() for i in range(7)]
    week = turns[turns["day"].astype(str).isin(days)] if not turns.empty else turns
    parts = []
    if week.empty:
        parts.append("no responses")
    else:
        versions = (sorted(week["version"].dropna().astype(str).unique(), key=version_key)
                    if "version" in week else [])
        span = "" if not versions else f" on {versions[0]}" + (f"–{versions[-1]}" if len(versions) > 1 else "")
        parts.append(f"{len(week):,} responses{span}")
        prompts = week[week["prompt_within_ttl"].astype(bool)]
        if prompts.empty:
            parts.append("no new-prompt turns")
        else:
            parts.append(f"cache ratio {prompts['cache_read_ratio'].mean():.3f} "
                         f"({prompts['is_miss'].astype(bool).mean():.1%} misses)")
        haiku = float(week["is_haiku"].mean())
        parts.append("no Haiku" if haiku == 0 else f"Haiku {haiku:.1%} of responses")
        parts.append(_loop_part(loops, days))
        if failures is not None:
            parts.append(digest_part(failures, days))
    parts.append(_count(sum(1 for i in state["incidents"] if i["status"] == "open"), "open incident"))
    parts.append(_count(sum(1 for c in state["settings"] if c["reported_on"] in days), "setting change"))
    parts.append(_count(sum(len(record["paths"]) for record in state.get("new_fields", [])
                            if record["reported_on"] in days), "new field"))
    ran = len(set(state.get("runs", [])) & set(days))
    parts.append(f"check ran on {ran} of 7 days")
    return f"Week of {days[0][5:]}: " + "; ".join(parts) + "."
