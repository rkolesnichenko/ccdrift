"""Early warning: a turn-by-turn CUSUM on new-prompt cache misses, so a caching
regression can show within hours instead of after 3 of 4 bad days.

It is a Bernoulli likelihood-ratio CUSUM testing the usual miss rate against P1, the
August 2026 regression's rate. A z-score CUSUM with k = 0.5 can't accumulate on a
rare event: at a 5% miss rate its sum drains between misses. lab/early_warning.py
measured it on the owner's logs and set THRESHOLD; None means the warning isn't
built."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import pandas as pd

from ccdrift.incidents import versions_text
from ccdrift.texts import clock_text

P1 = 0.05
MIN_P0 = 0.002
MAX_P0 = 0.025
THRESHOLD: Optional[float] = 4.0


def clamp_rate(rate: float) -> float:
    return min(max(rate, MIN_P0), MAX_P0)


def alarm_runs(misses: Sequence[bool], base_rate: float, h: float, p1: float = P1) -> list[tuple[int, int]]:
    """(first, alarm) for each alarm of the CUSUM for misses at `p1` against
    `base_rate`: the alarm is the position where the sum passes h, `first` the position
    after the sum last stood at 0. The sum restarts from 0 after each alarm."""
    p0 = clamp_rate(base_rate)
    on_miss, on_hit = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
    total = 0.0
    first = 0
    runs = []
    for position, missed in enumerate(misses):
        total = max(0.0, total + (on_miss if missed else on_hit))
        if total > h:
            runs.append((first, position))
            total = 0.0
        if total == 0.0:
            first = position + 1
    return runs


def miss_cusum(misses: Sequence[bool], base_rate: float, h: float, p1: float = P1) -> list[int]:
    """Positions at which the CUSUM passes h (see alarm_runs)."""
    return [alarm for _, alarm in alarm_runs(misses, base_rate, h, p1)]


def prompt_turns(df: pd.DataFrame) -> pd.DataFrame:
    """CLI main-thread new-prompt turns, in time order."""
    keep = df["main_thread"].astype(bool) & df["prompt_within_ttl"].astype(bool)
    if "entrypoint" in df:
        keep &= ~df["entrypoint"].fillna("").astype(str).str.startswith("sdk-")
    turns = df.loc[keep, ["timestamp", "day", "is_miss"]].copy()
    turns["day"] = turns["day"].astype(str)
    turns["is_miss"] = turns["is_miss"].astype(bool)
    return turns.sort_values("timestamp", kind="stable").reset_index(drop=True)


WINDOW_DAYS = 7      # the stretch the CUSUM runs over, today included
BASE_DAYS = 14       # complete days before it that give the usual miss rate
MIN_BASE_TURNS = 200
RECENT_HOURS = 24    # an alarm this recent is news
QUIET_DAYS = 7       # at most one warning in this many days


def _known_incident_days(days: pd.Series, incidents: Sequence[dict], today: str) -> pd.Series:
    known = pd.Series(False, index=days.index)
    for incident in incidents:
        if incident["metric"] == "cache_ratio" and incident["status"] != "dismissed":
            known |= days.between(incident["start"], incident["end"] or today)
    return known


def early_warning(responses: pd.DataFrame, incidents: Sequence[dict], state: dict[str, Any], now: datetime,
                  h: Optional[float] = None) -> Optional[dict[str, Any]]:
    """A warning when the CUSUM on the CLI main thread's new-prompt turns of the last
    WINDOW_DAYS days (up to `now`) alarmed in the last RECENT_HOURS hours, against the
    miss rate of the BASE_DAYS days before them. Days of recorded cache incidents are
    left out of both; no warning while a cache incident is open, within QUIET_DAYS of
    the last warning, or with under MIN_BASE_TURNS usual turns. The warning is also
    recorded in state["early_warnings"]."""
    h = THRESHOLD if h is None else h
    if h is None or responses.empty:
        return None
    if any(i["metric"] == "cache_ratio" and i["status"] == "open" for i in incidents):
        return None
    if any(now - datetime.fromisoformat(w["at"]) < timedelta(days=QUIET_DAYS) for w in state["early_warnings"]):
        return None
    today = now.astimezone(timezone.utc).date()
    first_day = (today - timedelta(days=WINDOW_DAYS - 1)).isoformat()
    base_start = (today - timedelta(days=WINDOW_DAYS - 1 + BASE_DAYS)).isoformat()
    turns = prompt_turns(responses)
    days = turns["day"]
    known = _known_incident_days(days, incidents, today.isoformat())
    base = (days >= base_start) & (days < first_day) & ~known
    if base.sum() < MIN_BASE_TURNS:
        return None
    base_rate = float(turns.loc[base, "is_miss"].mean())
    stretch = turns[(days >= first_day) & ~known & (turns["timestamp"] <= pd.Timestamp(now))].reset_index(drop=True)
    runs = alarm_runs(stretch["is_miss"].tolist(), base_rate, h)
    if not runs:
        return None
    first, alarm = runs[-1]
    at = stretch["timestamp"][alarm].to_pydatetime()
    if now - at > timedelta(hours=RECENT_HOURS):
        return None
    run = stretch.iloc[first:alarm + 1]
    main = responses[responses["main_thread"].astype(bool)]
    if "entrypoint" in main:
        main = main[~main["entrypoint"].fillna("").astype(str).str.startswith("sdk-")]
    warning = {"at": at.isoformat(timespec="seconds"),
               "since": run["timestamp"].iloc[0].to_pydatetime().isoformat(timespec="seconds"),
               "misses": int(run["is_miss"].sum()), "turns": len(run), "base_rate": round(base_rate, 4),
               "versions": versions_text(main, sorted(run["day"].unique())),
               "reported_on": now.date().isoformat()}
    state["early_warnings"].append(warning)
    return warning


def early_message(warning: dict[str, Any], now: datetime) -> str:
    on = f", on Claude Code {', '.join(warning['versions'])}" if warning["versions"] else ""
    return (f"{warning['misses']} of the last {warning['turns']} new-prompt turns missed the cache "
            f"(usually {warning['base_rate']:.1%}), since {clock_text(warning['since'], now)}{on}. "
            "The daily check confirms or clears it within a few days.")
