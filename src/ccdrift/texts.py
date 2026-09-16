"""Names and one-line texts shared by the check, the report and the status line. It
imports nothing heavy, so `ccdrift status --short` stays cheap enough for a status
line that refreshes often."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Optional

METRIC_ARGS = {"cache": "cache_ratio", "haiku": "haiku_fraction"}

INCIDENT_METRICS = {"cache_ratio": "Cache read ratio on new prompts",
                    "haiku_fraction": "Haiku share on the main thread"}
SHORT_NAMES = {"cache_ratio": "cache", "haiku_fraction": "haiku"}
MOVES = {"cache_ratio": "down", "haiku_fraction": "up"}
PERSISTENT_DAYS = 30

SETTING_NAMES = {"cache_tier": "cache tier", "effort": "effort", "speed": "speed", "service_tier": "service tier"}
TIER_NAMES = {"1h": "1-hour", "5m": "5-minute"}


def approx(value: float) -> str:
    """Two significant figures with a k, M or B suffix: 17_556_103 -> "18M"."""
    if value <= 0:
        return "0"
    rounded = float(f"{value:.2g}")
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if rounded >= size:
            return f"{rounded / size:g}{suffix}"
    return f"{rounded:g}"


def cost_text(metric: str, cost: float) -> str:
    """"~18M tokens re-cached" or "~120 extra Haiku responses"."""
    if metric == "cache_ratio":
        return f"~{approx(cost)} tokens re-cached" if cost > 0 else "no tokens re-cached"
    return f"~{approx(cost)} extra Haiku responses" if cost > 0 else "no extra Haiku responses"


def incident_line(incident: dict, cost: Optional[float] = None) -> str:
    """One line about an incident, as `incident list`, `report` and `status` show it."""
    status = {"open": "open", "persistent": f"still changed after {PERSISTENT_DAYS} days",
              "dismissed": "dismissed"}.get(incident["status"])
    if incident["status"] == "recovered":
        if incident["source"] == "user":
            status = "added by hand"
        elif incident["closed_by"] == "user":
            status = f"closed by hand on {incident['closed_on']}"
        else:
            status = f"back to normal from {incident['recovered_from']}"
    parts = [status, cost_text(incident["metric"], incident["cost"] if cost is None else cost)]
    if incident["versions"]:
        parts.append("on " + ", ".join(incident["versions"]))
    span = f"{incident['start']}..{incident['end'] or 'now'}"
    return f"{SHORT_NAMES[incident['metric']]:<5}  {span:<24}  {'; '.join(parts)}"


def change_line(change: dict[str, Any]) -> str:
    return (f"{SETTING_NAMES[change['setting']]} for {change['model']}: {change['from']} -> {change['to']} "
            f"from {change['since']}")


def clock_text(stamp: str, now: datetime) -> str:
    """`stamp` as a time in `now`'s time zone: "14:20" on now's day, "09-19 23:40" before it."""
    when = datetime.fromisoformat(stamp).astimezone(now.tzinfo)
    return when.strftime("%H:%M") if when.date() == now.date() else when.strftime("%m-%d %H:%M")


def version_key(version: str) -> tuple:
    """Sorts 2.1.99 before 2.1.233, and "unknown" last."""
    if version == "unknown":
        return (1,)
    return (0, *((0, int(part), "") if part.isdigit() else (1, 0, part) for part in re.split(r"[.+-]", version)))
