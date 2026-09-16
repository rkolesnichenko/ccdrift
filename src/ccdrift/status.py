"""ccdrift status: how the last check went and what ccdrift is following. It reads
only the state file, never the transcripts or the history, so a status line can
call it often."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from ccdrift.incidents import incident_line
from ccdrift.settings import change_line
from ccdrift.state import load_state

STALE_DAYS = 3
RECENT_DAYS = 30
LIVE_NAMES = {"cache_ratio": "cache ratio down", "haiku_fraction": "Haiku share up"}


def _when(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def short_status(state_path: Path, now: datetime) -> str:
    """One line for a status line, or "" when nothing needs attention. First match
    wins: an unreadable or malformed state, no check yet, a failed check, no successful
    check for more than STALE_DAYS days, open incidents."""
    try:
        state = load_state(state_path)
        last = state.get("last_run")
        if last is None:
            return "ccdrift: no check yet"
        if not last["ok"]:
            return f"ccdrift: check failed {_when(last['started']).strftime('%m-%d %H:%M')}"
        since_ok = now - _when(state["last_ok"])
        if since_ok > timedelta(days=STALE_DAYS):
            return f"ccdrift: no check for {since_ok.days} days"
        live = [f"{LIVE_NAMES[i['metric']]} since {i['start'][5:]}"
                for i in state["incidents"] if i["status"] == "open"]
        return f"ccdrift: {'; '.join(live)}" if live else ""
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return "ccdrift: can't read state"


def _section(title: str, items: list[str]) -> list[str]:
    return [f"{title}: none"] if not items else [f"{title}:"] + [f"  {item}" for item in items]


def status_report(state: dict[str, Any], now: datetime) -> str:
    last = state.get("last_run")
    if last is None:
        return "The daily check hasn't run yet. `ccdrift schedule install` sets it up.\n"
    outcome = "ok" if last["ok"] else f"failed: {last['error']}"
    lines = [f"Last check: {_when(last['started']).strftime('%Y-%m-%d %H:%M')}, {outcome}"]
    if not last["ok"] and state.get("last_ok"):
        lines.append(f"Last successful check: {_when(state['last_ok']).strftime('%Y-%m-%d %H:%M')}")
    since = (now.date() - timedelta(days=RECENT_DAYS)).isoformat()
    incidents = state["incidents"]
    lines += _section("Open incidents", [incident_line(i) for i in incidents if i["status"] == "open"])
    lines += _section(f"Closed in the last {RECENT_DAYS} days",
                      [incident_line(i) for i in incidents if i["status"] != "open" and (i["closed_on"] or "") >= since])
    lines += _section(f"Setting changes in the last {RECENT_DAYS} days",
                      [change_line(c) for c in state["settings"] if c["reported_on"] >= since])
    return "\n".join(lines) + "\n"


def run_status(state_path: Path, short: bool = False, now: Optional[datetime] = None) -> int:
    now = now or datetime.now().astimezone()
    if short:
        line = short_status(state_path, now)
        if line:
            print(line)
        return 0
    try:
        state = load_state(state_path)
    except (OSError, ValueError) as exc:
        print(f"Can't read the state file {state_path}: {exc}", file=sys.stderr)
        return 1
    print(status_report(state, now), end="")
    return 0
