"""ccdrift status: how the last check went and what ccdrift is following. It reads
only the state file, never the transcripts or the history, so a status line can
call it often."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from ccdrift.state import load_state
from ccdrift.texts import (COMMAND_LINES, LIVE_NAMES, LOOP_NAMES, STATUS_LINES, change_line, clock_text,
                           context_change_line, cut_short_line, early_warning_line, failure_line, field_gap_line,
                           hook_change_line, hook_failure_line, incident_line, loop_warning_line, new_field_line)

STALE_DAYS = 3
HOOK_DAYS = 3
RECENT_DAYS = 30
RISING_HOURS = 24


def _when(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp)


def short_status(state_path: Path, now: datetime) -> str:
    """One line for a status line, or "" when nothing needs attention. First match
    wins: an unreadable or malformed state, no check yet, a failed check, no successful
    check for more than STALE_DAYS days, open incidents, hooks failing since within
    the last HOOK_DAYS days, cache misses rising, tool-loop cache misses rising (the
    main thread, then subagents)."""
    try:
        state = load_state(state_path)
        last = state.get("last_run")
        if last is None:
            return STATUS_LINES["no_check"]
        if not last["ok"]:
            return STATUS_LINES["failed"].format(at=_when(last["started"]))
        since_ok = now - _when(state["last_ok"])
        if since_ok > timedelta(days=STALE_DAYS):
            return STATUS_LINES["stale"].format(days=since_ok.days)
        live = [STATUS_LINES["live"].format(name=LIVE_NAMES[i["metric"]], since=i["start"][5:])
                for i in state["incidents"] if i["status"] == "open"]
        if live:
            return STATUS_LINES["incidents"].format(incidents="; ".join(live))
        recent = (now.date() - timedelta(days=HOOK_DAYS)).isoformat()
        failing = [f for f in state.get("hook_failures", []) if f["reported_on"] >= recent]
        if failing:
            return STATUS_LINES["hooks"].format(since=failing[-1]["since"][5:])
        rising = [w for w in state.get("early_warnings", [])
                  if now - _when(w["at"]) < timedelta(hours=RISING_HOURS)]
        if rising:
            return STATUS_LINES["rising"].format(since=clock_text(rising[-1]["since"], now))
        for stream in ("main", "subagent"):
            loops = [w for w in state.get("loop_warnings", [])
                     if w["stream"] == stream and now - _when(w["at"]) < timedelta(hours=RISING_HOURS)]
            if loops:
                return STATUS_LINES["loop"].format(name=LOOP_NAMES[stream], since=clock_text(loops[-1]["since"], now))
        return ""
    except Exception:  # a status line must never show a traceback, whatever the state holds
        return STATUS_LINES["unreadable"]


def _section(title: str, items: list[str]) -> list[str]:
    if not items:
        return [STATUS_LINES["empty_section"].format(title=title)]
    return [STATUS_LINES["section"].format(title=title)] + [f"  {item}" for item in items]


def _hook_changes(records: list[dict[str, Any]], since: str) -> list[str]:
    """The hook coverage alerts reported from `since` on, one line each: the records of
    the streams an alert folded together share its thread, direction and day reported,
    and the line gives the earliest start. Never a record's `stream`, which names the
    project and tool, and an MCP server's among them."""
    alerts: dict[tuple[str, str, str], str] = {}
    for record in records:
        if record.get("alerted") and record["reported_on"] >= since:
            key = (record["reported_on"], record["thread"], record["direction"])
            alerts[key] = min(alerts.get(key, record["since"]), record["since"])
    return [hook_change_line(thread, direction, first) for (_, thread, direction), first in sorted(alerts.items())]


def status_report(state: dict[str, Any], now: datetime) -> str:
    last = state.get("last_run")
    if last is None:
        return STATUS_LINES["not_run"]
    outcome = STATUS_LINES["ok"] if last["ok"] else STATUS_LINES["outcome_failed"].format(error=last["error"])
    lines = [STATUS_LINES["last"].format(at=_when(last["started"]), outcome=outcome)]
    if not last["ok"] and state.get("last_ok"):
        lines.append(STATUS_LINES["last_ok"].format(at=_when(state["last_ok"])))
    since = (now.date() - timedelta(days=RECENT_DAYS)).isoformat()
    incidents = state["incidents"]
    lines += _section(STATUS_LINES["open"], [incident_line(i) for i in incidents if i["status"] == "open"])
    lines += _section(STATUS_LINES["closed"].format(days=RECENT_DAYS),
                      [incident_line(i) for i in incidents if i["status"] != "open" and (i["closed_on"] or "") >= since])
    lines += _section(STATUS_LINES["settings"].format(days=RECENT_DAYS),
                      [change_line(c) for c in state["settings"] if c["reported_on"] >= since])
    other = ([context_change_line(c) for c in state.get("context_changes", []) if c["reported_on"] >= since]
             + [hook_failure_line(f) for f in state.get("hook_failures", []) if f["reported_on"] >= since]
             + _hook_changes(state.get("hook_changes", []), since)
             + [field_gap_line(g) for g in state.get("field_gaps", []) if g["reported_on"] >= since]
             + [new_field_line(r) for r in state.get("new_fields", []) if r["reported_on"] >= since]
             + [early_warning_line(w) for w in state.get("early_warnings", []) if w["reported_on"] >= since]
             + [loop_warning_line(w) for w in state.get("loop_warnings", []) if w["reported_on"] >= since]
             + [failure_line(f) for f in state.get("failed_requests", []) if f["reported_on"] >= since]
             + [cut_short_line(c) for c in state.get("cut_short", []) if c["reported_on"] >= since])
    lines += _section(STATUS_LINES["other"].format(days=RECENT_DAYS), other)
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
        print(COMMAND_LINES["state_unreadable"].format(path=state_path, error=exc), file=sys.stderr)
        return 1
    print(status_report(state, now), end="")
    return 0
