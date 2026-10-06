"""The Claude Code plugin's hook. At each session start it shows the status line's verdict
and, when the last check began CHECK_INTERVAL or more ago, starts one in the background, so
Claude Code itself stands in for `ccdrift schedule install`. Claude's first response waits
for it, so like `status --short` it imports neither pandas nor numpy."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from ccdrift.state import load_state

# The cadence of `schedule install` without --at, so an hourly schedule keeps the last check
# fresh and the plugin starts one only when the schedule missed it. A warm check took 3.3-4.0 s
# on a 31.9 MB store and a cold one 34.4 s (2026-10-06): too long to run before a first response.
CHECK_INTERVAL = timedelta(minutes=60)


def check_due(state_path: Path, now: datetime) -> bool:
    """Whether a session start should begin a check: none has run, the last began
    CHECK_INTERVAL or more before `now`, or the state can't be read, which the check
    itself then reports."""
    try:
        started = datetime.fromisoformat(load_state(state_path)["last_run"]["started"])
        return now - started >= CHECK_INTERVAL
    except (OSError, ValueError, KeyError, TypeError):
        return True


def run_hook_check(state_path: Path) -> int:
    """`ccdrift hook check`: the check a session start began, with notifications. It gives
    up while another command holds the state, and skips the run when a check has begun
    since the session start found one due."""
    from ccdrift.check import run_check
    from ccdrift.logs import default_source
    return run_check(default_source(), state_path, notify_user=True,
                     due=lambda: check_due(state_path, datetime.now().astimezone()))
