"""The Claude Code plugin's hook. At each session start it shows the status line's verdict
and, when the last check began CHECK_INTERVAL or more ago, starts one in the background, so
Claude Code itself stands in for `ccdrift schedule install`. Claude's first response waits
for it, so like `status --short` it imports neither pandas nor numpy."""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, TextIO

from ccdrift.quota import read_payload
from ccdrift.state import LOG_FILE, load_state, make_private
from ccdrift.status import short_status
from ccdrift.texts import HOOK_LINES

# The cadence of `schedule install` without --at, so an hourly schedule keeps the last check
# fresh and the plugin starts one only when the schedule missed it. A warm check took 3.3-4.0 s
# on a 31.9 MB store and a cold one 34.4 s (2026-10-06): too long to run before a first response.
CHECK_INTERVAL = timedelta(minutes=60)
# The session starts that show the verdict; clear, compact and fork come in the middle of work.
SHOWN = ("startup", "resume")


def check_due(state_path: Path, now: datetime) -> bool:
    """Whether a session start should begin a check: none has run, the last began
    CHECK_INTERVAL or more before `now`, or the state can't be read, which the check
    itself then reports."""
    try:
        started = datetime.fromisoformat(load_state(state_path)["last_run"]["started"])
        return now - started >= CHECK_INTERVAL
    except (OSError, ValueError, KeyError, TypeError):
        return True


def hook_source(payload: Optional[str]) -> str:
    """The `source` of a SessionStart payload; "startup" when there is no payload, or it
    doesn't parse or carry one."""
    try:
        source = json.loads(payload)["source"]  # type: ignore[arg-type]
    except (TypeError, ValueError, KeyError):
        return "startup"
    return source if isinstance(source, str) else "startup"


def private_log(state_path: Path) -> Path:
    """The check's log beside `state_path`, created or narrowed to its owner."""
    log = state_path.with_name(LOG_FILE)
    log.parent.mkdir(parents=True, exist_ok=True)
    make_private(log)
    return log


def start_check(state_path: Path) -> None:
    """Start `ccdrift hook check` with this Python, so the same install, and return at once.
    It runs in a session of its own, so a `claude -p` that ends doesn't take it down, with
    its output appended to the check's log. It finds the state where this command did, in
    ccdrift's home, and starts there: Claude Code runs the hook in the project's folder, where
    `python -m` would import a ccdrift/ or a json.py the project holds before ccdrift's own."""
    with open(private_log(state_path), "a") as out:
        subprocess.Popen([sys.executable, "-m", "ccdrift", "hook", "check"], stdin=subprocess.DEVNULL,
                         stdout=out, stderr=out, cwd=state_path.parent, start_new_session=True)


def run_session_start(state_path: Path, stdin: TextIO, now: Optional[datetime] = None) -> int:
    """`ccdrift hook session-start`: on a startup or a resume, print the status line's
    verdict as the hook's systemMessage, which the user sees and Claude doesn't, then start
    a check when one is due. Of the payload only its `source` is read. Always 0: a failure
    shows as a line of its own, with its traceback appended to the check's log."""
    now = now or datetime.now().astimezone()
    message = ""
    try:
        if hook_source(read_payload(stdin)) in SHOWN:
            message = short_status(state_path, now)
        if check_due(state_path, now):
            start_check(state_path)
    except Exception as exc:
        message = HOOK_LINES["failed"].format(error=type(exc).__name__)
        try:
            with open(private_log(state_path), "a") as log:
                traceback.print_exc(file=log)
        except OSError:
            pass
    if message:
        print(json.dumps({"systemMessage": message}))
    return 0


def run_hook_check(state_path: Path) -> int:
    """`ccdrift hook check`: the check a session start began, with notifications. It gives
    up while another command holds the state, and skips the run when a check has begun
    since the session start found one due."""
    from ccdrift.check import run_check
    from ccdrift.logs import default_source
    return run_check(default_source(), state_path, notify_user=True,
                     due=lambda: check_due(state_path, datetime.now().astimezone()))
