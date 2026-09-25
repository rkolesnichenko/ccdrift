"""The daily check's state file: the incidents it follows, the setting changes, failure
days and blank-cache stretches it reported, and how its last run went."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, TextIO

try:
    import fcntl
except ImportError:  # Windows, where ccdrift sets up no schedule
    fcntl = None  # type: ignore[assignment]

from ccdrift.texts import STATE_LINES

STATE_VERSION = 2
# The session-start rule a state was written by: 2 judges each project against itself.
CONTEXT_RULE = 2


LOG_FILE = "check.log"  # the scheduled check's log, beside its state file


def ccdrift_home(environ: Mapping[str, str] = os.environ) -> Path:
    """Where the daily check keeps its state file, history and log: $CCDRIFT_HOME
    when set, otherwise ~/.ccdrift."""
    home = environ.get("CCDRIFT_HOME")
    return Path(home).expanduser() if home else Path.home() / ".ccdrift"


def make_private(path: Path) -> None:
    """Create `path` when it is missing, and take away everyone's access to it but its
    owner's, as Claude Code does for its transcripts. The owner's access stays as it
    is, so a file made read-only stays read-only."""
    path.touch(mode=0o600)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        path.chmod(mode & 0o700)


def make_stream_private(stream: TextIO, only: Optional[Path] = None) -> None:
    """Take everyone's access but its owner's off the file `stream` writes to, when it
    is a regular file and, given `only`, that file: the check's log, which launchd,
    systemd and cron recreate with the umask's permissions once it has been deleted,
    and not a file of the user's own that they sent the output to. A terminal, a pipe or
    a stream with no file behind it is left alone, and so is a file this can't change:
    the check runs on either way."""
    try:
        fd = stream.fileno()
        info = os.fstat(fd)
        mode = info.st_mode
        if only is not None and not os.path.samestat(info, os.stat(only)):
            return
        if stat.S_ISREG(mode) and stat.S_IMODE(mode) & 0o077:
            os.fchmod(fd, stat.S_IMODE(mode) & 0o700)
    except (OSError, ValueError, AttributeError):  # AttributeError: no os.fchmod on Windows before 3.13
        pass


def new_state() -> dict[str, Any]:
    return {"version": STATE_VERSION, "incidents": [], "settings": [], "blank_cache": [], "reported": {},
            "field_gaps": [], "new_fields": [], "hook_failures": [], "context_changes": [], "early_warnings": [],
            "loop_warnings": [], "failed_requests": [], "cut_short": [], "hook_changes": [], "unreadable": [],
            "no_responses": [], "context_rule": CONTEXT_RULE, "runs": []}


def load_state(path: Path) -> dict[str, Any]:
    """The state in `path` as version 2; a fresh state when the file doesn't exist.
    A version 1 file keeps its `reported` flags, so they aren't reported again.
    Raises OSError or ValueError when the file can't be read, including one written
    by a newer ccdrift, which this version must not overwrite."""
    if not path.exists():
        return new_state()
    state = json.loads(path.read_text())
    if not isinstance(state, dict):
        raise ValueError(STATE_LINES["not_object"].format(path=path))
    # Written before 0.8.0, so the check re-judges its session-start changes once. A
    # fresh state gets the current rule from new_state() below.
    state.setdefault("context_rule", 1)
    version = state.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(STATE_LINES["unknown_version"].format(path=path, version=version))
    if version > STATE_VERSION:
        raise ValueError(STATE_LINES["newer"].format(path=path, version=version))
    for key, empty in new_state().items():
        state.setdefault(key, empty)
    state["version"] = STATE_VERSION
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Write the state through a temporary file of its own, so a reader never sees
    half of it and two writers never write to the same one. The file is synced before
    it replaces the last one: a rename can land before the data it points at, and after
    a power cut the state would read back empty. (On macOS fsync leaves the drive's own
    cache unflushed, so this narrows that window rather than closing it.)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(state, indent=1) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@contextmanager
def state_lock(path: Path) -> Iterator[None]:
    """Hold the lock on the state file `path` from reading it to saving it, so the
    check and `ccdrift incident` never save over each other's changes, and two checks
    don't send the same alerts. Waits, saying so on stderr, while another command
    holds it. Raises OSError when the lock file next to the state file can't be opened."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as handle:
        if fcntl is not None:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(STATE_LINES["waiting"].format(path=path), file=sys.stderr, flush=True)
                fcntl.flock(handle, fcntl.LOCK_EX)
        yield


RUN_DAYS = 14


def record_run(state: dict[str, Any], started: datetime, error: Optional[str]) -> None:
    """Note when a run started and whether it worked; `error` is None when it did.
    Successful runs also keep their local date in state["runs"], the newest RUN_DAYS."""
    stamp = started.isoformat(timespec="seconds")
    state["last_run"] = {"started": stamp, "ok": error is None, "error": error}
    if error is None:
        state["last_ok"] = stamp
        runs = state.setdefault("runs", [])
        day = started.date().isoformat()
        if day not in runs:
            runs.append(day)
        del runs[:-RUN_DAYS]
