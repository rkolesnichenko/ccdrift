"""The daily check's state file: the incidents it follows, the setting changes and
blank-cache stretches it reported, and how its last run went."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

try:
    import fcntl
except ImportError:  # Windows, where ccdrift sets up no schedule
    fcntl = None  # type: ignore[assignment]

STATE_VERSION = 2


def ccdrift_home(environ: Mapping[str, str] = os.environ) -> Path:
    """Where the daily check keeps its state file, history and log: $CCDRIFT_HOME
    when set, otherwise ~/.ccdrift."""
    home = environ.get("CCDRIFT_HOME")
    return Path(home).expanduser() if home else Path.home() / ".ccdrift"


def new_state() -> dict[str, Any]:
    return {"version": STATE_VERSION, "incidents": [], "settings": [], "blank_cache": [], "reported": {},
            "field_gaps": [], "hook_failures": [], "context_changes": [], "early_warnings": [], "runs": []}


def load_state(path: Path) -> dict[str, Any]:
    """The state in `path` as version 2; a fresh state when the file doesn't exist.
    A version 1 file keeps its `reported` flags, so they aren't reported again.
    Raises OSError or ValueError when the file can't be read, including one written
    by a newer ccdrift, which this version must not overwrite."""
    if not path.exists():
        return new_state()
    state = json.loads(path.read_text())
    if not isinstance(state, dict):
        raise ValueError(f"{path} doesn't hold a JSON object")
    version = state.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError(f"{path} has an unknown state version: {version!r}")
    if version > STATE_VERSION:
        raise ValueError(f"{path} was written by a newer ccdrift (state version {version}); upgrade ccdrift")
    for key, empty in new_state().items():
        state.setdefault(key, empty)
    state["version"] = STATE_VERSION
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Write the state through a temporary file of its own, so a reader never sees
    half of it and two writers never write to the same one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(json.dumps(state, indent=1) + "\n")
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
                print(f"Waiting for another ccdrift command to finish with {path}...", file=sys.stderr, flush=True)
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
