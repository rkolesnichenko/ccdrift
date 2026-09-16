"""The daily check's state file: the incidents it follows, the setting changes and
blank-cache stretches it reported, and how its last run went."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

STATE_VERSION = 2


def new_state() -> dict[str, Any]:
    return {"version": STATE_VERSION, "incidents": [], "settings": [], "blank_cache": [], "reported": {}}


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
    """Write the state through a temporary file, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1) + "\n")
    os.replace(tmp, path)


def record_run(state: dict[str, Any], started: datetime, error: Optional[str]) -> None:
    """Note when a run started and whether it worked; `error` is None when it did."""
    stamp = started.isoformat(timespec="seconds")
    state["last_run"] = {"started": stamp, "ok": error is None, "error": error}
    if error is None:
        state["last_ok"] = stamp
