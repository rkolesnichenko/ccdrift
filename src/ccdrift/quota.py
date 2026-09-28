"""Quota samples from Claude Code's status line. Since 2.1.251 the JSON Claude Code
hands a status line command carries how much of the plan's 5-hour and 7-day limits
is used (Pro and Max only), which its transcripts never record. `ccdrift status
--short --stdin` keeps a sample of it beside the state file each time it changes, at
most once a minute, and nothing else from that JSON: no session id, path, project,
repository or branch. Nothing reads the samples back yet beyond a count; a rule on
quota used per token needs weeks of them first.

It runs on every status line refresh, so it imports only the standard library and
never raises."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TextIO

QUOTA_FILE = "quota.jsonl"
WINDOWS = ("five_hour", "seven_day", "spend_limit")
# Claude Code's documented example of the status line JSON is 1,322 bytes; 64 KiB leaves room
# for long paths and names while a runaway input still can't hold up a status line.
MAX_PAYLOAD = 64 * 1024
# A sample line with two windows is 216 bytes, so the last 4 KiB always hold the last whole one.
TAIL_BYTES = 4096
# At most one sample a minute: a rule compares 5-hour and 7-day windows, so finer adds
# nothing, and it caps the file at 1,440 lines a day, about 310 KB.
MIN_INTERVAL = 60


def quota_path(state_path: Path) -> Path:
    """The samples file, beside the state file as the history store is."""
    return state_path.parent / QUOTA_FILE


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sample_from(payload: Any, now: datetime) -> Optional[dict[str, Any]]:
    """The sample a parsed status line payload gives at `now`: the time, Claude Code's
    version, the model id and each window whose used share and reset time are numbers.
    None when no window is. Built from these keys alone, never by copying the payload."""
    if not isinstance(payload, dict) or not isinstance(payload.get("rate_limits"), dict):
        return None
    windows = {}
    for name in WINDOWS:
        window = payload["rate_limits"].get(name)
        if (isinstance(window, dict) and _number(window.get("used_percentage"))
                and _number(window.get("resets_at"))):
            windows[name] = {"used_percentage": window["used_percentage"], "resets_at": window["resets_at"]}
    if not windows:
        return None
    model = payload.get("model")
    version = payload.get("version")
    return {"at": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
            "version": version if isinstance(version, str) else None,
            "model": model.get("id") if isinstance(model, dict) and isinstance(model.get("id"), str) else None,
            **windows}


def last_sample(path: Path) -> Optional[dict[str, Any]]:
    """The file's last sample, from its last TAIL_BYTES; None when the file is missing,
    empty, or its last line doesn't parse as one, a time with its zone included: a time
    record_sample can't compare with now would otherwise stop every sample after it."""
    try:
        with path.open("rb") as file:
            file.seek(max(0, file.seek(0, os.SEEK_END) - TAIL_BYTES))
            tail = file.read().decode("utf-8").splitlines()
        sample = json.loads(tail[-1]) if tail else None
        if not (isinstance(sample, dict) and isinstance(sample.get("at"), str)):
            return None
        at = datetime.fromisoformat(sample["at"])
    except (OSError, ValueError):
        return None
    return sample if at.utcoffset() is not None else None


def _windows(sample: dict[str, Any]) -> dict[str, Any]:
    return {name: sample[name] for name in WINDOWS if name in sample}


def record_sample(text: str, path: Path, now: datetime) -> bool:
    """Append the sample `text` (a status line payload) gives to `path` when its windows
    differ from the last sample's and MIN_INTERVAL has passed since it; True when a line
    was written. The file is created owner-only, its folder too when missing, and one
    write of one line to an appending descriptor keeps refreshes running at once from
    interleaving. Raises nothing: a status line must never fail over a sample."""
    try:
        if len(text) > MAX_PAYLOAD:
            return False
        sample = sample_from(json.loads(text), now)
        if sample is None:
            return False
        last = last_sample(path)
        if last is not None:
            if _windows(last) == _windows(sample):
                return False
            if (now - datetime.fromisoformat(last["at"])).total_seconds() < MIN_INTERVAL:
                return False
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, (json.dumps(sample) + "\n").encode("utf-8"))
        finally:
            os.close(fd)
        return True
    except Exception:  # a status line must never show a traceback, whatever arrives on stdin
        return False


def read_payload(stream: TextIO) -> Optional[str]:
    """What arrived on `stream`, up to MAX_PAYLOAD bytes; None from a terminal, which a
    hand-run command would otherwise wait on, and for a payload over the cap or not UTF-8."""
    try:
        if stream.isatty():
            return None
        data = stream.buffer.read(MAX_PAYLOAD + 1)
        return None if len(data) > MAX_PAYLOAD else data.decode("utf-8")
    except (OSError, ValueError, AttributeError):
        return None


def samples_summary(path: Path) -> Optional[dict[str, Any]]:
    """How many samples `path` holds, with the first and last one's time; None when the
    file is missing or holds none. Lines that don't parse are skipped."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    times = []
    for line in text.splitlines():
        try:
            sample = json.loads(line)
        except ValueError:
            continue
        if isinstance(sample, dict) and isinstance(sample.get("at"), str):
            times.append(sample["at"])
    return {"count": len(times), "first": times[0], "last": times[-1]} if times else None
