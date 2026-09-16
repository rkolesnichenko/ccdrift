"""Desktop notifications and the user's own command for the daily check's alerts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Callable, Mapping, Optional


def notify(title: str, message: str, platform: str = sys.platform,
           run: Callable[..., object] = subprocess.run,
           which: Callable[[str], Optional[str]] = shutil.which) -> None:
    """Show a desktop notification: osascript on macOS, notify-send on Linux when it
    is installed, otherwise nothing. Failures are ignored, since every alert also
    goes to the log."""
    if platform == "darwin" and which("osascript"):
        script = (f"display notification {json.dumps(message, ensure_ascii=False)} "
                  f"with title {json.dumps(title, ensure_ascii=False)}")
        argv = ["osascript", "-e", script]
    elif platform.startswith("linux") and which("notify-send"):
        argv = ["notify-send", title, message]
    else:
        return
    try:
        run(argv, check=False, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


EXEC_TIMEOUT_SECONDS = 30


def run_exec(command: str, kind: str, title: str, message: str,
             run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
             environ: Mapping[str, str] = os.environ) -> Optional[str]:
    """Run the user's alert command through the shell, with the alert in
    CCDRIFT_ALERT (its kind), CCDRIFT_TITLE and CCDRIFT_MESSAGE. Returns None when it
    worked, otherwise why it didn't; a failing command never stops the check."""
    env = {**environ, "CCDRIFT_ALERT": kind, "CCDRIFT_TITLE": title, "CCDRIFT_MESSAGE": message}
    try:
        result = run(command, shell=True, env=env, capture_output=True, text=True, timeout=EXEC_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return f"timed out after {EXEC_TIMEOUT_SECONDS} s"
    except OSError as exc:
        return str(exc)
    if result.returncode != 0:
        first = (result.stderr or "").strip().splitlines()
        return f"exit {result.returncode}" + (f": {first[0]}" if first else "")
    return None
