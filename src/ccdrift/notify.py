"""Desktop notifications for the daily check's alerts."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from typing import Callable, Optional


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
