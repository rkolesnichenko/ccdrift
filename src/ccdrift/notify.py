"""Desktop notifications for the daily check's alerts."""

from __future__ import annotations

import json
import shutil
import subprocess


def notify(title: str, message: str) -> None:
    """Show a macOS notification; does nothing where osascript is missing."""
    if shutil.which("osascript"):
        script = (f"display notification {json.dumps(message, ensure_ascii=False)} "
                  f"with title {json.dumps(title, ensure_ascii=False)}")
        subprocess.run(["osascript", "-e", script], check=False)
