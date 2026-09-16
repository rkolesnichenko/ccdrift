"""Claude Code's release notes, read from the copy Claude Code keeps on disk
(`<config dir>/cache/changelog.md`), to explain an alert with what changed in the
versions around it. Nothing is fetched."""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence, Union

import pandas as pd

from ccdrift.texts import version_key

TOPICS: dict[str, tuple[str, ...]] = {
    "cache": ("cache",),
    "haiku": ("haiku", "model"),
    "effort": ("effort", "thinking"),
    "context": ("system prompt", "tool", "context", "mcp", "deferred"),
    "hooks": ("hook",),
    "subagents": ("subagent", "agent", "model"),
    "fields": ("transcript", "jsonl", "session file"),
}
NOTE_CHARS = 160


def changelog_path(source: Path) -> Path:
    """The changelog in the Claude Code config folder that holds `source`."""
    return source.expanduser().parent / "cache" / "changelog.md"


def load_changelog(path: Path) -> dict[str, list[str]]:
    """Bullet lines per version from a changelog with `## <version>` headers; empty
    when the file is missing or unreadable."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    notes: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        header = re.match(r"^##\s+v?(\d[\w.+-]*)", line)
        if header:
            current = header.group(1)
            notes.setdefault(current, [])
        elif current is not None and line.startswith("- "):
            notes[current].append(line[2:].strip())
    return notes


def days_before(day: str, days: int) -> str:
    return (date.fromisoformat(day) - timedelta(days=days)).isoformat()


def new_versions(turns: pd.DataFrame, first_day: str, last_day: str) -> list[str]:
    """Versions whose first day among `turns` lies in [first_day, last_day], oldest first."""
    if turns.empty or "version" not in turns:
        return []
    known = turns.dropna(subset=["version"])
    first_seen = known.groupby("version")["day"].min().astype(str)
    return sorted((str(v) for v, day in first_seen.items() if first_day <= day <= last_day), key=version_key)


def release_notes(changelog: dict[str, list[str]], versions: Sequence[str],
                  topics: Union[str, Sequence[str]], limit: int = 5) -> list[tuple[str, str]]:
    """Up to `limit` (version, line) pairs from `versions`, in that order, whose text
    mentions any keyword of `topics`; long lines are cut at NOTE_CHARS."""
    names = (topics,) if isinstance(topics, str) else tuple(topics)
    words = tuple(word for name in names for word in TOPICS[name])
    found = []
    for version in versions:
        for text in changelog.get(version, []):
            if any(word in text.lower() for word in words):
                found.append((version, text if len(text) <= NOTE_CHARS else text[:NOTE_CHARS - 1] + "…"))
                if len(found) == limit:
                    return found
    return found


def note_lines(notes: Sequence[tuple[str, str]]) -> list[str]:
    """Log lines for an alert."""
    return [f"release notes {version}: {text}" for version, text in notes]
