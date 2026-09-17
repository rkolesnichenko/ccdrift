"""Claude Code's release notes, read from the copy Claude Code keeps on disk
(`<config dir>/cache/changelog.md`), to explain an alert with what changed in the
versions around it. Nothing is fetched."""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence, Union

import pandas as pd

from ccdrift.logs import first_days_by_version
from ccdrift.texts import version_key

# Words per topic and their weight. A line is quoted once its words weigh QUOTE_WEIGHT,
# heaviest first. Checked against Claude Code's own changelog: "model", "tool", "agent"
# and "context" matched about half of each version's notes, and "mcp" mostly sign-in and
# menu fixes, so they aren't used; "effort", "transcript" and "usage" alone matched
# mostly display fixes, so they count only together with another word ("Now defaults
# to high effort", "reporting as 0 in transcript and result usage").
TOPICS: dict[str, dict[str, int]] = {
    "cache": {"cache": 2, "prompt-cache": 1, "prompt cache": 1, "cache miss": 1, "cache reuse": 1},
    "haiku": {"haiku": 2, "small model": 2, "small-model": 2, "fallback model": 2, "default model": 2},
    "effort": {"effort level": 2, "default effort": 2, "default-effort": 2, "reasoning effort": 2,
               "effortlevel": 2, "thinking budget": 2, "effort": 1, "defaults to": 1},
    "context": {"system prompt": 2, "tool definition": 2, "tool list": 2, "deferred": 2},
    "hooks": {"hook": 2, "stop hook": 1, "hook input": 1},
    "subagents": {"subagent model": 2, "subagent_model": 2},
    "fields": {"session transcript": 2, "transcript file": 2, "transcript writes": 2, "saved transcript": 2,
               "session file": 2, "transcript": 1, "usage": 1},
}
QUOTE_WEIGHT = 2
NOTES_PER_VERSION = 2
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
    first_seen = first_days_by_version(turns)
    return sorted((str(v) for v, day in first_seen.items() if first_day <= day <= last_day), key=version_key)


def note_versions(turns: pd.DataFrame, named: Sequence[str], first_day: str, last_day: str) -> list[str]:
    """The versions whose release notes an alert quotes: those its message names
    ("2.1.267 (since 09-10)"), then the others first seen from `first_day` to
    `last_day`, newest first."""
    new = new_versions(turns, first_day, last_day)
    return list(dict.fromkeys([text.split(" ")[0] for text in named] + new[::-1]))


def release_notes(changelog: dict[str, list[str]], versions: Sequence[str],
                  topics: Union[str, Sequence[str]], limit: int = 5) -> list[tuple[str, str]]:
    """Up to `limit` (version, line) pairs from `versions`, in that order, whose text
    mentions words of `topics` weighing QUOTE_WEIGHT or more: at most NOTES_PER_VERSION
    from each version, heaviest first, then in changelog order. Long lines are cut at
    NOTE_CHARS."""
    names = (topics,) if isinstance(topics, str) else tuple(topics)
    weights: dict[str, int] = {}
    for name in names:
        for word, weight in TOPICS[name].items():
            weights[word] = max(weight, weights.get(word, 0))
    found = []
    for version in versions:
        scored = [(sum(weight for word, weight in weights.items() if word in text.lower()), text)
                  for text in changelog.get(version, [])]
        lines = [text for score, text in sorted(scored, key=lambda item: -item[0]) if score >= QUOTE_WEIGHT]
        for text in lines[:NOTES_PER_VERSION]:
            found.append((version, text if len(text) <= NOTE_CHARS else text[:NOTE_CHARS - 1] + "…"))
            if len(found) == limit:
                return found
    return found


def note_lines(notes: Sequence[tuple[str, str]]) -> list[str]:
    """Log lines for an alert."""
    return [f"release notes {version}: {text}" for version, text in notes]
