"""Session starts: how much context a new Claude Code session sends with its first
request — system prompt, tool definitions, CLAUDE.md, skills and MCP servers. Every
new session, and every cache miss, pays for it again."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Sequence

import pandas as pd

WINDOW = 3          # the latest sessions judged together
BASELINE = 10       # sessions before them, at most
MIN_BASELINE = 5
CHANGE = 0.25       # the window's median moved at least this far from the baseline's
SIDE = 0.125        # and every window session lies beyond this, on the same side
DEDUPE_DAYS = 14
START_COLUMNS = ["source_file", "timestamp", "day", "version", "prompt_tokens"]


def session_starts(responses: pd.DataFrame) -> pd.DataFrame:
    """One row per main-thread CLI transcript, in time order: its first response's
    time, UTC day, Claude Code version and prompt size (input, cache creation and
    cache read tokens together)."""
    if responses.empty:
        return pd.DataFrame(columns=START_COLUMNS)
    keep = responses["main_thread"].astype(bool)
    if "entrypoint" in responses:
        keep &= ~responses["entrypoint"].fillna("").astype(str).str.startswith("sdk-")
    main = responses[keep]
    if main.empty:
        return pd.DataFrame(columns=START_COLUMNS)
    first = main.sort_values("timestamp", kind="stable").groupby("source_file", sort=False).head(1)
    starts = pd.DataFrame({
        "source_file": first["source_file"].astype(str),
        "timestamp": first["timestamp"],
        "day": first["day"].astype(str),
        "version": first["version"] if "version" in first else None,
        "prompt_tokens": (first["input_tokens"] + first["cache_creation"] + first["cache_read"]).astype(float),
    }, columns=START_COLUMNS)
    return starts.sort_values("timestamp", kind="stable").reset_index(drop=True)


@dataclass
class ContextChange:
    """A step in session-start size; `window` and `baseline` are row positions in the
    starts table."""
    since: str
    until: str
    before: float
    after: float
    window: list[int] = field(default_factory=list)
    baseline: list[int] = field(default_factory=list)

    @property
    def up(self) -> bool:
        return self.after > self.before


def context_changes_in(starts: pd.DataFrame) -> list[ContextChange]:
    """Every window of WINDOW consecutive session starts whose median moved at least
    CHANGE from the median of the up to BASELINE sessions before it, with each window
    session beyond SIDE on the same side. Several windows after one step qualify;
    first_of_each keeps one."""
    tokens = starts["prompt_tokens"].astype(float).tolist()
    days = starts["day"].astype(str).tolist()
    changes = []
    for end in range(MIN_BASELINE + WINDOW - 1, len(tokens)):
        window = list(range(end - WINDOW + 1, end + 1))
        baseline = list(range(max(0, window[0] - BASELINE), window[0]))
        before = statistics.median(tokens[i] for i in baseline)
        if before <= 0:
            continue
        after = statistics.median(tokens[i] for i in window)
        moves = [tokens[i] / before - 1 for i in window]
        same_side = all(m > SIDE for m in moves) or all(m < -SIDE for m in moves)
        if abs(after / before - 1) >= CHANGE and same_side:
            changes.append(ContextChange(days[window[0]], days[window[-1]], before, after, window, baseline))
    return changes


def first_of_each(changes: Sequence[ContextChange], recorded: Sequence[dict] = ()) -> list[ContextChange]:
    """One change per step: a change is dropped when a kept or recorded one in the same
    direction started no more than DEDUPE_DAYS days before it. `recorded` holds dicts
    with `since`, `from` and `to`."""
    kept = [(r["since"], r["to"] > r["from"]) for r in recorded]
    found = []
    for change in changes:
        earliest = (date.fromisoformat(change.since) - timedelta(days=DEDUPE_DAYS)).isoformat()
        if any(up == change.up and since >= earliest for since, up in kept):
            continue
        kept.append((change.since, change.up))
        found.append(change)
    return found
