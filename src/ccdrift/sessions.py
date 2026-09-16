"""Session starts: how much context a new Claude Code session sends with its first
request — system prompt, tool definitions, CLAUDE.md, skills and MCP servers. Every
new session, and every cache miss, pays for it again."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Sequence

import pandas as pd

from ccdrift.texts import approx

WINDOW = 3          # the latest sessions judged together
BASELINE = 10       # sessions before them, at most
MIN_BASELINE = 5
CHANGE = 0.25       # the window's median moved at least this far from the baseline's
SIDE = 0.125        # and every window session lies beyond this, on the same side
DEDUPE_DAYS = 14
MIN_SESSIONS = 3    # a version's typical session start needs at least this many sessions
START_COLUMNS = ["source_file", "timestamp", "day", "version", "prompt_tokens"]


def session_starts(responses: pd.DataFrame) -> pd.DataFrame:
    """One row per main-thread CLI transcript, in time order: its first response's
    time, UTC day, Claude Code version and prompt size (input, cache creation and
    cache read tokens together). A first response with no tokens logged isn't a start:
    every request sends context, so Claude Code has stopped logging usage."""
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
    starts = starts[starts["prompt_tokens"] > 0]
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
    direction either started no more than DEDUPE_DAYS days before it, or has `from` and
    `to` each within SIDE (relative) of the change's own `before` and `after` -- the
    same step re-detected, which sparse sessions can keep doing for weeks as the
    baseline slowly refills with post-step sessions. `recorded` holds dicts with
    `since`, `from` and `to`; kept changes are compared by their own `before`/`after`.
    One from or to 0 tokens, which a state file may hold, is compared by date only."""
    kept = [(r["since"], r["to"] > r["from"], r["from"], r["to"]) for r in recorded]
    found = []
    for change in changes:
        earliest = (date.fromisoformat(change.since) - timedelta(days=DEDUPE_DAYS)).isoformat()
        if any(up == change.up and (since >= earliest or (
                r_from > 0 and r_to > 0
                and abs(change.before / r_from - 1) <= SIDE and abs(change.after / r_to - 1) <= SIDE))
               for since, up, r_from, r_to in kept):
            continue
        kept.append((change.since, change.up, change.before, change.after))
        found.append(change)
    return found


RECENT_DAYS = 14


def context_alerts(starts: pd.DataFrame, state: dict[str, Any], today: date) -> list[dict[str, Any]]:
    """Steps in session-start size among the sessions of complete UTC days whose window
    ends within the last RECENT_DAYS days and that aren't recorded yet; each is
    recorded in state["context_changes"]."""
    complete = starts[starts["day"].astype(str) < today.isoformat()].reset_index(drop=True)
    since = (today - timedelta(days=RECENT_DAYS)).isoformat()
    changes = [c for c in context_changes_in(complete) if c.until >= since]
    new = []
    for change in first_of_each(changes, state["context_changes"]):
        record = {"since": change.since, "from": change.before, "to": change.after,
                  "days": [change.since, change.until], "reported_on": today.isoformat()}
        state["context_changes"].append(record)
        new.append(record)
    return new


def context_message(change: dict[str, Any], versions: Sequence[str]) -> str:
    on = f", on Claude Code {', '.join(versions)}" if versions else ""
    direction = "down" if change["to"] < change["from"] else "up"
    return (f"New sessions start with ~{approx(change['to'])} tokens of context from {change['since']}{on}, "
            f"{direction} from ~{approx(change['from'])}. Your MCP servers, plugins or CLAUDE.md can change this too.")
