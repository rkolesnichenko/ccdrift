"""Session starts: how much context a new Claude Code session sends with its first
request, meaning system prompt, tool definitions, CLAUDE.md, skills and MCP servers.
Every new session, and every cache miss, pays for it again."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Optional, Sequence

import pandas as pd

from ccdrift.logs import outside_sdk
from ccdrift.texts import project_path

WINDOW = 3          # the latest sessions judged together
BASELINE = 10       # sessions before them, at most
MIN_BASELINE = 5
CHANGE = 0.25       # the window's median moved at least this far from the baseline's
SIDE = 0.125        # and every window session lies beyond this, on the same side
DEDUPE_DAYS = 14
MIN_SESSIONS = 3    # a version's typical session start needs at least this many sessions
PROJECT_BASELINE = 10      # a project's own earlier sessions that set its level, at most
MIN_PROJECT_SESSIONS = 3   # earlier sessions a project needs before its own are judged
SIDE_DAYS = 14             # each way of a change, when working out which projects moved
START_COLUMNS = ["source_file", "project", "timestamp", "day", "version", "prompt_tokens"]
RATIO_COLUMNS = [*START_COLUMNS, "level", "ratio"]


SOURCE_PROJECT = ""   # the project of a transcript in no project folder: the source folder itself


def project_of(source_file: str) -> str:
    """The project folder a transcript belongs to: the first segment of its path under
    the transcripts folder, `<project>/<session>.jsonl` or
    `<project>/<session>/subagents/<agent>.jsonl`. With --source pointed at one project's
    own folder, a transcript lies directly in the source or in a session's subagents
    folder, and its first segment would be a session id, which ccdrift never prints; a
    project of each session would also leave none with the sessions to be judged. It
    belongs to the source folder instead, which is SOURCE_PROJECT."""
    parts = str(source_file).split("/")
    return SOURCE_PROJECT if len(parts) == 1 or parts[1] == "subagents" else parts[0]


def session_starts(responses: pd.DataFrame) -> pd.DataFrame:
    """One row per main-thread CLI transcript, in time order: its first response's
    time, UTC day, Claude Code version and prompt size (input, cache creation and
    cache read tokens together). A first response with no tokens logged isn't a start:
    every request sends context, so Claude Code has stopped logging usage. Neither is
    a resumed session's: its transcript opens with copies of the responses before it,
    and the first one it owns carries the whole resumed context."""
    if responses.empty:
        return pd.DataFrame(columns=START_COLUMNS)
    main = responses[responses["main_thread"].astype(bool) & responses["opens_transcript"].astype(bool)
                     & outside_sdk(responses)]
    if main.empty:
        return pd.DataFrame(columns=START_COLUMNS)
    first = main.sort_values("timestamp", kind="stable").groupby("source_file", sort=False).head(1)
    files = first["source_file"].astype(str)
    starts = pd.DataFrame({
        "source_file": files,
        "project": files.map(project_of),
        "timestamp": first["timestamp"],
        "day": first["day"].astype(str),
        "version": first["version"] if "version" in first else None,
        "prompt_tokens": (first["input_tokens"] + first["cache_creation"] + first["cache_read"]).astype(float),
    }, columns=START_COLUMNS)
    starts = starts[starts["prompt_tokens"] > 0]
    return starts.sort_values("timestamp", kind="stable").reset_index(drop=True)


def ratio_starts(starts: pd.DataFrame) -> pd.DataFrame:
    """Every session start with the level its own project was starting at (the median of
    that project's previous PROJECT_BASELINE sessions) and its ratio to it. A session
    whose project has fewer than MIN_PROJECT_SESSIONS earlier sessions is left out, but
    still counts towards the level of the sessions after it. Judging ratios rather than
    token counts is what keeps moving between projects from reading as a change: a project
    is only ever compared with itself."""
    if starts.empty:
        return pd.DataFrame(columns=RATIO_COLUMNS)
    seen: dict[str, list[float]] = {}
    rows = []
    for row in starts.sort_values("timestamp", kind="stable").to_dict("records"):
        earlier = seen.setdefault(str(row["project"]), [])
        if len(earlier) >= MIN_PROJECT_SESSIONS:
            level = statistics.median(earlier[-PROJECT_BASELINE:])
            if level > 0:
                rows.append({**row, "level": level, "ratio": float(row["prompt_tokens"]) / level})
        earlier.append(float(row["prompt_tokens"]))
    return pd.DataFrame(rows, columns=RATIO_COLUMNS)


@dataclass
class ContextChange:
    """A step in session-start size. `window` and `baseline` are row positions in the
    frame the change was found in, which is not always the judged table: `found_changes`
    runs the detector over all the judged sessions and over each project's own rows, and
    returns changes from several frames together, so a change's positions mean nothing
    outside the frame it came from. `project` names that frame: the project whose own
    sessions the step was found in, or None when it came from the pass over all of them,
    so a reader can find the step's own sessions without its positions."""
    since: str
    until: str
    before: float
    after: float
    window: list[int] = field(default_factory=list)
    baseline: list[int] = field(default_factory=list)
    # Whether a Claude Code version ran in the window that none of the baseline sessions
    # ran: the same test lab/session_start.py's G2 gate asks of a step.
    new_version: bool = False
    project: Optional[str] = None

    @property
    def up(self) -> bool:
        return self.after > self.before


def context_changes_in(starts: pd.DataFrame) -> list[ContextChange]:
    """Every window of WINDOW consecutive session starts whose median moved at least
    CHANGE from the median of the up to BASELINE sessions before it, with each window
    session beyond SIDE on the same side. Several windows after one step qualify;
    first_of_each keeps one."""
    judged = starts["ratio"].astype(float).tolist()
    tokens = starts["prompt_tokens"].astype(float).tolist()
    days = starts["day"].astype(str).tolist()
    changes = []
    for end in range(MIN_BASELINE + WINDOW - 1, len(judged)):
        window = list(range(end - WINDOW + 1, end + 1))
        baseline = list(range(max(0, window[0] - BASELINE), window[0]))
        level = statistics.median(judged[i] for i in baseline)
        if level <= 0:
            continue
        after = statistics.median(judged[i] for i in window)
        moves = [judged[i] / level - 1 for i in window]
        same_side = all(m > SIDE for m in moves) or all(m < -SIDE for m in moves)
        if abs(after / level - 1) >= CHANGE and same_side:
            # The ratios decide; the token medians describe, so the message can speak in
            # the numbers the owner sees.
            versions = starts["version"].fillna("unknown").astype(str).tolist() if "version" in starts else []
            fresh = bool({versions[i] for i in window} - {versions[i] for i in baseline}) if versions else False
            changes.append(ContextChange(days[window[0]], days[window[-1]],
                                         statistics.median(tokens[i] for i in baseline),
                                         statistics.median(tokens[i] for i in window), window, baseline, fresh))
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


def project_summary(starts: pd.DataFrame, days: Sequence[str]) -> list[dict[str, Any]]:
    """Each project's typical session start over `days`: the median tokens and how many
    sessions it had, largest first. `ccdrift report` prints it, so it names the folders."""
    window = starts[starts["day"].astype(str).isin(list(days))] if not starts.empty else starts
    if window.empty:
        return []
    rows = [{"path": project_path(str(project)), "sessions": len(group),
             "median_tokens": float(group["prompt_tokens"].median())}
            for project, group in window.groupby(window["project"].astype(str), sort=False)]
    return sorted(rows, key=lambda row: -row["median_tokens"])


def found_changes(judged: pd.DataFrame) -> list[ContextChange]:
    """Steps in the judged sessions: over all of them together, which is where a change
    that reaches every project shows, and over each project's own sessions, where a change
    in one project of several would otherwise be diluted by the others. A change from the
    second pass carries the project it was found in. Sorted by the day they start;
    `first_of_each` drops the same step found twice."""
    changes = list(context_changes_in(judged))
    if not judged.empty:
        for project, rows in judged.groupby(judged["project"].astype(str), sort=True):
            for change in context_changes_in(rows.reset_index(drop=True)):
                change.project = str(project)
                changes.append(change)
    return sorted(changes, key=lambda change: (change.since, change.until))


def moved_projects(starts: pd.DataFrame, change: ContextChange) -> dict[str, Any]:
    """Which projects moved with a change: for every project with at least
    MIN_PROJECT_SESSIONS sessions on each side of its first day, within SIDE_DAYS each
    way, whether its own median moved by at least SIDE in the change's direction. `seen`
    counts the projects that cleared that bar, so an alert can say "1 of 4".

    The bar is the same one a project must clear before any of its sessions are judged at
    all, and for the same reason: a median of one or two sessions is noise, and ordinary
    sessions of a settled project range over 0.92x-1.30x of its level, so a pair of them
    can differ by more than SIDE with nothing behind it. This count is what the alert's
    cause rests on -- "in every project" reads as Claude Code or the global config -- so a
    project with nothing to say does not vote."""
    days = starts["day"].astype(str)
    earliest = (date.fromisoformat(change.since) - timedelta(days=SIDE_DAYS)).isoformat()
    latest = (date.fromisoformat(change.until) + timedelta(days=SIDE_DAYS)).isoformat()
    window = starts[(days >= earliest) & (days <= latest)]
    moved, seen = [], 0
    for project, rows in window.groupby(window["project"].astype(str), sort=True):
        on_days = rows["day"].astype(str)
        before = rows.loc[on_days < change.since, "prompt_tokens"].astype(float)
        after = rows.loc[on_days >= change.since, "prompt_tokens"].astype(float)
        if len(before) < MIN_PROJECT_SESSIONS or len(after) < MIN_PROJECT_SESSIONS or before.median() <= 0:
            continue
        seen += 1
        move = after.median() / before.median() - 1
        if (move > SIDE) if change.up else (move < -SIDE):
            moved.append(str(project))
    return {"moved": moved, "seen": seen}


def context_alerts(starts: pd.DataFrame, state: dict[str, Any], today: date) -> list[dict[str, Any]]:
    """Steps in session-start size among the sessions of complete UTC days whose window
    ends within the last RECENT_DAYS days and that aren't recorded yet; each is recorded
    in state["context_changes"] with the projects that moved with it. `starts` is the
    session starts table; each session is judged against its own project's level."""
    complete = starts[starts["day"].astype(str) < today.isoformat()].reset_index(drop=True)
    judged = ratio_starts(complete)
    since = (today - timedelta(days=RECENT_DAYS)).isoformat()
    changes = [c for c in found_changes(judged) if c.until >= since]
    new = []
    for change in first_of_each(changes, state["context_changes"]):
        projects = moved_projects(complete, change)
        record = {"since": change.since, "from": change.before, "to": change.after,
                  "days": [change.since, change.until], "projects": projects["moved"],
                  "of_projects": projects["seen"], "new_version": change.new_version,
                  "reported_on": today.isoformat()}
        state["context_changes"].append(record)
        new.append(record)
    return new


def rejudged(starts: pd.DataFrame, state: dict[str, Any], today: date) -> list[dict[str, Any]]:
    """The recorded changes an older ccdrift found that this version's rule doesn't: the
    pooled rule counted a move between projects as a change. Two rules keep a record that
    this one can't reproduce exactly, because deleting it would only alert the owner about
    the same step again next run:

    - a record is the same step when the new rule finds a change in the same direction
      within DEDUPE_DAYS of it, not only on its own day: the two rules judge different
      sessions, so they date one step differently, and `first_of_each` already reads that
      as the same step;
    - a record before the earliest day the new rule could report on (the day of the
      judged session at MIN_BASELINE + WINDOW - 1, the first a window can end on) is kept
      unjudged, as are all of them when there are too few judged sessions to report
      anything. ccdrift doesn't drop what it can't re-check.

    The dropped records are removed from state["context_changes"]."""
    if starts.empty:
        return []
    complete = starts[starts["day"].astype(str) < today.isoformat()].reset_index(drop=True)
    judged = ratio_starts(complete)
    if len(judged) <= MIN_BASELINE + WINDOW - 1:
        return []
    earliest = str(judged["day"].astype(str).iloc[MIN_BASELINE + WINDOW - 1])
    found = [(date.fromisoformat(c.since), c.up) for c in found_changes(judged)]
    kept, dropped = [], []
    for record in state["context_changes"]:
        since, up = date.fromisoformat(record["since"]), record["to"] > record["from"]
        same_step = any(rose == up and abs((day - since).days) <= DEDUPE_DAYS for day, rose in found)
        if record["since"] >= earliest and not same_step:
            dropped.append(record)
        else:
            kept.append(record)
    state["context_changes"] = kept
    return dropped
