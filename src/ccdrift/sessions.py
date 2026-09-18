"""Session starts: how much context a new Claude Code session sends with its first
request — system prompt, tool definitions, CLAUDE.md, skills and MCP servers. Every
new session, and every cache miss, pays for it again."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Sequence

import pandas as pd

from ccdrift.logs import outside_sdk
from ccdrift.texts import approx, project_path

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


def project_of(source_file: str) -> str:
    """The project folder a transcript belongs to: the first segment of its path under
    the transcripts folder. A transcript lying directly in it is its own project."""
    head, sep, _ = str(source_file).partition("/")
    return head if sep else str(source_file)





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
    starts = pd.DataFrame({
        "source_file": first["source_file"].astype(str),
        "project": first["source_file"].astype(str).map(project_of),
        "timestamp": first["timestamp"],
        "day": first["day"].astype(str),
        "version": first["version"] if "version" in first else None,
        "prompt_tokens": (first["input_tokens"] + first["cache_creation"] + first["cache_read"]).astype(float),
    }, columns=START_COLUMNS)
    starts = starts[starts["prompt_tokens"] > 0]
    return starts.sort_values("timestamp", kind="stable").reset_index(drop=True)


def ratio_starts(starts: pd.DataFrame) -> pd.DataFrame:
    """Every session start with the level its own project was starting at — the median of
    that project's previous PROJECT_BASELINE sessions — and its ratio to it. A session
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
    """A step in session-start size; `window` and `baseline` are row positions in the
    starts table."""
    since: str
    until: str
    before: float
    after: float
    window: list[int] = field(default_factory=list)
    baseline: list[int] = field(default_factory=list)
    # Whether a Claude Code version ran in the window that none of the baseline sessions
    # ran: the same test lab/session_start.py's G2 gate asks of a step.
    new_version: bool = False

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


PROJECTS_IN_REPORT = 5


def project_summary(starts: pd.DataFrame, days: Sequence[str]) -> list[dict[str, Any]]:
    """Each project's typical session start over `days`: the median tokens and how many
    sessions it had, largest first. `ccdrift report` prints it, so it names the folders."""
    window = starts[starts["day"].astype(str).isin(list(days))] if not starts.empty else starts
    if window.empty:
        return []
    rows = [{"project": str(project), "path": project_path(str(project)), "sessions": len(group),
             "median_tokens": float(group["prompt_tokens"].median())}
            for project, group in window.groupby(window["project"].astype(str), sort=False)]
    return sorted(rows, key=lambda row: -row["median_tokens"])


def project_lines(summary: list[dict[str, Any]]) -> list[str]:
    """The report's session-starts-by-project line, starting with a blank line; empty
    when no project had a session over the days shown."""
    if not summary:
        return []
    shown = [f"{row['path']} ~{approx(row['median_tokens'])} ({row['sessions']} session"
             f"{'' if row['sessions'] == 1 else 's'})" for row in summary[:PROJECTS_IN_REPORT]]
    rest = len(summary) - PROJECTS_IN_REPORT
    return ["", "Session starts by project over these days: " + ", ".join(shown)
            + (f" and {rest} more" if rest > 0 else "")]


def found_changes(judged: pd.DataFrame) -> list[ContextChange]:
    """Steps in the judged sessions: over all of them together, which is where a change
    that reaches every project shows, and over each project's own sessions, where a change
    in one project of several would otherwise be diluted by the others. Sorted by the day
    they start; `first_of_each` drops the same step found twice."""
    changes = list(context_changes_in(judged))
    if not judged.empty:
        for _, rows in judged.groupby(judged["project"].astype(str), sort=True):
            changes += context_changes_in(rows.reset_index(drop=True))
    return sorted(changes, key=lambda change: (change.since, change.until))


def moved_projects(starts: pd.DataFrame, change: ContextChange) -> dict[str, Any]:
    """Which projects moved with a change: for every project with sessions on both sides
    of its first day, within SIDE_DAYS each way, whether its own median moved by at least
    SIDE in the change's direction. `seen` counts the projects that could be judged at
    all, so an alert can say "1 of 4"."""
    days = starts["day"].astype(str)
    earliest = (date.fromisoformat(change.since) - timedelta(days=SIDE_DAYS)).isoformat()
    latest = (date.fromisoformat(change.until) + timedelta(days=SIDE_DAYS)).isoformat()
    window = starts[(days >= earliest) & (days <= latest)]
    moved, seen = [], 0
    for project, rows in window.groupby(window["project"].astype(str), sort=True):
        on_days = rows["day"].astype(str)
        before = rows.loc[on_days < change.since, "prompt_tokens"].astype(float)
        after = rows.loc[on_days >= change.since, "prompt_tokens"].astype(float)
        if before.empty or after.empty or before.median() <= 0:
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


def _where(change: dict[str, Any], new_version: bool) -> str:
    """Which projects a change reached, and what that says about its cause; "" when no
    project could be judged on both sides of it."""
    moved, seen = len(change.get("projects", [])), change.get("of_projects", 0)
    if not seen or not moved:
        return ""
    if seen == 1:
        # One project is no evidence either way: Claude Code and that project's own files
        # both move it, and there is nothing to compare it with.
        if new_version:
            return (", in the one project ccdrift could compare with itself, and on a Claude Code version none of "
                    "the sessions before it ran: either that version or the project's own files explain it.")
        return (", in the one project ccdrift could compare with itself, so its CLAUDE.md, MCP servers or skills "
                "explain it as readily as Claude Code does.")
    if moved < seen:
        that = "That project's" if moved == 1 else "Those projects'"
        return (f", in {moved} of {seen} projects you used. "
                f"{that} CLAUDE.md, MCP servers or skills explain it, not Claude Code.")
    if new_version:
        return (f", in every project you used ({moved} of {seen}), on a Claude Code version none of the sessions "
                "before it ran — the likeliest cause.")
    return (f", in every project you used ({moved} of {seen}), with no new Claude Code version, so look at your "
            "global configuration in ~/.claude.")


def rejudged(starts: pd.DataFrame, state: dict[str, Any], today: date) -> list[dict[str, Any]]:
    """The recorded changes an older ccdrift found that this version's rule doesn't, among
    those inside the history read — the pooled rule counted a move between projects as a
    change. Records reaching further back than the starts are kept: ccdrift doesn't drop
    what it can't re-check. The records are removed from state["context_changes"]."""
    if starts.empty:
        return []
    complete = starts[starts["day"].astype(str) < today.isoformat()].reset_index(drop=True)
    earliest = str(complete["day"].astype(str).min()) if not complete.empty else today.isoformat()
    found = {c.since for c in context_changes_in(ratio_starts(complete))}
    kept, dropped = [], []
    for record in state["context_changes"]:
        if record["since"] >= earliest and record["since"] not in found:
            dropped.append(record)
        else:
            kept.append(record)
    state["context_changes"] = kept
    return dropped


def context_message(change: dict[str, Any], versions: Sequence[str]) -> str:
    on = f", on Claude Code {', '.join(versions)}" if versions else ""
    direction = "down" if change["to"] < change["from"] else "up"
    where = _where(change, bool(change.get("new_version", False)))
    tail = where or ". Your MCP servers, plugins or CLAUDE.md can change this too."
    return (f"New sessions start with ~{approx(change['to'])} tokens of context from {change['since']}{on}, "
            f"{direction} from ~{approx(change['from'])}{tail}")
