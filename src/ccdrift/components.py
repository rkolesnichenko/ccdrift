"""What changed in how sessions start: the skills, deferred tools, agent types, MCP
instructions, CLAUDE.md files, system prompt and tool definitions two sets of sessions
started with, from the component rows logs.parse_file keeps. Names and sizes only; the
sizes are characters, which G14 found no way to put in tokens (docs/findings.md)."""

from __future__ import annotations

import statistics
from typing import Any, Optional, Sequence

import pandas as pd

from ccdrift.sessions import project_of

# The parts compared, by the size column that says whether a session logged it. The
# first five have name sets of their own, under the part's name; CLAUDE.md files and the
# system prompt are sizes only.
PART_SIZES = {"skills": "skills_chars", "deferred": "deferred_chars", "agents": "agents_chars", "mcp": "mcp_chars",
              "claude_md": "claude_md_chars", "system": "system_chars", "tools": "tools_chars"}
NAMED_PARTS = ("skills", "deferred", "agents", "mcp", "tools")
MCP_TOOL = "mcp__"


def _logged(value: Any) -> bool:
    return value is not None and not (isinstance(value, float) and value != value)


def _part(row: dict[str, Any], part: str) -> Optional[tuple[list[str], float]]:
    """A session's names and size for one part; None when it didn't log the part. MCP
    instructions count as none wherever the deferred tools are logged: measured on
    2026-09-24, Claude Code wrote them in exactly the 49 transcripts of 1,059 whose
    deferred tools included an MCP tool."""
    size = row.get(PART_SIZES[part])
    if part == "mcp" and not _logged(size) and _logged(row.get("deferred_chars")):
        return [], 0.0
    if not _logged(size):
        return None
    names = row.get(part) if part in NAMED_PARTS else None
    return list(names or []), float(size)


def _by_server(names: Sequence[str]) -> dict[str, list[str]]:
    """MCP tools, `mcp__<server>__<tool>`, by server."""
    servers: dict[str, list[str]] = {}
    for name in sorted(names):
        servers.setdefault(name[len(MCP_TOOL):].split("__", 1)[0], []).append(name)
    return servers


def _split(changes: dict[str, list[str]]) -> dict[str, Any]:
    """Added or removed names by part, with the deferred tools split into MCP tools by
    server and the built-in ones, and parts with nothing left out."""
    out: dict[str, Any] = {}
    for part, names in changes.items():
        if part == "deferred":
            mcp = [name for name in names if name.startswith(MCP_TOOL)]
            if mcp:
                out["mcp_tools"] = _by_server(mcp)
            names = [name for name in names if not name.startswith(MCP_TOOL)]
        if names:
            out[part] = names
    return out


def _own_project_baseline(window: Sequence[str], baseline: Sequence[str]) -> Optional[list[str]]:
    """`baseline`, narrowed to the one project `window` belongs to; None when `window`
    itself spans more than one, so nothing is compared. `sessions.found_changes` runs its
    pooled pass over every project's sessions together, and hands compare_components a
    window and baseline that can each mix them: the any-move rule below was measured
    within one project and version, so comparing across projects would read a machine-wide
    move between projects, or the ordinary difference between two projects' own skills,
    MCP servers or CLAUDE.md, as something that changed."""
    projects = {project_of(path) for path in window}
    if len(projects) != 1:
        return None
    (only,) = projects
    return [path for path in baseline if project_of(path) == only]


def compare_components(components: pd.DataFrame, window: Sequence[str],
                       baseline: Sequence[str]) -> Optional[dict[str, Any]]:
    """What changed between the sessions of `baseline` and those of `window`, given as
    transcript paths, both narrowed to `window`'s own project first (_own_project_baseline).
    A part counts as logged on a side when more than half of that side's sessions logged
    it, and is compared over those sessions; otherwise it is `unknown`. A name is `added`
    when more than half of the window's sessions carry it and fewer than half of the
    baseline's do, and `removed` the other way round, so a one-off, like a server that
    failed to start once, drops out. `sizes` holds each part whose median size moved, as
    (baseline, window). Any move counts: within one project and version a part's size
    matched its group's median in all but 8 to 151 of the 47 to 1,055 sessions that logged
    it, measured on 2026-09-24, and those that didn't differed by thousands of characters.
    None when no part is logged on both sides, such as sessions from before ccdrift kept
    these rows, or when the baseline left after narrowing has too few of them."""
    if components.empty:
        return None
    baseline = _own_project_baseline(window, baseline)
    if baseline is None:
        return None
    rows = {str(row["source_file"]): row for row in components.to_dict("records")}
    after = [rows.get(str(path), {}) for path in window]
    before = [rows.get(str(path), {}) for path in baseline]
    added: dict[str, list[str]] = {}
    removed: dict[str, list[str]] = {}
    sizes: dict[str, tuple[float, float]] = {}
    unknown = []
    for part in PART_SIZES:
        now = [value for value in (_part(row, part) for row in after) if value is not None]
        then = [value for value in (_part(row, part) for row in before) if value is not None]
        if 2 * len(now) <= len(after) or 2 * len(then) <= len(before):
            unknown.append(part)
            continue
        was, became = statistics.median(size for _, size in then), statistics.median(size for _, size in now)
        if was != became:
            sizes[part] = (was, became)
        if part not in NAMED_PARTS:
            continue
        for name in sorted({name for names, _ in now + then for name in names}):
            # Twice the sessions carrying it against all of them: above means more than half.
            in_now = 2 * sum(name in names for names, _ in now) - len(now)
            in_then = 2 * sum(name in names for names, _ in then) - len(then)
            if in_now > 0 > in_then:
                added.setdefault(part, []).append(name)
            elif in_then > 0 > in_now:
                removed.setdefault(part, []).append(name)
    if len(unknown) == len(PART_SIZES):
        return None
    return {"added": _split(added), "removed": _split(removed), "sizes": sizes, "unknown": unknown}
