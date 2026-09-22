"""Names and one-line texts shared by the check, the report and the status line. It
imports nothing heavy, so `ccdrift status --short` stays cheap enough for a status
line that refreshes often."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

# NUL, line breaks, and the escapes that move a terminal's cursor or retitle its window.
# It lives here rather than in `logs`, so the two things that must strip it, parsing and
# printing a project folder, can both reach it without importing pandas.
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")

METRIC_ARGS = {"cache": "cache_ratio", "haiku": "haiku_fraction"}

INCIDENT_METRICS = {"cache_ratio": "Cache read ratio on new prompts",
                    "haiku_fraction": "Haiku share on the main thread"}
SHORT_NAMES = {"cache_ratio": "cache", "haiku_fraction": "haiku"}
MOVES = {"cache_ratio": "down", "haiku_fraction": "up"}
METRIC_WORDS = {"cache_ratio": "cache ratio", "haiku_fraction": "Haiku share"}
PERSISTENT_DAYS = 30

# What a tool-loop warning of each stream (see ccdrift.loops) says is happening.
LOOP_NAMES = {"main": "tool-loop cache misses rising", "subagent": "subagent cache misses rising"}

SETTING_NAMES = {"cache_tier": "cache tier", "effort": "effort", "speed": "speed", "service_tier": "service tier"}
TIER_NAMES = {"1h": "1-hour", "5m": "5-minute"}

# The reasons Claude Code records on a response whose prompt did not match what it had
# cached (message.diagnostics.cache_miss_reason.type). Five of them over 25 versions in
# one person's logs, on 925 of 147,500 responses.
REASON_NAMES = {"messages_changed": "the messages changed",
                "previous_message_not_found": "the previous message wasn't found",
                "system_changed": "the system prompt changed",
                "tools_changed": "the tools changed",
                "unavailable": "the cache was unavailable"}

# What each dimension of `ccdrift cost` is called in its heading.
DIMENSION_NAMES = {"thread": "thread", "agent": "agent", "skill": "skill", "plugin": "plugin",
                   "mcp": "MCP server", "model": "model", "project": "project", "branch": "branch"}


def number(value: Any, spec: str) -> str:
    """A number as the report prints it, or "-" when there is none."""
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def misses(count: int, turns: int, share: bool = False) -> str:
    """Tool-loop misses as "2/412", or as a share ("0.49%"); "-" without turns."""
    if not turns:
        return "-"
    return f"{count / turns:.2%}" if share else f"{count}/{turns}"


def approx(value: float) -> str:
    """Two significant figures with a k, M or B suffix: 17_556_103 -> "18M"."""
    if value <= 0:
        return "0"
    rounded = float(f"{value:.2g}")
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if rounded >= size:
            return f"{rounded / size:g}{suffix}"
    return f"{rounded:g}"


def spend_line(bucket: str, responses: int, tokens: float, share: float, dollars: Optional[float],
               unpriced: Sequence[str] = ()) -> str:
    """"  general-purpose            52,078   19.6B   53.1%   $412.18". A bucket holding a
    model ccdrift could not price shows "no price: claude-fable-5-1" where the money would
    be, since a column that simply goes blank reads as broken arithmetic. A bucket that is
    itself the model says "no price" alone: repeating its own name explains nothing."""
    if dollars is not None:
        money = f"  {'$' + format(dollars, ',.2f'):>12}"
    elif unpriced:
        named = [model for model in unpriced if model != bucket]
        money = "  no price" + (": " + ", ".join(named) if named else "")
    else:
        money = ""
    return f"  {bucket:<34}  {responses:>8,}  {approx(tokens):>7}  {share:>6.1%}{money}"


# Models named in the "no price" line before it gives up and counts the rest.
UNPRICED_SHOWN = 3


def unpriced_line(unpriced: Sequence[tuple[str, float]], total_withheld: bool, cutoff: float) -> str:
    """Why `ccdrift cost` shows no dollars somewhere: the models the price fit refused,
    the share of the window's tokens each carries, and where that was enough to withhold a
    figure. One cutoff decides the window's total and every bucket's, so it is stated once
    here with its number rather than repeated on each row it lets through. A missing
    dollar figure is only honest if its reason is printed."""
    together = sum(share for _, share in unpriced)
    if len(unpriced) == 1:
        # One model's own share is the summed share, so it is said once, not twice.
        names, share = unpriced[0][0], f"{together:.1%} of the window's tokens"
    else:
        shown = ", ".join(f"{model} ({each:.1%})" for model, each in unpriced[:UNPRICED_SHOWN])
        rest = len(unpriced) - UNPRICED_SHOWN
        names = shown + (f" and {rest} more" if rest > 0 else "")
        share = f"{together:.1%} of the window's tokens between them"
    if total_withheld:
        return (f"No total: no price for {names}, {share}. A bucket where it reaches "
                f"{cutoff:.0%} shows no dollars either.")
    return (f"No price for {names}, {share}: that spend is left out of the total, and out of any bucket "
            f"where it stays under {cutoff:.0%}. A bucket where it reaches {cutoff:.0%} shows no dollars at all.")


def cost_text(metric: str, cost: float) -> str:
    """"~18M tokens re-cached" or "~120 extra Haiku responses"."""
    if metric == "cache_ratio":
        return f"~{approx(cost)} tokens re-cached" if cost > 0 else "no tokens re-cached"
    return f"~{approx(cost)} extra Haiku responses" if cost > 0 else "no extra Haiku responses"


def incident_line(incident: dict, cost: Optional[float] = None) -> str:
    """One line about an incident, as `incident list`, `report` and `status` show it."""
    status = {"open": "open", "persistent": f"still changed after {PERSISTENT_DAYS} days",
              "dismissed": "dismissed"}.get(incident["status"])
    if incident["status"] == "recovered":
        if incident["source"] == "user":
            status = "added by hand"
        elif incident["closed_by"] == "user":
            status = f"closed by hand on {incident['closed_on']}"
        else:
            status = f"back to normal from {incident['recovered_from']}"
    parts = [status, cost_text(incident["metric"], incident["cost"] if cost is None else cost)]
    if incident["versions"]:
        parts.append("on " + ", ".join(incident["versions"]))
    span = f"{incident['start']}..{incident['end'] or 'now'}"
    return f"{SHORT_NAMES[incident['metric']]:<5}  {span:<24}  {'; '.join(parts)}"


def change_line(change: dict[str, Any]) -> str:
    return (f"{SETTING_NAMES[change['setting']]} for {change['model']}: {change['from']} -> {change['to']} "
            f"from {change['since']}")


def clock_text(stamp: str, now: datetime) -> str:
    """`stamp` as a time in `now`'s time zone: "14:20" on now's day, "09-19 23:40" before it."""
    when = datetime.fromisoformat(stamp).astimezone(now.tzinfo)
    return when.strftime("%H:%M") if when.date() == now.date() else when.strftime("%m-%d %H:%M")


def version_key(version: str) -> tuple:
    """Sorts 2.1.99 before 2.1.233, and "unknown" last."""
    if version == "unknown":
        return (1,)
    return (0, *((0, int(part), "") if part.isdigit() else (1, 0, part) for part in re.split(r"[.+-]", version)))


# What each kind of failed request is called after a count: "7 overloaded".
FAILURE_WORDS = {"overloaded": "overloaded", "stream": "cut off mid-stream", "retry": "retried",
                 "other": "of another kind", "slept": "while the Mac slept"}

# The window a failure rule judges a day against (ccdrift.failures reads it from here, so
# there is one number and the status line can name it without importing pandas).
BEFORE_DAYS = 14


def before_text(most: Any) -> str:
    """What a failure alert compared a day with: "against at most 3 a day on the days
    judged in the 14 before", or "against none" when nothing happened on them. The days
    judged, not the days that passed: only active days are baselines, so "in the 14 days
    before" would be false about a history holding quiet days."""
    if not most:
        return f"against none on the days judged in the {BEFORE_DAYS} before"
    return f"against at most {most} a day on the days judged in the {BEFORE_DAYS} before"


def run_text(days: int) -> str:
    """", leaving out the 2 days of this run", and nothing when the comparison left no day
    out. A cut-short day is compared with the days before it that aren't part of its own
    run, so without this the alert could call a fortnight clean that was not."""
    if not days:
        return ""
    return f", leaving out the {days} day{'' if days == 1 else 's'} of this run"


def worse_text(worse: dict[str, Any]) -> str:
    """What a second alert about the same regression compares with: "against the 0.60%
    reported on 2026-09-03". Not the baseline before the run: by the time a regression has
    tripled, the level the owner was actually told is the only number that says whether
    this is news."""
    return f"against the {worse['share']:.2%} reported on {worse['since']}"


def kinds_text(kinds: dict[str, int]) -> str:
    """"7 overloaded, 2 retried", the most first."""
    ordered = sorted(((kind, count) for kind, count in kinds.items() if count), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{count} {FAILURE_WORDS[kind]}" for kind, count in ordered)


def failure_line(episode: dict[str, Any]) -> str:
    named = kinds_text(episode["kinds"])
    return (f"requests failing on {episode['since']}: {episode['requests']}"
            f"{f' ({named})' if named else ''}, {before_text(episode['before'])}")


def cut_short_line(episode: dict[str, Any]) -> str:
    worse = episode.get("worse_than")
    tail = f", {worse_text(worse)}" if worse else run_text(episode.get("run_days", 0))
    return (f"responses cut short on {episode['since']}: {episode['cut']} of {episode['responses']:,} "
            f"main-thread responses{tail}")


def project_path(project: str) -> str:
    """A project folder read back as the path it stands for, for the owner's eyes: Claude
    Code writes "/Users/me/dev/app" as "-Users-me-dev-app". A directory whose own name
    holds a dash reads back with an extra slash, since the folder name is all Claude Code
    keeps, so this is a convenience, not a promise. The nameless project a source with no
    project folders makes (sessions.SOURCE_PROJECT) is the source folder itself.

    A folder name is the one piece of text ccdrift shows that never passed through
    `logs._text`: it comes from the transcript's own path, not from a field inside it. A
    directory name may hold control characters, and this path is printed to a terminal, so
    they are dropped here as they are everywhere else."""
    if not project:
        return "the source folder"
    return "/" + CONTROL_CHARS.sub("", project).lstrip("-").replace("-", "/")


PROJECTS_SHOWN = 3


def projects_text(paths: Sequence[str]) -> str:
    """"in /Users/me/dev/app", or "in 4 projects: /a, /b, /c and 1 more"; "" for none.
    Only `ccdrift status` and `ccdrift report` name a folder: an alert may be piped
    anywhere by --exec."""
    if not paths:
        return ""
    if len(paths) == 1:
        return f"in {paths[0]}"
    shown = ", ".join(paths[:PROJECTS_SHOWN])
    rest = len(paths) - PROJECTS_SHOWN
    return f"in {len(paths)} projects: {shown}" + (f" and {rest} more" if rest > 0 else "")


def context_change_line(change: dict[str, Any]) -> str:
    """"session start ~130k -> ~64k tokens from 2026-09-09 in /Users/me/a, 1 of 3 projects
    compared". The count says what the projects named are a share of, since ccdrift can
    only compare a project that has sessions each side of the change."""
    moved = [project_path(p) for p in change.get("projects", [])]
    where, seen = projects_text(moved), change.get("of_projects", 0)
    of = f", {len(moved)} of {seen} projects compared" if seen and len(moved) < seen else ""
    return (f"session start ~{approx(change['from'])} -> ~{approx(change['to'])} tokens from {change['since']}"
            + (f" {where}{of}" if where else ""))


def hook_failure_line(failure: dict[str, Any]) -> str:
    (first, second), (runs1, runs2), (failed1, failed2) = failure["days"], failure["runs"], failure["failed"]
    return (f"stop hooks failing from {failure['since']}: {failed1} of {runs1} runs on {first}, "
            f"{failed2} of {runs2} on {second}")


def field_gap_line(gap: dict[str, Any]) -> str:
    where = "" if gap["version"] == "unknown" else f" on {gap['version']}"
    return (f"{gap['field']} not logged{where}: {gap['share']:.0%} of {gap['responses']} responses, "
            f"{gap['share_before']:.0%} before")


def new_field_line(record: dict[str, Any]) -> str:
    paths = record["paths"]
    return (f"{len(paths)} new field{'' if len(paths) == 1 else 's'} on {record['version']}: "
            f"{', '.join(paths)}")


def loop_warning_line(warning: dict[str, Any]) -> str:
    sessions = f"{warning['sessions']} session{'' if warning['sessions'] == 1 else 's'}"
    return (f"{LOOP_NAMES[warning['stream']]} at {warning['at'][:16].replace('T', ' ')} UTC: {warning['misses']} of "
            f"{warning['turns']} turns in {sessions} (usually {warning['base_rate']:.2%}), "
            f"~{approx(warning['tokens'])} tokens rewritten")


def early_warning_line(warning: dict[str, Any]) -> str:
    return (f"cache misses rising at {warning['at'][:16].replace('T', ' ')} UTC: {warning['misses']} of "
            f"{warning['turns']} new-prompt turns (usually {warning['base_rate']:.1%})")


def reason_name(reason: str) -> str:
    """A cache-miss reason's English name, or the reason itself when ccdrift hasn't seen
    it, so a sixth one arriving is visible without a release."""
    return REASON_NAMES.get(reason, reason)


def miss_reason_line(counts: Mapping[str, int]) -> str:
    """"the messages changed 329, the system prompt changed 114", largest first, ties by
    name so two runs over one history read the same."""
    return ", ".join(f"{reason_name(reason)} {count:,}"
                     for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])))
