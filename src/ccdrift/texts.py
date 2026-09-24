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

NOT_LOGGED = "not logged"  # a setting's value on responses that carry none
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
               unpriced: Sequence[str] = (), projects: int = 1) -> str:
    """"  general-purpose                       52,078      20B   53.1%       $412.18". A bucket holding a
    model ccdrift could not price shows "no price: claude-fable-5-1" where the money would
    be, since a column that simply goes blank reads as broken arithmetic. A bucket that is
    itself the model says "no price" alone: repeating its own name explains nothing. A
    bucket drawn from more than one project folder says so, which only the branch
    dimension ever passes: every other key means the same thing wherever it appears."""
    if dollars is not None:
        money = f"  {'$' + format(dollars, ',.2f'):>12}"
    elif unpriced:
        named = [model for model in unpriced if model != bucket]
        money = "  no price" + (": " + ", ".join(named) if named else "")
    else:
        money = ""
    pooled = f"  {projects} projects" if projects > 1 else ""
    return f"  {bucket:<34}  {responses:>8,}  {approx(tokens):>7}  {share:>6.1%}{money}{pooled}"


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
    # isascii: str.isdigit also passes "²", which int() refuses.
    return (0, *((0, int(part), "") if part.isascii() and part.isdigit() else (1, 0, part)
                 for part in re.split(r"[.+-]", version)))


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


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------
# What each alert says. The module behind a rule decides when it goes out; check.py sends
# it under the title of its kind, which is also what `--exec` gets as CCDRIFT_ALERT.

ALERT_TITLES = {"flag": "ccdrift flag", "recovered": "ccdrift: back to normal",
                "persistent": "ccdrift: change persists", "history": "ccdrift: past incidents found",
                "early": "ccdrift: cache misses rising", "loop": f"ccdrift: {LOOP_NAMES['main']}",
                "subagent_loop": f"ccdrift: {LOOP_NAMES['subagent']}", "setting": "ccdrift: setting changed",
                "context": "ccdrift: session start changed",
                "context_dropped": "ccdrift: a recorded session-start change was dropped",
                "hooks": "ccdrift: hooks failing", "failed_requests": "ccdrift: requests failing",
                "cut_short": "ccdrift: responses cut short", "fields": "ccdrift: Claude Code stopped logging a field",
                "new_fields": "ccdrift: Claude Code logs a field ccdrift doesn't read",
                "blank_cache": "ccdrift can't compute the cache metric", "digest": "ccdrift: weekly summary",
                "failed": "ccdrift check failed"}


def incident_message(kind: str, incident: dict[str, Any], named: Sequence[str], z: Sequence[float],
                     run: Sequence[str]) -> tuple[str, list[str]]:
    """The message and log lines of an incident's `kind` of alert (flag, recovered or
    persistent), from its metric, start, cost and recovery day; `named` are the versions of
    the event's days, and `z` and `run` the flagging days' scores and dates."""
    metric, start = incident["metric"], incident["start"]
    label = INCIDENT_METRICS[metric]
    on = f", on Claude Code {', '.join(named)}" if named else ""
    cost = cost_text(metric, incident["cost"])
    if kind == "flag":
        scores = ", ".join(f"{v:+.1f}" for v in z)
        return (f"{label} {MOVES[metric]} from {start}{on}. {cost[0].upper()}{cost[1:]} so far.",
                [f"days {', '.join(run)}; z = {scores}"])
    if kind == "recovered":
        return f"{label} back to normal from {incident['recovered_from']}{on}. The incident from {start}: {cost}.", []
    return (f"{label} still {MOVES[metric]} {PERSISTENT_DAYS} days after {start}. ccdrift now treats it as the "
            "new normal; `ccdrift incident list` has the details.", [])


def context_dropped_message(record: dict[str, Any]) -> str:
    return (f"{record['since']}, ~{approx(record['from'])} -> ~{approx(record['to'])} tokens: "
            "judged against each project's own level, it isn't a change.")


def blank_cache_message(blank: dict[str, Any]) -> str:
    return (f"no usable cache values on {blank['days']} active days from {blank['first']} "
            f"({blank['responses']} responses, {blank['prompts']} prompts recognised). "
            "Claude Code's log format may have changed; run `ccdrift peek`.")


def state_unreadable(path: Any, exc: Exception) -> str:
    return f"can't read the state file {path}: {exc}"


FIELD_NAMES = {"version": "its version", "entrypoint": "the entrypoint", "effort": "effort",
               "speed": "the speed", "service_tier": "the service tier", "thinking_logged": "thinking token counts",
               "cache_split": "the 1-hour/5-minute cache split"}


CONSEQUENCES = {"version": "Alerts can't name versions",
                "entrypoint": "Agent SDK sessions can't be told apart",
                "effort": "Effort change alerts can't work",
                "speed": "The report can't show the speed",
                "service_tier": "The report can't show the service tier",
                "thinking_logged": "The lab can't compare logged thinking tokens",
                "cache_split": "Cache tier alerts can't work"}


def gap_message(gap: dict[str, Any]) -> str:
    where = "Claude Code" if gap["version"] == "unknown" else f"Claude Code {gap['version']}"
    return (f"{where} no longer logs {FIELD_NAMES[gap['field']]} (on {gap['share']:.0%} of {gap['responses']} "
            f"responses, {gap['share_before']:.0%} before). {CONSEQUENCES[gap['field']]} until ccdrift reads it "
            "again; run `ccdrift peek`.")


def new_fields_message(record: dict[str, Any]) -> str:
    paths = record["paths"]
    count = f"{len(paths)} field{'' if len(paths) == 1 else 's'}"
    subject = "It may be worth reading" if len(paths) == 1 else "They may be worth reading"
    return (f"Claude Code {record['version']} logs {count} ccdrift doesn't read: {', '.join(paths)} "
            f"(on {record['share']:.0%} of {record['responses']:,} responses). {subject}; "
            "please open an issue.")


def early_message(warning: dict[str, Any], now: datetime) -> str:
    on = f", on Claude Code {', '.join(warning['versions'])}" if warning["versions"] else ""
    return (f"{warning['misses']} of the last {warning['turns']} new-prompt turns missed the cache "
            f"(usually {warning['base_rate']:.1%}), since {clock_text(warning['since'], now)}{on}. "
            "The daily check confirms or clears it within a few days.")


def _on_versions(versions: Sequence[str]) -> str:
    return f", on Claude Code {', '.join(versions)}" if versions else ""


def requests_message(episode: dict[str, Any], versions: Sequence[str]) -> str:
    named = kinds_text(episode["kinds"])
    before = before_text(episode["before"])
    return (f"{episode['requests']} requests failed on {episode['since']}"
            f"{f' ({named})' if named else ''}, {before}{_on_versions(versions)}. Claude Code retries these itself; a run "
            "of them points at the API or your connection, not your setup.")


def cut_short_message(episode: dict[str, Any], versions: Sequence[str]) -> str:
    what = "stopped at the token limit or refused" if episode["refused"] else "stopped at the token limit"
    share = episode["cut"] / episode["responses"] if episode["responses"] else 0.0
    worse = episode.get("worse_than")
    against = (worse_text(worse) if worse else
               before_text(f"{episode['before_share']:.2%}" if episode["before_share"] > 0 else None)
               + run_text(episode.get("run_days", 0)))
    return (f"{episode['cut']} of {episode['responses']:,} main-thread responses {what} on {episode['since']} "
            f"({share:.2%}), {against}{_on_versions(versions)}. "
            "A Claude Code update may have changed the output limit.")


def hook_failure_message(failure: dict[str, Any], versions: Sequence[str]) -> str:
    on = f", on Claude Code {', '.join(versions)}" if versions else ""
    (first, second), (runs1, runs2), (failed1, failed2) = failure["days"], failure["runs"], failure["failed"]
    return (f"Stop hooks failed on {failed1} of {runs1} runs on {first} and {failed2} of {runs2} on {second}{on}. "
            "Check your hooks; a Claude Code update may have changed their input.")


STREAM_TURNS = {"main": "tool-loop turns", "subagent": "subagent tool-loop turns"}


def loop_message(warning: dict[str, Any], now: datetime) -> str:
    on = f", on Claude Code {', '.join(warning['versions'])}" if warning["versions"] else ""
    sessions = f"{warning['sessions']} session{'' if warning['sessions'] == 1 else 's'}"
    return (f"{warning['misses']} of the last {warning['turns']} {STREAM_TURNS[warning['stream']]} missed the cache "
            f"(usually {warning['base_rate']:.2%}), since {clock_text(warning['since'], now)}, in {sessions}, "
            f"rewriting ~{approx(warning['tokens'])} tokens{on}. `ccdrift report` shows whether it lasts.")


def replayed_line(incident: dict[str, Any]) -> str:
    """"cache ratio down 2026-08-18..2026-09-03, back to normal from 2026-09-04, ~16M
    tokens re-cached, on Claude Code 2.1.235 (since 08-19)"."""
    metric = incident["metric"]
    words = f"{METRIC_WORDS[metric]} {MOVES[metric]}"
    cost = cost_text(metric, incident["cost"])
    if incident["status"] == "open":
        text = f"{words} since {incident['start']}, still going, {cost} so far"
    elif incident["status"] == "persistent":
        text = f"{words} from {incident['start']}, still changed after {PERSISTENT_DAYS} days, {cost}"
    else:
        text = (f"{words} {incident['start']}..{incident['end']}, back to normal from "
                f"{incident['recovered_from']}, {cost}")
    return text + (f", on Claude Code {', '.join(incident['versions'])}" if incident["versions"] else "")


def history_message(found: Sequence[dict[str, Any]], first_day: str) -> str:
    """The first check's one alert about the incidents its replay found."""
    count = f"{len(found)} incident{'' if len(found) == 1 else 's'}"
    return (f"Replaying your history from {first_day} found {count} ccdrift would have followed: "
            f"{'; '.join(replayed_line(incident) for incident in found)}. `ccdrift incident list` has the "
            "details; `ccdrift incident dismiss` puts a false alarm's days back in the baseline.")


# What Claude Code logs about a session's start, as the session-start alert counts and
# names it: each kind of name by its singular and plural, and each part by what it is.
COMPONENT_NOUNS = {"agents": ("agent type", "agent types"), "skills": ("skill", "skills"),
                   "mcp_tools": ("MCP tool", "MCP tools"), "deferred": ("deferred tool", "deferred tools"),
                   "mcp": ("MCP server with instructions", "MCP servers with instructions"),
                   "tools": ("tool definition", "tool definitions")}
COMPONENT_PARTS = {"skills": "the skills listing", "deferred": "the deferred tools", "agents": "the agent types",
                   "mcp": "MCP instructions", "claude_md": "CLAUDE.md files", "system": "the system prompt",
                   "tools": "tool definitions"}
# Parts that are grammatically singular: only the skills listing and the system prompt.
SINGULAR_PARTS = frozenset({"skills", "system"})


def _joined(items: Sequence[str], last: str = "and") -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} {last} {items[-1]}"


def _name_count(names: Any) -> int:
    """How many names one kind of change holds; MCP tools come grouped by server."""
    return sum(len(tools) for tools in names.values()) if isinstance(names, Mapping) else len(names)


def _counted(changes: Mapping[str, Any]) -> list[str]:
    counts = [(kind, _name_count(changes.get(kind) or [])) for kind in COMPONENT_NOUNS]
    return [f"{n} {COMPONENT_NOUNS[kind][n != 1]}" for kind, n in counts if n]


def _were(items: Sequence[str]) -> str:
    return "was" if len(items) == 1 and items[0].startswith("1 ") else "were"


def components_text(what: Optional[Mapping[str, Any]]) -> str:
    """What the session-start alert adds about what Claude Code logged of the sessions'
    starts (components.compare_components): counts and characters, never a name, since
    the message reaches notifications and --exec. Empty when there was nothing to compare."""
    if what is None:
        return ""
    added, removed, sizes = _counted(what["added"]), _counted(what["removed"]), what["sizes"]
    if added and removed:
        changed = f"{_joined(added)} {_were(added)} added and {_joined(removed)} removed"
    elif added or removed:
        changed = f"{_joined(added or removed)} {_were(added or removed)} {'added' if added else 'removed'}"
    elif sizes:
        changed = f"only {_joined([COMPONENT_PARTS[part] for part in sizes])} changed"
    else:
        changed = ""
    moved = sum(after - before for before, after in sizes.values())
    size = (f", about {approx(moved)} more characters" if moved > 0 else
            f", about {approx(-moved)} fewer characters" if moved < 0 else "")
    text = (f" Of what Claude Code logs about a session's start, {changed}{size}, though the logs can't say how many "
            "of the tokens that is." if changed else
            " Nothing ccdrift could compare of what Claude Code logs about a session's start changed, so the step "
            "is in what it doesn't log or couldn't compare.")
    unknown = [COMPONENT_PARTS[part] for part in what["unknown"]]
    if unknown:
        joined = _joined(unknown, "or")
        verb = "wasn't" if len(unknown) == 1 and what["unknown"][0] in SINGULAR_PARTS else "weren't"
        text += (f" {joined[0].upper()}{joined[1:]} {verb} logged in every session compared, so ccdrift couldn't "
                 f"compare {'that part' if len(unknown) == 1 else 'those parts'}.")
    return text


def component_lines(what: Optional[Mapping[str, Any]]) -> list[str]:
    """The session-start alert's lines for the check's log alone: what was added and
    removed, by name, and each part whose size moved."""
    if what is None:
        return []
    lines = []
    for changes, word in ((what["added"], "added"), (what["removed"], "removed")):
        for kind, (_, plural) in COMPONENT_NOUNS.items():
            names = changes.get(kind)
            if names:
                listed = (", ".join(f"{server} ({len(tools)})" for server, tools in names.items())
                          if isinstance(names, Mapping) else ", ".join(names))
                lines.append(f"{plural} {word}: {listed}")
    return lines + [f"{COMPONENT_PARTS[part]}: {before:,.0f} -> {after:,.0f} characters"
                    for part, (before, after) in what["sizes"].items()]


def _context_where(change: dict[str, Any], new_version: bool) -> str:
    """Which projects a change reached, and what that says about its cause; "" when no
    project had the sessions each side to be compared with itself. Every branch counts the
    projects ccdrift could compare, not the projects the owner used: a machine with six
    active projects can have two that clear the bar, and "every project you used" would be
    false about the other four."""
    moved, seen = len(change.get("projects", [])), change.get("of_projects", 0)
    if not seen:
        return ""
    if not moved:
        # Every project that could be compared held its level, so whatever moved the
        # sessions isn't in any of them -- and isn't pinned on anything yet.
        if seen == 1:
            return (", though the one project ccdrift could compare with itself didn't move, so something outside "
                    "it changed.")
        return f", in none of the {seen} projects ccdrift could compare, so something outside them changed."
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
        # A version new to these sessions is named in the message's own opening clause, so
        # ruling Claude Code out here would contradict it: the projects that didn't move
        # say the cause isn't global, and a version that arrived says it might be.
        if new_version:
            return (f", in {moved} of the {seen} projects ccdrift could compare. {that} own files may explain it, "
                    "though a Claude Code version none of the sessions before it ran also arrived.")
        return (f", in {moved} of the {seen} projects ccdrift could compare. "
                f"{that} CLAUDE.md, MCP servers or skills explain it, not Claude Code.")
    if new_version:
        return (f", in every project ccdrift could compare ({moved} of {seen}), on a Claude Code version none of "
                "the sessions before it ran, the likeliest cause.")
    return (f", in every project ccdrift could compare ({moved} of {seen}), with no new Claude Code version, so "
            "look at your global configuration in ~/.claude.")


def context_message(change: dict[str, Any], versions: Sequence[str],
                    components: Optional[Mapping[str, Any]] = None) -> str:
    on = f", on Claude Code {', '.join(versions)}" if versions else ""
    direction = "down" if change["to"] < change["from"] else "up"
    where = _context_where(change, bool(change.get("new_version", False)))
    tail = where or ". Your MCP servers, plugins or CLAUDE.md can change this too."
    return (f"New sessions start with ~{approx(change['to'])} tokens of context from {change['since']}{on}, "
            f"{direction} from ~{approx(change['from'])}{tail}{components_text(components)}")


def change_message(change: dict[str, Any], versions: list[str]) -> str:
    on = f", on Claude Code {', '.join(versions)}" if versions else ""
    if change["setting"] == "cache_tier":
        old, new = (TIER_NAMES.get(change[k], change[k]) for k in ("from", "to"))
        return f"Cache writes for {change['model']} moved from the {old} to the {new} cache from {change['since']}{on}."
    return (f"Effort for {change['model']} changed from {change['from']} to {change['to']} from "
            f"{change['since']}{on}. If you didn't change it, Claude Code's default did.")


# ---------------------------------------------------------------------------
# Weekly summary
# ---------------------------------------------------------------------------

def _count_text(n: int, noun: str) -> str:
    return f"no {noun}s" if n == 0 else f"{n} {noun}{'' if n == 1 else 's'}"


# Each tool-loop stream in the summary: what its misses are called, and what it says with no turns.
DIGEST_LOOPS = (("loop", "tool-loop misses", "no tool-loop turns"),
                ("subagent_loop", "subagent", "no subagent loop turns"))


def week_failures_text(counted: int, cut: int) -> str:
    """The weekly summary's failures part: "no failed requests", or what there was."""
    if not counted and not cut:
        return "no failed requests"
    parts = [f"{counted} failed request{'' if counted == 1 else 's'}"] if counted else []
    if cut:
        parts.append(f"{cut} response{'' if cut == 1 else 's'} cut short")
    return ", ".join(parts)


def digest_text(summary: dict[str, Any]) -> str:
    """"Week of 09-14: 70 responses on 2.1.261–2.1.270; cache ratio 0.976 (1.4% misses);
    ..." from the numbers digest.week_summary counts."""
    parts = []
    if not summary["responses"]:
        parts.append("no responses")
    else:
        versions = summary["versions"]
        span = "" if not versions else f" on {versions[0]}" + (f"–{versions[-1]}" if len(versions) > 1 else "")
        parts.append(f"{summary['responses']:,} responses{span}")
        if summary["prompts"] is None:
            parts.append("no new-prompt turns")
        else:
            ratio, missed = summary["prompts"]
            parts.append(f"cache ratio {ratio:.3f} ({missed:.1%} misses)")
        haiku = summary["haiku"]
        parts.append("no Haiku" if haiku == 0 else f"Haiku {haiku:.1%} of responses")
        loops = []
        for prefix, found, missing in DIGEST_LOOPS:
            turns, misses = summary["loops"].get(prefix, (0, 0))
            loops.append(f"{found} {misses:,} of {turns:,}" if turns else missing)
        parts.append(", ".join(loops))
        if summary["failures"] is not None:
            parts.append(week_failures_text(*summary["failures"]))
    parts.append(_count_text(summary["open_incidents"], "open incident"))
    parts.append(_count_text(summary["setting_changes"], "setting change"))
    parts.append(_count_text(summary["new_fields"], "new field"))
    parts.append(f"check ran on {summary['ran']} of 7 days")
    return f"Week of {summary['week_start'][5:]}: " + "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# The status line and `ccdrift status`
# ---------------------------------------------------------------------------

LIVE_NAMES = {"cache_ratio": "cache ratio down", "haiku_fraction": "Haiku share up"}

# status.short_status fills these; its docstring says which one wins.
STATUS_LINES = {"no_check": "ccdrift: no check yet",
                "failed": "ccdrift: check failed {at:%m-%d %H:%M}",
                "stale": "ccdrift: no check for {days} days",
                "live": "{name} since {since}",
                "incidents": "ccdrift: {incidents}",
                "hooks": "ccdrift: hooks failing since {since}",
                "rising": "ccdrift: cache misses rising since {since}",
                "loop": "ccdrift: {name} since {since}",
                "unreadable": "ccdrift: can't read state",
                "not_run": "The check hasn't run yet. `ccdrift schedule install` sets it up.\n",
                "last": "Last check: {at:%Y-%m-%d %H:%M}, {outcome}",
                "ok": "ok",
                "outcome_failed": "failed: {error}",
                "last_ok": "Last successful check: {at:%Y-%m-%d %H:%M}",
                "open": "Open incidents",
                "closed": "Closed in the last {days} days",
                "settings": "Setting changes in the last {days} days",
                "other": "Other changes in the last {days} days",
                "section": "{title}:",
                "empty_section": "{title}: none"}


# ---------------------------------------------------------------------------
# What the commands print
# ---------------------------------------------------------------------------

COMMAND_LINES = {"no_scheduler": "ccdrift can't set up a scheduled job on this system. Run this command every hour, "
                                 "or once a day, with your system's scheduler:",
                 "not_installed": "Nothing installed: {error}",
                 "installed": "Installed a {backend} job: `ccdrift check` runs {when}.",
                 "log": "Log: {log}",
                 "first_run": "A first run has started. Check `ccdrift schedule status` in a minute.",
                 "not_removed": "Nothing removed: {error}",
                 "removed": "Removed the ccdrift job.",
                 "nothing_to_remove": "No ccdrift job was installed.",
                 "schedule_unreadable": "Couldn't read the schedule: {error}",
                 "state_unchangeable": "Can't change the state file {path}: {error}",
                 "state_unreadable": "Can't read the state file {path}: {error}",
                 "unchanged": "Nothing changed: {error}",
                 "added": "Added {metric} {start}..{end}. `ccdrift incident list` shows what it cost.",
                 "html_by_version": "--html draws the day view; drop --by version",
                 "bad_days": "expected a whole number of days, 1 or more, not {text!r}",
                 "bad_day": "expected a day like 2026-08-18, not {text!r}",
                 "unhandled": "unhandled command: {command}"}


# ---------------------------------------------------------------------------
# ccdrift report
# ---------------------------------------------------------------------------

REPORT_LINES = {"days": "Last {days} complete UTC days with main-thread activity.",
                "rule": "Flagged once {bins} of any {window} days in a row pass the cutoff: "
                        "z <= -{cache:.1f} for the cache ratio, z >= +{haiku:.1f} for Haiku share.",
                "versions": "Complete UTC days with main-thread activity, by Claude Code version.",
                "miss": "A miss is a new-prompt turn that reads less than half its input from the cache.",
                "loop_miss": "A loop miss is a tool-loop turn that reads less than half of what the response "
                             "before it had cached.",
                "release_note": "    release notes: {text}",
                "version_reasons": "    why the cache missed: {reasons}",
                "reasons": "Why the cache missed, as Claude Code recorded it: {reasons}",
                "incidents": "Incidents:",
                "no_incidents": "Incidents: none yet",
                "legacy_head": "Flags reported before ccdrift followed incidents:",
                "legacy": "  {label} from {day}",
                "page_by": "The page draws the day view; it has nothing to draw for by={by!r}.",
                "page_and_json": "The page and JSON are one output each; ask for one of them.",
                "unwritable": "Can't write {path}: {error}",
                "wrote": "wrote {path}"}

# Each table's columns: heading, then alignment and width.
DAY_TABLE = (("day", "<10"), ("responses", ">9"), ("cache ratio", ">11"), ("z", ">5"), ("haiku share", ">11"),
             ("z", ">5"), ("loop misses", ">11"), ("subagent misses", ">15"), ("flagged", ""))
VERSION_TABLE = (("version", "<11"), ("first day", "<10"), ("last day", "<10"), ("responses", ">9"),
                 ("prompt turns", ">12"), ("cache ratio", ">11"), ("misses", ">6"), ("loop misses", ">11"),
                 ("subagent misses", ">15"), ("haiku share", ">11"), ("session start", ">13"), ("compacts at", ">11"))


def table_row(columns: Sequence[tuple[str, str]], cells: Sequence[Any]) -> str:
    """One line of a report table: each cell aligned to its column, two spaces apart."""
    return "  ".join(format(cell, spec) for (_, spec), cell in zip(columns, cells)).rstrip()


def table_header(columns: Sequence[tuple[str, str]]) -> str:
    return table_row(columns, [heading for heading, _ in columns])


def size_text(value: float) -> str:
    """A token count as the version table prints it: "~130k", or "-" when there is none."""
    return "-" if math.isnan(value) else approx(value)


# ---------------------------------------------------------------------------
# ccdrift cost
# ---------------------------------------------------------------------------

COST_LINES = {"window": "{days} complete UTC days, {tokens} tokens{money}.",
              "money": ", ${total:,.2f}",
              "every": "Every section below accounts for all of them; a response can appear in more than one section.",
              "by": "By {name}"}

# What a response is called in a dimension that doesn't name it, and the two threads.
ABSENT_NAMES = {"agent": "no agent", "skill": "no skill", "plugin": "no plugin", "mcp": "no MCP server",
                "model": "unknown model", "branch": "no branch"}
THREAD_NAMES = {True: "subagent", False: "main thread"}
DETACHED_NAME = "detached HEAD"  # the branch bucket for gitBranch "HEAD": nothing checked out


# ---------------------------------------------------------------------------
# The report's sections from the rules: settings, subagents, projects, hooks, failures
# ---------------------------------------------------------------------------

def settings_lines(summary: list[dict[str, Any]]) -> list[str]:
    """The report's settings section, starting with a blank line; empty without models."""
    if not summary:
        return []
    lines = ["", "Settings on the CLI main thread over these days (share of responses):"]
    for model in summary:
        parts = [f"{SETTING_NAMES[s]} " + ", ".join(f"{v} {share:.0%}" for v, share in values.items())
                 for s, values in model["shares"].items() if values]
        lines.append(f"  {model['model']}: {'; '.join(parts)}")
        lines += [f"    {c['day']}: {SETTING_NAMES[c['setting']]} {c['from']} -> {c['to']}" for c in model["changes"]]
    return lines


CALLER_PICKED = "general-purpose"  # its model is whatever the caller asks for


def subagent_lines(summary: list[dict[str, Any]]) -> list[str]:
    """The report's subagent section, starting with a blank line; empty without subagents."""
    if not summary:
        return []
    lines = ["", "Subagent models over these days (share of responses):"]
    for agent in summary:
        label = agent["agent_type"] + (" (model picked by the caller)" if agent["agent_type"] == CALLER_PICKED else "")
        lines.append(f"  {label}: " + ", ".join(f"{model} {share:.0%}" for model, share in agent["models"].items()))
    return lines


PROJECTS_IN_REPORT = 5


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


def hooks_lines(summary: Optional[dict[str, Any]]) -> list[str]:
    """The report's hooks line, starting with a blank line; empty without runs."""
    if summary is None:
        return []
    days = summary["error_days"]
    errors = "no errors" if days == 0 else f"errors on {days} day{'s' if days != 1 else ''}"
    median = "" if summary["median_duration_ms"] is None else f", median {summary['median_duration_ms'] / 1000:.1f} s"
    return ["", f"Hooks over these days: {summary['runs']:,} stop-hook runs, {errors}{median}"]


def failure_lines(summary: Optional[dict[str, Any]]) -> list[str]:
    """The report's failures line, starting with a blank line; empty without failures."""
    if summary is None:
        return []
    parts = [kinds_text(summary["kinds"])] if summary["kinds"] else []
    if summary["cut"]:
        parts.append(f"{summary['cut']} response{'' if summary['cut'] == 1 else 's'} cut short")
    return ["", "Failures over these days: " + ", ".join(parts)]


# ---------------------------------------------------------------------------
# ccdrift schedule
# ---------------------------------------------------------------------------

SCHEDULE_LINES = {"exited": "`{command}` exited with {code}",
                  "output": ": {output}",
                  "hourly": "every hour",
                  "daily": "daily at {time}",
                  "clock": "{hour:02d}:{minute:02d}",
                  "unreadable_time": "an unreadable time",
                  "bad_at": "--at takes a 24-hour time like 09:00, not {text!r}",
                  "last_log": "last log line: {line}",
                  "no_log": "log: nothing written yet ({log})",
                  "restored": "{error}. The job installed before is back in place.",
                  "restored_unloaded": "{error}. The job installed before is back in place but didn't load; "
                                       "`ccdrift schedule status` shows it.",
                  "restored_disabled": "{error}. The job installed before is back in place but couldn't be "
                                       "enabled; `ccdrift schedule status` shows it.",
                  "not_installed": "not installed",
                  "bad_plist": "{plist} isn't a readable plist: {error}. Run `ccdrift schedule install` again.",
                  "launchd_installed": "installed: launchd agent {label}, {schedule}",
                  "not_loaded": "not loaded; run `ccdrift schedule install` again",
                  "field": "{key}: {value}",
                  "no_log_file": "log: the agent names no log file",
                  "systemd_note": "systemd user timers run only while you're logged in, unless lingering is on "
                                  "(loginctl enable-linger).",
                  "systemd_installed": "installed: systemd timer {timer}, {schedule}",
                  "enabled": "enabled: {state}",
                  "unknown": "unknown",
                  "last_run": "last run: {at}",
                  "never": "never",
                  "last_exit": "last exit code: {code}",
                  "unreadable_schedule": "an unreadable schedule",
                  "hourly_at": "every hour at minute {minute}",
                  "cron_schedule": "on the schedule `{fields}`",
                  "cron_catch_up": "Cron doesn't catch up on runs missed while the machine was off.",
                  "cron_notify": "Jobs started by cron usually can't show notifications, so alerts will mostly "
                                 "reach only the log.",
                  "cron_installed": "installed: crontab line, {schedule}",
                  "cron_history": "cron keeps no run history; the log shows each run",
                  "first_run_failed": "couldn't start a first run: {error}",
                  "control_char": "a line break or other control character can't go into a scheduled job: {value!r}",
                  "test_title": "ccdrift",
                  "test_message": "The check will run {when}. Alerts will look like this."}


# ---------------------------------------------------------------------------
# ccdrift incident draft
# ---------------------------------------------------------------------------
# draft.draft_facts counts; draft_text writes the Markdown issue from what it counted.

DRAFT_LINES = {"no_incident_on": "No {name} incident starts on {start}.",
               "no_incident": "No {name} incident is recorded.",
               "no_days": "The history holds no judged days during the {name} incident from {start}.",
               "macos": "macOS {version}",
               "system": "{system} {release}"}

# The pause before a prompt, by the upper bound of its bucket in seconds.
PAUSE_NAMES = {60: "≤1 min", 300: "1–5 min", 900: "5–15 min", 3600: "15–60 min"}

# The settings the environment names: the column, what it is called, and what its share counts.
DRAFT_SETTINGS = (("cache_tier", "cache tier", " that write to the cache"), ("effort", "effort", ""))

BASELINE_NOTE = ("Before is the baseline ccdrift judged the incident against: the days it compared with, which skip "
                 "the days of other incidents, except ones dismissed or taken as the new normal, so they need not "
                 "run up to the day it started.\n\n")


def version_span(names: Sequence[str]) -> str:
    """"2.1.233–2.1.258" for the versions a title names, oldest first; "" for none."""
    if not names:
        return ""
    return names[0] if len(names) == 1 else f"{names[0]}–{names[-1]}"


def _rate(part: float, whole: float) -> str:
    return f"{part / whole:.2%}" if whole else "-"


def _day_span(days: Sequence[str]) -> str:
    return f"{days[0][5:]}..{days[-1][5:]}"


def _days(count: int) -> str:
    return f"{count} day{'' if count == 1 else 's'}"


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


def _compared(parts: Mapping[str, tuple[int, int]], periods: Mapping[str, Sequence[str]]) -> str:
    """", against 1 of 200 (0.50%) on the 5 days before and 2 of 566 (0.35%) on the 11
    days after", for the periods that have days."""
    clauses = [f"{parts[name][0]:,} of {parts[name][1]:,} ({_rate(*parts[name])}) on the "
               f"{_days(len(periods[name]))} {name}" for name in ("before", "after") if periods[name]]
    return f", against {' and '.join(clauses)}" if clauses else ""


def _lead(incident: Mapping[str, Any], periods: Mapping[str, Sequence[str]]) -> str:
    if incident["status"] == "persistent":
        return f"From {incident['start']} to {incident['end']}, still changed after {PERSISTENT_DAYS} days"
    if incident["end"]:
        return f"From {incident['start']} to {incident['end']}"
    as_of = f" as of {periods['during'][-1]}" if periods["during"] else ""
    return f"From {incident['start']}, still going{as_of}"


def _version_table(rows: Sequence[tuple[str, str, int, int]], names: Sequence[str]) -> str:
    """One row per version and period (draft.draft_facts orders them): its rows, those
    that count, and their share; "" when no version is logged."""
    shown = [[version, period, f"{total:,}", f"{hits:,}", _rate(hits, total)] for version, period, total, hits in rows]
    return _markdown_table(["Version", "Period", *names], shown) if shown else ""


def _cache_draft(facts: Mapping[str, Any]) -> tuple[str, list[str]]:
    counts, periods, span = facts["counts"], facts["periods"], version_span(facts["span"])
    usually = f" (usually {_rate(*counts['before'])})" if counts["before"][1] else ""
    title = (f"New prompts miss the prompt cache {_rate(*counts['during'])} of the time"
             + (f" on Claude Code {span}" if span else "") + usually)
    beyond = (f"~{approx(facts['cost'])} tokens were written to the cache again" if facts["cost"] > 0
              else "no tokens were written to the cache again")
    sections = [
        "### What happened\n\n"
        f"{_lead(facts['incident'], periods)}, {counts['during'][0]:,} of {counts['during'][1]:,} main-thread turns "
        f"that open with a new prompt ({_rate(*counts['during'])}) missed the prompt cache"
        f"{_compared(counts, periods)}. ccdrift estimates {beyond} beyond the usual miss rate.",
        "### Before, during and after\n\n" + BASELINE_NOTE
        + _markdown_table(
            ["", "Days", "New-prompt turns", "Misses", "Miss rate", "Cache read ratio"],
            [[f"{name.capitalize()} ({_day_span(periods[name])})", len(periods[name]), f"{counts[name][1]:,}",
              f"{counts[name][0]:,}", _rate(*counts[name]),
              "-" if facts["ratios"][name] is None else f"{facts['ratios'][name]:.3f}"]
             for name in periods if periods[name]]),
    ]
    versions = _version_table(facts["versions"], ["Turns", "Misses", "Miss rate"])
    if versions:
        sections.append("### By Claude Code version\n\n" + versions)
    missed = facts["missed"]
    if missed:
        read, wrote = missed["read"], missed["wrote"]
        turns_text = f"{missed['turns']:,} missed turn{'' if missed['turns'] == 1 else 's'}"
        sections.append(
            "### What a missed turn looks like\n\n"
            f"The {turns_text} during read a median {read[0]:,} tokens from the cache (middle half "
            f"{read[1]:,}–{read[2]:,}) and wrote a median {wrote[0]:,} (middle half {wrote[1]:,}–{wrote[2]:,}), "
            "so each wrote most of its input to the cache again.")
    reasons = facts["reasons"]
    if reasons:
        rows = [[reason_name(reason).capitalize(), *(f"{count:,} ({_rate(count, total)})" for count, total in cells)]
                for reason, cells in reasons["rows"]]
        sections.append(
            "### Why the cache missed\n\n"
            "Claude Code records a reason on a response whose prompt did not match what it had cached. "
            "ccdrift counts them and does not judge them: no alert of its own turns on these numbers.\n\n"
            + _markdown_table(["Reason", *(name.capitalize() for name in reasons["shown"])], rows))
    if facts["pauses"]:
        rows = [[PAUSE_NAMES[bound], f"{turns:,}", f"{misses:,}", _rate(misses, turns)]
                for bound, turns, misses in facts["pauses"]]
        sections.append("### Pause before the prompt\n\nHow long the turn waited between the previous response and "
                        "the prompt that opened it.\n\n"
                        + _markdown_table(["Pause", "Turns", "Misses", "Miss rate"], rows))
    loop_parts = [f"{misses:,} of {total:,} ({_rate(misses, total)}) {name}" for name, misses, total in facts["loops"]]
    if loop_parts:
        joined = loop_parts[0] if len(loop_parts) == 1 else f"{', '.join(loop_parts[:-1])} and {loop_parts[-1]}"
        sections.append(f"### Tool-loop turns\n\nTurns inside the tool loop on the main thread missed {joined}.")
    return title, sections


def _haiku_draft(facts: Mapping[str, Any]) -> tuple[str, list[str]]:
    counts, periods, span = facts["counts"], facts["periods"], version_span(facts["span"])
    usually = f" (usually {_rate(*counts['before'])})" if counts["before"][1] else ""
    title = (f"Haiku answers {_rate(*counts['during'])} of main-thread responses"
             + (f" on Claude Code {span}" if span else "") + usually)
    extra = f"~{approx(facts['cost'])} extra Haiku responses" if facts["cost"] > 0 else "no extra Haiku responses"
    sections = [
        "### What happened\n\n"
        f"{_lead(facts['incident'], periods)}, Haiku answered {counts['during'][0]:,} of {counts['during'][1]:,} "
        f"main-thread responses ({_rate(*counts['during'])}){_compared(counts, periods)}: {extra} by ccdrift's "
        "estimate.",
        "### Before, during and after\n\n" + BASELINE_NOTE
        + _markdown_table(
            ["", "Days", "Responses", "Haiku responses", "Haiku share"],
            [[f"{name.capitalize()} ({_day_span(periods[name])})", len(periods[name]), f"{counts[name][1]:,}",
              f"{counts[name][0]:,}", _rate(counts[name][0], counts[name][1])]
             for name in periods if periods[name]]),
    ]
    versions = _version_table(facts["versions"], ["Responses", "Haiku responses", "Haiku share"])
    if versions:
        sections.append("### By Claude Code version\n\n" + versions)
    return title, sections


def _draft_environment(env: Mapping[str, Any]) -> str:
    entrypoints = env["entrypoints"]
    entry = f" (entrypoint{'s' if len(entrypoints) > 1 else ''} {', '.join(entrypoints)})" if entrypoints else ""
    lines = [f"- Claude Code: {', '.join(env['versions'])}{entry}" if env["versions"]
             else f"- Claude Code: version not logged{entry}"]
    if env["models"]:
        lines.append("- Models during: " + ", ".join(f"{model} ({share:.2%} of responses)"
                                                     for model, share in env["models"]))
    settings = []
    for column, name, suffix in DRAFT_SETTINGS:
        top = env["settings"][column]
        settings.append(f"{name} {top[0]} on {top[1]:.2%} of responses{suffix}" if top else f"{name} not logged")
    lines += [f"- Main thread: {', '.join(settings)}", f"- OS: {env['os']}",
              f"- Measured with ccdrift {env['ccdrift']} from local session transcripts (aggregates only)"]
    return "### Environment\n\n" + "\n".join(lines)


def _draft_method(method: Mapping[str, Any]) -> str:
    cache = method["metric"] == "cache_ratio"
    rule = (f"an incident opens when {method['bins']} of {method['window']} days in a row fall "
            f"{'below z = −' if cache else 'above z = +'}{method['cutoff']:.1f} and closes once "
            f"{method['recovery']} pooled days are back inside the cutoff on {method['recovery']} days in a row, "
            f"or after {PERSISTENT_DAYS} days, when it takes the change as the new normal.")
    if cache:
        counted = ("It counts main-thread turns that open with a new prompt within an hour of the previous response, "
                   "outside Agent SDK sessions and not right after a compaction. A turn misses the cache when it "
                   "reads less than half of its input from it.")
    else:
        counted = ("It counts main-thread responses outside Agent SDK sessions and the share answered by a Haiku "
                   "model.")
    return ("### How this was measured\n\nccdrift reads Claude Code's local session transcripts. " + counted
            + " A day is a UTC day, and only complete ones are judged. Each day is compared with the median of up "
            f"to {method['baseline']} days before it, in units of their spread, which never falls below the noise "
            "a day of that many turns shows anyway; those days skip the days of other incidents, except ones "
            "dismissed or taken as the new normal. Then "
            + rule)


def draft_text(facts: Mapping[str, Any]) -> str:
    """The draft issue from draft.draft_facts: a title line, a blank line and its sections."""
    title, sections = (_cache_draft if facts["metric"] == "cache_ratio" else _haiku_draft)(facts)
    if facts["notes"]:
        sections.append("### Release notes that may be related\n\n"
                        + "\n".join(f"- {version}: {text}" for version, text in facts["notes"]))
    sections += [_draft_environment(facts["environment"]), _draft_method(facts["method"])]
    return "\n\n".join([title, *sections]) + "\n"


# ---------------------------------------------------------------------------
# ccdrift incident and ccdrift replay
# ---------------------------------------------------------------------------

INCIDENT_LINES = {"version_since": "{version} (since {since})",
                  "bad_days": "expected START..END, e.g. 2026-08-16..2026-09-04, not {text!r}",
                  "end_before_start": "{end} is before {start}",
                  "not_over": "{end} isn't over yet in UTC; the last complete day is {yesterday}",
                  "overlaps": "it overlaps the {name} incident from {start}",
                  "none_open": "no {name} incident is open",
                  "none_starts": "no {name} incident starts on {start}",
                  "none_recorded": "No incidents recorded.",
                  "list": "Incidents, newest first:"}

REPLAY_LINES = {"recorded": "recorded",
                "dismissed": "dismissed",
                "overlap": "{label}: {incidents}",
                "overlap_item": "{name} {start}..{end}",
                "now": "now",
                "persistent": "not recorded: after {days} days the check takes the new level as normal, "
                              "so its days need no record",
                "open": "not recorded: `ccdrift incident add {name} {start}..{end}` records its days so far",
                "closed": "not recorded: `ccdrift incident add {name} {start}..{end}` records it",
                "nothing": "Nothing to replay: no complete UTC day with main-thread activity yet.",
                "header": "Replaying the check day by day from {first} to {today} (UTC) on an empty state, "
                          "incidents only.",
                "quiet": "Nothing is recorded and no alert is sent.",
                "event": "{day}  {title}: {message}",
                "none": "No incidents: the check would have sent no incident alert over these days.",
                "found": "Incidents the replay found:"}


# ---------------------------------------------------------------------------
# report --html, the state file and the alert command
# ---------------------------------------------------------------------------

PAGE_LINES = {"title": "ccdrift report {date}",
              "heading": "ccdrift report, {date}",
              "rule": "The last {days} complete UTC days with main-thread activity. A metric is flagged once {bins} of "
                      "any {window} days in a row pass the cutoff: z ≤ −{cache:.1f} for the cache ratio, "
                      "z ≥ +{haiku:.1f} for the Haiku share.",
              "cache_chart": "Cache read ratio per day",
              "cache_z": "Cache read ratio z",
              "haiku_chart": "Haiku share of main-thread responses per day",
              "haiku_z": "Haiku share z",
              "strip": "{label} per day",
              # The page is the artefact meant to be sent on, and its reader can't ask what a mark means.
              "key": "A ring marks a day ccdrift flagged; a shaded column is a day inside a recorded incident for "
                     "that metric; the bars under each chart are that day’s z, with the dashed line the cutoff. "
                     "Days with no main-thread activity are left out, so the line joins the days there are.",
              "incidents": "Incidents",
              "none_yet": "none yet",
              "legacy_head": "Flags reported before ccdrift followed incidents",
              "legacy": "{label} from {day}",
              "footer": "Written by ccdrift {version} on {date} from the transcripts in {source}. This page holds "
                        "local paths, and nothing left this machine to make it."}

STATE_LINES = {"not_object": "{path} doesn't hold a JSON object",
               "unknown_version": "{path} has an unknown state version: {version!r}",
               "newer": "{path} was written by a newer ccdrift (state version {version}); upgrade ccdrift",
               "waiting": "Waiting for another ccdrift command to finish with {path}..."}

NOTIFY_LINES = {"timed_out": "timed out after {seconds} s",
                "exit": "exit {code}",
                "stderr": ": {line}"}


# ---------------------------------------------------------------------------
# The check's log and reading the transcripts
# ---------------------------------------------------------------------------

CHECK_LINES = {"alert": "[check {at:%Y-%m-%d %H:%M}] {title}: {message}",
               "detail": "    {detail}",
               "notify_failed": '[check {at:%Y-%m-%d %H:%M}] notification failed for "{title}": {error}',
               "exec_failed": '[check {at:%Y-%m-%d %H:%M}] --exec failed for "{title}": {error}',
               "no_alerts": "[check {at:%Y-%m-%d %H:%M}] no alerts"}

LOG_LINES = {"unreadable": "  ! could not read {path}: {error}",
             "counts": "  files={files} lines={lines} bad_json={bad_json} assistant_lines={assistant_lines} "
                       "responses={responses}",
             "peek_line": "# first assistant line, text shown as its length",
             "peek_fields": "\n# resolved fields:"}


def note_lines(notes: Sequence[tuple[str, str]]) -> list[str]:
    """An alert's release notes as log lines, one per note."""
    return [f"release notes {version}: {text}" for version, text in notes]


def no_transcripts_message(source: Any) -> str:
    return (f"No Claude Code transcripts found in {source}. Pass --source DIR, or set "
            "CLAUDE_CONFIG_DIR if Claude Code keeps its files somewhere else.")


# ---------------------------------------------------------------------------
# The history store
# ---------------------------------------------------------------------------

HISTORY_LINES = {"busy": "Can't use the history store {path}: {error}. Another ccdrift command is using it; "
                         "try again once it has finished.",
                 "unusable": "Can't use the history store {path}: {error}. Move it aside to rebuild it from the "
                             "transcripts still on disk.",
                 "old_sqlite": "ccdrift needs SQLite 3.24 or newer for its history store; this Python has "
                               "SQLite {version}.",
                 "folder": "Can't use the history store {path}: it is a folder. Move it aside to rebuild it from "
                           "the transcripts still on disk.",
                 "cant_open": "Can't open the history store {path}: {error}",
                 "bad_schema": "its schema version {stored!r} isn't a number",
                 "newer": "The history store {path} was written by a newer ccdrift. Upgrade ccdrift, or move the "
                          "store aside to rebuild it from the transcripts still on disk.",
                 "other_source": "Not using ccdrift's history in {path}: it was built from {built_from}.",
                 "skipped": "Skipped {path}: {error}. Its rows stay as they were, and it is read again next time."}
