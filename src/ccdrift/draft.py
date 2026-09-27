"""A draft Claude Code issue about an incident: what happened before, during and after
it, by version, what a missed turn looks like, the release notes that may be related,
the environment and how ccdrift measured it, as aggregates only: no paths, project
names, session ids or prompt text. The owner wrote the August 2026 caching regression
up by hand from the lab; `ccdrift incident draft` prints that write-up for any
incident."""

from __future__ import annotations

import platform
import statistics
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ccdrift import __version__
from ccdrift.changelog import (TOPIC_OF, changelog_path, days_before, load_changelog, note_versions,
                               release_notes)
from ccdrift.components import MCP_TOOL, compare_components
from ccdrift.detector import DetectorConfig, baseline_bins
from ccdrift.failures import (ACTIVE_RESPONSES, CUT_FLOOR, CUT_RATIO, CUT_SHARE, CUT_USUAL, MIN_BEFORE_DAYS,
                              cut_shares, failure_counts, judged_failures)
from ccdrift.history import HistoryError, load_history
from ccdrift.hookcover import MERGE_DAYS, SETTING, merged, stream_changes, transcript_states
from ccdrift.incidents import OPEN_END, RECOVERY_BINS, exclusions, incident_cost, incident_versions, versions_text
from ccdrift.logs import HOOK_EVENTS, LOOP_GAP_SECONDS, LOOP_MISS_SHARE, Tables, judged_turns, outside_sdk
from ccdrift.loops import BASE_DAYS, LOOP_SETTINGS, WINDOW_DAYS, loop_turns
from ccdrift.report import reason_counts
from ccdrift.sessions import (BASELINE, CHANGE, MIN_BASELINE, MIN_PROJECT_SESSIONS, PROJECT_BASELINE, SIDE, SIDE_DAYS,
                              WINDOW, found_changes, ratio_starts, session_starts)
from ccdrift.state import load_state
from ccdrift.texts import (ALERT_NAMES, BEFORE_DAYS, COMMAND_LINES, DRAFT_LINES, DRAFT_SETTINGS, SHORT_NAMES,
                           draft_text, no_transcripts_message, version_key)

AFTER_DAYS = 14  # judged days after an incident that the draft compares with
PAUSE_BOUNDS = (60, 300, 900, 3600)  # seconds before the prompt: the upper bound of each bucket
PERIODS = ("before", "during", "after")


def find_incident(incidents: Sequence[dict[str, Any]], metric: str,
                  start: Optional[str] = None) -> Optional[dict[str, Any]]:
    """The incident of `metric` that starts on `start`, one not dismissed first; without
    `start`, the latest one not dismissed. None when there is none."""
    matching = [i for i in incidents if i["metric"] == metric and (start is None or i["start"] == start)]
    kept = [i for i in matching if i["status"] != "dismissed"]
    if start is None:
        return max(kept, key=lambda i: i["start"]) if kept else None
    return (kept or matching or [None])[0]


def draft_periods(turns: pd.DataFrame, incident: dict[str, Any], incidents: Sequence[dict[str, Any]],
                  cfg: DetectorConfig) -> dict[str, list[str]]:
    """The judged days before the incident (the baseline its cost compares with), during
    it (through the last judged day while it is open) and, for one with an end that
    isn't persistent, up to AFTER_DAYS after it and before the start of the next
    incident of the same metric that isn't dismissed; an open or persistent incident
    has none."""
    days = sorted(turns["day"].astype(str).unique()) if not turns.empty else []
    first = next((k for k, day in enumerate(days) if day >= incident["start"]), len(days))
    mask = np.array(exclusions(days, incidents)[incident["metric"]], dtype=bool)
    end = incident["end"] or OPEN_END
    after: list[str] = []
    if incident["end"] and incident["status"] != "persistent":
        later = [i["start"] for i in incidents if i["metric"] == incident["metric"] and i["status"] != "dismissed"
                 and i["start"] > incident["end"]]
        limit = min(later) if later else None
        after = [day for day in days if incident["end"] < day and (limit is None or day < limit)][:AFTER_DAYS]
    return {"before": [days[j] for j in baseline_bins(first, mask, cfg.baseline_window)],
            "during": [day for day in days if incident["start"] <= day <= end],
            "after": after}


def _on_days(frame: pd.DataFrame, days: Sequence[str]) -> pd.DataFrame:
    return frame[frame["day"].astype(str).isin(list(days))]


def _shares(values: pd.Series) -> list[tuple[str, float]]:
    """Each value with its share of `values` (nulls left out), largest first."""
    counts = values.dropna().astype(str).value_counts()
    return [(str(value), count / counts.sum()) for value, count in counts.items()]


TITLE_SHARE = 0.02  # of the turns during, for a version to be named in the title


def title_versions(versions: pd.Series) -> list[str]:
    """The versions the title names, oldest first: those behind at least TITLE_SHARE of the
    turns. A session left open on an old version runs a handful of turns weeks later, and
    naming it would widen the range past what the incident was about. The floor is low on
    purpose: the version an incident starts on may carry only a tenth of its turns, and
    dropping it would understate when the regression began."""
    known = versions.dropna().astype(str)
    if known.empty:
        return []
    shares = known.value_counts(normalize=True)
    named = sorted((version for version, share in shares.items() if share >= TITLE_SHARE), key=version_key)
    if not named:
        named = sorted(shares.index.astype(str), key=version_key)
    return [named[0]] if len(named) == 1 else [named[0], named[-1]]


def _version_rows(by_period: dict[str, pd.DataFrame], column: str) -> list[tuple[str, str, int, int]]:
    """(version, period, rows, rows where `column` holds) for each Claude Code version and
    period that has rows, ordered by version (`version_key`) then period."""
    rows = []
    frames = [frame.assign(period=name) for name, frame in by_period.items() if len(frame)]
    if frames:
        both = pd.concat(frames)
        both = both[both["version"].notna()]
        for version in sorted(both["version"].astype(str).unique(), key=version_key):
            for name in PERIODS:
                group = both[(both["version"].astype(str) == version) & (both["period"] == name)]
                if len(group):
                    rows.append((version, name, len(group), int(group[column].astype(float).sum())))
    return rows


def _cache_facts(responses: pd.DataFrame, turns: pd.DataFrame, periods: dict[str, list[str]]) -> dict[str, Any]:
    prompts = turns[turns["prompt_within_ttl"].astype(bool)]
    by_period = {name: _on_days(prompts, periods[name]) for name in PERIODS}
    during = by_period["during"]
    facts: dict[str, Any] = {
        "counts": {name: (int(rows["is_miss"].astype(bool).sum()), len(rows)) for name, rows in by_period.items()},
        "span": title_versions(during["version"]) if "version" in during else [],
        "ratios": {name: float(rows["cache_read_ratio"].mean()) if len(rows) else None
                   for name, rows in by_period.items()},
        "versions": _version_rows(by_period, "is_miss"),
        "missed": None, "reasons": None, "pauses": None, "loops": []}
    missed = during[during["is_miss"].astype(bool)]
    if len(missed):
        facts["missed"] = {"turns": len(missed),
                           "read": missed["cache_read"].quantile([0.5, 0.25, 0.75]).round().astype(int).tolist(),
                           "wrote": missed["cache_creation"].quantile([0.5, 0.25, 0.75]).round().astype(int).tolist()}
    # Counted over every judged response of the period, not over the new-prompt turns
    # the rest of this draft is about: Claude Code records a reason on any response
    # whose prompt did not match what it had cached.
    over = {name: _on_days(turns, periods[name]) for name in PERIODS}
    reasons = {name: reason_counts(rows) for name, rows in over.items()}
    if reasons["during"]:
        shown = [name for name in PERIODS if periods[name]]
        named = sorted({reason for counts in reasons.values() for reason in counts},
                       key=lambda reason: (-reasons["during"].get(reason, 0), reason))
        facts["reasons"] = {"shown": shown,
                            "rows": [(reason, [(reasons[name].get(reason, 0), len(over[name])) for name in shown])
                                     for reason in named]}
    if len(during):
        pauses, low = [], -1.0
        for high in PAUSE_BOUNDS:
            pause = during[(during["gap_seconds"] > low) & (during["gap_seconds"] <= high)]
            pauses.append((high, len(pause), int(pause["is_miss"].astype(bool).sum())))
            low = float(high)
        facts["pauses"] = pauses
    loops = loop_turns(responses, "main")
    for name in PERIODS:
        rows = _on_days(loops, periods[name])
        if len(rows):
            facts["loops"].append((name, int(rows["is_loop_miss"].sum()), len(rows)))
    return facts


def _haiku_facts(turns: pd.DataFrame, periods: dict[str, list[str]]) -> dict[str, Any]:
    by_period = {name: _on_days(turns, periods[name]) for name in PERIODS}
    during = by_period["during"]
    return {"counts": {name: (int(rows["is_haiku"].sum()), len(rows)) for name, rows in by_period.items()},
            "span": title_versions(during["version"]) if "version" in during else [],
            "versions": _version_rows(by_period, "is_haiku")}


def os_text() -> str:
    """"macOS 26.5.2" on a Mac, else the system and its release."""
    if sys.platform == "darwin" and platform.mac_ver()[0]:
        return DRAFT_LINES["macos"].format(version=platform.mac_ver()[0])
    return DRAFT_LINES["system"].format(system=platform.system(), release=platform.release()).strip()


def _environment_facts(during: pd.DataFrame, os_name: str, thread: str = "main") -> dict[str, Any]:
    entrypoints = _shares(during["entrypoint"]) if "entrypoint" in during and len(during) else []
    settings = {}
    for column, _, _ in DRAFT_SETTINGS:
        shares = _shares(during[column]) if column in during and len(during) else []
        settings[column] = shares[0] if shares else None
    return {"versions": sorted(during["version"].dropna().astype(str).unique(), key=version_key),
            "entrypoints": [name for name, _ in entrypoints], "models": _shares(during["model"]) if len(during) else [],
            "settings": settings, "os": os_name, "ccdrift": __version__, "thread": thread}


def draft_facts(responses: pd.DataFrame, incident: dict[str, Any], incidents: Sequence[dict[str, Any]],
                changelog: dict[str, list[str]], today: date, cfg: DetectorConfig,
                os_name: str) -> Optional[dict[str, Any]]:
    """What the draft says, as numbers (texts.draft_text words them): the incident and its
    periods, the counts before, during and after, by version, and for a cache incident what
    a missed turn looks like, why the cache missed, the pause before the prompt and the
    tool-loop turns, then the release notes, the environment and the rule. None when the
    history holds no judged days during the incident."""
    turns = judged_turns(responses, today)
    periods = draft_periods(turns, incident, incidents, cfg)
    if not periods["during"]:
        return None
    metric = incident["metric"]
    facts = (_cache_facts(responses, turns, periods) if metric == "cache_ratio" else _haiku_facts(turns, periods))
    versions = incident["versions"] or incident_versions(turns, incident)
    first_days = periods["during"][:RECOVERY_BINS]
    quoted = note_versions(turns, versions, days_before(incident["start"], 7),
                           first_days[-1] if first_days else incident["start"])
    facts.update({"kind": metric, "metric": metric, "incident": incident, "periods": periods,
                  "cost": incident_cost(turns, incident, incidents, cfg),
                  "notes": release_notes(changelog, quoted, TOPIC_OF[metric]),
                  "environment": _environment_facts(_on_days(turns, periods["during"]), os_name),
                  "method": {"metric": metric, "bins": cfg.deviant_bins, "window": cfg.flag_window,
                             "cutoff": cfg.metric_z_thresholds.get(metric, cfg.z_threshold),
                             "recovery": RECOVERY_BINS, "baseline": cfg.baseline_window}})
    return facts


def draft_markdown(responses: pd.DataFrame, incident: dict[str, Any], incidents: Sequence[dict[str, Any]],
                   changelog: dict[str, list[str]], today: date, cfg: DetectorConfig,
                   os_name: str) -> Optional[str]:
    """The draft: a title line, a blank line and the body's sections. None when the
    history holds no judged days during the incident."""
    facts = draft_facts(responses, incident, incidents, changelog, today, cfg, os_name)
    return None if facts is None else draft_text(facts)


def _days_apart(one: str, other: str) -> int:
    return abs((date.fromisoformat(one) - date.fromisoformat(other)).days)


def hook_groups(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """state["hook_changes"], one record per stream, as the alerts they make up: the records
    of one thread and direction whose `since` lies within MERGE_DAYS of the group's first,
    as hookcover.merged groups changes. Every record counts, whether it alerted or was
    recorded quietly on the first check after upgrading. Each group carries its thread,
    direction, first `since` and records."""
    groups: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda r: (r["thread"], r["direction"], r["since"], r["stream"])):
        last = groups[-1] if groups else None
        if (last is None or (last["thread"], last["direction"]) != (record["thread"], record["direction"])
                or _days_apart(record["since"], last["since"]) > MERGE_DAYS):
            last = {"thread": record["thread"], "direction": record["direction"], "since": record["since"],
                    "records": []}
            groups.append(last)
        last["records"].append(record)
    return groups


def find_alert(records: Sequence[dict[str, Any]], day: Optional[str] = None) -> Optional[dict[str, Any]]:
    """The alert record that starts on `day`, the UTC date its `since` opens with; without
    `day`, the latest. None when there is none."""
    matching = [record for record in records if day is None or str(record["since"])[:10] == day]
    return max(matching, key=lambda record: str(record["since"])) if matching else None


def _quartiles(values: pd.Series) -> list[int]:
    return values.quantile([0.5, 0.25, 0.75]).round().astype(int).tolist()


def _loop_facts(tables: Tables, warning: dict[str, Any], today: date) -> Optional[dict[str, Any]]:
    """What a tool-loop draft says: the turns of the warning's stream on the BASE_DAYS
    days that gave its usual rate (those before the WINDOW_DAYS it ran over, taken to end
    on the day of the alarm), the rise it warned about, from its first turn to the alarm,
    and up to AFTER_DAYS complete days after it. None when the history no longer holds
    the rise."""
    stream, responses = warning["stream"], tables.responses
    turns = loop_turns(responses, stream)
    since, at = pd.Timestamp(warning["since"]), pd.Timestamp(warning["at"])
    first = date.fromisoformat(warning["at"][:10]) - timedelta(days=WINDOW_DAYS - 1)
    days = turns["day"]
    frames = {"before": turns[(days >= (first - timedelta(days=BASE_DAYS)).isoformat()) & (days < first.isoformat())],
              "during": turns[(turns["timestamp"] >= since) & (turns["timestamp"] <= at)]}
    later = sorted(days[(days > warning["at"][:10]) & (days < today.isoformat())].unique())[:AFTER_DAYS]
    frames["after"] = turns[days.isin(later)]
    during = frames["during"]
    if during.empty:
        return None
    periods = {name: sorted(frame["day"].unique()) for name, frame in frames.items()}
    missed = during[during["is_loop_miss"]]
    main = responses["main_thread"].astype(bool)
    stamps = responses["timestamp"]
    in_rise = responses[(main if stream == "main" else ~main) & outside_sdk(responses)
                        & (stamps >= since) & (stamps <= at)]
    setting = LOOP_SETTINGS[stream]
    return {"counts": {name: (int(frame["is_loop_miss"].sum()), len(frame)) for name, frame in frames.items()},
            "periods": periods, "span": title_versions(during["version"]),
            "versions": _version_rows(frames, "is_loop_miss"),
            "missed": {"turns": len(missed), "wrote": _quartiles(missed["cache_creation"])} if len(missed) else None,
            "environment_rows": in_rise, "thread": stream, "topic": "cache",
            "method": {"kind": "loop_warnings", "stream": stream, "window": WINDOW_DAYS, "base": BASE_DAYS,
                       "gap": LOOP_GAP_SECONDS, "share": LOOP_MISS_SHARE,
                       "setting": setting._asdict() if setting else None},
            "notes_from": (warning["versions"], days_before(warning["since"][:10], 7), warning["at"][:10])}


def _cut_facts(tables: Tables, episode: dict[str, Any], today: date) -> Optional[dict[str, Any]]:
    """What a cut-short draft says, over main-thread responses on active days: the
    BEFORE_DAYS before the episode's day that it was compared with, the run it opened (its
    day and each active day after it still at or above CUT_SHARE, unbroken) and up to
    AFTER_DAYS active days after the run. None when the history no longer holds its day."""
    turns = judged_turns(tables.responses, today)
    counts = failure_counts(judged_failures(tables.failures, today), turns)
    active = counts[counts["responses"] >= ACTIVE_RESPONSES].reset_index(drop=True)
    days, since = active["day"].astype(str), episode["since"]
    if since not in set(days):
        return None
    earliest = (date.fromisoformat(since) - timedelta(days=BEFORE_DAYS)).isoformat()
    run = []
    for day, share in zip(days[days >= since], cut_shares(active[days >= since])):
        if share < CUT_SHARE:
            break
        run.append(day)
    periods = {"before": sorted(days[(days < since) & (days >= earliest)]), "during": run,
               "after": sorted(days[days > run[-1]])[:AFTER_DAYS]}
    stop = turns["stop_reason"].astype("string")
    marked = turns.assign(cut=stop.isin(["max_tokens", "refusal"]).fillna(False))
    frames = {name: _on_days(marked, periods[name]) for name in PERIODS}
    by_day = active.set_index(days)
    counts_of = {name: (int(by_day.loc[periods[name], ["truncated", "refused"]].to_numpy().sum()),
                        int(by_day.loc[periods[name], "responses"].sum())) for name in PERIODS}
    stops = {name: (int(by_day.loc[periods[name], "truncated"].sum()), int(by_day.loc[periods[name], "refused"].sum()))
             for name in PERIODS}
    return {"counts": counts_of, "periods": periods, "stops": stops,
            "span": title_versions(frames["during"]["version"]), "versions": _version_rows(frames, "cut"),
            "environment_rows": frames["during"], "thread": "main", "topic": "errors",
            "method": {"kind": "cut_short", "floor": CUT_FLOOR, "share": CUT_SHARE, "ratio": CUT_RATIO,
                       "usual": CUT_USUAL, "active": ACTIVE_RESPONSES, "days": BEFORE_DAYS,
                       "min_days": MIN_BEFORE_DAYS},
            "notes_from": (versions_text(turns, episode["days"]), days_before(since, 7), episode["days"][-1])}


def _median_rows(frames: dict[str, pd.DataFrame]) -> list[tuple[str, str, int, float, float]]:
    """(version, period, sessions, median start tokens, median ratio to the project's
    level) for each Claude Code version and period with sessions, ordered by version
    (`version_key`) then period."""
    rows = []
    for version in sorted({str(v) for frame in frames.values() for v in frame["version"].dropna()}, key=version_key):
        for name, frame in frames.items():
            group = frame[frame["version"].astype(str) == version]
            if len(group):
                rows.append((version, name, len(group), float(group["prompt_tokens"].median()),
                             float(group["ratio"].median())))
    return rows


def _builtin_tools(what: Optional[dict[str, Any]]) -> dict[str, list[str]]:
    """The built-in tools a session-start comparison found added or removed, by name: the
    tool definitions and deferred tools that aren't an MCP server's. Claude Code names
    those; a draft names nothing else it finds."""
    found: dict[str, list[str]] = {}
    for word in ("added", "removed") if what else ():
        names = sorted({name for part in ("tools", "deferred") for name in what[word].get(part, [])
                        if not name.startswith(MCP_TOOL)})
        if names:
            found[word] = names
    return found


def step_ratios(starts: pd.DataFrame, since: str) -> pd.DataFrame:
    """The session starts of projects with at least MIN_PROJECT_SESSIONS sessions before
    `since`, each with its ratio to its project's level when the step began: the median of
    that project's last PROJECT_BASELINE sessions before it. A fixed level, unlike
    ratio_starts' rolling one, which catches up with a step within PROJECT_BASELINE
    sessions and would read every later session as usual."""
    before = starts[starts["day"].astype(str) < since]
    levels = {project: statistics.median(rows["prompt_tokens"].astype(float).tolist()[-PROJECT_BASELINE:])
              for project, rows in before.groupby(before["project"].astype(str), sort=True)
              if len(rows) >= MIN_PROJECT_SESSIONS}
    known = starts[starts["project"].astype(str).isin(levels)]
    level = known["project"].astype(str).map(levels).astype(float)
    return known.assign(ratio=known["prompt_tokens"].astype(float) / level)


def _start_facts(tables: Tables, record: dict[str, Any], today: date) -> Optional[dict[str, Any]]:
    """What a session-start draft says: each project's sessions against its level when the
    step began (step_ratios) over the SIDE_DAYS before the step and from it through SIDE_DAYS
    after its window, which is what the check compared projects over, and what the sessions
    started with, compared as the check compared it. The step is found again as the check
    found it (found_changes over sessions.ratio_starts), by its first day. None when the
    history no longer shows it."""
    starts = session_starts(tables.responses)
    starts = starts[starts["day"].astype(str) < today.isoformat()].reset_index(drop=True)
    change = next((found for found in found_changes(ratio_starts(starts)) if found.since == record["since"]), None)
    if change is None:
        return None
    rated = step_ratios(starts, change.since)
    days = rated["day"].astype(str)
    earliest = (date.fromisoformat(change.since) - timedelta(days=SIDE_DAYS)).isoformat()
    latest = (date.fromisoformat(change.until) + timedelta(days=SIDE_DAYS)).isoformat()
    frames = {"before": rated[(days >= earliest) & (days < change.since)],
              "after": rated[(days >= change.since) & (days <= latest)]}
    what = compare_components(tables.components, change.window_files, change.baseline_files)
    turns = judged_turns(tables.responses, today)
    periods = {name: sorted(frame["day"].astype(str).unique()) for name, frame in frames.items()}
    return {"periods": periods, "up": change.up,
            "medians": {name: (len(frame), float(frame["prompt_tokens"].median()) if len(frame) else 0.0,
                               float(frame["ratio"].median()) if len(frame) else 0.0) for name, frame in frames.items()},
            "span": title_versions(frames["after"]["version"]), "versions": _median_rows(frames),
            "what": what, "builtin": _builtin_tools(what),
            "environment_rows": _on_days(turns, periods["after"]), "thread": "main", "topic": "context",
            "method": {"kind": "context_changes", "window": WINDOW, "baseline": BASELINE, "min_baseline": MIN_BASELINE,
                       "change": CHANGE, "side": SIDE, "project_baseline": PROJECT_BASELINE,
                       "min_project": MIN_PROJECT_SESSIONS, "side_days": SIDE_DAYS},
            "notes_from": (versions_text(turns, record["days"]), days_before(record["since"], 7), record["days"][-1])}


def _call_rows(coverage: pd.DataFrame, files: Sequence[str]) -> dict[tuple[str, str], tuple[int, int]]:
    """(calls, hooked) by (event, tool) over the coverage rows of `files`, each MCP
    server's tools folded into one "mcp__" tool."""
    rows = coverage[coverage["source_file"].astype(str).isin(list(files))]
    tool = rows["tool"].astype(str)
    rows = rows.assign(tool=tool.where(~tool.str.startswith(MCP_TOOL), MCP_TOOL))
    sums = rows.groupby(["event", "tool"], sort=True)[["calls", "hooked"]].sum()
    return {(str(event), str(name)): (int(calls), int(hooked))
            for (event, name), calls, hooked in zip(sums.index, sums["calls"], sums["hooked"])}


def _call_table(coverage: pd.DataFrame, files: dict[str, list[str]]) -> list[tuple[str, str, dict[str, tuple[int, int]]]]:
    """(event, tool, {period: (calls, hooked)}) for every event and tool the transcripts of
    any period called, as HOOK_EVENTS then tool order with the MCP tools last; a period
    that never called one has no entry for it."""
    periods = {name: _call_rows(coverage, names) for name, names in files.items()}
    keys = sorted({key for rows in periods.values() for key in rows},
                  key=lambda key: (HOOK_EVENTS.index(key[0]), key[1] == MCP_TOOL, key[1]))
    return [(event, tool, {name: rows[(event, tool)] for name, rows in periods.items() if (event, tool) in rows})
            for event, tool in keys]


def _hook_facts(tables: Tables, group: dict[str, Any], today: date) -> Optional[dict[str, Any]]:
    """What a hook-coverage draft says: the CLI transcripts of the change's thread and
    projects, each judged hooked or not per hook event and tool as the rule judges them
    (transcript_states at the shipped SETTING), over the SIDE_DAYS before the change and
    from it through SIDE_DAYS after its window, and their tool calls by event and tool.
    The change is found again as the check found it (hookcover.merged), the one of the
    group's thread and direction starting nearest the group, within MERGE_DAYS. None when
    the history no longer shows it."""
    coverage = tables.hook_coverage
    states = transcript_states(coverage, today, SETTING.min_calls)
    found = [alert for alert in merged(stream_changes(states, SETTING))
             if (alert["thread"], alert["direction"]) == (group["thread"], group["direction"])
             and _days_apart(alert["since"], group["since"]) <= MERGE_DAYS]
    if not found:
        return None
    alert = min(found, key=lambda change: _days_apart(change["since"], group["since"]))
    rows = states[(states["thread"] == alert["thread"]) & states["project"].isin(alert["projects"])
                  & states["event"].isin(alert["events"]) & states["tool"].isin(alert["tools"])]
    days = rows["day"].astype(str)
    earliest = (date.fromisoformat(alert["since"]) - timedelta(days=SIDE_DAYS)).isoformat()
    latest = (date.fromisoformat(alert["until"]) + timedelta(days=SIDE_DAYS)).isoformat()
    frames = {"before": rows[(days >= earliest) & (days < alert["since"])],
              "after": rows[(days >= alert["since"]) & (days <= latest)]}
    by_version = {name: frame.explode("versions").rename(columns={"versions": "version"}).assign(
        on=lambda f: f["on"].astype(bool)) for name, frame in frames.items()}
    files = {name: sorted(frame["source_file"].astype(str).unique()) for name, frame in frames.items()}
    main = tables.responses["main_thread"].astype(bool)
    thread = alert["thread"]
    after_rows = tables.responses[(main if thread == "main" else ~main) & outside_sdk(tables.responses)
                                  & tables.responses["source_file"].astype(str).isin(files["after"])]
    turns = judged_turns(tables.responses, today)
    tools = coverage.loc[coverage["source_file"].astype(str).isin(files["before"] + files["after"]), "tool"].astype(str)
    servers = {tool[len(MCP_TOOL):] for tool in tools if tool.startswith(MCP_TOOL)}
    changed = coverage[coverage["event"].isin(alert["events"]) & coverage["tool"].isin(alert["tools"])]
    ran = {name: {event: tuple(sum(pair[i] for (e, _), pair in rows.items() if e == event) for i in (0, 1))
                  for event in alert["events"]}
           for name, rows in ((name, _call_rows(changed, names)) for name, names in files.items())}
    return {"periods": {name: sorted(frame["day"].astype(str).unique()) for name, frame in frames.items()},
            "counts": {name: (int(frame["on"].astype(bool).sum()), len(frame)) for name, frame in frames.items()},
            "transcripts": {name: len(names) for name, names in files.items()},
            "calls": _call_table(coverage, files), "ran": ran,
            "change": {key: alert[key] for key in ("thread", "direction", "since", "events")},
            "projects": len(alert["projects"]), "servers": len(servers),
            "span": title_versions(by_version["after"]["version"]),
            "versions": _version_rows(by_version, "on"), "environment_rows": after_rows, "thread": thread,
            "topic": "hooks",
            "method": {"kind": "hook_changes", "window": SETTING.window, "baseline": SETTING.baseline,
                       "agree": SETTING.agree, "min_calls": SETTING.min_calls, "merge": MERGE_DAYS},
            "notes_from": (versions_text(turns, alert["days"]), days_before(alert["since"], 7), alert["days"][-1])}


ALERT_FACTS = {"context_changes": _start_facts, "hook_changes": _hook_facts, "loop_warnings": _loop_facts,
               "cut_short": _cut_facts}
# How a kind's records become the alerts a draft is about: hook coverage records one
# change per stream, and the check alerts on them grouped.
ALERT_RECORDS = {"hook_changes": hook_groups}


def alert_facts(tables: Tables, kind: str, alert: dict[str, Any], changelog: dict[str, list[str]],
                today: date, os_name: str) -> Optional[dict[str, Any]]:
    """What the draft about `alert`, a record of state key `kind`, says: its kind's facts
    (ALERT_FACTS, from the history's tables), then the release notes and the environment
    every draft carries. A kind's facts name the versions and days whose release notes the
    check quoted with the alert (`notes_from`), the rows the environment describes and the
    thread they ran on. None when the history no longer holds what the alert was about."""
    facts = ALERT_FACTS[kind](tables, alert, today)
    if facts is None:
        return None
    named, first, last = facts.pop("notes_from")
    quoted = note_versions(judged_turns(tables.responses, today), named, first, last)
    facts.update({"kind": kind, "alert": alert,
                  "notes": release_notes(changelog, quoted, facts.pop("topic")),
                  "environment": _environment_facts(facts.pop("environment_rows"), os_name, facts.pop("thread"))})
    return facts


def run_draft(source: Path, state_path: Path, metric: str, start: Optional[str] = None,
              today: Optional[date] = None, cfg: Optional[DetectorConfig] = None,
              os_name: Optional[str] = None) -> int:
    """Print a GitHub issue draft about the incident of `metric` starting on `start`, or
    the latest, from the history. It saves no state and creates or claims no history
    store; like `ccdrift report`, it brings a store the check has claimed up to date.
    Nothing is sent."""
    try:
        state = load_state(state_path)
    except (OSError, ValueError) as exc:
        print(COMMAND_LINES["state_unreadable"].format(path=state_path, error=exc), file=sys.stderr)
        return 1
    if metric in ALERT_NAMES:
        return _run_alert_draft(source, state_path, state, metric, start, today, os_name)
    incidents = state["incidents"]
    incident = find_incident(incidents, metric, start)
    if incident is None:
        name = SHORT_NAMES[metric]
        print(DRAFT_LINES["no_incident_on"].format(name=name, start=start) if start
              else DRAFT_LINES["no_incident"].format(name=name), file=sys.stderr)
        return 2
    try:
        responses = load_history(source, state_path, claim=False).responses
    except HistoryError as exc:
        print(exc, file=sys.stderr)
        return 1
    if responses.empty:
        print(no_transcripts_message(source), file=sys.stderr)
        return 2
    today = today or datetime.now(timezone.utc).date()
    text = draft_markdown(responses, incident, incidents, load_changelog(changelog_path(source)), today,
                          cfg or DetectorConfig(), os_name or os_text())
    if text is None:
        print(DRAFT_LINES["no_days"].format(name=SHORT_NAMES[metric], start=incident["start"]), file=sys.stderr)
        return 2
    print(text, end="")
    return 0


def _run_alert_draft(source: Path, state_path: Path, state: dict[str, Any], kind: str, start: Optional[str],
                     today: Optional[date], os_name: Optional[str]) -> int:
    """run_draft for an alert that is not an incident: the record of `kind` starting on
    `start`, or the latest."""
    name = ALERT_NAMES[kind]
    alert = find_alert(ALERT_RECORDS.get(kind, list)(state[kind]), start)
    if alert is None:
        print(DRAFT_LINES["no_alert_on"].format(name=name, start=start) if start
              else DRAFT_LINES["no_alert"].format(name=name), file=sys.stderr)
        return 2
    try:
        tables = load_history(source, state_path, claim=False)
    except HistoryError as exc:
        print(exc, file=sys.stderr)
        return 1
    if tables.responses.empty:
        print(no_transcripts_message(source), file=sys.stderr)
        return 2
    today = today or datetime.now(timezone.utc).date()
    facts = alert_facts(tables, kind, alert, load_changelog(changelog_path(source)), today, os_name or os_text())
    if facts is None:
        print(DRAFT_LINES["no_alert_days"].format(name=name, start=str(alert["since"])[:10]), file=sys.stderr)
        return 2
    print(draft_text(facts), end="")
    return 0
