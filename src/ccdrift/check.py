"""The daily check: follow incidents from their first flagged day until they
recover, and alert when the check fails or can't compute the cache metric."""

from __future__ import annotations

import copy
import traceback
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ccdrift.changelog import changelog_path, days_before, load_changelog, new_versions, note_lines, release_notes
from ccdrift.detector import DetectorConfig
from ccdrift.fields import field_gaps, gap_message
from ccdrift.history import load_history
from ccdrift.hooks import failure_message, hook_failures, judged_hook_runs
from ccdrift.incidents import describe, incident_cost, update_incidents, versions_text
from ccdrift.logs import judged_turns, no_transcripts_message
from ccdrift.notify import notify, run_exec
from ccdrift.sessions import context_alerts, context_message, session_starts
from ccdrift.settings import change_message, setting_changes
from ccdrift.state import ccdrift_home, load_state, record_run, save_state

# kind, title, message, and lines for the log only
Alert = tuple[str, str, str, list[str]]

# Which release notes explain an alert about each metric or setting.
TOPIC_OF = {"cache_ratio": "cache", "haiku_fraction": "haiku", "cache_tier": "cache", "effort": "effort",
            "subagent_model": "subagents"}


# A stretch of active days without usable cache values means the cache metric
# can't be computed, most likely because Claude Code's log format changed: it
# goes blank when prompts aren't recognised, and reads as all misses when cache
# usage isn't read.
CHECK_BLANK_DAYS = 3
CHECK_ACTIVE_RESPONSES = 50  # main-thread responses; the quietest of 29 real days had 81


def blank_cache_stretch(turns: pd.DataFrame, state: dict[str, Any],
                        days: int = CHECK_BLANK_DAYS,
                        active: int = CHECK_ACTIVE_RESPONSES) -> Optional[dict[str, Any]]:
    """The latest run of active days (at least `active` judged responses) on which no
    new-prompt turn has cache token counts, once it is `days` long and wasn't
    reported before; it is recorded in state["blank_cache"]. Every Claude Code
    response reads or writes the prompt cache, so such days mean the parser has
    lost track of it."""
    usable = turns["prompt_within_ttl"].astype(bool) & ((turns["cache_read"] + turns["cache_creation"]) > 0)
    per_day = pd.DataFrame({"responses": turns.groupby("day").size(),
                            "usable": usable.groupby(turns["day"]).sum()})
    per_day = per_day[per_day["responses"] >= active]
    blank = (per_day["usable"] == 0).to_numpy()
    if len(blank) < days or not blank[-days:].all():
        return None
    start = len(blank) - days
    while start > 0 and blank[start - 1]:
        start -= 1
    stretch = per_day.iloc[start:]
    first = str(stretch.index[0])
    if first in state["blank_cache"]:
        return None
    state["blank_cache"].append(first)
    prompts = int(turns.loc[turns["day"].isin(stretch.index), "new_prompt"].sum())
    return {"first": first, "days": len(stretch), "responses": int(stretch["responses"].sum()),
            "prompts": prompts}


def _alerts(source: Path, state_path: Path, state: dict[str, Any], cfg: DetectorConfig,
            today: date) -> list[Alert]:
    """Everything that changed since the last run, in the order alerts go out;
    `state` is updated to match."""
    tables = load_history(source, state_path, claim=True)
    df = tables.responses
    if df.empty:
        raise RuntimeError(no_transcripts_message(source))
    turns = judged_turns(df, today)
    changelog = load_changelog(changelog_path(source))
    incidents = state["incidents"]
    events = update_incidents(turns, state, today, cfg)
    alerts: list[Alert] = []
    for event in events:
        kind, title, message, details = describe(event, turns, incidents, cfg)
        notes = []
        if event.kind != "persistent" and event.days:
            first = days_before(event.incident["start"], 7) if event.kind == "flag" else event.incident["start"]
            notes = release_notes(changelog, new_versions(turns, first, max(event.days)),
                                  TOPIC_OF[event.incident["metric"]])
        alerts.append((kind, title, message, details + note_lines(notes)))
    # `ccdrift status` reads only the state file, so it shows the cost and versions
    # saved here: for open incidents, and for incidents added by hand, which start
    # without either. describe() has just refreshed the incidents it alerted about.
    described = {id(event.incident) for event in events}
    for incident in incidents:
        if id(incident) in described or (incident["status"] != "open" and incident["source"] != "user"):
            continue
        incident["cost"] = round(incident_cost(turns, incident, incidents, cfg))
        if incident["source"] == "user" and not incident["versions"]:
            days = turns["day"].astype(str)
            first_days = sorted(days[days.between(incident["start"], incident["end"])].unique())[:3]
            incident["versions"] = versions_text(turns, first_days)
    for change in setting_changes(turns, state, today):
        notes = release_notes(changelog, new_versions(turns, days_before(change["since"], 7), change["days"][-1]),
                              TOPIC_OF[change["setting"]])
        alerts.append(("setting", "ccdrift: setting changed",
                       change_message(change, versions_text(turns, change["days"])), note_lines(notes)))
    for change in context_alerts(session_starts(df), state, today):
        notes = release_notes(changelog, new_versions(turns, days_before(change["since"], 7), change["days"][-1]),
                              "context")
        alerts.append(("context", "ccdrift: session start changed",
                       context_message(change, versions_text(turns, change["days"])), note_lines(notes)))
    for failure in hook_failures(judged_hook_runs(tables.hook_runs, today), state, today):
        notes = release_notes(changelog, new_versions(turns, days_before(failure["since"], 7), failure["days"][-1]),
                              "hooks")
        alerts.append(("hooks", "ccdrift: hooks failing",
                       failure_message(failure, versions_text(turns, failure["days"])), note_lines(notes)))
    for gap in field_gaps(turns, state, today):
        notes = release_notes(changelog, [] if gap["version"] == "unknown" else [gap["version"]], "fields")
        alerts.append(("fields", "ccdrift: Claude Code stopped logging a field", gap_message(gap), note_lines(notes)))
    blank = blank_cache_stretch(turns, state)
    if blank:
        alerts.append(("blank_cache", "ccdrift can't compute the cache metric",
                       f"no usable cache values on {blank['days']} active days from {blank['first']} "
                       f"({blank['responses']} responses, {blank['prompts']} prompts recognised). "
                       "Claude Code's log format may have changed; run `ccdrift peek`.", []))
    return alerts


def run_check(source: Path, state_path: Path, cfg: Optional[DetectorConfig] = None,
              notify_user: bool = False, today: Optional[date] = None,
              exec_command: Optional[str] = None, now: Optional[datetime] = None) -> int:
    """Run the daily check and print a line per alert; with notify_user, also show a
    notification for each; with exec_command, also run it for each (see
    notify.run_exec). The state file is read once and written once, with how the
    run went, so `ccdrift status` can tell a broken check from a quiet week. A
    state file that can't be read is left as it is."""
    started = now or datetime.now().astimezone()
    stamp = started.strftime("%Y-%m-%d %H:%M")

    def alert(kind: str, title: str, message: str) -> None:
        print(f"[check {stamp}] {title}: {message}")
        if notify_user:
            notify(title, message)
        if exec_command:
            # The state already records the alerts as sent, so nothing may stop the rest.
            try:
                failure = run_exec(exec_command, kind, title, message)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
            if failure:
                print(f'[check {stamp}] --exec failed for "{title}": {failure}')

    try:
        state = load_state(state_path)
    except (OSError, ValueError) as exc:
        traceback.print_exc()
        alert("failed", "ccdrift check failed", f"can't read the state file {state_path}: {exc}")
        return 1
    updated = copy.deepcopy(state)
    try:
        alerts = _alerts(source, state_path, updated, cfg or DetectorConfig(),
                         today or datetime.now(timezone.utc).date())
        record_run(updated, started, None)
        save_state(state_path, updated)
    except Exception as exc:
        traceback.print_exc()
        record_run(state, started, f"{type(exc).__name__}: {exc}")
        try:
            save_state(state_path, state)
        except OSError:
            pass
        alert("failed", "ccdrift check failed", f"{type(exc).__name__}: {exc}")
        return 1
    for kind, title, message, details in alerts:
        alert(kind, title, message)
        for detail in details:
            print(f"    {detail}")
    if not alerts:
        print(f"[check {stamp}] no alerts")
    return 0
