"""The daily check: follow incidents from their first flagged day until they
recover, and alert when the check fails or can't compute the cache metric."""

from __future__ import annotations

import copy
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ccdrift.changelog import changelog_path, days_before, load_changelog, new_versions, note_lines, release_notes
from ccdrift.detector import DetectorConfig
from ccdrift.digest import digest_due, digest_week, weekly_digest
from ccdrift.early import early_message, early_warning
from ccdrift.fields import field_gaps, gap_message
from ccdrift.history import load_history
from ccdrift.hooks import failure_message, hook_failures, judged_hook_runs
from ccdrift.incidents import describe, incident_cost, incident_versions, update_incidents, versions_text
from ccdrift.logs import judged_turns, no_transcripts_message
from ccdrift.notify import notify, run_exec
from ccdrift.sessions import context_alerts, context_message, session_starts
from ccdrift.settings import change_message, setting_changes
from ccdrift.state import ccdrift_home, load_state, record_run, save_state, state_lock

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
    reported before; it is recorded in state["blank_cache"], and its last day in
    state["blank_cache_seen"]. Every Claude Code response reads or writes the prompt
    cache, so such days mean the parser has lost track of it."""
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
    # A stretch reaching back to the earliest day the check read may have begun before
    # it: it is the one already reported when an earlier run saw it go on to that day,
    # or, in a state from before ccdrift kept that day, when one was reported earlier.
    seen = state.get("blank_cache_seen")
    continues = start == 0 and (seen >= first if seen else any(day < first for day in state["blank_cache"]))
    state["blank_cache_seen"] = str(stretch.index[-1])
    if first in state["blank_cache"] or continues:
        return None
    state["blank_cache"].append(first)
    prompts = int(turns.loc[turns["day"].isin(stretch.index), "new_prompt"].sum())
    return {"first": first, "days": len(stretch), "responses": int(stretch["responses"].sum()),
            "prompts": prompts}


# How much history a check reads, so hourly runs don't slow down as the history grows:
# the last HISTORY_DAYS days, or the last HISTORY_ACTIVE_DAYS days with at least
# ACTIVE_DAY_RESPONSES main-thread responses when those reach further back (after a
# break, or for occasional use), reaching back HISTORY_DAYS before any incident whose
# cost it works out too. The longest look back is a flag within the last 14 days,
# judged against 14 active days that skip an incident of up to 30 days.
HISTORY_DAYS = 90
HISTORY_ACTIVE_DAYS = 60
ACTIVE_DAY_RESPONSES = 20


def _needs_cost(incident: dict[str, Any]) -> bool:
    """Whether the check works out an incident's cost and versions for `ccdrift status`:
    on every run while it is open, and once after it is added, closed or dismissed by
    hand, since its days don't change after that."""
    if incident["status"] == "open":
        return True
    by_hand = incident["source"] == "user" or incident["closed_by"] == "user"
    return by_hand and (incident.get("costed_on") or "") < (incident["closed_on"] or "")


def history_start(incidents: list[dict[str, Any]], today: date) -> str:
    """The UTC day HISTORY_DAYS before today, or before the start of the earliest
    incident whose cost the check works out."""
    starts = [incident["start"] for incident in incidents if _needs_cost(incident)]
    return days_before(min([today.isoformat(), *starts]), HISTORY_DAYS)


def _note_versions(turns: pd.DataFrame, named: list[str], first_day: str, last_day: str) -> list[str]:
    """The versions whose release notes an alert quotes: those its message names
    ("2.1.267 (since 09-10)"), then the others first seen from `first_day` to
    `last_day`, newest first."""
    new = new_versions(turns, first_day, last_day)
    return list(dict.fromkeys([text.split(" ")[0] for text in named] + new[::-1]))


def _change_notes(turns: pd.DataFrame, changelog: dict[str, list[str]], change: dict[str, Any],
                  topic: str) -> tuple[list[str], list[str]]:
    """For a change seen on `change["days"]` from `change["since"]`: the versions behind
    those days, which its message names, and the log lines quoting release notes on
    `topic` from those and the other versions first seen from a week before it."""
    versions = versions_text(turns, change["days"])
    quoted = _note_versions(turns, versions, days_before(change["since"], 7), change["days"][-1])
    return versions, note_lines(release_notes(changelog, quoted, topic))


def _alerts(source: Path, state_path: Path, state: dict[str, Any], cfg: DetectorConfig,
            today: date, now: datetime, digest: bool) -> list[Alert]:
    """Everything that changed since the last run, in the order alerts go out;
    `state` is updated to match."""
    tables = load_history(source, state_path, claim=True, since=history_start(state["incidents"], today),
                          active_days=HISTORY_ACTIVE_DAYS, active_responses=ACTIVE_DAY_RESPONSES)
    df = tables.responses
    if df.empty:
        raise RuntimeError(no_transcripts_message(source))
    turns = judged_turns(df, today)
    changelog = load_changelog(changelog_path(source))
    incidents = state["incidents"]
    events = update_incidents(turns, state, today, cfg)
    alerts: list[Alert] = []
    for event in events:
        kind, title, message, details, named = describe(event, turns, incidents, cfg)
        notes = []
        if event.kind != "persistent" and event.days:
            first = days_before(event.incident["start"], 7) if event.kind == "flag" else event.incident["start"]
            notes = release_notes(changelog, _note_versions(turns, named, first, max(event.days)),
                                  TOPIC_OF[event.incident["metric"]])
        alerts.append((kind, title, message, details + note_lines(notes)))
    warning = early_warning(df, incidents, state, now)
    if warning:
        alerts.append(("early", "ccdrift: cache misses rising", early_message(warning, now), []))
    # `ccdrift status` reads only the state file, so it shows the cost and versions
    # saved here: for open incidents, and for incidents added or closed by hand, which
    # start without a cost or keep the one from before they were closed. describe()
    # has just refreshed the incidents it alerted about.
    described = {id(event.incident) for event in events}
    for incident in incidents:
        if id(incident) in described or not _needs_cost(incident):
            continue
        incident["cost"] = round(incident_cost(turns, incident, incidents, cfg))
        if incident["status"] != "open":
            incident["costed_on"] = today.isoformat()
        if incident["source"] == "user" and not incident["versions"]:
            incident["versions"] = incident_versions(turns, incident)
    for change in setting_changes(turns, state, today):
        versions, notes = _change_notes(turns, changelog, change, TOPIC_OF[change["setting"]])
        alerts.append(("setting", "ccdrift: setting changed", change_message(change, versions), notes))
    for change in context_alerts(session_starts(df), state, today):
        versions, notes = _change_notes(turns, changelog, change, "context")
        alerts.append(("context", "ccdrift: session start changed", context_message(change, versions), notes))
    for failure in hook_failures(judged_hook_runs(tables.hook_runs, today), state, today):
        versions, notes = _change_notes(turns, changelog, failure, "hooks")
        alerts.append(("hooks", "ccdrift: hooks failing", failure_message(failure, versions), notes))
    for gap in field_gaps(turns, state, today):
        notes = release_notes(changelog, [] if gap["version"] == "unknown" else [gap["version"]], "fields")
        alerts.append(("fields", "ccdrift: Claude Code stopped logging a field", gap_message(gap), note_lines(notes)))
    blank = blank_cache_stretch(turns, state)
    if blank:
        alerts.append(("blank_cache", "ccdrift can't compute the cache metric",
                       f"no usable cache values on {blank['days']} active days from {blank['first']} "
                       f"({blank['responses']} responses, {blank['prompts']} prompts recognised). "
                       "Claude Code's log format may have changed; run `ccdrift peek`.", []))
    week_start = digest_due(state, now) if digest else None
    if week_start is not None:
        state["digest_week"] = digest_week(now)
        alerts.append(("digest", "ccdrift: weekly summary", weekly_digest(turns, state, week_start), []))
    return alerts


def _run_on_state(source: Path, state_path: Path, cfg: DetectorConfig, today: date, started: datetime,
                  digest: bool) -> tuple[list[Alert], Optional[str]]:
    """The alerts of a run, or why it failed, with the run saved in the state file. A
    state file that can't be read is left as it is."""
    try:
        state = load_state(state_path)
    except (OSError, ValueError) as exc:
        traceback.print_exc()
        return [], f"can't read the state file {state_path}: {exc}"
    updated = copy.deepcopy(state)
    try:
        alerts = _alerts(source, state_path, updated, cfg, today, started, digest)
        record_run(updated, started, None)
        save_state(state_path, updated)
    except Exception as exc:
        traceback.print_exc()
        error = f"{type(exc).__name__}: {exc}"
        record_run(state, started, error)
        try:
            save_state(state_path, state)
        except OSError:
            pass
        return [], error
    return alerts, None


FAILURE_NOTICE_HOURS = 20


def run_check(source: Path, state_path: Path, cfg: Optional[DetectorConfig] = None,
              notify_user: bool = False, today: Optional[date] = None,
              exec_command: Optional[str] = None, now: Optional[datetime] = None,
              digest: bool = True) -> int:
    """Run the daily check and print a line per alert; with notify_user, also show a
    notification for each; with exec_command, also run it for each (see
    notify.run_exec). The state file is read once and written once, with how the
    run went, so `ccdrift status` can tell a broken check from a quiet week, and
    stays locked in between (see state.state_lock). A state file that can't be read
    is left as it is. Without `digest`, no weekly summary."""
    started = now or datetime.now().astimezone()
    stamp = started.strftime("%Y-%m-%d %H:%M")
    notice = state_path.with_name(state_path.name + ".last-failure-notice")

    def alert(kind: str, title: str, message: str, send: bool = True) -> None:
        print(f"[check {stamp}] {title}: {message}")
        if not send:
            return
        # The state already records the alerts as sent, so nothing may stop the rest.
        if notify_user:
            try:
                notify(title, message)
            except Exception as exc:
                print(f'[check {stamp}] notification failed for "{title}": {type(exc).__name__}: {exc}')
        if exec_command:
            try:
                failure = run_exec(exec_command, kind, title, message)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
            if failure:
                print(f'[check {stamp}] --exec failed for "{title}": {failure}')

    def failure_notice_due() -> bool:
        """A failure notifies and runs --exec at most once per FAILURE_NOTICE_HOURS
        until a run succeeds. A file next to the state file notes when it last did, so
        a state file that can't be read or saved doesn't notify on every run; when that
        file can't be read or written either, the failure notifies."""
        try:
            if started - datetime.fromisoformat(notice.read_text().strip()) < timedelta(hours=FAILURE_NOTICE_HOURS):
                return False
        except (OSError, ValueError, TypeError):
            pass
        try:
            notice.parent.mkdir(parents=True, exist_ok=True)
            notice.write_text(started.isoformat(timespec="seconds") + "\n")
        except OSError:
            pass
        return True

    try:
        with state_lock(state_path):
            alerts, failure = _run_on_state(source, state_path, cfg or DetectorConfig(),
                                            today or datetime.now(timezone.utc).date(), started, digest)
    except OSError as exc:
        traceback.print_exc()
        alerts, failure = [], f"can't read the state file {state_path}: {exc}"
    # Alerts go out once the lock is released: notifications and --exec take their time.
    if failure is not None:
        alert("failed", "ccdrift check failed", failure, failure_notice_due())
        return 1
    try:
        notice.unlink(missing_ok=True)
    except OSError:
        pass
    for kind, title, message, details in alerts:
        alert(kind, title, message)
        for detail in details:
            print(f"    {detail}")
    if not alerts:
        print(f"[check {stamp}] no alerts")
    return 0
