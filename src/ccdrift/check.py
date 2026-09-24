"""The daily check: follow incidents from their first flagged day until they
recover, and alert when the check fails or can't compute the cache metric."""

from __future__ import annotations

import copy
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ccdrift.changelog import (TOPIC_OF, changelog_path, days_before, load_changelog, note_versions,
                               release_notes)
from ccdrift.detector import DetectorConfig
from ccdrift.digest import digest_due, digest_week, weekly_digest
from ccdrift.early import early_warning
from ccdrift.failures import cut_short, failing_requests, failure_counts, judged_failures
from ccdrift.fields import field_gaps, new_fields
from ccdrift.history import load_history
from ccdrift.hooks import hook_failures, judged_hook_runs
from ccdrift.incidents import (RECOVERY_BINS, describe, incident_cost, incident_versions, update_incidents,
                               versions_text)
from ccdrift.logs import judged_turns
from ccdrift.loops import STREAMS, loop_counts, loop_warning
from ccdrift.notify import notify, run_exec
from ccdrift.replay import REPLAY_SOURCE, first_run, replay_incidents
from ccdrift.sessions import context_alerts, rejudged, session_starts
from ccdrift.settings import setting_changes
from ccdrift.state import (CONTEXT_RULE, LOG_FILE, ccdrift_home, load_state, make_stream_private, record_run,
                           save_state, state_lock)
from ccdrift.texts import (ALERT_TITLES, CHECK_LINES, blank_cache_message, change_message, context_dropped_message,
                           context_message, cut_short_message, early_message, gap_message, history_message,
                           hook_failure_message, loop_message, new_fields_message, no_transcripts_message, note_lines,
                           requests_message, state_unreadable)

# kind, title, message, and lines for the log only
Alert = tuple[str, str, str, list[str]]

# Alerts that only go to the log: nothing changed that the owner can act on. A field
# that vanishes silences an alert and is worth a notification; a field that arrives
# breaks nothing, and arrives about once a week.
LOG_ONLY = frozenset({"context_dropped", "new_fields"})


# The alert kind of a tool-loop warning for each stream.
LOOP_KINDS = {"main": "loop", "subagent": "subagent_loop"}

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


def _change_notes(turns: pd.DataFrame, changelog: dict[str, list[str]], change: dict[str, Any],
                  topic: str) -> tuple[list[str], list[str]]:
    """For a change seen on `change["days"]` from `change["since"]`: the versions behind
    those days, which its message names, and the log lines quoting release notes on
    `topic` from those and the other versions first seen from a week before it."""
    versions = versions_text(turns, change["days"])
    quoted = note_versions(turns, versions, days_before(change["since"], 7), change["days"][-1])
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
    alerts: list[Alert] = []
    # A first check replays its history day by day instead, so a regression that began
    # more than 14 days ago is recorded too; its last replayed day is today.
    replaying = first_run(state)
    events = [] if replaying else update_incidents(turns, state, today, cfg)
    if replaying:
        replayed = replay_incidents(df, today, cfg, state)
        found = sorted((i for i in incidents if i["source"] == REPLAY_SOURCE), key=lambda i: i["start"])
        if found:
            notes = []
            judged_days = sorted(turns["day"].astype(str).unique())
            for incident in found:
                if incident["status"] != "persistent":
                    # As its flag alert would have: versions first seen from a week before
                    # the incident through its first RECOVERY_BINS days.
                    first_days = [day for day in judged_days if day >= incident["start"]][:RECOVERY_BINS]
                    quoted = note_versions(turns, incident["versions"], days_before(incident["start"], 7),
                                           first_days[-1])
                    notes += release_notes(changelog, quoted, TOPIC_OF[incident["metric"]])
            alerts.append(("history", ALERT_TITLES["history"],
                           history_message(found, str(turns["day"].min())), note_lines(notes)))
        # A regression still going is why someone installs ccdrift mid-flight, and the
        # summary above reads as history. It also gets the flag alert it would have had,
        # with the days and z values that opened it; the release notes stay on the summary
        # rather than being quoted twice in the same run.
        for _, event in replayed:
            if event.kind == "flag" and event.incident["status"] == "open":
                kind, title, message, details, _ = describe(event, turns, incidents, cfg)
                alerts.append((kind, title, message, details))
    for event in events:
        kind, title, message, details, named = describe(event, turns, incidents, cfg)
        notes = []
        if event.kind != "persistent" and event.days:
            first = days_before(event.incident["start"], 7) if event.kind == "flag" else event.incident["start"]
            notes = release_notes(changelog, note_versions(turns, named, first, max(event.days)),
                                  TOPIC_OF[event.incident["metric"]])
        alerts.append((kind, title, message, details + note_lines(notes)))
    warning = early_warning(df, incidents, state, now)
    if warning:
        alerts.append(("early", ALERT_TITLES["early"], early_message(warning, now), []))
    for stream in STREAMS:
        loop = loop_warning(df, stream, state, now)
        if loop:
            since, alarm_day = loop["since"][:10], loop["at"][:10]
            quoted = note_versions(turns, loop["versions"], days_before(since, 7), alarm_day)
            alerts.append((LOOP_KINDS[stream], ALERT_TITLES[LOOP_KINDS[stream]], loop_message(loop, now),
                           note_lines(release_notes(changelog, quoted, "cache"))))
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
        alerts.append(("setting", ALERT_TITLES["setting"], change_message(change, versions), notes))
    starts = session_starts(df)
    for change in context_alerts(starts, state, today):
        versions, notes = _change_notes(turns, changelog, change, "context")
        alerts.append(("context", ALERT_TITLES["context"], context_message(change, versions), notes))
    # A state written before the rule judged each project against itself may hold changes
    # that were only a move between projects. They are re-judged once, and the run says so
    # in its log without alerting: nothing changed for the owner to act on.
    if state.get("context_rule", 1) < CONTEXT_RULE:
        for record in rejudged(starts, state, today):
            alerts.append(("context_dropped", ALERT_TITLES["context_dropped"], context_dropped_message(record), []))
        state["context_rule"] = CONTEXT_RULE
    for failure in hook_failures(judged_hook_runs(tables.hook_runs, today), state, today):
        versions, notes = _change_notes(turns, changelog, failure, "hooks")
        alerts.append(("hooks", ALERT_TITLES["hooks"], hook_failure_message(failure, versions), notes))
    counts = failure_counts(judged_failures(tables.failures, today), turns)
    for episode in failing_requests(counts, state, today):
        versions, notes = _change_notes(turns, changelog, episode, "errors")
        alerts.append(("failed_requests", ALERT_TITLES["failed_requests"], requests_message(episode, versions), notes))
    for episode in cut_short(counts, state, today):
        versions, notes = _change_notes(turns, changelog, episode, "errors")
        alerts.append(("cut_short", ALERT_TITLES["cut_short"], cut_short_message(episode, versions), notes))
    for gap in field_gaps(turns, state, today):
        notes = release_notes(changelog, [] if gap["version"] == "unknown" else [gap["version"]], "fields")
        alerts.append(("fields", ALERT_TITLES["fields"], gap_message(gap), note_lines(notes)))
    for record in new_fields(tables.field_census, turns, state, today):
        alerts.append(("new_fields", ALERT_TITLES["new_fields"], new_fields_message(record), []))
    blank = blank_cache_stretch(turns, state)
    if blank:
        alerts.append(("blank_cache", ALERT_TITLES["blank_cache"], blank_cache_message(blank), []))
    week_start = digest_due(state, now) if digest else None
    if week_start is not None:
        state["digest_week"] = digest_week(now)
        alerts.append(("digest", ALERT_TITLES["digest"],
                       weekly_digest(turns, state, week_start, loop_counts(df, today), counts), []))
    return alerts


def _run_on_state(source: Path, state_path: Path, cfg: DetectorConfig, today: date, started: datetime,
                  digest: bool) -> tuple[list[Alert], Optional[str]]:
    """The alerts of a run, or why it failed, with the run saved in the state file. A
    state file that can't be read is left as it is."""
    try:
        state = load_state(state_path)
    except (OSError, ValueError) as exc:
        traceback.print_exc()
        return [], state_unreadable(state_path, exc)
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
    make_stream_private(sys.stdout, only=state_path.with_name(LOG_FILE))
    started = now or datetime.now().astimezone()
    notice = state_path.with_name(state_path.name + ".last-failure-notice")

    def alert(kind: str, title: str, message: str, send: bool = True) -> None:
        print(CHECK_LINES["alert"].format(at=started, title=title, message=message))
        if not send:
            return
        # The state already records the alerts as sent, so nothing may stop the rest.
        if notify_user:
            try:
                notify(title, message)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                print(CHECK_LINES["notify_failed"].format(at=started, title=title, error=error))
        if exec_command:
            try:
                failure = run_exec(exec_command, kind, title, message)
            except Exception as exc:
                failure = f"{type(exc).__name__}: {exc}"
            if failure:
                print(CHECK_LINES["exec_failed"].format(at=started, title=title, error=failure))

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
        alerts, failure = [], state_unreadable(state_path, exc)
    # Alerts go out once the lock is released: notifications and --exec take their time.
    if failure is not None:
        alert("failed", ALERT_TITLES["failed"], failure, failure_notice_due())
        return 1
    try:
        notice.unlink(missing_ok=True)
    except OSError:
        pass
    for kind, title, message, details in alerts:
        alert(kind, title, message, send=kind not in LOG_ONLY)
        for detail in details:
            print(CHECK_LINES["detail"].format(detail=detail))
    if not alerts:
        print(CHECK_LINES["no_alerts"].format(at=started))
    return 0
