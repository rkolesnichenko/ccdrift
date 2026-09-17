"""Replaying incident detection over past days, as if the check had run on each of
them. The first check does it, so a regression from before ccdrift ran is recorded, and
`ccdrift replay` shows what it finds on any install without changing anything.

Without it a first check opens incidents only for flags from the last 14 days (see
incidents.RECENT_DAYS): on the owner's logs it found nothing, although August held a
16M-token caching regression. Replayed day by day, that incident opens on Aug 22 from
Aug 18 and closes on Sep 7, as the daily check had followed it. A single pass without
the 14-day limit finds the same days, but it settles an incident by the date of the one
run, so it can call recovered what the daily check would have closed as persistent on
its 30th day."""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from ccdrift.detector import DetectorConfig
from ccdrift.history import HistoryError, load_history
from ccdrift.incidents import OPEN_END, Event, describe, incident_cost, incident_versions, update_incidents
from ccdrift.logs import judged_turns, no_transcripts_message
from ccdrift.state import load_state, new_state
from ccdrift.texts import METRIC_WORDS, MOVES, PERSISTENT_DAYS, SHORT_NAMES, cost_text, incident_line

REPLAY_SOURCE = "replay"


def first_run(state: dict[str, Any]) -> bool:
    """Whether no check has run successfully on `state` and nothing is recorded in it:
    no incident, found or added by hand, and no flag a version 1 state reported."""
    return state.get("last_ok") is None and not state["incidents"] and not state["reported"]


def replay_incidents(responses: pd.DataFrame, today: date, cfg: DetectorConfig,
                     state: dict[str, Any]) -> list[tuple[str, Event]]:
    """Follow incidents on `state` as the check would have on each UTC day from the day
    after the first judged day through `today`, each on the judged turns before it, and
    return every event with its day. Incidents it opens get source "replay" and, once
    the replay is done, their cost and versions over the judged turns before `today`."""
    turns = judged_turns(responses, today)
    if turns.empty:
        return []
    events = []
    day = date.fromisoformat(str(turns["day"].min())) + timedelta(days=1)
    while day <= today:
        for event in update_incidents(judged_turns(responses, day), state, day, cfg):
            if event.kind == "flag":
                event.incident["source"] = REPLAY_SOURCE
            events.append((day.isoformat(), event))
        day += timedelta(days=1)
    for incident in state["incidents"]:
        if incident["source"] == REPLAY_SOURCE:
            incident["cost"] = round(incident_cost(turns, incident, state["incidents"], cfg))
            incident["versions"] = incident_versions(turns, incident)
    return events


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


def _recorded_note(found: dict[str, Any], recorded: Sequence[dict[str, Any]], today: date) -> str:
    """How an incident the replay found stands against the recorded ones."""
    end = found["end"] or OPEN_END
    overlapping = [i for i in recorded if i["metric"] == found["metric"]
                   and i["start"] <= end and found["start"] <= (i["end"] or OPEN_END)]
    kept = [i for i in overlapping if i["status"] != "dismissed"]
    if overlapping:
        label = "recorded" if kept else "dismissed"
        return f"{label}: " + ", ".join(f"{SHORT_NAMES[i['metric']]} {i['start']}..{i['end'] or 'now'}"
                                        for i in (kept or overlapping))
    short = SHORT_NAMES[found["metric"]]
    if found["status"] == "persistent":
        return (f"not recorded: after {PERSISTENT_DAYS} days the check takes the new level as normal, "
                "so its days need no record")
    if found["status"] == "open":
        yesterday = (today - timedelta(days=1)).isoformat()
        return (f"not recorded: `ccdrift incident add {short} {found['start']}..{yesterday}` "
                "records its days so far")
    return f"not recorded: `ccdrift incident add {short} {found['start']}..{found['end']}` records it"


def run_replay(source: Path, state_path: Path, today: Optional[date] = None,
               cfg: Optional[DetectorConfig] = None) -> int:
    """Print what the check would have followed in the whole history, replayed day by
    day on an empty state, against the incidents recorded in `state_path`. Nothing is
    recorded, no alert is sent, and a store no check has claimed isn't written."""
    try:
        recorded = load_state(state_path)["incidents"]
    except (OSError, ValueError) as exc:
        print(f"Can't read the state file {state_path}: {exc}", file=sys.stderr)
        return 1
    try:
        responses = load_history(source, state_path, claim=False).responses
    except HistoryError as exc:
        print(exc, file=sys.stderr)
        return 1
    if responses.empty:
        print(no_transcripts_message(source), file=sys.stderr)
        return 2
    cfg = cfg or DetectorConfig()
    today = today or datetime.now(timezone.utc).date()
    turns = judged_turns(responses, today)
    if turns.empty:
        print("Nothing to replay: no complete UTC day with main-thread activity yet.")
        return 0
    state = new_state()
    events = replay_incidents(responses, today, cfg, state)
    first = date.fromisoformat(str(turns["day"].min())) + timedelta(days=1)
    lines = [f"Replaying the check day by day from {first.isoformat()} to {today.isoformat()} (UTC) on an empty "
             "state, incidents only.",
             "Nothing is recorded and no alert is sent.", ""]
    for day, event in events:
        _, title, message, _, _ = describe(event, judged_turns(responses, date.fromisoformat(day)),
                                           state["incidents"], cfg)
        lines.append(f"{day}  {title}: {message}")
    if not events:
        lines.append("No incidents: the check would have sent no incident alert over these days.")
    else:
        lines += ["", "Incidents the replay found:"]
        for incident in sorted(state["incidents"], key=lambda i: i["start"]):
            cost = incident_cost(turns, incident, state["incidents"], cfg)
            lines += [f"  {incident_line(incident, cost)}", f"    {_recorded_note(incident, recorded, today)}"]
    print("\n".join(lines))
    return 0
