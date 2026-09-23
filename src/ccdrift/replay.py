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
from ccdrift.texts import COMMAND_LINES, PERSISTENT_DAYS, REPLAY_LINES, SHORT_NAMES, incident_line

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


def _recorded_note(found: dict[str, Any], recorded: Sequence[dict[str, Any]], today: date) -> str:
    """How an incident the replay found stands against the recorded ones."""
    end = found["end"] or OPEN_END
    overlapping = [i for i in recorded if i["metric"] == found["metric"]
                   and i["start"] <= end and found["start"] <= (i["end"] or OPEN_END)]
    kept = [i for i in overlapping if i["status"] != "dismissed"]
    if overlapping:
        items = [REPLAY_LINES["overlap_item"].format(name=SHORT_NAMES[i["metric"]], start=i["start"],
                                                     end=i["end"] or REPLAY_LINES["now"])
                 for i in (kept or overlapping)]
        return REPLAY_LINES["overlap"].format(label=REPLAY_LINES["recorded" if kept else "dismissed"],
                                              incidents=", ".join(items))
    short = SHORT_NAMES[found["metric"]]
    if found["status"] == "persistent":
        return REPLAY_LINES["persistent"].format(days=PERSISTENT_DAYS)
    if found["status"] == "open":
        yesterday = (today - timedelta(days=1)).isoformat()
        return REPLAY_LINES["open"].format(name=short, start=found["start"], end=yesterday)
    return REPLAY_LINES["closed"].format(name=short, start=found["start"], end=found["end"])


def run_replay(source: Path, state_path: Path, today: Optional[date] = None,
               cfg: Optional[DetectorConfig] = None) -> int:
    """Print what the check would have followed in the whole history, replayed day by
    day on an empty state, against the incidents recorded in `state_path`. Nothing is
    recorded, no alert is sent, and a store no check has claimed isn't written."""
    try:
        recorded = load_state(state_path)["incidents"]
    except (OSError, ValueError) as exc:
        print(COMMAND_LINES["state_unreadable"].format(path=state_path, error=exc), file=sys.stderr)
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
        print(REPLAY_LINES["nothing"])
        return 0
    state = new_state()
    events = replay_incidents(responses, today, cfg, state)
    first = date.fromisoformat(str(turns["day"].min())) + timedelta(days=1)
    lines = [REPLAY_LINES["header"].format(first=first.isoformat(), today=today.isoformat()), REPLAY_LINES["quiet"], ""]
    for day, event in events:
        _, title, message, _, _ = describe(event, judged_turns(responses, date.fromisoformat(day)),
                                           state["incidents"], cfg)
        lines.append(REPLAY_LINES["event"].format(day=day, title=title, message=message))
    if not events:
        lines.append(REPLAY_LINES["none"])
    else:
        lines += ["", REPLAY_LINES["found"]]
        for incident in sorted(state["incidents"], key=lambda i: i["start"]):
            cost = incident_cost(turns, incident, state["incidents"], cfg)
            lines += [f"  {incident_line(incident, cost)}", f"    {_recorded_note(incident, recorded, today)}"]
    print("\n".join(lines))
    return 0
