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

from datetime import date, timedelta
from typing import Any, Sequence

import pandas as pd

from ccdrift.detector import DetectorConfig
from ccdrift.incidents import Event, incident_cost, incident_versions, update_incidents
from ccdrift.logs import judged_turns
from ccdrift.texts import METRIC_WORDS, MOVES, PERSISTENT_DAYS, cost_text

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
