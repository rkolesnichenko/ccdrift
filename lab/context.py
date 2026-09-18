"""Does judging each project against itself stop a move between projects from reading as
a change in context? (G12)

Until 0.8.0 every project's session starts were pooled, so the owner's logs produced one
alert — "context halved on 2026-09-10" — when all that happened was that work moved to two
newly created projects. The project that carried on was starting at the same size a week
later. The rule now measures each session against its own project's recent level.

The gate asks three things of the new rule and reports what the pooled one did beside it:

- a history where work simply moves to a smaller project holds no change (the pooled rule
  finds one);
- every change it does report on the real logs names a project whose own level moved, so
  no alert is left without something behind it;
- a planted step — every project's sessions from one day on multiplied by PLANT_FACTOR —
  is caught, planted on each of the last PLANT_DAYS judgeable days in turn, and credited
  only when the unplanted history was quiet on that day.

Run from the repo root:

  uv run --group lab python -m lab.context
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from ccdrift.logs import default_source, parse_all
from ccdrift.sessions import (MIN_BASELINE, MIN_PROJECT_SESSIONS, WINDOW, context_alerts, context_changes_in,
                              first_of_each, found_changes, moved_projects, project_path, ratio_starts,
                              session_starts)
from ccdrift.state import new_state

PLANT_FACTOR = 0.5   # what a planted step does to every project's session starts
PLANT_DAYS = 5       # days a step is planted on in turn, one replay each
GATE = "G12"


def replay(starts: pd.DataFrame) -> list[str]:
    """The days the rule alerts on, judging each day as the check would the morning after
    it, with one state carried along."""
    state = new_state()
    alerts = []
    for day in sorted(starts["day"].astype(str).unique()):
        today = date.fromisoformat(day) + timedelta(days=1)
        seen = starts[starts["day"].astype(str) <= day]
        alerts += [change["since"] for change in context_alerts(seen, state, today)]
    return alerts


def pooled_changes(starts: pd.DataFrame) -> list[str]:
    """What the rule before 0.8.0 found: the same detector over the token counts, every
    project pooled. Kept here so the gate can print the difference rather than claim it."""
    pooled = starts.assign(ratio=starts["prompt_tokens"].astype(float), level=1.0)
    return [change.since for change in first_of_each(context_changes_in(pooled))]


def plant(starts: pd.DataFrame, day: str, factor: float = PLANT_FACTOR) -> pd.DataFrame:
    """The starts with every session from `day` on, in every project, multiplied by
    `factor` — a step that reaches the whole machine, as a Claude Code change would."""
    planted = starts.copy()
    from_day = planted["day"].astype(str) >= str(day)
    planted.loc[from_day, "prompt_tokens"] = planted.loc[from_day, "prompt_tokens"].astype(float) * factor
    return planted


def plant_days(starts: pd.DataFrame, how_many: int = PLANT_DAYS) -> list[str]:
    """The last `how_many` days a step could be judged on: days with at least
    MIN_BASELINE judged sessions before them to compare with, and at least WINDOW judged
    sessions from that day on, since a step is only visible once that many sessions have
    run under it. Planting where the logs stop would ask whether the rule can see a step
    that nothing has followed yet — a question about the corpus's last day, not the rule."""
    judged = ratio_starts(starts)
    if judged.empty:
        return []
    days = judged["day"].astype(str).tolist()
    before_day = starts["day"].astype(str)
    judgeable = []
    for i, day in enumerate(days):
        if i < MIN_BASELINE:
            continue
        # A session can only show a step if its own project was already being judged
        # before the step: otherwise its level is made of halved sessions too, and the
        # ratio comes back to 1 with nothing to see.
        after = judged[judged["day"].astype(str) >= day]
        visible = sum(1 for _, row in after.iterrows()
                      if (before_day[(starts["project"].astype(str) == str(row["project"]))
                                     & (before_day < day)].count()) >= MIN_PROJECT_SESSIONS)
        if visible >= WINDOW:
            judgeable.append(day)
    ordered = sorted(set(judgeable))
    return ordered[-how_many:]


def every_alert_names_a_project(starts: pd.DataFrame) -> tuple[bool, list[str]]:
    """Whether every change the rule reports names a project whose own level moved."""
    judged = ratio_starts(starts)
    notes, sound = [], True
    for change in first_of_each(found_changes(judged)):
        moved = moved_projects(starts, change)
        notes.append(f"{change.since}: {change.before / 1000:.0f}k -> {change.after / 1000:.0f}k, "
                     f"{len(moved['moved'])} of {moved['seen']} projects moved"
                     + (f" ({', '.join(project_path(p) for p in moved['moved'])})" if moved["moved"] else ""))
        if not moved["moved"]:
            sound = False
    return sound, notes


def switch_history() -> pd.DataFrame:
    """The shape of the real case this gate stands for: twelve sessions in one project at
    128k, then five in a new, smaller one at 54k, then two more back in the first project
    a week later, still at 128k — the project that had been running kept running,
    unchanged, rather than vanishing once the work moved. Nothing inside either project
    changed."""
    rows = []
    for i in range(12):
        rows.append({"project": "-big", "day": f"2026-09-{i + 1:02d}", "prompt_tokens": 128_000.0})
    for i in range(5):
        rows.append({"project": "-small", "day": f"2026-09-{i + 13:02d}", "prompt_tokens": 54_000.0})
    for i in range(2):
        rows.append({"project": "-big", "day": f"2026-09-{i + 18:02d}", "prompt_tokens": 128_000.0})
    starts = pd.DataFrame(rows)
    starts["source_file"] = [f"{row['project']}/{i}.jsonl" for i, row in enumerate(rows)]
    starts["timestamp"] = pd.to_datetime([f"{row['day']}T10:00:00Z" for row in rows])
    starts["version"] = "2.1.261"
    return starts[["source_file", "project", "timestamp", "day", "version", "prompt_tokens"]]


def gate(starts: pd.DataFrame) -> tuple[bool, list[str]]:
    """Whether the new rule passes G12 on `starts`, and what to say about it."""
    switch = switch_history()
    switch_alerts = replay(switch)
    switch_pooled = pooled_changes(switch)
    notes = [f"a move to a smaller project: {len(switch_alerts)} alerts, where pooling found "
             f"{len(switch_pooled)}"]

    sound, alert_notes = every_alert_names_a_project(starts)
    notes += [f"on the logs: {note}" for note in alert_notes] or ["on the logs: no change found"]
    notes.append(f"pooling these logs found {len(pooled_changes(starts))}: "
                 f"{', '.join(pooled_changes(starts)) or 'none'}")

    quiet = replay(starts)
    # A plant is credited only when the planted history alerts on a day the unplanted one
    # was quiet on, so a day the logs already report a change on can give no answer:
    # planting there is skipped rather than counted as a miss.
    plants = [day for day in plant_days(starts) if day not in quiet]
    skipped = [day for day in plant_days(starts) if day in quiet]
    caught = sum(1 for day in plants if any(one not in quiet for one in replay(plant(starts, day))))
    notes.append(f"planted steps caught: {caught} of {len(plants)}"
                 + (f" (planted on {', '.join(plants)})" if plants else "")
                 + (f"; skipped {', '.join(skipped)}, where the logs already report a change" if skipped else ""))
    passed = not switch_alerts and sound and bool(plants) and caught == len(plants)
    return passed, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    args = parser.parse_args(argv)

    starts = session_starts(parse_all(args.source).responses)
    starts = starts[starts["day"].astype(str) < args.today.isoformat()].reset_index(drop=True)
    judged = ratio_starts(starts)
    print(f"{len(starts)} session starts in {starts['project'].nunique()} projects; "
          f"{len(judged)} judged against their own project "
          f"(a project's first {MIN_PROJECT_SESSIONS} sessions only set its level)")
    passed, notes = gate(starts)
    print(f"{GATE}: {'PASS' if passed else 'FAIL'}")
    for note in notes:
        print(f"  {note}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
