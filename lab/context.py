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
- two planted steps are caught, planted on each of the last PLANT_DAYS judgeable days in
  turn and credited only when the unplanted history was quiet on that day.

The two plants are the two shapes `found_changes` exists for, and each tests a different
half of it. A step in every project — every project's sessions from one day on multiplied
by PLANT_FACTOR — is the shape a Claude Code change takes, and the pooled pass finds it on
its own. A step in one project, the shape its own CLAUDE.md, skills or MCP servers take, is
diluted by the other projects' sessions and only the per-project pass sees it; planting
only the first shape would leave that half of the rule — the half both fixed Criticals of
this build were about — unexercised.

The one-project plant asks an answerable question only where the project it lands in is a
minority of the judged sessions: a project holding more than half of them *is* the pooled
set, so halving it halves the pooled ratios too and no credit rule can say which pass found
the step. Where that holds, the plant is credited only when the pooled pass —
`context_changes_in` over every judged session's ratio, the half `found_changes` contrasts
with — is silent on the day the alert lands. Each plant day is judged on its own, since
each picks its own project and they need not be the same one: the days landing in a
minority project are scored, and any others are reported beside them as not measurable.
Where no plant is measurable, or no project has the sessions to carry one at all, the
gate prints "not measurable here" with the reason and neither passes nor fails on it: a
missing number is a question this history couldn't put, never a silent pass, and a
screened-out plant never helps the gate pass. The owner's logs are such a history — one
project holds 19 of the 20 judged sessions — so what pins the per-project pass there is
`lab/test_context.py`'s balanced two-project history, where halving one project moves no
pooled median.

Run from the repo root:

  uv run --group lab python -m lab.context
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ccdrift.logs import default_source, parse_all
from ccdrift.sessions import (MIN_BASELINE, MIN_PROJECT_SESSIONS, WINDOW, context_alerts, context_changes_in,
                              first_of_each, found_changes, moved_projects, project_path, ratio_starts,
                              session_starts)
from ccdrift.state import new_state

PLANT_FACTOR = 0.5   # what a planted step multiplies the session starts it reaches by
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


def plant_in(starts: pd.DataFrame, day: str, project: str, factor: float = PLANT_FACTOR) -> pd.DataFrame:
    """The starts with one project's sessions from `day` on multiplied by `factor`, every
    other project untouched — a step in that project's own files, which the pooled pass
    dilutes and only the per-project pass can see."""
    planted = starts.copy()
    step = (planted["day"].astype(str) >= str(day)) & (planted["project"].astype(str) == str(project))
    planted.loc[step, "prompt_tokens"] = planted.loc[step, "prompt_tokens"].astype(float) * factor
    return planted


def plantable_project(starts: pd.DataFrame, day: str) -> str | None:
    """The largest project a one-project step can be planted in on `day`: one with
    MIN_PROJECT_SESSIONS sessions to set its level and MIN_BASELINE more before the day to
    be a baseline, and WINDOW sessions from the day on for the step to fill a window with.
    None when no project of this history has that many — where the question can't be put,
    not where the rule fails it."""
    enough = []
    for project, rows in starts.groupby(starts["project"].astype(str), sort=True):
        on_days = rows["day"].astype(str)
        if (int((on_days < str(day)).sum()) >= MIN_PROJECT_SESSIONS + MIN_BASELINE
                and int((on_days >= str(day)).sum()) >= WINDOW):
            enough.append((len(rows), str(project)))
    return max(enough)[1] if enough else None


def pooled_pass(starts: pd.DataFrame) -> set[str]:
    """The days the pooled half of `found_changes` starts a change on: the detector over
    the ratios of every judged session together. This, not `pooled_changes`, is the half
    the per-project pass is contrasted with — `pooled_changes` is the pre-0.8.0 rule over
    raw token counts, and a day can clear that one merely because the two rules date the
    same step a day apart. Every window counts, not just the first of each step: a step the
    pooled pass finds on any window is one it can see."""
    return {change.since for change in context_changes_in(ratio_starts(starts))}


def separable(starts: pd.DataFrame, project: str) -> tuple[int, int]:
    """How many of the judged sessions belong to `project`, and how many there are. A
    project holding more than half of them is the pooled set: halving it halves the pooled
    ratios too, so no credit rule can tell the two passes apart on that history, and the
    question the one-project plant asks can't be put there at all."""
    judged = ratio_starts(starts)
    return int((judged["project"].astype(str) == str(project)).sum()), len(judged)


def caught_per_project(starts: pd.DataFrame, day: str, project: str, quiet: list[str]) -> bool:
    """Whether a step planted in `project` alone on `day` is caught by the pass that exists
    for it: the planted history alerts on a day the unplanted one (`quiet`) didn't, and the
    pooled pass is silent on that day. Without the second half the line would credit the
    per-project pass for a step the pooled pass found on its own."""
    planted = plant_in(starts, day, project)
    pooled = pooled_pass(planted)
    return any(one not in quiet and one not in pooled for one in replay(planted))


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
    pooled = pooled_changes(starts)
    notes.append(f"pooling these logs found {len(pooled)}: {', '.join(pooled) or 'none'}")

    quiet = replay(starts)
    # A plant is credited only when the planted history alerts on a day the unplanted one
    # was quiet on, so a day the logs already report a change on can give no answer:
    # planting there is skipped rather than counted as a miss.
    days = plant_days(starts)
    plants = [day for day in days if day not in quiet]
    skipped = [day for day in days if day in quiet]
    caught = sum(1 for day in plants if any(one not in quiet for one in replay(plant(starts, day))))
    notes.append(f"planted steps caught: {caught} of {len(plants)}"
                 + (f" (planted on {', '.join(plants)})" if plants else "")
                 + (f"; skipped {', '.join(skipped)}, where the logs already report a change" if skipped else ""))

    # The same days, stepped in one project only: the half of the rule the pooled pass
    # can't answer for. It only answers it where that project is a minority of the judged
    # sessions and the pooled pass stays quiet on the day the alert lands; where neither
    # holds, the gate says the question couldn't be put rather than scoring it.
    # Each day picks its own project, so the screen is per plant, not per run: one day
    # landing in a project that is the judged sessions says nothing about another day that
    # lands in a minority one, and throwing that day away would let the gate pass on a
    # claim it never measured.
    in_one = [(day, plantable_project(starts, day)) for day in plants]
    one_project = [(day, project) for day, project in in_one if project]
    shares = {project: separable(starts, project) for _, project in one_project}
    measurable = [(day, project) for day, project in one_project if 2 * shares[project][0] <= shares[project][1]]
    screened = [(day, project) for day, project in one_project if 2 * shares[project][0] > shares[project][1]]
    corpus = f"the project the plant lands in holds {shares[screened[0][1]][0]} of {shares[screened[0][1]][1]} " \
             "judged sessions, so halving it also halves the pooled set" if screened else ""
    caught_one = sum(1 for day, project in measurable if caught_per_project(starts, day, project, quiet))
    if measurable:
        notes.append(f"planted one-project steps caught by the per-project pass: {caught_one} of "
                     f"{len(measurable)} (planted in "
                     + ", ".join(f"{project_path(project)} on {day}" for day, project in measurable) + ")"
                     + (f"; {len(screened)} more not measurable: {corpus}" if screened else ""))
    elif screened:
        notes.append(f"planted one-project steps: not measurable here — {corpus}")
    else:
        notes.append("planted one-project steps: not measurable here — no project has "
                     f"{MIN_PROJECT_SESSIONS + MIN_BASELINE} sessions before a plantable day and {WINDOW} "
                     "from it on")
    passed = (not switch_alerts and sound and bool(plants) and caught == len(plants)
              and caught_one == len(measurable))
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
