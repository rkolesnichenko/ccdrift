"""The G12 gate: how judging each project against itself is measured."""

import pandas as pd

from ccdrift.sessions import START_COLUMNS
from lab.context import every_alert_names_a_project, gate, plant, plant_days, pooled_changes, replay, switch_history
from tests.helpers import nth_day


def starts_of(rows):
    """rows: (project, prompt tokens) in time order, one session a day."""
    frame = pd.DataFrame({"source_file": [f"{project}/{i}.jsonl" for i, (project, _) in enumerate(rows)],
                          "project": [project for project, _ in rows],
                          "timestamp": pd.to_datetime([f"{nth_day(i)}T10:00:00Z" for i in range(len(rows))]),
                          "day": [nth_day(i) for i in range(len(rows))],
                          "version": ["2.1.261"] * len(rows),
                          "prompt_tokens": [float(tokens) for _, tokens in rows]})
    return frame[START_COLUMNS]


def test_a_move_to_a_smaller_project_reads_as_a_change_only_when_projects_are_pooled():
    switch = switch_history()
    assert replay(switch) == []
    assert len(pooled_changes(switch)) == 1


def test_a_planted_step_reaches_every_project_and_is_found():
    starts = starts_of([("-a", 100_000)] * 8 + [("-b", 50_000)] * 8)
    day = plant_days(starts)[0]
    planted = plant(starts, day)
    assert planted.loc[planted["day"].astype(str) >= day, "prompt_tokens"].max() <= 50_000
    assert replay(planted) != []
    assert replay(starts) == []


def test_a_step_is_only_planted_where_the_rule_could_see_it():
    # The last days hold too few judged sessions after them for a step to show yet.
    starts = starts_of([("-a", 100_000)] * 12)
    days = plant_days(starts)
    assert days and all(day <= starts["day"].iloc[-3] for day in days)


def test_an_alert_that_names_no_moved_project_fails_the_gate():
    steady = starts_of([("-a", 100_000)] * 12)
    sound, notes = every_alert_names_a_project(steady)
    assert (sound, notes) == (True, [])


def test_the_gate_passes_when_the_rule_is_quiet_on_a_switch_and_catches_a_planted_step():
    starts = starts_of([("-a", 100_000), ("-b", 60_000)] * 10)
    passed, notes = gate(starts)
    assert passed, notes
    assert any("planted steps caught" in note for note in notes)
