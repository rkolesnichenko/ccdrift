"""The G12 gate: how judging each project against itself is measured."""

import pandas as pd

from ccdrift.sessions import START_COLUMNS, context_changes_in, first_of_each, found_changes, ratio_starts
from lab.context import (caught_per_project, every_alert_names_a_project, gate, plant, plant_days, plant_in,
                         plantable_project, pooled_changes, pooled_pass, replay, separable, switch_history)
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
    # A 3-session spike the rule reports as a step, diluted back to the project's usual
    # level by the ten sessions that follow it: before and after the step's own day, the
    # project's median is unchanged, so moved_projects finds nothing behind the alert.
    starts = starts_of([("-a", 100_000)] * 8 + [("-a", 140_000)] * 3 + [("-a", 100_000)] * 10)
    sound, notes = every_alert_names_a_project(starts)
    assert sound is False
    assert any("0 of 1 projects moved" in note for note in notes)


def test_a_planted_step_in_one_project_is_invisible_to_the_pooled_pass_and_found_anyway():
    # The half of found_changes the every-project plant can't test: -b's sessions dilute
    # -a's step until the pooled pass sees nothing, and the per-project pass still finds it.
    starts = starts_of([("-a", 100_000), ("-a", 100_000), ("-b", 60_000)] * 8)
    day = plant_days(starts)[0]
    assert plantable_project(starts, day) == "-a"
    planted = plant_in(starts, day, "-a")
    assert set(planted.loc[planted["day"].astype(str) >= day, "prompt_tokens"]) == {50_000.0, 60_000.0}
    judged = ratio_starts(planted)
    assert first_of_each(context_changes_in(judged)) == []
    assert [c.since for c in first_of_each(found_changes(judged))] == ["2026-09-19"]
    assert replay(planted) == ["2026-09-19"]
    assert replay(starts) == []


def corpus_history():
    """One project holding all but one of the judged sessions, as the owner's logs do: -a
    is 17 of 18, so a window of the pooled pass is made of -a's sessions whatever it is."""
    return starts_of(([("-a", 100_000)] * 5 + [("-b", 60_000)]) * 4)


def test_a_one_project_plant_the_pooled_pass_also_finds_is_not_credited():
    # Halving a project that is the judged sessions halves the pooled ratios with it: the
    # planted history alerts, and the pooled pass names the same day, so the alert is no
    # evidence that the pass this plant exists for is what found it.
    starts = corpus_history()
    quiet = replay(starts)
    assert replay(plant_in(starts, "2026-09-20", "-a")) == ["2026-09-20"]
    assert "2026-09-20" in pooled_pass(plant_in(starts, "2026-09-20", "-a"))
    assert caught_per_project(starts, "2026-09-20", "-a", quiet) is False
    # Where the planted project is half the judged sessions, the same plant is credited:
    # the pooled pass stays quiet and only the per-project pass has the step.
    balanced = starts_of([("-a", 100_000), ("-b", 100_000)] * 12)
    assert "2026-09-18" not in pooled_pass(plant_in(balanced, "2026-09-18", "-b"))
    assert caught_per_project(balanced, "2026-09-18", "-b", replay(balanced)) is True


def test_a_history_where_one_project_is_the_corpus_cannot_answer_the_one_project_plant():
    # The owner's shape. The gate says so and scores neither a catch nor a miss on it: the
    # question needs a project that halving doesn't take the pooled set with it.
    starts = corpus_history()
    assert separable(starts, "-a") == (17, 18)
    passed, notes = gate(starts)
    assert passed, notes
    assert notes[-1] == ("planted one-project steps: not measurable here — the project the plant lands in holds "
                         "17 of 18 judged sessions, so halving it also halves the pooled set")


def test_the_gate_passes_when_the_rule_is_quiet_on_a_switch_and_catches_both_planted_steps():
    # Both projects start at the same size, so halving one moves no pooled median: every
    # one-project plant that is caught here is caught by the per-project pass.
    starts = starts_of([("-a", 100_000), ("-b", 100_000)] * 12)
    passed, notes = gate(starts)
    assert passed, notes
    assert [note for note in notes if note.startswith("planted")] == [
        "planted steps caught: 5 of 5 (planted on 2026-09-18, 2026-09-19, 2026-09-20, 2026-09-21, 2026-09-22)",
        "planted one-project steps caught by the per-project pass: 3 of 3 (planted in /b on 2026-09-18, "
        "/b on 2026-09-19, /b on 2026-09-20)"]


def test_a_history_too_thin_for_a_one_project_plant_says_so_rather_than_passing_silently():
    # Every project alternates, so none has the sessions each side of a plantable day to
    # carry a step of its own. The gate says the question couldn't be put here.
    starts = starts_of([("-a", 100_000), ("-b", 60_000)] * 10)
    assert [plantable_project(starts, day) for day in plant_days(starts)] == [None] * 5
    passed, notes = gate(starts)
    assert passed, notes
    assert notes[-1] == ("planted one-project steps: not measurable here — no project has 8 sessions before a "
                         "plantable day and 3 from it on")
