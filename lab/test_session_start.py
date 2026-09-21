"""The G2 spike: session-start size per version and whether its alerts would be sound."""

import numpy as np
import pandas as pd

from ccdrift.sessions import (MIN_SESSIONS, context_changes_in, first_of_each, found_changes,
                             ratio_starts)
from lab.harness import generate_synthetic
from lab.session_start import (MAX_SPREAD, SWEEP_SEED, false_alert_rate, gate, main,
                               project_version_table, residual_table, steady_history, step_versions,
                               version_table)
from tests.helpers import nth_day


def starts_of(rows, project="p"):
    """rows: (version, prompt tokens) in time order, one session a day, all in one
    project so ratio_starts judges every one of them against the same level."""
    return pd.DataFrame({"source_file": [f"{project}/{i}.jsonl" for i in range(len(rows))],
                         "project": [project] * len(rows),
                         "timestamp": pd.to_datetime([f"{nth_day(i)}T10:00:00Z" for i in range(len(rows))]),
                         "day": [nth_day(i) for i in range(len(rows))],
                         "version": [v for v, _ in rows], "prompt_tokens": [float(t) for _, t in rows]})


def projects_of(rows, version="2.1.261"):
    """rows: (project, prompt tokens) in time order, one session a day, every session on
    the same version."""
    return starts_of([(version, tokens) for _, tokens in rows]).assign(
        project=[project for project, _ in rows],
        source_file=[f"{project}/{i}.jsonl" for i, (project, _) in enumerate(rows)])


def test_version_table_shows_each_versions_median_and_spread():
    table = version_table(starts_of([("2.1.261", 128_000), ("2.1.261", 130_000), ("2.1.261", 126_000),
                                     ("2.1.267", 54_000)]))
    assert table[["version", "sessions", "median"]].values.tolist() == [["2.1.261", 3, 128_000.0],
                                                                        ["2.1.267", 1, 54_000.0]]
    assert round(float(table["spread"].iloc[0]), 3) == 0.023


def test_gate_passes_when_versions_are_steady_and_steps_come_with_a_new_version():
    rows = [("2.1.261", 128_000 + (i % 3) * 1_000) for i in range(8)] + [("2.1.267", 54_000)] * 4
    passed, notes = gate(starts_of(rows))
    assert passed, notes


def test_gate_passes_when_a_step_comes_without_a_new_version_and_says_so():
    # A project's own CLAUDE.md, skills or MCP servers step its sessions with no new
    # Claude Code version, which is what G12 measures, so the version is reported beside
    # the step rather than deciding the gate.
    rows = [("2.1.261", 128_000)] * 8 + [("2.1.261", 54_000)] * 4
    passed, notes = gate(starts_of(rows))
    assert passed, notes
    assert notes[1] == "steps found: 1; with a version new to their baseline: 0"


def test_the_gate_reads_the_steps_the_shipped_rule_finds_not_the_pooled_pass_alone():
    # One project of two steps; pooled, the other project's sessions dilute it away.
    rows = [("-a", 128_000), ("-a", 128_000), ("-b", 60_000)] * 4 + [("-a", 54_000), ("-a", 54_000),
                                                                     ("-b", 60_000)] * 3
    starts = projects_of(rows)
    assert first_of_each(context_changes_in(ratio_starts(starts))) == []
    _, notes = gate(starts)
    assert notes[1] == "steps found: 1; with a version new to their baseline: 0"


def test_a_step_is_reported_with_its_own_projects_versions_only():
    # -a steps on 2.1.267 while -b runs 9.9.9 throughout, untouched. Reading the step's
    # days across the judged table would credit it with 9.9.9, which no session of the
    # step ever ran.
    rows = [("-a", 128_000), ("-a", 128_000), ("-b", 60_000)] * 4 + [("-a", 54_000), ("-a", 54_000),
                                                                     ("-b", 60_000)] * 3
    starts = projects_of(rows).assign(
        version=["2.1.261" if project == "-a" else "9.9.9" for project, _ in rows[:12]]
        + ["2.1.267" if project == "-a" else "9.9.9" for project, _ in rows[12:]])
    judged = ratio_starts(starts)
    steps = first_of_each(found_changes(judged))
    assert [(c.since, c.until, c.project) for c in steps] == [("2026-09-13", "2026-09-16", "-a")]
    days = judged["day"].astype(str)
    on_the_days = judged[(days >= steps[0].since) & (days <= steps[0].until)]
    assert sorted(set(on_the_days["version"])) == ["2.1.267", "9.9.9"]
    assert step_versions(judged, steps[0]) == ["2.1.267"]


def test_one_version_across_several_projects_is_not_read_as_a_spread():
    # The failure this grouping fixes: a version seen once in each of three projects of
    # very different sizes. Pooled it reads as a wide spread; per project each row holds
    # one session, so the gate reads none of them and the projects say nothing about it.
    starts = projects_of([("-a", 61_000), ("-b", 72_000), ("-c", 140_000)], version="2.1.278")
    pooled = version_table(starts)
    assert float(pooled["spread"].iloc[0]) > MAX_SPREAD
    per_project = project_version_table(starts)
    assert per_project["sessions"].tolist() == [1, 1, 1]
    assert (per_project["sessions"] >= MIN_SESSIONS).sum() == 0


def test_a_version_steady_in_one_project_passes_beside_a_second_project():
    # The same version in two projects, steady inside each: the gate reads two groups and
    # neither is troubled by the other project's size.
    rows = [("-a", 128_000), ("-b", 54_000)] * 4
    passed, notes = gate(projects_of(rows))
    assert passed, notes
    assert notes[0].startswith("projects with 3+ sessions: 2")


def test_a_version_step_is_not_counted_as_noise():
    # A project whose sessions step with the version: raw spread is the step, and the
    # residual is what a session varies by once its own version's level is taken out.
    # Reading the raw figure would call the signal the alert exists to find noise.
    rows = [("2.1.261", 100_000), ("2.1.261", 101_000), ("2.1.261", 99_000),
            ("2.1.276", 140_000), ("2.1.276", 141_000), ("2.1.276", 139_000)]
    table = residual_table(starts_of(rows))
    assert float(table["raw"].iloc[0]) > MAX_SPREAD
    assert float(table["residual"].iloc[0]) < MAX_SPREAD
    passed, notes = gate(starts_of(rows * 2))
    assert passed, notes


def test_gate_fails_when_a_version_varies_too_much():
    rows = [("2.1.261", t) for t in (100_000, 130_000, 160_000, 100_000, 130_000, 160_000)]
    passed, _ = gate(starts_of(rows))
    assert not passed


def test_the_spread_bar_is_where_the_shipped_rule_stops_raising_false_alerts():
    # MAX_SPREAD is measured, not chosen: at the bar a steady history raises nothing, and
    # twice it the shipped rule reports changes that were never planted. Seeded, so this
    # pins the measurement rather than resampling it.
    assert false_alert_rate(MAX_SPREAD, 30, 50, SWEEP_SEED) == 0.0
    assert false_alert_rate(MAX_SPREAD * 2, 30, 50, SWEEP_SEED) > 0.0


def test_a_steady_history_holds_no_change_to_find():
    rng = np.random.default_rng(SWEEP_SEED)
    starts = steady_history(0.02, 30, rng)
    assert len(starts) == 30
    assert first_of_each(context_changes_in(ratio_starts(starts))) == []


def test_session_start_spike_runs_on_synthetic_logs(tmp_path, capsys):
    source = generate_synthetic(tmp_path / "syn", days=20, seed=1)
    assert main(["--source", str(source)]) == 0
    assert "G2:" in capsys.readouterr().out
