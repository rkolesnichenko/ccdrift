"""The G2 spike: session-start size per version and whether its alerts would be sound."""

import pandas as pd

from lab.harness import generate_synthetic
from lab.session_start import gate, main, version_table
from tests.helpers import nth_day


def starts_of(rows, project="p"):
    """rows: (version, prompt tokens) in time order, one session a day, all in one
    project so ratio_starts judges every one of them against the same level."""
    return pd.DataFrame({"source_file": [f"{project}/{i}.jsonl" for i in range(len(rows))],
                         "project": [project] * len(rows),
                         "timestamp": pd.to_datetime([f"{nth_day(i)}T10:00:00Z" for i in range(len(rows))]),
                         "day": [nth_day(i) for i in range(len(rows))],
                         "version": [v for v, _ in rows], "prompt_tokens": [float(t) for _, t in rows]})


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


def test_gate_fails_when_a_step_comes_without_a_new_version():
    rows = [("2.1.261", 128_000)] * 8 + [("2.1.261", 54_000)] * 4
    passed, _ = gate(starts_of(rows))
    assert not passed


def test_gate_fails_when_a_version_varies_too_much():
    rows = [("2.1.261", t) for t in (100_000, 130_000, 160_000, 100_000, 130_000, 160_000)]
    passed, _ = gate(starts_of(rows))
    assert not passed


def test_session_start_spike_runs_on_synthetic_logs(tmp_path, capsys):
    source = generate_synthetic(tmp_path / "syn", days=20, seed=1)
    assert main(["--source", str(source)]) == 0
    assert "G2:" in capsys.readouterr().out
