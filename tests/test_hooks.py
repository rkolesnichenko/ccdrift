"""Stop hooks per day, from the summaries Claude Code logs."""

from datetime import date

from ccdrift.hooks import hook_days, hooks_lines, hooks_summary, judged_hook_runs
from ccdrift.logs import parse_all
from tests.helpers import DAY, at, stop_hook_summary, write


def hook_logs(tmp_path):
    write(tmp_path / "s1.jsonl", [
        stop_hook_summary(at(0), 1, durations=(1000,), uuid="h1"),
        stop_hook_summary(at(60), 1, durations=(3000,), errors=("exit 1",), uuid="h2"),
        stop_hook_summary(at(DAY), 1, durations=(2000,), uuid="h3"),
        stop_hook_summary(at(2 * DAY), 1, durations=(9000,), uuid="h4"),
    ])
    write(tmp_path / "sdk.jsonl", [stop_hook_summary(at(30), 1, errors=("x",), uuid="s1", entrypoint="sdk-py")])
    return judged_hook_runs(parse_all(tmp_path).hook_runs, date(2026, 9, 3))


def test_hook_days_count_runs_failures_and_median_duration(tmp_path):
    assert hook_days(hook_logs(tmp_path)).values.tolist() == [["2026-09-01", 2, 1, 2000.0],
                                                              ["2026-09-02", 1, 0, 2000.0]]


def test_hooks_summary_reads_as_one_line(tmp_path):
    summary = hooks_summary(hook_logs(tmp_path), ["2026-09-01", "2026-09-02"])
    assert summary == {"runs": 3, "error_days": 1, "median_duration_ms": 2000.0}
    assert hooks_lines(summary) == ["", "Hooks over these days: 3 stop-hook runs, errors on 1 day, median 2.0 s"]
    assert hooks_lines(None) == []
