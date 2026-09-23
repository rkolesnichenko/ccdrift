"""Stop hooks per day, from the summaries Claude Code logs."""

from datetime import date

from ccdrift.hooks import hook_days, hook_failures, hooks_lines, hooks_summary, judged_hook_runs
from ccdrift.logs import parse_all
from ccdrift.state import new_state
from ccdrift.texts import hook_failure_message
from tests.helpers import DAY, at, hook_days_logs, stop_hook_summary, write


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


def failing_runs(tmp_path, failing_days, days=16, per_day=10):
    hook_days_logs(tmp_path, failing_days, days, per_day)
    return judged_hook_runs(parse_all(tmp_path).hook_runs, date(2026, 9, 1 + days))


def test_hooks_failing_two_days_running_after_two_quiet_weeks_are_reported_once(tmp_path):
    state = new_state()
    runs = failing_runs(tmp_path, {14, 15})
    failures = hook_failures(runs, state, date(2026, 9, 17))
    assert [(f["since"], f["days"], f["runs"], f["failed"]) for f in failures] == [
        ("2026-09-15", ["2026-09-15", "2026-09-16"], [10, 10], [10, 10])]
    assert hook_failures(runs, state, date(2026, 9, 17)) == []
    assert hook_failure_message(failures[0], ["2.1.280 (since 09-15)"]) == (
        "Stop hooks failed on 10 of 10 runs on 2026-09-15 and 10 of 10 on 2026-09-16, on Claude Code 2.1.280 "
        "(since 09-15). Check your hooks; a Claude Code update may have changed their input.")


def test_one_failing_day_is_not_reported(tmp_path):
    assert hook_failures(failing_runs(tmp_path, {15}), new_state(), date(2026, 9, 17)) == []


def test_hooks_that_already_failed_in_the_weeks_before_are_not_a_new_failure(tmp_path):
    assert hook_failures(failing_runs(tmp_path, {5, 14, 15}), new_state(), date(2026, 9, 17)) == []


def test_days_with_few_hook_runs_dont_count(tmp_path):
    assert hook_failures(failing_runs(tmp_path, {14, 15}, per_day=5), new_state(), date(2026, 9, 17)) == []
