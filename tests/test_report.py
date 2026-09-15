"""ccdrift report: recent daily metrics, their z-scores and flags."""

import json
from datetime import date

from ccdrift.cli import main
from ccdrift.report import daily_rows, run_report
from tests.helpers import HAIKU, QUIET, busy_days, daily_turns


def test_report_lists_recent_days_with_their_metrics(tmp_path, capsys):
    busy_days(tmp_path / "logs", days=3, per_day=60, cache_read=900, cache_creation=100)
    assert run_report(tmp_path / "logs", tmp_path / "state.json", days=2, today=date(2026, 9, 4)) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Last 2 complete UTC days with main-thread activity.",
        "Flagged once 3 of any 4 days in a row pass the cutoff: z <= -3.0 for the cache ratio, "
        "z >= +3.5 for Haiku share.",
        "",
        "day         responses  cache ratio      z  haiku share      z  flagged",
        "2026-09-02         60        0.900      -        0.000      -",
        "2026-09-03         60        0.900      -        0.000      -",
        "",
        "Flags reported by the daily check: none yet",
    ]


def test_report_lists_the_flags_the_check_reported(tmp_path, capsys):
    busy_days(tmp_path / "logs", days=1, per_day=60, cache_read=900, cache_creation=100)
    (tmp_path / "state.json").write_text(json.dumps({"reported": {"cache_ratio": ["2026-08-18"]}}))
    run_report(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4))
    assert capsys.readouterr().out.splitlines()[-2:] == [
        "Flags reported by the daily check:",
        "  Cache read ratio on new prompts from 2026-08-18",
    ]


def test_report_marks_the_metrics_flagged_each_day():
    df = daily_turns([QUIET] * 14 + [HAIKU] * 3)
    df["main_thread"] = True
    assert daily_rows(df, date(2026, 9, 18), days=4)["flagged"].tolist() == ["", "haiku", "haiku", "haiku"]


def test_report_exits_2_without_transcripts(tmp_path):
    (tmp_path / "empty").mkdir()
    assert main(["report", "--source", str(tmp_path / "empty")]) == 2


def test_report_explains_an_unreadable_state_file(tmp_path, capsys):
    busy_days(tmp_path / "logs", days=1, per_day=60, cache_read=900, cache_creation=100)
    (tmp_path / "state.json").write_text("not json")
    assert run_report(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4)) == 1
    assert "Can't read the state file" in capsys.readouterr().err
