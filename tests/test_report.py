"""ccdrift report: recent daily metrics, their z-scores and flags."""

import json
from datetime import date

import pytest

from ccdrift.cli import main
from ccdrift.logs import judged_turns
from ccdrift.report import daily_rows, run_report, version_key
from tests.helpers import HAIKU, QUIET, busy_days, daily_turns, main_thread_days


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
        "Incidents: none yet",
        "",
        "Settings on the CLI main thread over these days (share of responses):",
        "  claude-opus-5: effort not logged 100%; speed not logged 100%; service tier not logged 100%",
    ]


def test_report_lists_flags_reported_before_incidents_were_followed(tmp_path, capsys):
    busy_days(tmp_path / "logs", days=1, per_day=60, cache_read=900, cache_creation=100)
    (tmp_path / "state.json").write_text(json.dumps({"reported": {"cache_ratio": ["2026-08-18"]}}))
    run_report(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4))
    lines = capsys.readouterr().out.splitlines()
    i = lines.index("Flags reported before ccdrift followed incidents:")
    assert lines[i + 1] == "  Cache read ratio on new prompts from 2026-08-18"


def test_report_lists_incidents_with_their_cost(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    (tmp_path / "state.json").write_text(json.dumps({"version": 2, "incidents": [
        {"metric": "cache_ratio", "start": "2026-09-02", "end": None, "status": "open", "source": "check",
         "closed_by": None, "recovered_from": None, "opened_on": "2026-09-03", "closed_on": None,
         "versions": ["2.1.226 (since 09-01)"], "cost": 0}]}))
    run_report(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4))
    assert "  cache  2026-09-02..now           open; no tokens re-cached; on 2.1.226 (since 09-01)" \
        in capsys.readouterr().out.splitlines()


def test_report_marks_the_metrics_flagged_each_day():
    df = daily_turns([QUIET] * 14 + [HAIKU] * 3)
    df["main_thread"] = True
    rows = daily_rows(judged_turns(df, date(2026, 9, 18)), days=4)
    assert rows["flagged"].tolist() == ["", "haiku", "haiku", "haiku"]


def test_report_judges_days_in_an_incident_against_the_days_before_it():
    df = daily_turns([QUIET] * 14 + [HAIKU] * 20)
    df["main_thread"] = True
    incident = {"metric": "haiku_fraction", "start": "2026-09-15", "end": None, "status": "open"}
    rows = daily_rows(judged_turns(df, date(2026, 10, 5)), days=1, incidents=[incident])
    assert rows["haiku_z"].iloc[0] > 3.5


def test_report_exits_2_without_transcripts(tmp_path):
    (tmp_path / "empty").mkdir()
    assert main(["report", "--source", str(tmp_path / "empty"), "--state", str(tmp_path / "state.json")]) == 2


def test_report_explains_an_unreadable_state_file(tmp_path, capsys):
    busy_days(tmp_path / "logs", days=1, per_day=60, cache_read=900, cache_creation=100)
    (tmp_path / "state.json").write_text("not json")
    assert run_report(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4)) == 1
    assert "Can't read the state file" in capsys.readouterr().err


def test_report_by_version_compares_claude_code_versions_oldest_first(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{"version": "2.1.99"}] * 2 + [{"version": "2.1.233"}] * 2)
    assert run_report(tmp_path / "logs", tmp_path / "state.json", by="version", today=date(2026, 9, 5)) == 0
    assert capsys.readouterr().out.splitlines()[:6] == [
        "Complete UTC days with main-thread activity, by Claude Code version.",
        "A miss is a new-prompt turn that reads less than half its input from the cache.",
        "",
        "version      first day   last day    responses  prompt turns  cache ratio  misses  haiku share",
        "2.1.99       2026-09-01  2026-09-02        120           118        0.900    0.0%        0.000",
        "2.1.233      2026-09-03  2026-09-04        120           118        0.900    0.0%        0.000",
    ]


def test_versions_sort_by_their_numbers():
    assert sorted(["2.1.233", "unknown", "2.1.99", "2.0.300"], key=version_key) == \
        ["2.0.300", "2.1.99", "2.1.233", "unknown"]


def test_report_json_holds_aggregates_without_paths_or_session_ids(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    assert run_report(tmp_path / "logs", tmp_path / "state.json", as_json=True, today=date(2026, 9, 4)) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert list(payload) == ["view", "days", "incidents", "reported_before_incidents", "settings", "cutoffs",
                             "flag_rule"]
    assert payload["days"][0] == {"day": "2026-09-01", "responses": 60, "cache_ratio": pytest.approx(0.9),
                                  "cache_z": None, "haiku_share": 0.0, "haiku_z": None, "flagged": []}
    assert payload["cutoffs"] == {"cache_ratio": -3.0, "haiku_fraction": 3.5}
    assert str(tmp_path) not in out and ".jsonl" not in out and '"s0"' not in out


def test_report_options_reach_the_report(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    assert main(["report", "--by", "version", "--json", "--days", "2", "--source", str(tmp_path / "logs"),
                 "--state", str(tmp_path / "state.json")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["view"] == "version"
    assert [v["version"] for v in payload["versions"]] == ["2.1.226"]
