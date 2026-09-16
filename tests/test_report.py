"""ccdrift report: recent daily metrics, their z-scores and flags."""

import json
from datetime import date

import pytest

from ccdrift.cli import main
from ccdrift.history import load_history
from ccdrift.logs import judged_turns
from ccdrift.report import daily_rows, run_report, version_key
from tests.helpers import (DAY, HAIKU, QUIET, at, busy_days, compact_boundary, daily_turns,
                           damage_responses_table, line, main_thread_days, stop_hook_summary, text, write)


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


def test_report_explains_a_history_store_whose_rows_cant_be_read(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    # report no longer claims a store on its own; build one the way the check would.
    load_history(tmp_path / "logs", tmp_path / "state.json", claim=True)
    damage_responses_table(tmp_path / "history.sqlite")
    assert run_report(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4)) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"Can't use the history store {tmp_path / 'history.sqlite'}: ")
    assert captured.err.endswith("Move it aside to rebuild it from the transcripts still on disk.\n")


def test_report_by_version_compares_claude_code_versions_oldest_first(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{"version": "2.1.99"}] * 2 + [{"version": "2.1.233"}] * 2)
    assert run_report(tmp_path / "logs", tmp_path / "state.json", by="version", today=date(2026, 9, 5)) == 0
    assert capsys.readouterr().out.splitlines()[:6] == [
        "Complete UTC days with main-thread activity, by Claude Code version.",
        "A miss is a new-prompt turn that reads less than half its input from the cache.",
        "",
        "version      first day   last day    responses  prompt turns  cache ratio  misses  haiku share  session start  compacts at",
        "2.1.99       2026-09-01  2026-09-02        120           118        0.900    0.0%        0.000              -            -",
        "2.1.233      2026-09-03  2026-09-04        120           118        0.900    0.0%        0.000              -            -",
    ]


def test_report_by_version_shows_session_start_size_and_where_compaction_starts(tmp_path, capsys):
    # A session-start median over 1 or 2 sessions misleads (128k and 508k made 320k), so it needs 3.
    main_thread_days(tmp_path / "logs", [{"version": "2.1.99"}] * 3 + [{"version": "2.1.233"}] * 2)
    write(tmp_path / "logs" / "compacted.jsonl",
          [compact_boundary(at(3 * DAY + 30), trigger="auto", pre_tokens=971_000, version="2.1.233")])
    run_report(tmp_path / "logs", tmp_path / "state.json", by="version", today=date(2026, 9, 6))
    assert capsys.readouterr().out.splitlines()[3:6] == [
        "version      first day   last day    responses  prompt turns  cache ratio  misses  haiku share  session start  compacts at",
        "2.1.99       2026-09-01  2026-09-03        180           177        0.900    0.0%        0.000             1k            -",
        "2.1.233      2026-09-04  2026-09-05        120           118        0.900    0.0%        0.000              -         970k",
    ]


def test_report_shows_hooks_and_subagent_models_over_its_days(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    write(tmp_path / "logs" / "s0" / "subagents" / "agent-a.jsonl", [
        line("p1", text(40), ts=at(100), sidechain=True, agent_type="Plan"),
        line("g1", text(40), ts=at(200), sidechain=True, agent_type="general-purpose", model="claude-sonnet-5"),
    ])
    write(tmp_path / "logs" / "hooks.jsonl", [stop_hook_summary(at(300), 1, durations=(1500,), uuid="h1"),
                                             stop_hook_summary(at(DAY + 300), 1, errors=("x",), durations=(500,),
                                                               uuid="h2")])
    run_report(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4))
    lines = capsys.readouterr().out.splitlines()
    assert "Hooks over these days: 2 stop-hook runs, errors on 1 day, median 1.0 s" in lines
    i = lines.index("Subagent models over these days (share of responses):")
    assert lines[i + 1:i + 3] == ["  Plan: claude-opus-5 100%",
                                  "  general-purpose (model picked by the caller): claude-sonnet-5 100%"]


def test_versions_sort_by_their_numbers():
    assert sorted(["2.1.233", "unknown", "2.1.99", "2.0.300"], key=version_key) == \
        ["2.0.300", "2.1.99", "2.1.233", "unknown"]


def test_report_json_holds_aggregates_without_paths_or_session_ids(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    assert run_report(tmp_path / "logs", tmp_path / "state.json", as_json=True, today=date(2026, 9, 4)) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert list(payload) == ["view", "days", "incidents", "reported_before_incidents", "settings", "hooks",
                             "subagents", "cutoffs", "flag_rule"]
    assert payload["days"][0] == {"day": "2026-09-01", "responses": 60, "cache_ratio": pytest.approx(0.9),
                                  "cache_z": None, "haiku_share": 0.0, "haiku_z": None, "flagged": []}
    assert payload["cutoffs"] == {"cache_ratio": -3.0, "haiku_fraction": 3.5}
    assert str(tmp_path) not in out and ".jsonl" not in out and '"s0"' not in out


@pytest.mark.parametrize("days", ["0", "-3", "two"])
def test_report_days_must_be_a_positive_whole_number(tmp_path, capsys, days):
    # --by version --days 0 showed every day, and negative values dropped days.
    with pytest.raises(SystemExit) as exited:
        main(["report", "--by", "version", "--days", days, "--source", str(tmp_path / "logs"),
              "--state", str(tmp_path / "state.json")])
    assert exited.value.code == 2
    assert capsys.readouterr().err.endswith(
        f"ccdrift report: error: argument --days: expected a whole number of days, 1 or more, not '{days}'\n")


def test_report_options_reach_the_report(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    assert main(["report", "--by", "version", "--json", "--days", "2", "--source", str(tmp_path / "logs"),
                 "--state", str(tmp_path / "state.json")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["view"] == "version"
    assert [v["version"] for v in payload["versions"]] == ["2.1.226"]


def test_report_by_version_quotes_release_notes_under_each_version(tmp_path, capsys):
    logs = tmp_path / "cfg" / "projects"
    main_thread_days(logs, [{"version": "2.1.99"}] * 2 + [{"version": "2.1.233"}] * 2)
    (tmp_path / "cfg" / "cache").mkdir(parents=True)
    (tmp_path / "cfg" / "cache" / "changelog.md").write_text(
        "## 2.1.233\n\n- Fixed prompt cache misses at turn boundaries\n- Fixed the /model picker showing disabled models\n"
        "- Hooks now receive the session's effort level\n- Fixed a cache warning after /rewind\n")
    run_report(logs, tmp_path / "state.json", by="version", today=date(2026, 9, 5))
    lines = capsys.readouterr().out.splitlines()
    row = next(i for i, text in enumerate(lines) if text.startswith("2.1.233"))
    # Heaviest first: the hooks note also names the effort level.
    assert lines[row + 1:row + 4] == ["    release notes: Hooks now receive the session's effort level",
                                      "    release notes: Fixed prompt cache misses at turn boundaries",
                                      ""]
