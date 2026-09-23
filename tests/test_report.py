"""ccdrift report: recent daily metrics, their z-scores and flags."""

import json
import os
import stat
from datetime import date

import pytest

from ccdrift.cli import main
from ccdrift.history import load_history
from ccdrift.logs import judged_turns
from ccdrift.report import _incident_days, daily_rows, run_report, version_key
from ccdrift.state import new_state, save_state
from ccdrift.texts import miss_reason_line
from tests.helpers import (DAY, HAIKU, QUIET, at, busy_days, compact_boundary, daily_turns,
                           damage_responses_table, line, main_thread_days, prompt, stop_hook_summary, text,
                           tool_result, write)


def test_report_lists_recent_days_with_their_metrics(tmp_path, capsys):
    # Under a project folder, as Claude Code writes them, so the report can name it.
    busy_days(tmp_path / "logs" / "-Users-me-app", days=3, per_day=60, cache_read=900, cache_creation=100)
    assert run_report(tmp_path / "logs", tmp_path / "state.json", days=2, today=date(2026, 9, 4)) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Last 2 complete UTC days with main-thread activity.",
        "Flagged once 3 of any 4 days in a row pass the cutoff: z <= -3.0 for the cache ratio, "
        "z >= +3.5 for Haiku share.",
        "",
        "day         responses  cache ratio      z  haiku share      z  loop misses  subagent misses  flagged",
        "2026-09-02         60        0.900      -        0.000      -            -                -",
        "2026-09-03         60        0.900      -        0.000      -            -                -",
        "",
        "Incidents: none yet",
        "",
        "Settings on the CLI main thread over these days (share of responses):",
        "  claude-opus-5: effort not logged 100%; speed not logged 100%; service tier not logged 100%",
        "",
        "Session starts by project over these days: /Users/me/app ~1k (2 sessions)",
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
    assert capsys.readouterr().out.splitlines()[:7] == [
        "Complete UTC days with main-thread activity, by Claude Code version.",
        "A miss is a new-prompt turn that reads less than half its input from the cache.",
        "A loop miss is a tool-loop turn that reads less than half of what the response before it had cached.",
        "",
        "version      first day   last day    responses  prompt turns  cache ratio  misses  loop misses  subagent misses"
        "  haiku share  session start  compacts at",
        "2.1.99       2026-09-01  2026-09-02        120           118        0.900    0.0%            -                -"
        "        0.000              -            -",
        "2.1.233      2026-09-03  2026-09-04        120           118        0.900    0.0%            -                -"
        "        0.000              -            -",
    ]


def test_report_by_version_shows_session_start_size_and_where_compaction_starts(tmp_path, capsys):
    # A session-start median over 1 or 2 sessions misleads (128k and 508k made 320k), so it needs 3.
    main_thread_days(tmp_path / "logs", [{"version": "2.1.99"}] * 3 + [{"version": "2.1.233"}] * 2)
    write(tmp_path / "logs" / "compacted.jsonl",
          [compact_boundary(at(3 * DAY + 30), trigger="auto", pre_tokens=971_000, version="2.1.233")])
    run_report(tmp_path / "logs", tmp_path / "state.json", by="version", today=date(2026, 9, 6))
    assert capsys.readouterr().out.splitlines()[4:7] == [
        "version      first day   last day    responses  prompt turns  cache ratio  misses  loop misses  subagent misses"
        "  haiku share  session start  compacts at",
        "2.1.99       2026-09-01  2026-09-03        180           177        0.900    0.0%            -                -"
        "        0.000             1k            -",
        "2.1.233      2026-09-04  2026-09-05        120           118        0.900    0.0%            -                -"
        "        0.000              -         970k",
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


def tool_loop_logs(path):
    """Sep 1-2 on 2.1.280 and Sep 3 on 2.1.281: a main-thread session a day of 1 prompt
    and 4 tool-loop turns a minute apart, the last missing the cache on Sep 2, and on
    Sep 3 a subagent of 1 prompt and 2 tool-loop turns, the second a miss."""
    for d, version in enumerate(["2.1.280", "2.1.280", "2.1.281"]):
        records = [prompt(at(d * DAY)), line(f"m{d}-0", text(40), ts=at(d * DAY), sid=f"s{d}", cache_creation=1000,
                                             version=version, entrypoint="cli")]
        for k in range(1, 5):
            read = 0 if (d == 1 and k == 4) else 1000 * k
            records += [tool_result(at(d * DAY + 60 * k), sid=f"s{d}"),
                        line(f"m{d}-{k}", text(40), ts=at(d * DAY + 60 * k), sid=f"s{d}", cache_read=read,
                             cache_creation=1000 if read else 1000 * (k + 1), version=version, entrypoint="cli")]
        write(path / f"s{d}.jsonl", records)
    write(path / "s2" / "subagents" / "agent-a.jsonl", [
        prompt(at(2 * DAY + 10), sid="s2", sidechain=True),
        line("a0", text(40), ts=at(2 * DAY + 10), sid="s2", sidechain=True, cache_creation=500, version="2.1.281",
             entrypoint="cli"),
        tool_result(at(2 * DAY + 20), sid="s2", sidechain=True),
        line("a1", text(40), ts=at(2 * DAY + 20), sid="s2", sidechain=True, cache_read=500, cache_creation=100,
             version="2.1.281", entrypoint="cli"),
        tool_result(at(2 * DAY + 30), sid="s2", sidechain=True),
        line("a2", text(40), ts=at(2 * DAY + 30), sid="s2", sidechain=True, cache_read=0, cache_creation=700,
             version="2.1.281", entrypoint="cli"),
    ])


def test_report_counts_tool_loop_misses_by_day_and_version(tmp_path, capsys):
    tool_loop_logs(tmp_path / "logs")
    run_report(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4))
    lines = capsys.readouterr().out.splitlines()
    assert lines[4:7] == [
        "2026-09-01          5            -      -        0.000      -          0/4                -",
        "2026-09-02          5            -      -        0.000      -          1/4                -",
        "2026-09-03          5            -      -        0.000      -          0/4              1/2",
    ]
    run_report(tmp_path / "logs", tmp_path / "state.json", by="version", today=date(2026, 9, 4))
    lines = capsys.readouterr().out.splitlines()
    assert lines[5:7] == [
        "2.1.280      2026-09-01  2026-09-02         10             0            -       -       12.50%                -"
        "        0.000              -            -",
        "2.1.281      2026-09-03  2026-09-03          5             0            -       -        0.00%           50.00%"
        "        0.000              -            -",
    ]
    run_report(tmp_path / "logs", tmp_path / "state.json", by="version", as_json=True, today=date(2026, 9, 4))
    versions = json.loads(capsys.readouterr().out)["versions"]
    assert [(v["loop_turns"], v["loop_misses"], v["subagent_loop_turns"], v["subagent_loop_misses"])
            for v in versions] == [(8, 1, 0, 0), (4, 0, 2, 1)]


def test_versions_sort_by_their_numbers():
    assert sorted(["2.1.233", "unknown", "2.1.99", "2.0.300"], key=version_key) == \
        ["2.0.300", "2.1.99", "2.1.233", "unknown"]


def test_a_version_part_python_calls_a_digit_but_cant_read_as_a_number_sorts_as_text():
    # "²" passes str.isdigit and fails int(); a version comes from the transcript as written.
    assert sorted(["2.1.²", "2.1.9"], key=version_key) == ["2.1.9", "2.1.²"]


def test_report_json_holds_aggregates_without_paths_or_session_ids(tmp_path, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    assert run_report(tmp_path / "logs", tmp_path / "state.json", as_json=True, today=date(2026, 9, 4)) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert list(payload) == ["view", "days", "incidents", "reported_before_incidents", "settings", "hooks",
                             "subagents", "failures", "miss_reasons", "cutoffs", "flag_rule"]
    assert payload["days"][0] == {"day": "2026-09-01", "responses": 60, "cache_ratio": pytest.approx(0.9),
                                  "cache_z": None, "haiku_share": 0.0, "haiku_z": None, "loop_turns": 0,
                                  "loop_misses": 0, "subagent_loop_turns": 0, "subagent_loop_misses": 0,
                                  "flagged": []}
    assert payload["cutoffs"] == {"cache_ratio": -3.0, "haiku_fraction": 3.5}
    assert str(tmp_path) not in out and ".jsonl" not in out and '"s0"' not in out


def test_the_day_view_says_why_the_cache_missed(tmp_path, capsys):
    main_thread_days(tmp_path / "p", [{}, {}])
    write(tmp_path / "p" / "extra.jsonl", [line("x1", text(40), ts=at(0), version="2.1.226",
                                                entrypoint="cli", miss_reason="system_changed")])
    state = tmp_path / "state.json"
    save_state(state, new_state())
    run_report(tmp_path / "p", state, today=date(2026, 9, 5))
    assert "the system prompt changed 1" in capsys.readouterr().out


def test_the_day_view_says_nothing_when_claude_code_recorded_no_reason(tmp_path, capsys):
    main_thread_days(tmp_path / "p", [{}, {}])
    state = tmp_path / "state.json"
    save_state(state, new_state())
    run_report(tmp_path / "p", state, today=date(2026, 9, 5))
    assert "Why the cache missed" not in capsys.readouterr().out


def test_a_reason_ccdrift_hasnt_seen_is_shown_as_itself():
    assert miss_reason_line({"something_new": 3}) == "something_new 3"


def test_reasons_are_listed_largest_first():
    assert miss_reason_line({"tools_changed": 2, "system_changed": 9}).startswith("the system prompt changed 9")


def test_reasons_tied_in_count_are_ordered_by_name():
    # "unavailable" is inserted first; "tools_changed" prints first anyway, by name.
    assert miss_reason_line({"unavailable": 2, "tools_changed": 2}) == \
        "the tools changed 2, the cache was unavailable 2"


def test_the_json_holds_the_reason_counts(tmp_path, capsys):
    main_thread_days(tmp_path / "p", [{}, {}])
    write(tmp_path / "p" / "extra.jsonl", [line("x1", text(40), ts=at(0), version="2.1.226",
                                                entrypoint="cli", miss_reason="tools_changed")])
    state = tmp_path / "state.json"
    save_state(state, new_state())
    run_report(tmp_path / "p", state, as_json=True, today=date(2026, 9, 5))
    assert json.loads(capsys.readouterr().out)["miss_reasons"] == {"tools_changed": 1}


def test_the_json_orders_tied_reasons_by_name(tmp_path, capsys):
    main_thread_days(tmp_path / "p", [{}, {}])
    # Written unavailable-then-tools_changed, so an unsorted report--json would show
    # them in that, hash-dependent, order; this pins the tie-break on that surface.
    write(tmp_path / "p" / "extra.jsonl", [line("x1", text(40), ts=at(0), version="2.1.226",
                                                entrypoint="cli", miss_reason="unavailable"),
                                          line("x2", text(40), ts=at(60), version="2.1.226",
                                                entrypoint="cli", miss_reason="tools_changed")])
    state = tmp_path / "state.json"
    save_state(state, new_state())
    run_report(tmp_path / "p", state, as_json=True, today=date(2026, 9, 5))
    payload = json.loads(capsys.readouterr().out)
    assert list(payload["miss_reasons"]) == ["tools_changed", "unavailable"]


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


def test_report_html_writes_a_self_contained_page_and_prints_where(tmp_path, capsys):
    main_thread_days(tmp_path / "logs" / "-Users-me-app", [{}] * 3)
    page = tmp_path / "report.html"
    assert run_report(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4),
                      html_path=page) == 0
    assert capsys.readouterr().out == f"wrote {page}\n"
    text = page.read_text()
    assert text.startswith("<!DOCTYPE html>\n")
    assert "<h1>ccdrift report, 2026-09-04</h1>" in text
    assert "2026-09-03" in text and "<script" not in text
    # Nothing is fetched when the page is opened: no links out, no images, no scripts.
    assert "http://" not in text and "https://" not in text and "<img" not in text


def entry(metric, start, end, status):
    return ({"metric": metric, "start": start, "end": end, "status": status}, 0.0)


def test_only_a_live_incident_on_the_charts_own_metric_shades_its_days():
    days = ["2026-09-02", "2026-09-03", "2026-09-04"]
    live = entry("cache_ratio", "2026-09-04", None, "open")          # open: runs to the last day shown
    recovered = entry("cache_ratio", "2026-09-01", "2026-09-02", "recovered")
    persistent = entry("cache_ratio", "2026-09-03", "2026-09-03", "persistent")
    # Dismissed: ccdrift put these days back in the baseline and scores them like any other,
    # so shading them would tell the reader to discount a dip ccdrift counts as normal.
    dismissed = entry("cache_ratio", "2026-09-02", "2026-09-04", "dismissed")
    haiku = entry("haiku_fraction", "2026-09-02", "2026-09-04", "recovered")
    older = entry("cache_ratio", "2026-08-01", "2026-08-10", "recovered")
    assert _incident_days([live], days, "cache_ratio") == ["2026-09-04"]
    assert _incident_days([recovered], days, "cache_ratio") == ["2026-09-02"]
    assert _incident_days([persistent], days, "cache_ratio") == ["2026-09-03"]
    assert _incident_days([dismissed], days, "cache_ratio") == []
    assert _incident_days([haiku], days, "cache_ratio") == []
    assert _incident_days([haiku], days, "haiku_fraction") == days
    assert _incident_days([older], days, "cache_ratio") == []
    assert _incident_days([live, recovered, persistent, dismissed, haiku, older], days, "cache_ratio") == days


def test_run_report_refuses_a_page_it_cannot_draw(tmp_path):
    page = tmp_path / "report.html"
    # The CLI refuses both, but the function doesn't trust its caller.
    with pytest.raises(ValueError, match="day view"):
        run_report(tmp_path / "logs", tmp_path / "state.json", by="version", html_path=page)
    with pytest.raises(ValueError, match="one output each"):
        run_report(tmp_path / "logs", tmp_path / "state.json", as_json=True, html_path=page)
    assert not page.exists()


def test_report_html_says_when_it_cannot_write_the_file(tmp_path, capsys):
    main_thread_days(tmp_path / "logs" / "-Users-me-app", [{}] * 3)
    assert run_report(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4),
                      html_path=tmp_path / "missing" / "report.html") == 1
    assert "Can't write" in capsys.readouterr().err


@pytest.mark.parametrize("existing", [False, True])
def test_report_html_writes_a_page_only_its_owner_can_read(tmp_path, existing):
    # The page names project folders, as the terminal report does, and says it isn't shareable.
    main_thread_days(tmp_path / "logs" / "-Users-me-app", [{}] * 3)
    page = tmp_path / "report.html"
    if existing:
        page.write_text("")
        page.chmod(0o644)
    umask = os.umask(0o022)
    try:
        assert run_report(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4), html_path=page) == 0
    finally:
        os.umask(umask)
    assert stat.S_IMODE(page.stat().st_mode) == 0o600
    assert "Users/me/app" in page.read_text()


def test_report_html_writes_to_a_path_under_the_home_folder_given_with_a_tilde(tmp_path, monkeypatch):
    # A quoted ~, or --html=~/..., reaches ccdrift without the shell expanding it.
    main_thread_days(tmp_path / "logs" / "-Users-me-app", [{}] * 3)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert main(["report", "--html", "~/report.html", "--source", str(tmp_path / "logs"),
                 "--state", str(tmp_path / "state.json")]) == 0
    assert (tmp_path / "report.html").exists()


def test_the_html_flag_takes_the_day_view_only_and_not_with_json(tmp_path, capsys):
    main_thread_days(tmp_path / "logs" / "-Users-me-app", [{}] * 3)
    page = tmp_path / "report.html"
    for argv in (["report", "--json", "--html", str(page)],
                 ["report", "--by", "version", "--html", str(page)]):
        with pytest.raises(SystemExit) as exited:
            main([*argv, "--source", str(tmp_path / "logs"), "--state", str(tmp_path / "state.json")])
        assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "--html draws the day view; drop --by version" in err
    # Both refusals print the report's own usage, the way argparse's own do.
    assert err.count("usage: ccdrift report") == 2 and "usage: ccdrift [-h]" not in err
