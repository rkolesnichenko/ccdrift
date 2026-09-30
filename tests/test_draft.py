"""ccdrift incident draft: a GitHub issue draft about an incident, aggregates only."""

import platform
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from ccdrift import __version__
import ccdrift.draft
from ccdrift.changelog import TOPIC_OF, load_changelog
from ccdrift.cli import main
from ccdrift.detector import DetectorConfig
from ccdrift.draft import (alert_facts, draft_markdown, draft_periods, find_alert, find_incident, hook_groups, os_text,
                          run_draft, title_versions)
from ccdrift.failures import cut_short, failure_counts, judged_failures
from ccdrift.hookcover import hook_coverage_alerts
from ccdrift.logs import judged_turns, parse_all, parse_source
from ccdrift.loops import LoopSetting, loop_warning
from ccdrift.sessions import context_alerts, session_starts
from ccdrift.state import new_state, save_state
from ccdrift.texts import draft_text, version_span
import tests.helpers as helpers
from tests.helpers import (DAY, PRIVATE_PATH, PRIVATE_TEXT, at, attachment, busy_days, deferred_tools, failure_days,
                           hook_record, line, main_thread_days, nth_day, partial_readback_days, prompt, prompt_snapshot,
                           skill_listing, subagent_history, text, tool_loop_days, tool_use, write)


BASELINE_NOTE = ("Before is the baseline ccdrift judged the incident against: the days it compared with, which skip "
                 "the days of other incidents, except ones dismissed or taken as the new normal, so they need not "
                 "run up to the day it started.\n\n")


def incident(metric, start, end, **fields):
    return {"metric": metric, "start": start, "end": end, "status": "recovered" if end else "open",
            "source": "check", "closed_by": "check" if end else None, "recovered_from": None, "opened_on": start,
            "closed_on": end, "versions": [], "cost": 0, **fields}


@pytest.mark.parametrize("metric, start, expected", [
    ("cache_ratio", None, "2026-09-15"),
    ("cache_ratio", "2026-08-18", "2026-08-18"),
    ("cache_ratio", "2026-09-20", "2026-09-20"),
    ("haiku_fraction", None, "2026-09-10"),
    ("haiku_fraction", "2026-08-18", None),
])
def test_a_draft_is_about_the_incident_that_starts_on_the_day_given_or_the_latest_not_dismissed(
        metric, start, expected):
    incidents = [incident("cache_ratio", "2026-08-18", "2026-09-03"),
                 incident("cache_ratio", "2026-09-15", "2026-09-16"),
                 incident("cache_ratio", "2026-09-20", "2026-09-21", status="dismissed"),
                 incident("haiku_fraction", "2026-09-10", None)]
    found = find_incident(incidents, metric, start)
    assert (found["start"] if found else None) == expected


def test_the_days_compared_are_the_baseline_the_incident_was_judged_against_and_14_days_after(tmp_path):
    # An earlier incident's days stay out of the baseline, a dismissed one's don't.
    main_thread_days(tmp_path / "logs", [{}] * 40)
    turns = judged_turns(parse_source(tmp_path / "logs"), date(2026, 10, 20))
    earlier = incident("cache_ratio", "2026-09-10", "2026-09-12")
    dismissed = incident("cache_ratio", "2026-09-05", "2026-09-06", status="dismissed")
    closed = incident("cache_ratio", "2026-09-20", "2026-09-22")
    periods = draft_periods(turns, closed, [earlier, closed, dismissed], DetectorConfig())
    assert periods["before"] == [f"2026-09-{day:02d}" for day in (3, 4, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 18, 19)]
    assert (periods["during"], periods["after"][0], len(periods["after"])) == (
        ["2026-09-20", "2026-09-21", "2026-09-22"], "2026-09-23", 14)
    still_open = incident("cache_ratio", "2026-09-20", None)
    periods = draft_periods(turns, still_open, [earlier, still_open], DetectorConfig())
    assert (periods["during"][-1], len(periods["during"]), periods["after"]) == ("2026-10-10", 21, [])


def test_after_stops_before_a_later_incident_a_dismissed_one_doesnt_and_persistent_has_none(tmp_path):
    main_thread_days(tmp_path / "logs", [{}] * 40)
    turns = judged_turns(parse_source(tmp_path / "logs"), date(2026, 10, 20))
    closed = incident("cache_ratio", "2026-09-20", "2026-09-22")
    later = incident("cache_ratio", "2026-09-30", "2026-10-02")
    periods = draft_periods(turns, closed, [closed, later], DetectorConfig())
    assert periods["after"] == [f"2026-09-{day:02d}" for day in range(23, 30)]

    dismissed = incident("cache_ratio", "2026-09-30", "2026-10-02", status="dismissed")
    periods = draft_periods(turns, closed, [closed, dismissed], DetectorConfig())
    assert (periods["after"][0], len(periods["after"])) == ("2026-09-23", 14)

    persistent = incident("cache_ratio", "2026-09-20", "2026-09-22", status="persistent")
    periods = draft_periods(turns, persistent, [persistent], DetectorConfig())
    assert periods["after"] == []


def cache_logs(path):
    """Sep 1-14 clean on 2.1.279; Sep 15-18 on 2.1.280, the last 6 of 60 prompts a day
    missing the cache; Sep 19-21 clean on 2.1.281; release notes for both new versions."""
    main_thread_days(path / "logs", [{"version": "2.1.279"}] * 14 + [{"misses": 6, "version": "2.1.280"}] * 4
                     + [{"version": "2.1.281"}] * 3)
    (path / "cache").mkdir()
    (path / "cache" / "changelog.md").write_text(
        "## 2.1.281\n\n- Fixed prompt cache misses on new prompts\n\n"
        "## 2.1.280\n\n- Changed how the prompt cache is keyed\n- Added a theme picker\n")


def drafted(path, found, today, os_name="macOS 26.5.2"):
    return draft_markdown(parse_source(path / "logs"), found, [found], load_changelog(path / "cache" / "changelog.md"),
                          today, DetectorConfig(), os_name)


CACHE_METHOD = (
    "### How this was measured\n\nccdrift reads Claude Code's local session transcripts. It counts main-thread turns "
    "that open with a new prompt within an hour of the previous response, outside Agent SDK sessions and not right "
    "after a compaction. A turn misses the cache when it reads less than half of its input from it. "
    "A day is a UTC day, and only complete ones are judged. Each day is compared with the median of "
    "up to 14 days before it, in units of their spread, which never falls below the noise a day of "
    "that many turns shows anyway; those days skip the days of other incidents, except ones dismissed "
    "or taken as the new normal. Then an incident "
    "opens when 3 of 4 days in a row fall below z = −3.0 and closes once 3 pooled days are back "
    "inside the cutoff on 3 days in a row, or after 30 days, when it takes the change as the new "
    "normal.\n")


def test_a_cache_incident_draft_holds_the_evidence_as_aggregates(tmp_path):
    cache_logs(tmp_path)
    found = incident("cache_ratio", "2026-09-15", "2026-09-18", recovered_from="2026-09-19")
    assert drafted(tmp_path, found, date(2026, 9, 25)) == (
        "New prompts miss the prompt cache 10.17% of the time on Claude Code 2.1.280 (usually 0.00%)\n\n"
        "### What happened\n\n"
        "From 2026-09-15 to 2026-09-18, 24 of 236 main-thread turns that open with a new prompt (10.17%) missed the "
        "prompt cache, against 0 of 826 (0.00%) on the 14 days before and 0 of 177 (0.00%) on the 3 days after. "
        "ccdrift estimates ~24k tokens were written to the cache again beyond the usual miss rate.\n\n"
        "### Before, during and after\n\n"
        + BASELINE_NOTE +
        "|  | Days | New-prompt turns | Misses | Miss rate | Cache read ratio |\n"
        "|---|---|---|---|---|---|\n"
        "| Before (09-01..09-14) | 14 | 826 | 0 | 0.00% | 0.900 |\n"
        "| During (09-15..09-18) | 4 | 236 | 24 | 10.17% | 0.808 |\n"
        "| After (09-19..09-21) | 3 | 177 | 0 | 0.00% | 0.900 |\n\n"
        "### By Claude Code version\n\n"
        "| Version | Period | Turns | Misses | Miss rate |\n"
        "|---|---|---|---|---|\n"
        "| 2.1.279 | before | 826 | 0 | 0.00% |\n"
        "| 2.1.280 | during | 236 | 24 | 10.17% |\n"
        "| 2.1.281 | after | 177 | 0 | 0.00% |\n\n"
        "### What a missed turn looks like\n\n"
        "The 24 missed turns during read a median 0 tokens from the cache (middle half 0–0) and wrote a median 1,000 "
        "(middle half 1,000–1,000), so each wrote most of its input to the cache again.\n\n"
        "### Pause before the prompt\n\n"
        "How long the turn waited between the previous response and the prompt that opened it.\n\n"
        "| Pause | Turns | Misses | Miss rate |\n"
        "|---|---|---|---|\n"
        "| ≤1 min | 236 | 24 | 10.17% |\n"
        "| 1–5 min | 0 | 0 | - |\n"
        "| 5–15 min | 0 | 0 | - |\n"
        "| 15–60 min | 0 | 0 | - |\n\n"
        "### Release notes that may be related\n\n"
        "- 2.1.280: Changed how the prompt cache is keyed\n\n"
        "### Environment\n\n"
        "- Claude Code: 2.1.280 (entrypoint cli)\n"
        "- Models during: claude-opus-5 (100.00% of responses)\n"
        "- Main thread: cache tier 1h on 100.00% of responses that write to the cache, effort xhigh on 100.00% of "
        "responses\n"
        "- OS: macOS 26.5.2\n"
        f"- Measured with ccdrift {__version__} from local session transcripts (aggregates only)\n\n"
        + CACHE_METHOD)


def test_a_draft_of_an_open_incident_says_it_is_still_going_and_has_no_after(tmp_path):
    cache_logs(tmp_path)
    text = drafted(tmp_path, incident("cache_ratio", "2026-09-15", None), date(2026, 9, 19), os_name="Linux 6.8.0")
    sections = text.split("\n\n")
    assert sections[2] == (
        "From 2026-09-15, still going as of 2026-09-18, 24 of 236 main-thread turns that open with a new prompt "
        "(10.17%) missed the prompt cache, against 0 of 826 (0.00%) on the 14 days before. ccdrift estimates ~24k "
        "tokens were written to the cache again beyond the usual miss rate.")
    assert "| After" not in text
    assert "- OS: Linux 6.8.0" in text


def test_a_persistent_incidents_draft_says_it_still_changed_and_has_no_after(tmp_path):
    cache_logs(tmp_path)
    found = incident("cache_ratio", "2026-09-15", "2026-09-18", status="persistent", closed_by="check")
    text = drafted(tmp_path, found, date(2026, 9, 25))
    assert text.split("\n\n")[2] == (
        "From 2026-09-15 to 2026-09-18, still changed after 30 days, 24 of 236 main-thread turns that open with a "
        "new prompt (10.17%) missed the prompt cache, against 0 of 826 (0.00%) on the 14 days before. ccdrift "
        "estimates ~24k tokens were written to the cache again beyond the usual miss rate.")
    assert "| After" not in text


def test_a_draft_leaves_out_usually_when_there_are_no_turns_before(tmp_path):
    cache_logs(tmp_path)
    found = incident("cache_ratio", "2026-09-01", "2026-09-03")
    text = drafted(tmp_path, found, date(2026, 9, 25))
    assert text.split("\n")[0] == "New prompts miss the prompt cache 0.00% of the time on Claude Code 2.1.279"


def test_a_draft_leaves_out_what_the_history_cant_say(tmp_path):
    # No misses during, no versions or settings logged, no changelog, no tool-loop turns.
    busy_days(tmp_path / "logs", days=20, per_day=60, cache_read=900, cache_creation=100)
    found = incident("cache_ratio", "2026-09-15", "2026-09-17")
    text = draft_markdown(parse_source(tmp_path / "logs"), found, [found], {}, date(2026, 9, 25), DetectorConfig(),
                          "macOS 26.5.2")
    assert text.split("\n")[0] == "New prompts miss the prompt cache 0.00% of the time (usually 0.00%)"
    assert [block.split("\n")[0] for block in text.split("\n\n") if block.startswith("###")] == [
        "### What happened", "### Before, during and after", "### Pause before the prompt", "### Environment",
        "### How this was measured"]
    assert ("### Environment\n\n- Claude Code: version not logged\n"
            "- Models during: claude-opus-5 (100.00% of responses)\n"
            "- Main thread: cache tier not logged, effort not logged\n") in text


def test_a_cache_draft_says_how_tool_loop_turns_fared(tmp_path):
    tool_loop_days(tmp_path / "logs", 21, misses=10)
    found = incident("cache_ratio", "2026-09-15", "2026-09-18")
    text = draft_markdown(parse_source(tmp_path / "logs"), found, [found], {}, date(2026, 9, 25), DetectorConfig(),
                          "macOS 26.5.2")
    assert ("### Tool-loop turns\n\nTurns inside the tool loop on the main thread missed 0 of 1,386 (0.00%) before, "
            "0 of 396 (0.00%) during and 10 of 297 (3.37%) after.") in text


def test_a_cache_draft_says_why_claude_code_recorded_the_misses(tmp_path):
    days = [{}] * 20 + [{"reason": "system_changed", "reasons": 4, "misses": 6}] * 3 + [{}] * 5
    main_thread_days(tmp_path / "logs", days)
    responses = parse_source(tmp_path / "logs")
    inc = incident("cache_ratio", nth_day(20), nth_day(22))
    drafted = draft_markdown(responses, inc, [inc], {}, date(2026, 10, 20), DetectorConfig(), "macOS 26.5.2")
    section = drafted.split("### Why the cache missed")[1]
    # 4 responses on each of the 3 days of the incident, of that period's 180.
    assert "| The system prompt changed | 0 (0.00%) | 12 (6.67%) |" in section
    # Answers the question "What a missed turn looks like" raises, so it follows that section
    # and comes before the next one, "Pause before the prompt".
    assert (drafted.index("### What a missed turn looks like") < drafted.index("### Why the cache missed")
           < drafted.index("### Pause before the prompt"))


def test_a_cache_draft_leaves_the_reason_section_out_when_none_was_recorded(tmp_path):
    main_thread_days(tmp_path / "logs", [{}] * 28)
    responses = parse_source(tmp_path / "logs")
    inc = incident("cache_ratio", nth_day(20), nth_day(22))
    drafted = draft_markdown(responses, inc, [inc], {}, date(2026, 10, 20), DetectorConfig(), "macOS 26.5.2")
    assert "Why the cache missed" not in drafted


def test_a_cache_draft_leaves_the_reason_section_out_when_only_before_or_after_carries_one(tmp_path):
    # The incident's during window (the middle 3 days) carries no reason; only the
    # days after it do. The section is about the during window, not any window.
    days = [{}] * 20 + [{}] * 3 + [{"reason": "unavailable", "reasons": 2}] * 5
    main_thread_days(tmp_path / "logs", days)
    responses = parse_source(tmp_path / "logs")
    inc = incident("cache_ratio", nth_day(20), nth_day(22))
    drafted = draft_markdown(responses, inc, [inc], {}, date(2026, 10, 20), DetectorConfig(), "macOS 26.5.2")
    assert "Why the cache missed" not in drafted


def test_a_haiku_incident_draft_compares_haiku_share_and_the_models_during(tmp_path):
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"haiku": 12, "version": "2.1.233"}] * 3
                     + [{"version": "2.1.259"}] * 5)
    found = incident("haiku_fraction", "2026-09-15", "2026-09-19", recovered_from="2026-09-20")
    text = draft_markdown(parse_source(tmp_path / "logs"), found, [found], {}, date(2026, 10, 10), DetectorConfig(),
                          "macOS 26.5.2")
    assert text == (
        "Haiku answers 12.00% of main-thread responses on Claude Code 2.1.233–2.1.259 (usually 0.00%)\n\n"
        "### What happened\n\n"
        "From 2026-09-15 to 2026-09-19, Haiku answered 36 of 300 main-thread responses (12.00%), against 0 of 840 "
        "(0.00%) on the 14 days before and 0 of 180 (0.00%) on the 3 days after: ~36 extra Haiku responses by "
        "ccdrift's estimate.\n\n"
        "### Before, during and after\n\n"
        + BASELINE_NOTE +
        "|  | Days | Responses | Haiku responses | Haiku share |\n"
        "|---|---|---|---|---|\n"
        "| Before (09-01..09-14) | 14 | 840 | 0 | 0.00% |\n"
        "| During (09-15..09-19) | 5 | 300 | 36 | 12.00% |\n"
        "| After (09-20..09-22) | 3 | 180 | 0 | 0.00% |\n\n"
        "### By Claude Code version\n\n"
        "| Version | Period | Responses | Haiku responses | Haiku share |\n"
        "|---|---|---|---|---|\n"
        "| 2.1.226 | before | 840 | 0 | 0.00% |\n"
        "| 2.1.233 | during | 180 | 36 | 20.00% |\n"
        "| 2.1.259 | during | 120 | 0 | 0.00% |\n"
        "| 2.1.259 | after | 180 | 0 | 0.00% |\n\n"
        "### Environment\n\n"
        "- Claude Code: 2.1.233, 2.1.259 (entrypoint cli)\n"
        "- Models during: claude-opus-5 (88.00% of responses), claude-haiku-4-5 (12.00% of responses)\n"
        "- Main thread: cache tier 1h on 100.00% of responses that write to the cache, effort xhigh on 100.00% of "
        "responses\n"
        "- OS: macOS 26.5.2\n"
        f"- Measured with ccdrift {__version__} from local session transcripts (aggregates only)\n\n"
        "### How this was measured\n\nccdrift reads Claude Code's local session transcripts. It counts main-thread "
        "responses outside Agent SDK sessions and the share answered by a Haiku model. A day is a UTC day, and only "
        "complete ones are judged. Each day is compared with the median of up to 14 days before it, in units of "
        "their spread, which never falls below the noise a day of that many turns shows anyway; those days skip the "
        "days of other incidents, except ones dismissed or taken as the new normal. Then an incident opens when 3 "
        "of 4 days in a row fall above z = +3.5 and closes "
        "once 3 pooled days are back inside the cutoff on 3 days in a row, or after 30 days, when it takes the "
        "change as the new normal.\n")


def test_the_environment_names_several_entrypoints_most_responses_first(tmp_path):
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"version": "2.1.280"}] * 3
                     + [{"version": "2.1.280", "entrypoint": "claude-vscode"}])
    found = incident("cache_ratio", "2026-09-15", "2026-09-18")
    text = draft_markdown(parse_source(tmp_path / "logs"), found, [found], {}, date(2026, 9, 25), DetectorConfig(),
                          "macOS 26.5.2")
    assert "- Claude Code: 2.1.280 (entrypoints cli, claude-vscode)" in text


@pytest.mark.parametrize("platform_name, mac_version, system, release, expected", [
    ("darwin", "26.5.2", "Darwin", "25.5.0", "macOS 26.5.2"),
    ("linux", "", "Linux", "6.8.0-45-generic", "Linux 6.8.0-45-generic"),
])
def test_the_os_is_named_with_its_version(monkeypatch, platform_name, mac_version, system, release, expected):
    monkeypatch.setattr(sys, "platform", platform_name)
    monkeypatch.setattr(platform, "mac_ver", lambda: (mac_version, ("", "", ""), ""))
    monkeypatch.setattr(platform, "system", lambda: system)
    monkeypatch.setattr(platform, "release", lambda: release)
    assert os_text() == expected


def test_run_draft_prints_the_draft_and_writes_nothing(tmp_path, capsys):
    cache_logs(tmp_path)
    save_state(tmp_path / "state.json",
               {**new_state(), "incidents": [incident("cache_ratio", "2026-09-15", "2026-09-18")]})
    before = sorted(path.name for path in tmp_path.iterdir())
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "cache_ratio", today=date(2026, 9, 25),
                     os_name="macOS 26.5.2") == 0
    assert capsys.readouterr().out.startswith(
        "New prompts miss the prompt cache 10.17% of the time on Claude Code 2.1.280 (usually 0.00%)\n\n")
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_run_draft_says_when_there_is_no_such_incident_or_it_cant_read(tmp_path, capsys):
    cache_logs(tmp_path)
    save_state(tmp_path / "state.json", new_state())
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "cache_ratio", today=date(2026, 9, 25)) == 2
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "haiku_fraction", "2026-09-15",
                     today=date(2026, 9, 25)) == 2
    err = capsys.readouterr().err
    assert "No cache incident is recorded.\n" in err and "No haiku incident starts on 2026-09-15.\n" in err
    save_state(tmp_path / "state.json",
               {**new_state(), "incidents": [incident("cache_ratio", "2026-09-15", "2026-09-18")]})
    (tmp_path / "empty").mkdir()
    assert run_draft(tmp_path / "empty", tmp_path / "state.json", "cache_ratio", today=date(2026, 9, 25)) == 2
    (tmp_path / "state.json").write_text("not json")
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "cache_ratio", today=date(2026, 9, 25)) == 1
    assert "Can't read the state file" in capsys.readouterr().err


def test_run_draft_refuses_an_incident_with_no_judged_days_during(tmp_path, capsys):
    cache_logs(tmp_path)
    save_state(tmp_path / "state.json",
               {**new_state(), "incidents": [incident("cache_ratio", "2026-10-01", "2026-10-03")]})
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "cache_ratio", today=date(2026, 10, 10)) == 2
    captured = capsys.readouterr()
    assert "The history holds no judged days during the cache incident from 2026-10-01.\n" in captured.err
    assert captured.out == ""


def test_the_draft_command_reads_the_metric_day_and_options(tmp_path, capsys):
    cache_logs(tmp_path)
    save_state(tmp_path / "state.json",
               {**new_state(), "incidents": [incident("cache_ratio", "2026-09-15", "2026-09-18")]})
    assert main(["incident", "draft", "cache", "2026-09-15", "--source", str(tmp_path / "logs"),
                 "--state", str(tmp_path / "state.json")]) == 0
    assert capsys.readouterr().out.startswith("New prompts miss the prompt cache 10.17% of the time")
    with pytest.raises(SystemExit) as exited:
        main(["incident", "draft", "cache", "Sep 15", "--state", str(tmp_path / "state.json")])
    assert exited.value.code == 2
    assert "expected a day like 2026-08-18, not 'Sep 15'" in capsys.readouterr().err


def test_a_subagent_cache_draft_counts_subagent_loop_turns_and_their_read_back(tmp_path, capsys):
    subagent_history(tmp_path / "logs", 25, miss_days=[16, 17, 18], version_from=16)
    save_state(tmp_path / "state.json", {**new_state(), "incidents": [
        incident("subagent_cache", "2026-09-17", "2026-09-21", recovered_from="2026-09-22")]})
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "subagent_cache", today=date(2026, 9, 26),
                     os_name="macOS 26.5") == 0
    out = capsys.readouterr().out
    assert out.startswith("Tool-loop turns in subagents read back 96.00% of what they had cached on Claude Code "
                          "2.1.300 (usually 100.00%)\n")
    assert ("From 2026-09-17 to 2026-09-21, tool-loop turns in subagents read back 96.00% of what the turn before had "
            "cached on average, against 100.00% on the 14 days before and 100.00% on the 4 days after; 60 of 1,500 "
            "(4.00%) missed the prompt cache, against 0 of 4,200 (0.00%) on the 14 days before and 0 of 1,200 (0.00%) "
            "on the 4 days after. ccdrift estimates ~1.8M tokens were written to the cache again beyond the usual "
            "read-back.") in out
    assert "| During (09-17..09-21) | 5 | 1,500 | 60 | 4.00% | 0.9600 |" in out
    assert "| 2.1.300 | during | 1,500 | 60 | 4.00% |" in out
    assert "- Subagents: cache tier not logged, effort not logged" in out
    assert ("It counts tool-loop turns in subagents, outside Agent SDK sessions: responses that follow a tool result "
            "within 5 minutes") in out
    assert "fall below z = −3.5 and closes" in out


def test_a_subagent_cache_draft_about_a_regression_without_misses_leads_with_its_read_back_and_its_cost(
        tmp_path, capsys):
    partial_readback_days(tmp_path / "logs", 22, from_day=16, share=0.8)
    save_state(tmp_path / "state.json", {**new_state(), "incidents": [incident("subagent_cache", "2026-09-17", None)]})
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "subagent_cache", today=date(2026, 9, 23),
                     os_name="macOS 26.5") == 0
    out = capsys.readouterr().out
    assert out.startswith("Tool-loop turns in subagents read back 80.00% of what they had cached on Claude Code "
                          "2.1.226 (usually 100.00%)\n")
    assert "; 0 of 1,800 (0.00%) missed the prompt cache" in out
    assert "no tokens were written to the cache again" not in out


def test_the_draft_command_takes_the_subagent_cache_metric(tmp_path, capsys):
    subagent_history(tmp_path / "logs", 25, miss_days=[16, 17, 18])
    save_state(tmp_path / "state.json", {**new_state(), "incidents": [
        incident("subagent_cache", "2026-09-17", "2026-09-21", recovered_from="2026-09-22")]})
    assert main(["incident", "draft", "subagent-cache", "--source", str(tmp_path / "logs"),
                 "--state", str(tmp_path / "state.json")]) == 0
    assert capsys.readouterr().out.startswith("Tool-loop turns in subagents read back 96.00% of what they had cached")


def test_the_title_names_the_versions_the_incident_ran_on_not_a_stale_session(tmp_path):
    # A session left open on an old version runs a few turns during the incident; naming it
    # would widen the range past what the incident was about.
    main_thread_days(tmp_path / "logs", [{"version": "2.1.279"}] * 14
                     + [{"misses": 6, "version": "2.1.280"}] * 4 + [{"version": "2.1.281"}] * 3)
    # Five turns a minute apart, so they count: the metric needs a previous response within the hour.
    stale = []
    for k in range(5):
        ts = at(15 * DAY + 60 * k)
        stale += [prompt(ts, sid="stale"),
                  line(f"stale-{k}", text(40), ts=ts, sid="stale", cache_read=900, cache_creation=100,
                       version="2.1.100", entrypoint="cli")]
    write(tmp_path / "logs" / "stale.jsonl", stale)
    found = incident("cache_ratio", "2026-09-15", "2026-09-18")
    drafted = draft_markdown(parse_source(tmp_path / "logs"), found, [found], {}, date(2026, 9, 25),
                             DetectorConfig(), "macOS 26.5.2")
    assert drafted.splitlines()[0].endswith("on Claude Code 2.1.280 (usually 0.00%)")
    # The floor is a share, not a rank: a version behind a tenth of the turns is still named.
    assert version_span(title_versions(pd.Series(["2.1.233"] * 73 + ["2.1.235"] * 221))) == "2.1.233–2.1.235"
    # It is still counted everywhere else: the version table keeps its row.
    assert "| 2.1.100 | during |" in drafted


def test_the_draft_picks_release_notes_without_the_check():
    # TOPIC_OF lives beside the release notes it chooses, so a draft needs nothing of the check.
    assert "ccdrift.check" not in Path(ccdrift.draft.__file__).read_text()
    assert TOPIC_OF["cache_ratio"] == "cache"


def test_run_draft_says_when_the_history_cant_be_read(tmp_path, capsys):
    cache_logs(tmp_path)
    save_state(tmp_path / "state.json",
               {**new_state(), "incidents": [incident("cache_ratio", "2026-09-15", "2026-09-18")]})
    (tmp_path / "history.sqlite").write_text("not a database")
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "cache_ratio", today=date(2026, 9, 25)) == 1
    assert "history store" in capsys.readouterr().err


def test_a_dismissed_incident_is_drafted_when_its_day_is_asked_for(tmp_path, capsys):
    cache_logs(tmp_path)
    save_state(tmp_path / "state.json", {**new_state(), "incidents": [
        incident("cache_ratio", "2026-09-15", "2026-09-18", status="dismissed", closed_by="user")]})
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "cache_ratio", "2026-09-15",
                     today=date(2026, 9, 25), os_name="macOS 26.5.2") == 0
    assert capsys.readouterr().out.startswith("New prompts miss the prompt cache 10.17% of the time")


# ---------------------------------------------------------------------------
# Drafts about alerts that are not incidents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("day, expected", [(None, "2026-09-22T11:33:00+00:00"),
                                           ("2026-09-22", "2026-09-22T11:33:00+00:00"),
                                           ("2026-09-10", "2026-09-10T08:00:00+00:00"),
                                           ("2026-09-11", None)])
def test_an_alert_draft_is_about_the_alert_that_starts_on_the_day_given_or_the_latest(day, expected):
    # A tool-loop warning's `since` is a time; the day asked for is its UTC date. Two
    # starting that day, as a main-thread and a subagent warning could, give the later.
    records = [{"since": "2026-09-10T08:00:00+00:00"}, {"since": "2026-09-22T11:33:00+00:00"},
               {"since": "2026-09-22T09:00:00+00:00"}, {"since": "2026-09-01T23:00:00+00:00"}]
    found = find_alert(records, day)
    assert (found["since"] if found else None) == expected
    assert find_alert([], None) is None


LOOP_VERSIONS = ["2.1.279"] * 21 + ["2.1.280"] + ["2.1.281"] * 3
LOOP_NOTES = {"2.1.280": ["Changed how the prompt cache is keyed in tool loops", "Added a theme picker"]}


def loop_warned(path, stream="main"):
    """Tool-loop turns from Sep 1 to 25 whose last 8 turns on Sep 22 miss the cache, and the
    warning the check raised about them that day, through the shipped rule. Subagents no
    longer get a warning, so theirs is raised at the setting they had before 0.16.0, as the
    owner's own subagent warning was."""
    tool_loop_days(path / "logs", 25, misses=8, miss_day=21, subagent=stream == "subagent", versions=LOOP_VERSIONS)
    tables = parse_all(path / "logs")
    setting = LoopSetting(p1=0.02, h=8.0, min_sessions=1) if stream == "subagent" else None
    state = new_state()
    for hour in range(24):
        warning = loop_warning(tables.responses, stream, state, datetime(2026, 9, 22, hour, 59, tzinfo=timezone.utc),
                               setting)
        if warning:
            return tables, warning
    raise AssertionError("the fixture raised no warning")


def test_a_tool_loop_draft_holds_the_rise_the_warning_was_about_and_the_days_around_it(tmp_path):
    tables, warning = loop_warned(tmp_path)
    # The draft counts the rise the warning counted, from the same turns.
    assert (warning["misses"], warning["turns"], warning["since"], warning["at"]) == (
        8, 8, "2026-09-22T11:33:00+00:00", "2026-09-22T11:40:00+00:00")
    facts = alert_facts(tables, "loop_warnings", warning, LOOP_NOTES, date(2026, 9, 26), "macOS 26.5.2")
    assert draft_text(facts) == (
        "Tool-loop turns on the main thread miss the prompt cache 100.00% of the time on Claude Code 2.1.280 "
        "(usually 0.00%)\n\n"
        "### What happened\n\n"
        "From 2026-09-22 11:33 to 11:40 UTC, 8 of 8 tool-loop turns on the main thread (100.00%) missed the prompt "
        "cache, in 1 session, against 0 of 1,386 (0.00%) on the 14 days before and 0 of 297 (0.00%) on the 3 days "
        "after; ~85k tokens were written to the cache again.\n\n"
        "### Before, during and after\n\n"
        "Before is the 14 days the warning took its usual miss rate from; during is the rise it warned about, from "
        "its first turn to the turn that raised it.\n\n"
        "|  | Days | Tool-loop turns | Misses | Miss rate |\n"
        "|---|---|---|---|---|\n"
        "| Before (09-02..09-15) | 14 | 1,386 | 0 | 0.00% |\n"
        "| During (09-22..09-22) | 1 | 8 | 8 | 100.00% |\n"
        "| After (09-23..09-25) | 3 | 297 | 0 | 0.00% |\n\n"
        "### By Claude Code version\n\n"
        "| Version | Period | Turns | Misses | Miss rate |\n"
        "|---|---|---|---|---|\n"
        "| 2.1.279 | before | 1,386 | 0 | 0.00% |\n"
        "| 2.1.280 | during | 8 | 8 | 100.00% |\n"
        "| 2.1.281 | after | 297 | 0 | 0.00% |\n\n"
        "### What a missed turn looks like\n\n"
        "The 8 missed turns during wrote a median 10,650 tokens to the cache (middle half 10,475–10,825), where the "
        "turn before had left what they needed cached.\n\n"
        "### Release notes that may be related\n\n"
        "- 2.1.280: Changed how the prompt cache is keyed in tool loops\n\n"
        "### Environment\n\n"
        "- Claude Code: 2.1.280 (entrypoint cli)\n"
        "- Models during: claude-opus-5 (100.00% of responses)\n"
        "- Main thread: cache tier not logged, effort not logged\n"
        "- OS: macOS 26.5.2\n"
        f"- Measured with ccdrift {__version__} from local session transcripts (aggregates only)\n\n"
        "### How this was measured\n\n"
        "ccdrift reads Claude Code's local session transcripts. It counts tool-loop turns on the main thread, outside "
        "Agent SDK sessions: responses that follow a tool result within 5 minutes of the previous response, not "
        "right after a compaction, where the response before had left tokens cached. A turn misses when it reads "
        "back less than 50% of what the turn before had cached. It runs a CUSUM over the turns of the last 7 days "
        "against the miss rate of the 14 days before them, and warns when it passes h = 3 with a miss rate of "
        "p1 = 2% in mind and misses from at least 1 session.\n")


def test_a_subagent_tool_loop_draft_says_where_it_ran_and_that_the_stream_no_longer_warns(tmp_path):
    tables, warning = loop_warned(tmp_path, "subagent")
    text = draft_text(alert_facts(tables, "loop_warnings", warning, {}, date(2026, 9, 26), "macOS 26.5.2"))
    assert text.startswith("Tool-loop turns in subagents miss the prompt cache 100.00% of the time")
    assert "\n- Subagents: cache tier not logged, effort not logged\n" in text
    assert "\n- Main thread:" not in text
    assert text.endswith("It ran a CUSUM over the turns of the last 7 days against the miss rate of the 14 days "
                         "before them; ccdrift no longer warns on this stream, since no setting it measured passes "
                         "its gate.\n")


def test_a_tool_loop_draft_of_one_missed_turn_says_what_it_wrote(tmp_path):
    # Found drafting the owner's own warning on 2026-09-27: a median of one turn, and its
    # middle half, read as nonsense. The shipped setting needs more than one miss; a looser
    # one, as subagents had, warned on one.
    tool_loop_days(tmp_path / "logs", 22, misses=1)
    tables = parse_all(tmp_path / "logs")
    warning = next(w for hour in range(24) for w in [loop_warning(tables.responses, "main", new_state(),
                                                                  datetime(2026, 9, 22, hour, 59, tzinfo=timezone.utc),
                                                                  LoopSetting(p1=0.02, h=2.0, min_sessions=1))]
                   if w)
    assert (warning["misses"], warning["turns"]) == (1, 1)
    text = draft_text(alert_facts(tables, "loop_warnings", warning, {}, date(2026, 9, 23), "macOS 26.5.2"))
    assert ("### What a missed turn looks like\n\nThe missed turn during wrote 11,000 tokens to the cache, where the "
            "turn before had left what it needed cached.\n") in text


def test_a_tool_loop_draft_drafted_the_day_it_fired_has_its_rise_and_no_after(tmp_path):
    # The warning fires mid-day, so the rise is counted by time, not by complete days.
    tables, warning = loop_warned(tmp_path)
    facts = alert_facts(tables, "loop_warnings", warning, {}, date(2026, 9, 22), "macOS 26.5.2")
    assert (facts["counts"]["during"], facts["periods"]["after"]) == ((8, 8), [])


def test_run_draft_drafts_a_tool_loop_warning_and_writes_nothing(tmp_path, capsys):
    tables, warning = loop_warned(tmp_path)
    save_state(tmp_path / "state.json", {**new_state(), "loop_warnings": [warning]})
    before = sorted(path.name for path in tmp_path.iterdir())
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "loop_warnings", "2026-09-22",
                     today=date(2026, 9, 26), os_name="macOS 26.5.2") == 0
    assert capsys.readouterr().out.startswith("Tool-loop turns on the main thread miss the prompt cache 100.00%")
    assert sorted(path.name for path in tmp_path.iterdir()) == before


def test_run_draft_says_when_there_is_no_such_alert_or_the_history_no_longer_holds_it(tmp_path, capsys):
    tables, warning = loop_warned(tmp_path)
    save_state(tmp_path / "state.json", new_state())
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "loop_warnings", today=date(2026, 9, 26)) == 2
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "loop_warnings", "2026-09-22",
                     today=date(2026, 9, 26)) == 2
    err = capsys.readouterr().err
    assert "No tool-loop alert is recorded.\n" in err and "No tool-loop alert starts on 2026-09-22.\n" in err
    moved = {**warning, "since": "2026-10-02T11:33:00+00:00", "at": "2026-10-02T11:40:00+00:00"}
    save_state(tmp_path / "state.json", {**new_state(), "loop_warnings": [moved]})
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "loop_warnings", today=date(2026, 10, 5)) == 2
    captured = capsys.readouterr()
    assert captured.err == "The history no longer holds the tool-loop alert from 2026-10-02.\n"
    assert captured.out == ""


def test_the_draft_command_takes_an_alert_kind(tmp_path, capsys):
    tables, warning = loop_warned(tmp_path)
    save_state(tmp_path / "state.json", {**new_state(), "loop_warnings": [warning]})
    assert main(["incident", "draft", "tool-loop", "2026-09-22", "--source", str(tmp_path / "logs"),
                 "--state", str(tmp_path / "state.json")]) == 0
    assert capsys.readouterr().out.startswith("Tool-loop turns on the main thread miss the prompt cache")


CUT_NOTES = {"2.1.280": ["Lowered the output limit for long responses", "Added a theme picker"]}


def cut_episodes(path, spec, per_day=200, days=range(15, 19)):
    """The main-thread responses of `spec` (failure_days) and the cut-short episodes the
    check recorded over them, running each morning of `days` (days after Sep 1) through the
    shipped rule."""
    failure_days(path / "logs", spec, per_day=per_day)
    tables = parse_all(path / "logs")
    state = new_state()
    for d in days:
        today = date(2026, 9, d + 1)
        cut_short(failure_counts(judged_failures(tables.failures, today), judged_turns(tables.responses, today)),
                  state, today)
    return tables, state["cut_short"]


CUT_SPEC = ([{"version": "2.1.279"}] * 14 + [{"version": "2.1.280", "truncated": 5, "refused": 1}] * 3
            + [{"version": "2.1.281"}] * 4)


def test_a_cut_short_draft_holds_the_run_the_check_reported_and_how_the_responses_stopped(tmp_path):
    tables, episodes = cut_episodes(tmp_path, CUT_SPEC)
    # One run, one episode: its first day, as the check reported it.
    assert [(e["since"], e["cut"], e["responses"]) for e in episodes] == [("2026-09-15", 6, 200)]
    facts = alert_facts(tables, "cut_short", episodes[0], CUT_NOTES, date(2026, 9, 22), "macOS 26.5.2")
    assert draft_text(facts) == (
        "Main-thread responses stop at the token limit or refuse 3.00% of the time on Claude Code 2.1.280 "
        "(usually 0.00%)\n\n"
        "### What happened\n\n"
        "From 2026-09-15 to 2026-09-17, 18 of 600 main-thread responses (3.00%) stopped at the token limit or "
        "refused, against 0 of 2,800 (0.00%) on the 14 days before and 0 of 800 (0.00%) on the 4 days after.\n\n"
        "### Before, during and after\n\n"
        "Before is the active days the check compared the first day with; during is that day and each day after it "
        "that stayed as high.\n\n"
        "|  | Days | Responses | Cut short | Share |\n"
        "|---|---|---|---|---|\n"
        "| Before (09-01..09-14) | 14 | 2,800 | 0 | 0.00% |\n"
        "| During (09-15..09-17) | 3 | 600 | 18 | 3.00% |\n"
        "| After (09-18..09-21) | 4 | 800 | 0 | 0.00% |\n\n"
        "### By Claude Code version\n\n"
        "| Version | Period | Responses | Cut short | Share |\n"
        "|---|---|---|---|---|\n"
        "| 2.1.279 | before | 2,800 | 0 | 0.00% |\n"
        "| 2.1.280 | during | 600 | 18 | 3.00% |\n"
        "| 2.1.281 | after | 800 | 0 | 0.00% |\n\n"
        "### How they stopped\n\n"
        "|  | At the token limit | Refused |\n"
        "|---|---|---|\n"
        "| Before | 0 | 0 |\n"
        "| During | 15 | 3 |\n"
        "| After | 0 | 0 |\n\n"
        "### Release notes that may be related\n\n"
        "- 2.1.280: Lowered the output limit for long responses\n\n"
        "### Environment\n\n"
        "- Claude Code: 2.1.280 (entrypoint cli)\n"
        "- Models during: claude-opus-5 (100.00% of responses)\n"
        "- Main thread: cache tier 1h on 100.00% of responses that write to the cache, effort xhigh on 100.00% of "
        "responses\n"
        "- OS: macOS 26.5.2\n"
        f"- Measured with ccdrift {__version__} from local session transcripts (aggregates only)\n\n"
        "### How this was measured\n\n"
        "ccdrift reads Claude Code's local session transcripts. It counts main-thread responses outside Agent SDK "
        "sessions, and those that stopped at the token limit or refused, on UTC days with at least 50 of them. A day "
        "is reported when at least 5 did, on at least 0.5% of its responses and 3 times the worst share of the days "
        "in the 14 before it that stand for the usual level, a clean day counting as 0.1% and the days of its own run "
        "left out; it needs 5 such days to compare with. A run is reported once, and again when a day stands 3 times "
        "above what was last reported of it.\n")


def test_a_cut_short_run_ends_at_the_first_day_under_the_share_and_only_truncation_says_so(tmp_path):
    # Sep 17 stays at 0.5% exactly and belongs to the run; Sep 18 at 0.4% ends it, and the
    # days after it are after however high they climb again.
    spec = ([{"version": "2.1.279"}] * 14 + [{"version": "2.1.280", "truncated": 6}] * 2
            + [{"version": "2.1.280", "truncated": 1}, {"version": "2.1.280", "truncated": 0}]
            + [{"version": "2.1.281", "truncated": 6}])
    tables, episodes = cut_episodes(tmp_path, spec, days=range(15, 16))
    facts = alert_facts(tables, "cut_short", episodes[0], {}, date(2026, 9, 25), "macOS 26.5.2")
    assert facts["periods"]["during"] == ["2026-09-15", "2026-09-16", "2026-09-17"]
    assert facts["periods"]["after"] == ["2026-09-18", "2026-09-19"]
    text = draft_text(facts)
    assert text.startswith("Main-thread responses stop at the token limit 2.17% of the time")
    assert "responses (2.17%) stopped at the token limit, against" in text


def test_a_cut_short_draft_of_a_run_that_deepened_says_what_was_reported_before(tmp_path):
    spec = ([{"version": "2.1.279"}] * 14 + [{"version": "2.1.280", "truncated": 5}]
            + [{"version": "2.1.280", "truncated": 16}] + [{"version": "2.1.281"}] * 2)
    tables, episodes = cut_episodes(tmp_path, spec, per_day=400, days=range(15, 17))
    assert [(e["since"], e.get("worse_than")) for e in episodes] == [
        ("2026-09-15", None), ("2026-09-16", {"share": 0.0125, "since": "2026-09-15"})]
    text = draft_text(alert_facts(tables, "cut_short", episodes[1], {}, date(2026, 9, 19), "macOS 26.5.2"))
    # Before is the check's own usual level: the days of the run it deepened are left out, as
    # the method paragraph says. Found in the final review of 2026-09-27.
    assert ("On 2026-09-16, 16 of 400 main-thread responses (4.00%) stopped at the token limit, against 0 of 5,200 "
            "(0.00%) on the 13 days before and 0 of 800 (0.00%) on the 2 days after. The check had reported this run "
            "at 1.25% on 2026-09-15; this day stood 3 times above it.") in text


def test_the_draft_command_takes_a_cut_short_alert(tmp_path, capsys):
    tables, episodes = cut_episodes(tmp_path, CUT_SPEC)
    save_state(tmp_path / "state.json", {**new_state(), "cut_short": episodes})
    assert main(["incident", "draft", "cut-short", "--source", str(tmp_path / "logs"),
                 "--state", str(tmp_path / "state.json")]) == 0
    assert capsys.readouterr().out.startswith("Main-thread responses stop at the token limit or refuse 3.00%")
    save_state(tmp_path / "state.json", {**new_state(), "cut_short": [{**episodes[0], "since": "2026-10-02",
                                                                      "days": ["2026-10-02"]}]})
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "cut_short", today=date(2026, 10, 5)) == 2
    assert capsys.readouterr().err == "The history no longer holds the cut-short alert from 2026-10-02.\n"


def test_an_alert_drafts_after_is_at_most_14_days(tmp_path):
    tool_loop_days(tmp_path / "loops", 40, misses=8, miss_day=21)
    loops = parse_all(tmp_path / "loops")
    warning = next(w for hour in range(24) for w in [loop_warning(loops.responses, "main", new_state(),
                                                                  datetime(2026, 9, 22, hour, 59, tzinfo=timezone.utc))]
                   if w)
    facts = alert_facts(loops, "loop_warnings", warning, {}, date(2026, 10, 15), "macOS 26.5.2")
    assert (facts["periods"]["after"][0], len(facts["periods"]["after"])) == ("2026-09-23", 14)
    spec = [{}] * 14 + [{"truncated": 6}] + [{}] * 16
    cuts, episodes = cut_episodes(tmp_path / "cut", spec, per_day=60, days=range(15, 16))
    facts = alert_facts(cuts, "cut_short", episodes[0], {}, date(2026, 10, 15), "macOS 26.5.2")
    assert (facts["periods"]["after"][0], len(facts["periods"]["after"])) == ("2026-09-16", 14)


def start_sessions(root, project, sizes, first_day, version, skills, deferred, tools):
    """One CLI session a day in `project` from `first_day`, each starting with the next of
    `sizes` tokens of context, the skills and deferred tools listed before its first
    response and the tool definitions in the snapshot after it."""
    for i, size in enumerate(sizes):
        d = first_day + i
        sid = f"{project}-{d}"
        write(root / project / f"{sid}.jsonl",
              [attachment(at(d * DAY), skill_listing(skills), sid=sid, version=version),
               attachment(at(d * DAY), deferred_tools(deferred), sid=sid, version=version),
               prompt(at(d * DAY + 1), sid=sid),
               line(f"m{sid}", text(40), ts=at(d * DAY + 2), sid=sid, cache_creation=100, cache_read=size - 110,
                    version=version, entrypoint="cli"),
               attachment(at(d * DAY + 3), prompt_snapshot(4000, tools=tools), sid=sid, version=version)])


def stepped(path, projects=(("-Users-me-alpha", 50_000),)):
    """A session a day in each project, 20 on 2.1.266 (the first ten 10% smaller, too little
    to be a step) and then 20 on 2.1.267 starting 60% larger than the ten before them, with
    a skill, an MCP server's tools and the built-in Monitor and WebFetch tools added; the
    step the check recorded about them on Sep 25, through the shipped rule."""
    for project, level in projects:
        start_sessions(path / "logs", project, [int(level * 0.9)] * 10 + [level] * 10, 0, "2.1.266", ["review"],
                       ["Read", "mcp__gh__pr"], {"Bash": 100, "Read": 50})
        start_sessions(path / "logs", project, [int(level * 1.6)] * 20, 20, "2.1.267", ["review", "deploy"],
                       ["Read", "WebFetch", "mcp__gh__pr", "mcp__private__x"],
                       {"Bash": 100, "Read": 50, "Monitor": 400, "mcp__private__x": 300})
    tables = parse_all(path / "logs")
    return tables, context_alerts(session_starts(tables.responses), new_state(), date(2026, 9, 25))


START_NOTES = {"2.1.267": ["Added the Monitor tool to the tool definitions", "Added a theme picker"]}


def test_a_session_start_draft_holds_the_step_against_each_projects_level_and_what_changed(tmp_path):
    tables, records = stepped(tmp_path)
    assert [(r["since"], r["from"], r["to"]) for r in records] == [("2026-09-21", 50_000, 80_000)]
    text = draft_text(alert_facts(tables, "context_changes", records[0], START_NOTES, date(2026, 10, 11),
                                  "macOS 26.5.2"))
    assert text == (
        "Session start grew from ~50k to ~80k tokens on Claude Code 2.1.267\n\n"
        "### What happened\n\n"
        "From 2026-09-21, CLI sessions started at a median ~80k tokens, against ~50k in the sessions before them. "
        "Against each project's level when the step began they started at 1.60 times it, against 1.00 times on the "
        "14 days before. The step showed in 1 of 1 project compared.\n\n"
        "### Before and after\n\n"
        "A session's start is the context its first response sent: input, cache writes and cache reads. Each is also "
        "given against its project's level when the step began, the median of that project's last sessions before "
        "it, so moving between projects doesn't read as a change.\n\n"
        "|  | Days | Sessions | Median start | Against its project |\n"
        "|---|---|---|---|---|\n"
        "| Before (09-07..09-20) | 14 | 14 | ~50k | 1.00 |\n"
        "| After (09-21..10-07) | 17 | 17 | ~80k | 1.60 |\n\n"
        "### By Claude Code version\n\n"
        "| Version | Period | Sessions | Median start | Against its project |\n"
        "|---|---|---|---|---|\n"
        "| 2.1.266 | before | 14 | ~50k | 1.00 |\n"
        "| 2.1.267 | after | 17 | ~80k | 1.60 |\n\n"
        "### What changed at the start of the session\n\n"
        "Of what Claude Code logs about a session's start, 1 skill, 1 MCP tool, 1 deferred tool and 2 tool "
        "definitions were added, about 920 more characters, though the logs can't say how many of the tokens that "
        "is. The agent types, CLAUDE.md files or the system prompt weren't logged in every session compared, so "
        "ccdrift couldn't compare those parts. Built-in tools added: Monitor, WebFetch.\n\n"
        "### Release notes that may be related\n\n"
        "- 2.1.267: Added the Monitor tool to the tool definitions\n\n"
        "### Environment\n\n"
        "- Claude Code: 2.1.267 (entrypoint cli)\n"
        "- Models during: claude-opus-5 (100.00% of responses)\n"
        "- Main thread: cache tier not logged, effort not logged\n"
        "- OS: macOS 26.5.2\n"
        f"- Measured with ccdrift {__version__} from local session transcripts (aggregates only)\n\n"
        "### How this was measured\n\n"
        "ccdrift reads Claude Code's local session transcripts. It takes the first response of each CLI main-thread "
        "session, outside Agent SDK sessions and not a resumed one, and the context it sent. Each session is measured "
        "against its own project's level, the median of up to 10 of that project's earlier sessions once it has 3. A "
        "step is reported when 3 sessions in a row move at least 25% from the median of up to 10 before them, with at "
        "least 5, each of them more than 12.5% on the same side. A project counts as compared when it has 3 sessions "
        "each side of the step within 14 days.\n")
    # A skill or an MCP server is the user's own configuration: counted, never named.
    assert "deploy" not in text and "private" not in text


def test_a_session_start_draft_of_a_step_in_every_project_says_so_and_cant_compare_across_them(tmp_path):
    tables, records = stepped(tmp_path, (("-Users-me-alpha", 50_000), ("-Users-me-beta", 30_000)))
    text = draft_text(alert_facts(tables, "context_changes", records[0], {}, date(2026, 10, 11), "macOS 26.5.2"))
    assert "The step showed in every project compared (2 of 2).\n" in text
    assert ("### What changed at the start of the session\n\nThe sessions compared don't log enough of how they "
            "started for ccdrift to compare its parts.\n") in text
    assert "alpha" not in text and "beta" not in text


def test_the_draft_command_takes_a_session_start_alert_the_history_still_shows(tmp_path, capsys):
    tables, records = stepped(tmp_path)
    save_state(tmp_path / "state.json", {**new_state(), "context_changes": records})
    assert main(["incident", "draft", "session-start", "2026-09-21", "--source", str(tmp_path / "logs"),
                 "--state", str(tmp_path / "state.json")]) == 0
    assert capsys.readouterr().out.startswith("Session start grew from ~50k to ~80k tokens on Claude Code 2.1.267\n")
    # A step the rule no longer finds, as after a transcript was deleted, can't be drafted.
    save_state(tmp_path / "state.json", {**new_state(), "context_changes": [{**records[0], "since": "2026-09-05"}]})
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "context_changes", today=date(2026, 10, 11)) == 2
    assert capsys.readouterr().err == "The history no longer holds the session-start alert from 2026-09-05.\n"


def hook_change(stream, since, thread="subagent", direction="started", alerted=False):
    return {"stream": f"{stream}|{thread}|PreToolUse|Bash", "thread": thread, "direction": direction,
            "after": since, "since": since, "reported_on": since, "alerted": alerted}


def test_hook_records_are_drafted_as_the_alerts_they_make_up_alerted_or_not():
    # One record per stream; a Claude Code update reaches every stream within days, and
    # the first check after upgrading records such a change without alerting. 14 days
    # after a group's first is still that group, as in hookcover.merged; 15 isn't.
    records = [hook_change("b", "2026-09-06"), hook_change("a", "2026-09-05", alerted=True),
               hook_change("d", "2026-09-19"), hook_change("c", "2026-09-20"),
               hook_change("a", "2026-09-20", direction="stopped"), hook_change("a", "2026-09-06", thread="main")]
    groups = hook_groups(records)
    assert [(g["thread"], g["direction"], g["since"], [r["stream"][0] for r in g["records"]]) for g in groups] == [
        ("main", "started", "2026-09-06", ["a"]),
        ("subagent", "started", "2026-09-05", ["a", "b", "d"]),
        ("subagent", "started", "2026-09-20", ["c"]),
        ("subagent", "stopped", "2026-09-20", ["a"])]
    assert find_alert(groups, "2026-09-05")["records"][1]["stream"].startswith("b|")
    assert find_alert(groups)["since"] == "2026-09-20"


HOOK_TOOLS = ("Bash", "Read", "mcp__vault__read")
ALL_HOOKS = frozenset((event, tool) for event in ("PreToolUse", "PostToolUse") for tool in (*HOOK_TOOLS, "Grep"))


def hooked_sessions(root, project, days, first_day, version, subagent_hooks, tools=HOOK_TOOLS):
    """A CLI session a day in `project` from `first_day`, calling each of `tools` (Bash,
    Read and an MCP server's tool) four times on the main thread and in a subagent. Every
    main-thread call gets a PreToolUse and a PostToolUse hook record; the subagent's get the
    (event, tool) pairs in `subagent_hooks`."""
    for i in range(days):
        d = first_day + i
        sid = f"{project}-{d}"
        main, sub = [prompt(at(d * DAY), sid=sid)], []
        for k, tool in enumerate(t for t in tools for _ in range(4)):
            for records, sidechain, hooks in ((main, False, ALL_HOOKS), (sub, True, subagent_hooks)):
                tid = f"toolu_{'s' if sidechain else 'm'}{d}_{k}"
                ts = d * DAY + 20 * k + (10 if sidechain else 0)
                records.append(line(f"{tid}-r", tool_use(tid, tool), ts=at(ts), sid=sid, sidechain=sidechain,
                                    version=version, entrypoint="cli", cache_read=900, cache_creation=100))
                records += [hook_record(at(ts + 1), event, tid, tool, sid=sid, version=version, sidechain=sidechain)
                            for event in ("PreToolUse", "PostToolUse") if (event, tool) in hooks]
        write(root / project / f"{sid}.jsonl", main)
        write(root / project / sid / "subagents" / "agent-a.jsonl", sub)


def hooks_started(path, today=date(2026, 9, 25), before=frozenset(), after=ALL_HOOKS):
    """Two projects' sessions, 16 on 2.1.247 with the subagent hook records of `before`
    (none), then 20 on 2.1.261 with those of `after` (all) and Grep called too, beside a
    third project whose subagents never had any; the hook coverage records the check made
    on `today`."""
    for project in ("-Users-me-alpha", "-Users-me-beta"):
        hooked_sessions(path / "logs", project, 16, 0, "2.1.247", before)
        hooked_sessions(path / "logs", project, 20, 16, "2.1.261", after, tools=(*HOOK_TOOLS, "Grep"))
    hooked_sessions(path / "logs", "-Users-me-gamma", 36, 0, "2.1.261", frozenset())
    tables = parse_all(path / "logs")
    state = new_state()
    hook_coverage_alerts(tables.hook_coverage, state, today)
    return tables, state["hook_changes"]


def test_a_hook_coverage_draft_holds_the_change_by_transcript_version_event_and_tool(tmp_path):
    tables, records = hooks_started(tmp_path)
    group = find_alert(hook_groups(records))
    assert (group["since"], len(group["records"])) == ("2026-09-17", 12)
    text = draft_text(alert_facts(tables, "hook_changes", group, {"2.1.261": ["Hooks now run on subagent tool calls"]},
                                  date(2026, 10, 8), "macOS 26.5.2"))
    assert text == (
        "Hooks started running on subagent tool calls from Claude Code 2.1.261\n\n"
        "### What happened\n\n"
        "Before 2026-09-17, PreToolUse hooks ran on 0 of 336 and PostToolUse hooks on 0 of 336 CLI subagent tool calls; "
        "from then, on 408 of 408 and 408 of 408. The change showed in 2 projects.\n\n"
        "### Before and after\n\n"
        "Judged is a transcript's calls of one tool for one hook event, where it made enough of them; it counts as "
        "hooked when a hook record came on at least half of those calls.\n\n"
        "|  | Days | Transcripts | Judged | Hooked | Share |\n"
        "|---|---|---|---|---|---|\n"
        "| Before (09-03..09-16) | 14 | 28 | 168 | 0 | 0.00% |\n"
        "| After (09-17..10-03) | 17 | 34 | 204 | 204 | 100.00% |\n\n"
        "### By Claude Code version\n\n"
        "| Version | Period | Judged | Hooked | Share |\n"
        "|---|---|---|---|---|\n"
        "| 2.1.247 | before | 168 | 0 | 0.00% |\n"
        "| 2.1.261 | after | 204 | 204 | 100.00% |\n\n"
        "### Which events and tools\n\n"
        "Tool calls a hook record came on, of all the calls.\n\n"
        "| Event | Tool | Before | After |\n"
        "|---|---|---|---|\n"
        "| PreToolUse | Bash | 0 of 112 (0.00%) | 136 of 136 (100.00%) |\n"
        "| PreToolUse | Grep | - | 136 of 136 (100.00%) |\n"
        "| PreToolUse | Read | 0 of 112 (0.00%) | 136 of 136 (100.00%) |\n"
        "| PreToolUse | MCP tools (1 server) | 0 of 112 (0.00%) | 136 of 136 (100.00%) |\n"
        "| PostToolUse | Bash | 0 of 112 (0.00%) | 136 of 136 (100.00%) |\n"
        "| PostToolUse | Grep | - | 136 of 136 (100.00%) |\n"
        "| PostToolUse | Read | 0 of 112 (0.00%) | 136 of 136 (100.00%) |\n"
        "| PostToolUse | MCP tools (1 server) | 0 of 112 (0.00%) | 136 of 136 (100.00%) |\n\n"
        "### Release notes that may be related\n\n"
        "- 2.1.261: Hooks now run on subagent tool calls\n\n"
        "### Environment\n\n"
        "- Claude Code: 2.1.261 (entrypoint cli)\n"
        "- Models during: claude-opus-5 (100.00% of responses)\n"
        "- Subagents: cache tier not logged, effort not logged\n"
        "- OS: macOS 26.5.2\n"
        f"- Measured with ccdrift {__version__} from local session transcripts (aggregates only)\n\n"
        "### How this was measured\n\n"
        "ccdrift reads Claude Code's local session transcripts. For each CLI transcript it counts the calls of each "
        "tool, an MCP server's tools together, and whether Claude Code logged a PreToolUse or PostToolUse hook record "
        "on each. A transcript is hooked for an event and tool when a hook record came on at least half of its calls, "
        "with at least 3 of them. A project's transcripts in one thread, for one event and tool, make a stream; a "
        "change is reported when its last 3 transcripts all turned the other way from 100% of the 10 before them. "
        "Changes in one thread and direction starting within 14 days of each other are one alert.\n")
    # The hook commands, tool ids, MCP server and projects are the user's own.
    for private in ("vault", "alpha", "beta", "gamma", "toolu_", PRIVATE_PATH, PRIVATE_TEXT):
        assert private not in text


def test_the_draft_command_takes_a_hook_change_the_check_recorded_without_alerting(tmp_path, capsys):
    # A first check long after the change records it quietly; it is still the one to file.
    tables, records = hooks_started(tmp_path, today=date(2026, 10, 20))
    assert records and not any(record["alerted"] for record in records)
    save_state(tmp_path / "state.json", {**new_state(), "hook_changes": records})
    assert main(["incident", "draft", "hooks", "2026-09-17", "--source", str(tmp_path / "logs"),
                 "--state", str(tmp_path / "state.json")]) == 0
    assert capsys.readouterr().out.startswith("Hooks started running on subagent tool calls from Claude Code 2.1.261\n")
    # An alert is named by the day it starts, not by a later stream's; one the history
    # doesn't show, in that direction or near that day, can't be drafted.
    later = [{**records[0], "since": "2026-09-19"}, *records[1:]]
    save_state(tmp_path / "state.json", {**new_state(), "hook_changes": later})
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "hook_changes", "2026-09-19",
                     today=date(2026, 10, 20)) == 2
    for changed in ({"direction": "stopped"}, {"since": "2026-08-20", "after": "2026-08-20"}):
        save_state(tmp_path / "state.json", {**new_state(), "hook_changes": [{**r, **changed} for r in records]})
        assert run_draft(tmp_path / "logs", tmp_path / "state.json", "hook_changes", today=date(2026, 10, 20)) == 2
    assert capsys.readouterr().err == ("No hooks alert starts on 2026-09-19.\n"
                                       "The history no longer holds the hooks alert from 2026-09-17.\n"
                                       "The history no longer holds the hooks alert from 2026-08-20.\n")


def test_a_hook_change_on_some_events_and_tools_counts_only_the_streams_that_changed(tmp_path):
    # PostToolUse hooks ran on every subagent call throughout, and PreToolUse ones on the MCP
    # server's; 2.1.261 started PreToolUse hooks on Bash and Read alone.
    always = frozenset({("PostToolUse", tool) for tool in HOOK_TOOLS} | {("PreToolUse", "mcp__vault__read")})
    tables, records = hooks_started(tmp_path, before=always, after=ALL_HOOKS)
    facts = alert_facts(tables, "hook_changes", find_alert(hook_groups(records)), {}, date(2026, 10, 8),
                        "macOS 26.5.2")
    assert (facts["change"]["events"], facts["counts"]) == (["PreToolUse"], {"before": (0, 56), "after": (68, 68)})
    assert "Before 2026-09-17, PreToolUse hooks ran on 0 of 224 CLI subagent tool calls; from then, on 272 of 272." \
        in draft_text(facts)


@pytest.mark.parametrize("kind, word", [("context_changes", "session-start"), ("hook_changes", "hooks"),
                                        ("loop_warnings", "tool-loop"), ("cut_short", "cut-short")])
def test_no_alert_draft_names_a_folder_session_skill_mcp_server_hook_command_or_tool_id(tmp_path, capsys, kind, word):
    # A draft is meant for a public issue. Each kind's fixture, run through the command with
    # its transcripts in a project folder whose name is private; the fixtures also carry
    # a skill, an MCP server, hook commands, a working directory and tool ids of their own.
    records = {"context_changes": lambda: stepped(tmp_path)[1],
               "hook_changes": lambda: hooks_started(tmp_path)[1],
               "loop_warnings": lambda: [loop_warned(tmp_path)[1]],
               "cut_short": lambda: cut_episodes(tmp_path, CUT_SPEC)[1]}[kind]()
    logs = tmp_path / "logs"
    (logs / "-Users-me-secretproject").mkdir()
    for path in logs.glob("*.jsonl"):
        path.rename(logs / "-Users-me-secretproject" / path.name)
    # A subagent of the user's own on a branch of theirs; one response, too few to move any draft.
    write(logs / "-Users-me-secretproject" / "s-own" / "subagents" / "agent-own.jsonl",
          [line("own", text(40), ts=at(10 * DAY), sid="s-own", sidechain=True, agent_type="ownagentname",
                branch="ownbranchname", version="2.1.261", entrypoint="cli")])
    save_state(tmp_path / "state.json", {**new_state(), kind: records})
    assert main(["incident", "draft", word, "--source", str(logs), "--state", str(tmp_path / "state.json")]) == 0
    out = capsys.readouterr().out
    for private in (str(tmp_path), "secretproject", "-Users-me-", "alpha", "beta", "gamma", "deploy", "vault",
                    "private", "ownagentname", "ownbranchname", "toolu_", ".jsonl", PRIVATE_PATH, PRIVATE_TEXT):
        assert private not in out
    if kind == "hook_changes":  # Claude Code's own tools are named: the rule is names, not the lack of them
        assert "| PreToolUse | Bash |" in out


# ---------------------------------------------------------------------------
# Found in the final review of 2026-09-27
# ---------------------------------------------------------------------------

def test_a_tool_loop_draft_counts_the_turn_that_raised_the_warning_though_claude_code_logs_milliseconds(
        tmp_path, monkeypatch):
    # The record keeps `since` and `at` to the second; Claude Code logs milliseconds, so the
    # alarm turn fell after `at` and a one-turn rise couldn't be drafted at all.
    monkeypatch.setattr(helpers, "at", lambda seconds: (helpers.T0 + timedelta(seconds=seconds, milliseconds=537))
                        .isoformat().replace("+00:00", "Z"))
    tables, warning = loop_warned(tmp_path)
    facts = alert_facts(tables, "loop_warnings", warning, {}, date(2026, 9, 26), "macOS 26.5.2")
    assert facts["counts"]["during"] == (warning["misses"], warning["turns"]) == (8, 8)
    tool_loop_days(tmp_path / "one", 22, misses=1)
    one = parse_all(tmp_path / "one")
    warning = next(w for hour in range(24) for w in [loop_warning(one.responses, "main", new_state(),
                                                                  datetime(2026, 9, 22, hour, 59, tzinfo=timezone.utc),
                                                                  LoopSetting(p1=0.02, h=2.0, min_sessions=1))]
                   if w)
    assert alert_facts(one, "loop_warnings", warning, {}, date(2026, 9, 23), "macOS 26.5.2")["counts"]["during"] == (1, 1)


def test_a_cut_short_alert_whose_day_no_longer_reaches_the_share_is_no_longer_held(tmp_path, capsys):
    # Claude Code's cleanup can delete the transcript that held the cut responses while
    # another from that day survives: the day is still active, its share is gone.
    tables, episodes = cut_episodes(tmp_path, CUT_SPEC)
    save_state(tmp_path / "state.json", {**new_state(), "cut_short": [{**episodes[0], "since": "2026-09-10",
                                                                      "days": ["2026-09-10"]}]})
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "cut_short", today=date(2026, 9, 22)) == 2
    assert capsys.readouterr().err == "The history no longer holds the cut-short alert from 2026-09-10.\n"


def main_thread_hooks(root, project, days, hooked_tools, version, tools=("Bash", "Read"), inline_subagent=0):
    """A CLI session a day in `project` on `days`, calling each of `tools` four times on the
    main thread, with a PreToolUse hook record on the calls of `hooked_tools`, and
    `inline_subagent` unhooked Bash calls of a subagent written into the same transcript,
    as Claude Code wrote subagents before it gave them transcripts of their own."""
    for d in days:
        sid = f"{project}-{d}"
        records = [prompt(at(d * DAY), sid=sid)]
        for k, tool in enumerate(t for t in tools for _ in range(4)):
            tid, ts = f"toolu_m{d}_{k}", d * DAY + 20 * k
            records.append(line(f"{tid}-r", tool_use(tid, tool), ts=at(ts), sid=sid, version=version, entrypoint="cli",
                                cache_read=900, cache_creation=100))
            if tool in hooked_tools:
                records.append(hook_record(at(ts + 1), "PreToolUse", tid, tool, sid=sid, version=version))
        for k in range(inline_subagent):
            tid = f"toolu_s{d}_{k}"
            records.append(line(f"{tid}-r", tool_use(tid, "Bash"), ts=at(d * DAY + 1000 + k), sid=sid, sidechain=True,
                                version=version, entrypoint="cli", cache_read=900, cache_creation=100))
        write(root / project / f"{sid}.jsonl", records)


def hook_facts_of(path, today):
    tables = parse_all(path / "logs")
    state = new_state()
    hook_coverage_alerts(tables.hook_coverage, state, today)
    return alert_facts(tables, "hook_changes", find_alert(hook_groups(state["hook_changes"])), {}, today, "macOS 26.5.2")


def test_a_hook_draft_counts_the_streams_that_changed_not_every_pairing_of_their_projects_events_and_tools(tmp_path):
    # One project hooked Bash only and the other Read only; one update stopped both.
    main_thread_hooks(tmp_path / "logs", "-Users-me-alpha", range(0, 14), {"Bash"}, "2.1.260")
    main_thread_hooks(tmp_path / "logs", "-Users-me-alpha", range(14, 20), set(), "2.1.270")
    main_thread_hooks(tmp_path / "logs", "-Users-me-beta", range(0, 14), {"Read"}, "2.1.260")
    main_thread_hooks(tmp_path / "logs", "-Users-me-beta", range(14, 20), set(), "2.1.270")
    facts = hook_facts_of(tmp_path, date(2026, 9, 22))
    assert facts["counts"] == {"before": (28, 28), "after": (0, 12)}
    assert ("Before 2026-09-15, PreToolUse hooks ran on 112 of 112 CLI main-thread tool calls; from then, on 0 of 48."
            in draft_text(facts))


def test_a_hook_draft_of_a_stream_idle_across_the_step_compares_with_what_the_rule_compared(tmp_path):
    # The stream's last transcripts before the step came three weeks earlier; the rule
    # compared with them, and so does the draft.
    main_thread_hooks(tmp_path / "logs", "-Users-me-alpha", range(0, 12), {"Bash"}, "2.1.260", tools=("Bash",))
    main_thread_hooks(tmp_path / "logs", "-Users-me-alpha", range(32, 37), set(), "2.1.270", tools=("Bash",))
    facts = hook_facts_of(tmp_path, date(2026, 10, 8))
    assert (facts["counts"], len(facts["periods"]["before"])) == ({"before": (10, 10), "after": (0, 5)}, 10)
    assert "Before 2026-10-03, PreToolUse hooks ran on 40 of 40 CLI main-thread tool calls; from then, on 0 of 20." \
        in draft_text(facts)


def test_a_session_start_draft_of_a_step_in_one_project_of_several_measures_that_project(tmp_path):
    # Six larger projects held steady while one stepped: pooled, the step read as 1.00 against 1.00.
    start_sessions(tmp_path / "logs", "-Users-me-alpha", [50_000] * 20, 0, "2.1.266", ["r"], ["Read"], {"Bash": 100})
    start_sessions(tmp_path / "logs", "-Users-me-alpha", [80_000] * 20, 20, "2.1.266", ["r"], ["Read"], {"Bash": 100})
    for project in ("-Users-me-beta", "-Users-me-gamma", "-Users-me-delta"):
        start_sessions(tmp_path / "logs", project, [120_000] * 40, 0, "2.1.266", ["r"], ["Read"], {"Bash": 100})
    tables = parse_all(tmp_path / "logs")
    records = context_alerts(session_starts(tables.responses), new_state(), date(2026, 9, 25))
    assert [(r["projects"], r["of_projects"]) for r in records] == [(["-Users-me-alpha"], 4)]
    facts = alert_facts(tables, "context_changes", records[0], {}, date(2026, 10, 11), "macOS 26.5.2")
    assert (facts["medians"]["before"][1:], facts["medians"]["after"][1:]) == ((50_000, 1.0), (80_000, 1.6))


def test_a_hook_draft_of_a_later_change_leaves_out_the_streams_of_an_earlier_one(tmp_path):
    # Two stops a month apart in one thread are two alerts; the later one's draft counts
    # only its own stream, not the stream that stopped weeks before.
    main_thread_hooks(tmp_path / "logs", "-Users-me-alpha", range(0, 14), {"Bash"}, "2.1.260", tools=("Bash",))
    main_thread_hooks(tmp_path / "logs", "-Users-me-alpha", range(14, 20), set(), "2.1.270", tools=("Bash",))
    main_thread_hooks(tmp_path / "logs", "-Users-me-beta", range(0, 40), {"Read"}, "2.1.260", tools=("Read",))
    main_thread_hooks(tmp_path / "logs", "-Users-me-beta", range(40, 46), set(), "2.1.280", tools=("Read",))
    facts = hook_facts_of(tmp_path, date(2026, 10, 18))
    assert (facts["change"]["since"], facts["counts"]) == ("2026-10-11", {"before": (14, 14), "after": (0, 6)})


# ---------------------------------------------------------------------------
# 0.17.1: what the final review of 2026-09-27 deferred
# ---------------------------------------------------------------------------

def test_a_hook_draft_names_no_unknown_version(tmp_path):
    # transcript_states reads a transcript that logged no version as "unknown"; the title
    # and the version table leave it out, as every other draft does.
    main_thread_hooks(tmp_path / "logs", "-Users-me-alpha", range(0, 14), {"Bash"}, "2.1.260", tools=("Bash",))
    main_thread_hooks(tmp_path / "logs", "-Users-me-alpha", range(14, 20), set(), None, tools=("Bash",))
    text = draft_text(hook_facts_of(tmp_path, date(2026, 9, 22)))
    assert text.startswith("Hooks stopped running on main-thread tool calls\n\n")
    assert "unknown" not in text


def test_a_tool_loop_draft_of_a_rise_across_midnight_gives_both_days(tmp_path):
    tables, warning = loop_warned(tmp_path)
    facts = alert_facts(tables, "loop_warnings", warning, {}, date(2026, 9, 26), "macOS 26.5.2")
    facts["alert"] = {**warning, "since": "2026-09-21T23:50:00+00:00", "at": "2026-09-22T00:10:00+00:00"}
    assert "From 2026-09-21 23:50 to 2026-09-22 00:10 UTC, 8 of 8 tool-loop turns" in draft_text(facts)


def test_a_session_start_record_is_drafted_only_from_a_change_in_its_own_direction(tmp_path):
    # A change found on the record's day but going the other way is a different step.
    tables, records = stepped(tmp_path)
    shrank = {**records[0], "from": records[0]["to"], "to": records[0]["from"]}
    assert alert_facts(tables, "context_changes", shrank, {}, date(2026, 10, 11), "macOS 26.5.2") is None


def test_a_main_thread_hook_draft_leaves_out_subagent_calls_written_into_the_same_transcripts(tmp_path):
    # Older Claude Code wrote a subagent's tool calls into its session's own transcript; they
    # are the subagent thread's, whatever file they sit in.
    main_thread_hooks(tmp_path / "logs", "-Users-me-alpha", range(0, 14), {"Bash"}, "2.1.260", tools=("Bash",),
                      inline_subagent=2)
    main_thread_hooks(tmp_path / "logs", "-Users-me-alpha", range(14, 20), set(), "2.1.270", tools=("Bash",),
                      inline_subagent=2)
    facts = hook_facts_of(tmp_path, date(2026, 9, 22))
    assert (facts["change"]["thread"], facts["ran"]["before"]["PreToolUse"]) == ("main", (56, 56))
    assert [(event, tool, cells["before"]) for event, tool, cells in facts["calls"]] == [
        ("PreToolUse", "Bash", (56, 56)), ("PostToolUse", "Bash", (56, 0))]


def test_two_hook_alerts_starting_the_same_day_are_both_drafted(tmp_path, capsys):
    # The subagents started running hooks on 2026-09-17, the day another project's main
    # thread stopped: one day, two alerts, and a day alone can't say which is meant.
    tables, records = hooks_started(tmp_path)
    main_thread_hooks(tmp_path / "logs", "-Users-me-delta", range(0, 16), {"Bash"}, "2.1.247")
    main_thread_hooks(tmp_path / "logs", "-Users-me-delta", range(16, 36), set(), "2.1.261")
    state = new_state()
    hook_coverage_alerts(parse_all(tmp_path / "logs").hook_coverage, state, date(2026, 9, 25))
    assert sorted({(r["thread"], r["direction"], r["since"]) for r in state["hook_changes"]}) == [
        ("main", "stopped", "2026-09-17"), ("subagent", "started", "2026-09-17")]
    save_state(tmp_path / "state.json", state)
    assert run_draft(tmp_path / "logs", tmp_path / "state.json", "hook_changes", "2026-09-17",
                     today=date(2026, 10, 8), os_name="macOS 26.5.2") == 0
    captured = capsys.readouterr()
    drafts = captured.out.split("\n---\n\n")
    assert [draft.splitlines()[0] for draft in drafts] == [
        "Hooks stopped running on main-thread tool calls from Claude Code 2.1.261",
        "Hooks started running on subagent tool calls from Claude Code 2.1.261"]
    assert captured.err == "2 hooks alerts start on 2026-09-17; each is drafted below.\n"

