"""ccdrift incident draft: a GitHub issue draft about an incident, aggregates only."""

import platform
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from ccdrift import __version__
import ccdrift.draft
from ccdrift.changelog import TOPIC_OF, load_changelog
from ccdrift.cli import main
from ccdrift.detector import DetectorConfig
from ccdrift.draft import (alert_facts, draft_markdown, draft_periods, find_alert, find_incident, os_text, run_draft,
                          title_versions)
from ccdrift.failures import cut_short, failure_counts, judged_failures
from ccdrift.logs import judged_turns, parse_all, parse_source
from ccdrift.loops import LoopSetting, loop_warning
from ccdrift.state import new_state, save_state
from ccdrift.texts import draft_text, version_span
from tests.helpers import (DAY, at, busy_days, failure_days, line, main_thread_days, nth_day, prompt, text,
                           tool_loop_days, write)


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
    assert ("On 2026-09-16, 16 of 400 main-thread responses (4.00%) stopped at the token limit, against 5 of 5,600 "
            "(0.09%) on the 14 days before and 0 of 800 (0.00%) on the 2 days after. The check had reported this run "
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
