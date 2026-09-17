"""ccdrift incident draft: a GitHub issue draft about an incident, aggregates only."""

import platform
import sys
from datetime import date

import pytest

from ccdrift import __version__
from ccdrift.changelog import load_changelog
from ccdrift.detector import DetectorConfig
from ccdrift.draft import draft_markdown, draft_periods, find_incident, os_text
from ccdrift.logs import judged_turns, parse_source
from tests.helpers import busy_days, main_thread_days, tool_loop_days


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
    "after a compaction. A turn misses the cache when it reads less than half of its input from it. Each day is "
    "compared with up to 14 days before it; an incident opens when 3 of 4 days in a row fall below z = −3.0 and "
    "closes once 3 pooled days are back inside the cutoff on 3 days in a row.\n")


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
        "| Pause | Turns | Misses | Miss rate |\n"
        "|---|---|---|---|\n"
        "| ≤1 min | 236 | 24 | 10.17% |\n"
        "| 1–5 min | 0 | 0 | - |\n"
        "| 5–15 min | 0 | 0 | - |\n"
        "| 15–60 min | 0 | 0 | - |\n\n"
        "### Release notes that may be related\n\n"
        "- 2.1.280: Changed how the prompt cache is keyed\n\n"
        "### Environment\n\n"
        "- Claude Code: 2.1.280 (CLI)\n"
        "- Models during: claude-opus-5 (100.00% of responses)\n"
        "- Main thread: cache tier 1h on 100.00% of responses, effort xhigh on 100.00% of responses\n"
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
        "| 2.1.259 | during, after | 300 | 0 | 0.00% |\n\n"
        "### Models during\n\n"
        "claude-opus-5 88.00%, claude-haiku-4-5 12.00%\n\n"
        "### Environment\n\n"
        "- Claude Code: 2.1.233, 2.1.259 (CLI)\n"
        "- Models during: claude-opus-5 (88.00% of responses), claude-haiku-4-5 (12.00% of responses)\n"
        "- Main thread: cache tier 1h on 100.00% of responses, effort xhigh on 100.00% of responses\n"
        "- OS: macOS 26.5.2\n"
        f"- Measured with ccdrift {__version__} from local session transcripts (aggregates only)\n\n"
        "### How this was measured\n\nccdrift reads Claude Code's local session transcripts. It counts main-thread "
        "responses outside Agent SDK sessions and the share answered by a Haiku model. Each day is compared with up "
        "to 14 days before it; an incident opens when 3 of 4 days in a row fall above z = +3.5 and closes once 3 "
        "pooled days are back inside the cutoff on 3 days in a row.\n")


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
