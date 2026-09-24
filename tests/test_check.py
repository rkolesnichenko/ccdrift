"""The daily check: new flags once, and alerts when it fails or can't compute the
cache metric."""

import os
import stat
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

import ccdrift.check
import ccdrift.history
import ccdrift.loops
from ccdrift.check import blank_cache_stretch, run_check
from ccdrift.incidents import add_incident, close_incident
from ccdrift.logs import parse_source
from ccdrift.loops import LoopSetting
from ccdrift.sessions import context_found, session_starts
from ccdrift.state import load_state, new_state, save_state
from ccdrift.status import short_status, status_report
from tests.helpers import (DAY, agent_listing, at, attachment, busy_days, damage_responses_table, deferred_tools,
                           hook_days_logs, line, main_thread_days, nth_day, prompt, skill_listing, text, tool_loop_days,
                           write)


@pytest.fixture
def sent(monkeypatch):
    titles = []
    monkeypatch.setattr(ccdrift.check, "notify", lambda title, message: titles.append(title))
    return titles


def ran_before(tmp_path):
    """The state of a check that has run successfully before, so the next one follows
    incidents as usual instead of replaying the history as a first check does."""
    save_state(tmp_path / "state.json", {**new_state(), "last_ok": "2026-09-01T09:00:00+00:00"})


def check_logs(tmp_path, today=date(2026, 9, 4), **options):
    """Run the check as the schedule would at 09:00 UTC on `today`, without the weekly digest."""
    options.setdefault("now", datetime.combine(today, time(9, 0), tzinfo=timezone.utc))
    options.setdefault("digest", False)
    return run_check(tmp_path / "logs", tmp_path / "state.json", notify_user=True, today=today, **options)


def test_check_alerts_when_busy_days_show_no_cache_usage(tmp_path, sent):
    # Every Claude Code response reads or writes the prompt cache, so busy days
    # without cache token counts mean the parser no longer finds them.
    busy_days(tmp_path / "logs", days=3, per_day=60)
    check_logs(tmp_path)
    assert sent == ["ccdrift can't compute the cache metric"]


def test_check_alerts_when_busy_days_show_no_prompts(tmp_path, sent):
    busy_days(tmp_path / "logs", days=3, per_day=60, prompts=False, cache_read=900, cache_creation=100)
    check_logs(tmp_path)
    assert sent == ["ccdrift can't compute the cache metric"]


def test_check_stays_quiet_when_busy_days_have_cache_values(tmp_path, sent):
    busy_days(tmp_path / "logs", days=3, per_day=60, cache_read=900, cache_creation=100)
    check_logs(tmp_path)
    assert sent == []


def test_check_alerts_about_unusable_cache_values_once(tmp_path, sent):
    busy_days(tmp_path / "logs", days=4, per_day=60)
    check_logs(tmp_path, today=date(2026, 9, 4))
    check_logs(tmp_path, today=date(2026, 9, 5))
    assert sent == ["ccdrift can't compute the cache metric"]


def test_check_ignores_quiet_days_without_cache_values(tmp_path, sent):
    # A quick question a day can leave no new-prompt turn to measure. The quietest
    # of 29 real days still had 81 main-thread responses and 3 cache values.
    busy_days(tmp_path / "logs", days=3, per_day=5)
    check_logs(tmp_path)
    assert sent == []


def test_check_alerts_when_it_cannot_run(tmp_path, sent):
    # A broken check would otherwise look like a quiet week.
    (tmp_path / "logs").mkdir()
    assert check_logs(tmp_path) != 0
    assert sent == ["ccdrift check failed"]


def test_the_check_takes_everyone_but_its_owner_off_the_file_its_output_goes_to(tmp_path, monkeypatch):
    # launchd, systemd and cron recreate a deleted log with the umask's permissions, and
    # the log holds alert text and error messages that name folders.
    busy_days(tmp_path / "logs", days=3, per_day=60)
    log = tmp_path / "check.log"
    with open(log, "a") as out:
        log.chmod(0o644)
        monkeypatch.setattr(sys, "stdout", out)
        run_check(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4))
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert "ccdrift" in log.read_text()


def test_the_check_leaves_a_file_of_the_users_own_its_output_is_sent_to_as_it_is(tmp_path, monkeypatch):
    # `ccdrift check >> ~/logs/all.log` is the user's file, shared or not as they chose.
    busy_days(tmp_path / "logs", days=3, per_day=60)
    log = tmp_path / "all.log"
    with open(log, "a") as out:
        log.chmod(0o644)
        monkeypatch.setattr(sys, "stdout", out)
        run_check(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4))
    assert stat.S_IMODE(log.stat().st_mode) == 0o644


def test_the_check_carries_on_past_a_transcript_the_parser_fails_on_and_its_log_names_it(
        tmp_path, monkeypatch, capsys):
    # Before, it failed every hour with that exception until Claude Code deleted the transcript.
    main_thread_days(tmp_path / "logs" / "p", [{}] * 3)
    write(tmp_path / "logs" / "p" / "bad.jsonl", [line("b1", text(40), ts=at(0))])
    real = ccdrift.history.parse_file

    def parse(fp, rel):
        if rel == "p/bad.jsonl":
            raise RuntimeError("the parser tripped on this transcript")
        return real(fp, rel)

    monkeypatch.setattr(ccdrift.history, "parse_file", parse)
    assert run_check(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4)) == 0
    assert load_state(tmp_path / "state.json")["last_run"]["ok"]
    assert "Skipped p/bad.jsonl: RuntimeError: the parser tripped" in capsys.readouterr().err


def test_cache_metric_alert_points_to_ccdrift_peek(tmp_path, sent, capsys):
    # The alert suggested --schema-peek, the lab harness's option, which the
    # installed command doesn't have.
    busy_days(tmp_path / "logs", days=3, per_day=60)
    check_logs(tmp_path)
    out = capsys.readouterr().out
    assert "run `ccdrift peek`" in out
    assert "--schema-peek" not in out


def test_check_alerts_when_a_flag_opens_an_incident(tmp_path, sent, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"haiku": 12, "version": "2.1.233"}] * 3)
    ran_before(tmp_path)
    assert check_logs(tmp_path, today=date(2026, 9, 18)) == 0
    assert sent == ["ccdrift flag"]
    assert ("ccdrift flag: Haiku share on the main thread up from 2026-09-15, on Claude Code 2.1.233 "
            "(since 09-15). ~36 extra Haiku responses so far.") in capsys.readouterr().out


def test_check_alerts_when_an_incident_is_back_to_normal(tmp_path, sent, capsys):
    days = [{}] * 14 + [{"haiku": 12, "version": "2.1.233"}] * 3 + [{"version": "2.1.259"}] * 5
    main_thread_days(tmp_path / "logs", days)
    ran_before(tmp_path)
    check_logs(tmp_path, today=date(2026, 9, 18))
    check_logs(tmp_path, today=date(2026, 9, 23))
    assert sent == ["ccdrift flag", "ccdrift: back to normal"]
    assert ("ccdrift: back to normal: Haiku share on the main thread back to normal from 2026-09-20, on Claude "
            "Code 2.1.259 (since 09-18). The incident from 2026-09-15: ~36 extra Haiku responses.") \
        in capsys.readouterr().out


def test_an_incident_back_to_normal_keeps_the_versions_it_started_on(tmp_path, sent):
    # `ccdrift status` and `incident list` showed the versions of the recovery days instead.
    days = [{}] * 14 + [{"haiku": 12, "version": "2.1.233"}] * 3 + [{"version": "2.1.259"}] * 5
    main_thread_days(tmp_path / "logs", days)
    ran_before(tmp_path)
    check_logs(tmp_path, today=date(2026, 9, 18))
    check_logs(tmp_path, today=date(2026, 9, 23))
    assert sent == ["ccdrift flag", "ccdrift: back to normal"]
    assert load_state(tmp_path / "state.json")["incidents"][0]["versions"] == ["2.1.233 (since 09-15)"]


def test_a_back_to_normal_alert_quotes_the_version_it_names_first(tmp_path, sent, capsys):
    logs = tmp_path / "cfg" / "projects"
    days = [{}] * 14 + [{"haiku": 12, "version": "2.1.233"}] * 3 + [{"version": "2.1.259"}] * 5
    main_thread_days(logs, days)
    (tmp_path / "cfg" / "cache").mkdir(parents=True)
    (tmp_path / "cfg" / "cache" / "changelog.md").write_text(
        "## 2.1.259\n\n- Search subagents run on Sonnet again instead of Haiku\n\n"
        "## 2.1.233\n\n- Search subagents now run on the Haiku model\n")
    ran_before(tmp_path)
    run_check(logs, tmp_path / "state.json", today=date(2026, 9, 18))
    capsys.readouterr()
    run_check(logs, tmp_path / "state.json", today=date(2026, 9, 23))
    assert [text for text in capsys.readouterr().out.splitlines() if "release notes" in text] == [
        "    release notes 2.1.259: Search subagents run on Sonnet again instead of Haiku",
        "    release notes 2.1.233: Search subagents now run on the Haiku model",
    ]


def test_check_works_out_the_cost_and_versions_of_an_incident_added_by_hand(tmp_path, sent):
    # `ccdrift status` reads only the state file, so the check keeps both there.
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"misses": 6, "version": "2.1.233"}] * 3)
    state = new_state()
    add_incident(state["incidents"], "cache_ratio", "2026-09-15", "2026-09-17", date(2026, 9, 18))
    save_state(tmp_path / "state.json", state)
    now = datetime(2026, 9, 18, 9, 0, tzinfo=timezone.utc)
    assert run_check(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 18), now=now) == 0
    saved = load_state(tmp_path / "state.json")
    assert (saved["incidents"][0]["cost"], saved["incidents"][0]["versions"]) == (18_000, ["2.1.233 (since 09-15)"])
    assert status_report(saved, now).splitlines()[1:4] == [
        "Open incidents: none",
        "Closed in the last 30 days:",
        "  cache  2026-09-15..2026-09-17    added by hand; ~18k tokens re-cached; on 2.1.233 (since 09-15)",
    ]
    assert sent == []


def test_check_refreshes_the_cost_of_an_incident_closed_by_hand(tmp_path, sent):
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"misses": 6, "version": "2.1.233"}] * 3)
    state = new_state()
    state["incidents"].append({"metric": "cache_ratio", "start": "2026-09-15", "end": None, "status": "open",
                               "source": "check", "closed_by": None, "recovered_from": None,
                               "opened_on": "2026-09-17", "closed_on": None,
                               "versions": ["2.1.233 (since 09-15)"], "cost": 0})
    close_incident(state["incidents"], "cache_ratio", date(2026, 9, 18))
    save_state(tmp_path / "state.json", state)
    check_logs(tmp_path, today=date(2026, 9, 18))
    assert load_state(tmp_path / "state.json")["incidents"][0]["cost"] == 18_000
    assert sent == []


def test_check_works_out_the_cost_of_an_incident_added_by_hand_once(tmp_path, sent):
    # Its days don't change once it's closed, and reading back to an old incident's
    # start on every hourly run would read the whole history.
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"misses": 6, "version": "2.1.233"}] * 3)
    state = new_state()
    add_incident(state["incidents"], "cache_ratio", "2026-09-15", "2026-09-17", date(2026, 9, 18))
    save_state(tmp_path / "state.json", state)
    check_logs(tmp_path, today=date(2026, 9, 18))
    saved = load_state(tmp_path / "state.json")
    assert saved["incidents"][0]["cost"] == 18_000
    saved["incidents"][0]["cost"] = 1_234
    save_state(tmp_path / "state.json", saved)
    check_logs(tmp_path, today=date(2026, 9, 19))
    assert load_state(tmp_path / "state.json")["incidents"][0]["cost"] == 1_234


def test_check_reads_back_to_an_incident_added_by_hand_from_months_ago(tmp_path, sent):
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"misses": 6}] * 3 + [{}] * 113)
    state = new_state()
    add_incident(state["incidents"], "cache_ratio", "2026-09-15", "2026-09-17", date(2027, 1, 9))
    save_state(tmp_path / "state.json", state)
    check_logs(tmp_path, today=date(2027, 1, 9))
    assert load_state(tmp_path / "state.json")["incidents"][0]["cost"] == 18_000


def test_check_flags_a_regression_right_after_a_long_break(tmp_path, sent):
    # 60 active days, 100 days away, then 4 days on a new version missing the cache:
    # the baseline comes from before the break.
    main_thread_days(tmp_path / "logs", [{}] * 60)
    main_thread_days(tmp_path / "logs", [{"misses": 8, "version": "2.1.300"}] * 4, first_day=160)
    ran_before(tmp_path)
    check_logs(tmp_path, today=date(2027, 2, 12))
    assert sent == ["ccdrift flag"]


def test_check_names_the_day_a_version_first_ran_from_before_the_history_it_reads(tmp_path, sent, capsys):
    # 107 days of the same version: more than the 90 days a check reads.
    main_thread_days(tmp_path / "logs", [{}] * 105 + [{"tier": "5m"}] * 2)
    check_logs(tmp_path, today=date(2026, 12, 17))
    assert sent == ["ccdrift: setting changed"]
    assert ("ccdrift: setting changed: Cache writes for claude-opus-5 moved from the 1-hour to the 5-minute "
            "cache from 2026-12-15, on Claude Code 2.1.226 (since 09-01).") in capsys.readouterr().out


def test_check_doesnt_alert_again_about_a_blank_cache_stretch_longer_than_the_history_it_reads(tmp_path, sent):
    busy_days(tmp_path / "logs", days=105, per_day=60)
    check_logs(tmp_path, today=date(2026, 12, 15))
    check_logs(tmp_path, today=date(2026, 12, 16))
    assert sent == ["ccdrift can't compute the cache metric"]


def blank_turns(days):
    """60 new-prompt turns a day with no cache token counts."""
    return pd.DataFrame([{"day": nth_day(d), "prompt_within_ttl": True, "new_prompt": True,
                          "cache_read": 0.0, "cache_creation": 0.0} for d in days for _ in range(60)])


def test_a_new_blank_stretch_after_a_break_is_reported_even_when_it_starts_the_history_read():
    state = {**new_state(), "blank_cache": [nth_day(0)], "blank_cache_seen": nth_day(9)}
    stretch = blank_cache_stretch(blank_turns(range(160, 163)), state)
    assert (stretch["first"], stretch["days"]) == (nth_day(160), 3)


def test_a_blank_stretch_reported_before_ccdrift_kept_its_last_day_isnt_reported_again():
    state = {**new_state(), "blank_cache": [nth_day(0)]}
    assert blank_cache_stretch(blank_turns(range(10, 100)), state) is None


def test_a_blank_stretch_seen_on_the_last_run_isnt_reported_again_when_the_history_read_moves_on():
    state = {**new_state(), "blank_cache": [nth_day(15)], "blank_cache_seen": nth_day(104)}
    assert blank_cache_stretch(blank_turns(range(16, 105)), state) is None


def test_check_says_so_when_there_is_nothing_to_report(tmp_path, sent, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    check_logs(tmp_path)
    assert capsys.readouterr().out.endswith("no alerts\n")


def test_check_saves_each_run_in_the_state_file(tmp_path, sent):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    now = datetime(2026, 9, 4, 9, 0, tzinfo=timezone(timedelta(hours=3)))
    run_check(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4), now=now)
    state = load_state(tmp_path / "state.json")
    assert state["last_run"] == {"started": "2026-09-04T09:00:00+03:00", "ok": True, "error": None}
    assert state["last_ok"] == "2026-09-04T09:00:00+03:00"


def test_a_failed_run_is_saved_in_the_state_file(tmp_path, sent):
    # So `ccdrift status` can tell a broken check from a quiet week.
    (tmp_path / "logs").mkdir()
    check_logs(tmp_path)
    last_run = load_state(tmp_path / "state.json")["last_run"]
    assert last_run["ok"] is False
    assert last_run["error"].startswith("RuntimeError: No Claude Code transcripts found")


def test_check_leaves_an_unreadable_state_file_as_it_is(tmp_path, sent):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    (tmp_path / "state.json").write_text("not json")
    assert check_logs(tmp_path) == 1
    assert sent == ["ccdrift check failed"]
    assert (tmp_path / "state.json").read_text() == "not json"


def test_an_unreadable_state_file_notifies_at_most_once_in_20_hours(tmp_path, sent, capsys):
    # The check can't note the notice in a state file it can't read, and it runs every hour.
    main_thread_days(tmp_path / "logs", [{}] * 3)
    (tmp_path / "state.json").write_text("not json")
    first = datetime(2026, 9, 4, 9, 0, tzinfo=timezone.utc)
    for hours in (0, 1, 20):
        assert check_logs(tmp_path, now=first + timedelta(hours=hours)) == 1
    assert sent == ["ccdrift check failed", "ccdrift check failed"]
    assert capsys.readouterr().out.count("ccdrift check failed: can't read the state file") == 3


def test_check_leaves_a_state_file_from_a_newer_ccdrift_as_it_is(tmp_path, sent):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    newer = '{"version": 3, "incidents": [], "settings": [], "blank_cache": [], "reported": {}, "new": 1}\n'
    (tmp_path / "state.json").write_text(newer)
    assert check_logs(tmp_path) == 1
    assert sent == ["ccdrift check failed"]
    assert (tmp_path / "state.json").read_text() == newer


def test_check_names_an_unusable_history_store(tmp_path, sent, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    (tmp_path / "history.sqlite").write_text("not a database")
    assert check_logs(tmp_path) == 1
    assert "Can't use the history store" in capsys.readouterr().out


def test_check_says_how_to_rebuild_a_history_store_whose_rows_cant_be_read(tmp_path, sent, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 3)
    check_logs(tmp_path)
    capsys.readouterr()
    damage_responses_table(tmp_path / "history.sqlite")
    assert check_logs(tmp_path) == 1
    assert sent == ["ccdrift check failed"]
    assert capsys.readouterr().out.endswith(
        f"ccdrift check failed: HistoryError: Can't use the history store {tmp_path / 'history.sqlite'}: "
        "no such column: ts. Move it aside to rebuild it from the transcripts still on disk.\n")


def test_check_alerts_when_the_main_thread_moves_to_the_5_minute_cache(tmp_path, sent, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"tier": "5m", "version": "2.1.280"}] * 2)
    check_logs(tmp_path, today=date(2026, 9, 17))
    assert sent == ["ccdrift: setting changed"]
    assert ("ccdrift: setting changed: Cache writes for claude-opus-5 moved from the 1-hour to the 5-minute "
            "cache from 2026-09-15, on Claude Code 2.1.280 (since 09-15).") in capsys.readouterr().out


def test_check_runs_the_exec_command_for_each_alert(tmp_path, sent, capsys):
    (tmp_path / "logs").mkdir()
    out = tmp_path / "alerts.txt"
    run_check(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 4),
              exec_command=f'echo "$CCDRIFT_ALERT" >> "{out}"')
    assert out.read_text() == "failed\n"


def test_a_failing_exec_command_is_logged_and_the_check_goes_on(tmp_path, sent, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"haiku": 12}] * 3)
    ran_before(tmp_path)
    assert run_check(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 18),
                     exec_command="exit 7") == 0
    assert '--exec failed for "ccdrift flag": exit 7' in capsys.readouterr().out


def test_an_exec_command_that_raises_is_logged_and_every_alert_still_goes_out(tmp_path, sent, capsys, monkeypatch):
    # The state already marks these alerts as sent, so none may be lost.
    def broken(command, kind, title, message):
        raise RuntimeError("boom")

    monkeypatch.setattr(ccdrift.check, "run_exec", broken)
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"haiku": 12, "tier": "5m"}] * 3)
    ran_before(tmp_path)
    assert run_check(tmp_path / "logs", tmp_path / "state.json", notify_user=True, today=date(2026, 9, 18),
                     exec_command="notify-me") == 0
    assert sent == ["ccdrift flag", "ccdrift: setting changed"]
    out = capsys.readouterr().out
    assert '--exec failed for "ccdrift flag": RuntimeError: boom' in out
    assert '--exec failed for "ccdrift: setting changed": RuntimeError: boom' in out


def a_notifier_that_raises_on_the_first_alert(monkeypatch):
    titles = []

    def notify(title, message):
        titles.append(title)
        if len(titles) == 1:
            raise ValueError("embedded null byte")

    monkeypatch.setattr(ccdrift.check, "notify", notify)
    return titles


def test_a_notification_that_raises_is_logged_and_every_alert_still_goes_out(tmp_path, capsys, monkeypatch):
    # The state already marks these alerts as sent, so none may be lost.
    titles = a_notifier_that_raises_on_the_first_alert(monkeypatch)
    out = tmp_path / "alerts.txt"
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"haiku": 12, "tier": "5m"}] * 3)
    ran_before(tmp_path)
    assert run_check(tmp_path / "logs", tmp_path / "state.json", notify_user=True, today=date(2026, 9, 18),
                     exec_command=f'echo "$CCDRIFT_ALERT" >> "{out}"') == 0
    assert titles == ["ccdrift flag", "ccdrift: setting changed"]
    assert out.read_text() == "flag\nsetting\n"
    assert 'notification failed for "ccdrift flag": ValueError: embedded null byte' in capsys.readouterr().out


def test_alerts_quote_release_notes_on_their_topic_from_new_versions(tmp_path, sent, capsys):
    logs = tmp_path / "cfg" / "projects"
    main_thread_days(logs, [{}] * 14 + [{"haiku": 12, "version": "2.1.233"}] * 3)
    (tmp_path / "cfg" / "cache").mkdir(parents=True)
    (tmp_path / "cfg" / "cache" / "changelog.md").write_text(
        "## 2.1.233\n\n- Search subagents now run on the Haiku model\n- Added a theme picker\n")
    ran_before(tmp_path)
    run_check(logs, tmp_path / "state.json", notify_user=True, today=date(2026, 9, 18))
    out = capsys.readouterr().out
    assert "    release notes 2.1.233: Search subagents now run on the Haiku model" in out.splitlines()
    assert "theme picker" not in out
    assert sent == ["ccdrift flag"]


def test_alerts_quote_the_version_they_name_first_then_other_new_versions_newest_first(tmp_path, sent, capsys):
    # Sessions start smaller from Sep 9 on 2.1.265; 2.1.263 came before it and 2.1.267
    # showed up on Sep 10 in a resumed session, so the alert names only 2.1.265.
    logs = tmp_path / "cfg" / "projects"
    for d, version in enumerate(["2.1.261"] * 4 + ["2.1.263"] * 4 + ["2.1.265"] * 3):
        tokens = 54_000 if version == "2.1.265" else 128_000
        records = [prompt(at(d * DAY), sid=f"s{d}"),
                   line(f"m{d}", text(40), ts=at(d * DAY), sid=f"s{d}", cache_creation=100, cache_read=tokens - 110,
                        version=version, entrypoint="cli")]
        if d == 9:
            records.append(line("resumed", text(40), ts=at(d * DAY + 3600), sid=f"s{d}", cache_read=54_000,
                                version="2.1.267", entrypoint="cli"))
        write(logs / "-Users-me-app" / f"session-{d}.jsonl", records)
    (tmp_path / "cfg" / "cache").mkdir(parents=True)
    (tmp_path / "cfg" / "cache" / "changelog.md").write_text(
        "## 2.1.267\n\n- Fixed the tool list changing when an MCP server reconnects\n\n"
        "## 2.1.265\n\n- Fixed tool definitions re-rendering after a login\n- Added a theme picker\n"
        "- Changed the system prompt to load later\n- Changed deferred tools to load on first use\n\n"
        "## 2.1.263\n\n- Fixed background agents losing their tool results\n- Changed the system prompt on Bedrock\n\n"
        "## 2.1.261\n\n- Fixed the system prompt on Vertex\n")
    run_check(logs, tmp_path / "state.json", today=date(2026, 9, 12), digest=False,
              now=datetime(2026, 9, 12, 9, 0, tzinfo=timezone.utc))
    out = capsys.readouterr().out
    assert "ccdrift: session start changed: New sessions start with ~54k tokens of context from 2026-09-09, on " \
           "Claude Code 2.1.265 (since 09-09)" in out
    assert [text for text in out.splitlines() if "release notes" in text] == [
        "    release notes 2.1.265: Fixed tool definitions re-rendering after a login",
        "    release notes 2.1.265: Changed the system prompt to load later",
        "    release notes 2.1.267: Fixed the tool list changing when an MCP server reconnects",
        "    release notes 2.1.263: Changed the system prompt on Bedrock",
    ]


def test_check_alerts_when_a_new_version_stops_logging_effort(tmp_path, sent, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"version": "2.1.280", "effort": None}] * 2)
    check_logs(tmp_path, today=date(2026, 9, 17))
    assert sent == ["ccdrift: Claude Code stopped logging a field"]
    assert ("ccdrift: Claude Code stopped logging a field: Claude Code 2.1.280 no longer logs effort "
            "(on 0% of 120 responses, 100% before). Effort change alerts can't work until ccdrift reads it again; "
            "run `ccdrift peek`.") in capsys.readouterr().out


def test_check_alerts_when_stop_hooks_start_failing(tmp_path, sent, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 16)
    hook_days_logs(tmp_path / "logs", {14, 15}, 16)
    check_logs(tmp_path, today=date(2026, 9, 17))
    assert sent == ["ccdrift: hooks failing"]
    assert ("ccdrift: hooks failing: Stop hooks failed on 10 of 10 runs on 2026-09-15 and 10 of 10 on 2026-09-16, "
            "on Claude Code 2.1.226 (since 09-01). Check your hooks; a Claude Code update may have changed their "
            "input.") in capsys.readouterr().out


def test_check_alerts_when_sessions_start_with_much_less_context(tmp_path, sent, capsys):
    # One session a day; its first request reads the rest from the cache, so the cache ratio stays high.
    for d, (tokens, version) in enumerate([(128_000, "2.1.261")] * 8 + [(54_000, "2.1.267")] * 3):
        write(tmp_path / "logs" / "-Users-me-app" / f"session-{d}.jsonl",
              [prompt(at(d * DAY), sid=f"s{d}"),
               line(f"m{d}", text(40), ts=at(d * DAY), sid=f"s{d}", cache_creation=100, cache_read=tokens - 110,
                    version=version, entrypoint="cli")])
    check_logs(tmp_path, today=date(2026, 9, 12))
    assert sent == ["ccdrift: session start changed"]
    assert ("ccdrift: session start changed: New sessions start with ~54k tokens of context from 2026-09-09, on "
            "Claude Code 2.1.267 (since 09-09), down from ~130k, in the one project ccdrift could compare with "
            "itself, and on a Claude Code version none of the sessions before it ran: either that version or the "
            "project's own files explain it.") in capsys.readouterr().out


def test_a_session_start_alert_counts_what_the_sessions_started_with_and_logs_the_names(tmp_path, sent, capsys):
    # What changed around the 2026-08-27 step over 17 sessions before it and 9 after: 11
    # agent types, 4 skills and 11 MCP tools, the skills listing growing from 21,077 to
    # 22,976 characters. (Over the detector's own 10 and 3 sessions the real step shows
    # less; docs/findings.md has it.) The step itself is 100k -> 130k: against a baseline
    # this even, the real 106k -> 129k is under CHANGE.
    for d in range(11):
        after = d >= 8
        sid, tokens = f"s{d}", 130_000 if after else 100_000
        deferred = [f"Tool{i}" for i in range(113)] + ([f"mcp__tracker__t{i}" for i in range(11)] if after else [])
        write(tmp_path / "logs" / "-Users-me-app" / f"session-{d}.jsonl", [
            attachment(at(d * DAY), skill_listing([f"skill-{i}" for i in range(72 if after else 68)],
                                                  chars=22_976 if after else 21_077), sid=sid, version="2.1.250"),
            attachment(at(d * DAY), deferred_tools(deferred), sid=sid, version="2.1.250"),
            attachment(at(d * DAY), agent_listing([f"agent-{i}" for i in range(27 if after else 16)]), sid=sid,
                       version="2.1.250"),
            prompt(at(d * DAY), sid=sid),
            line(f"m{d}", text(40), ts=at(d * DAY), sid=sid, cache_creation=100, cache_read=tokens - 110,
                 version="2.1.250", entrypoint="cli")])
    sent_to_exec = tmp_path / "exec.txt"
    check_logs(tmp_path, today=date(2026, 9, 12), exec_command=f'echo "$CCDRIFT_MESSAGE" >> "{sent_to_exec}"')
    out = capsys.readouterr().out
    assert sent == ["ccdrift: session start changed"]
    message = sent_to_exec.read_text()
    assert message.endswith(
        " Of what Claude Code logs about a session's start, 11 agent types, 4 skills and 11 MCP tools were added, "
        "about 2.4k more characters, though the logs can't say how many of the tokens that is. CLAUDE.md files, "
        "the system prompt or tool definitions weren't logged in every session compared, so ccdrift couldn't "
        "compare those parts.\n")
    assert "agent-26" not in message and "skill-71" not in message and "tracker" not in message
    lines = out.splitlines()
    assert "    agent types added: " + ", ".join(f"agent-{i}" for i in range(16, 27)) in lines
    assert "    skills added: skill-68, skill-69, skill-70, skill-71" in lines
    assert "    MCP tools added: tracker (11)" in lines
    assert "    the skills listing: 21,077 -> 22,976 characters" in lines


def test_a_session_start_alert_names_no_addition_when_its_window_and_baseline_are_two_different_projects(
        tmp_path, sent, capsys):
    # -Users-me-b runs first, with its own skill and MCP server; work then moves to
    # -Users-me-a, whose sessions genuinely start with half the context. Each project has
    # too few sessions on its own for its own pass to judge it, so only the pooled pass
    # finds the step, and first_of_each keeps it: its window is -a's own three sessions,
    # its baseline -b's, a different project entirely.
    # Neither project's own skill or MCP server changed; the alert must say nothing changed
    # rather than reading -b's as removed and -a's as added.
    def start(project, day, tokens, skill, mcp_tool, sid):
        write(tmp_path / "logs" / project / f"{sid}.jsonl", [
            attachment(at(day * DAY), skill_listing([skill]), sid=sid, version="2.1.250"),
            attachment(at(day * DAY), deferred_tools(["Read", mcp_tool]), sid=sid, version="2.1.250"),
            prompt(at(day * DAY), sid=sid),
            line(f"m{sid}", text(40), ts=at(day * DAY), sid=sid, cache_creation=100, cache_read=tokens - 110,
                 version="2.1.250", entrypoint="cli")])

    for d in range(8):
        start("-Users-me-b", d, 40_000, "skill-b", "mcp__toolb__x", f"b{d}")
    for d in (8, 9, 10):
        start("-Users-me-a", d, 128_000, "skill-a", "mcp__toola__y", f"a{d}")
    for d in (11, 12, 13):
        start("-Users-me-a", d, 64_000, "skill-a", "mcp__toola__y", f"a{d}")

    # The scenario actually exercises the pooled pass's change, not either project's own.
    starts = session_starts(parse_source(tmp_path / "logs"))
    (record, change), = context_found(starts, new_state(), date(2026, 9, 15))
    assert change.project is None
    assert {f.split("/")[0] for f in change.window_files} == {"-Users-me-a"}
    assert {f.split("/")[0] for f in change.baseline_files} == {"-Users-me-b"}

    sent_to_exec = tmp_path / "exec.txt"
    check_logs(tmp_path, today=date(2026, 9, 15), exec_command=f'echo "$CCDRIFT_MESSAGE" >> "{sent_to_exec}"')
    assert sent == ["ccdrift: session start changed"]
    message = sent_to_exec.read_text()
    assert "Of what Claude Code logs" not in message
    assert "added" not in message and "removed" not in message
    out = capsys.readouterr().out
    assert "added:" not in out and "removed:" not in out
    assert "skill-b" not in out and "skill-a" not in out and "toolb" not in out and "toola" not in out


def test_a_state_from_an_older_ccdrift_is_rejudged_once_and_keeps_a_real_change(tmp_path, sent, capsys):
    # -Users-me-a steps alone among two projects: the shape of
    # test_a_step_in_one_project_of_several_blames_that_projects_own_files in test_sessions.py.
    # Only the per-project pass finds it; the pooled-only rule `rejudged` used to re-check
    # against would have deleted it in the same run it was alerted.
    for project, tokens in [("-Users-me-a", [128_000] * 8 + [64_000] * 3), ("-Users-me-b", [64_000] * 11)]:
        for d, size in enumerate(tokens):
            write(tmp_path / "logs" / project / f"s{d}.jsonl",
                  [prompt(at(d * DAY), sid=f"{project}-{d}"),
                   line(f"m{project}-{d}", text(40), ts=at(d * DAY), sid=f"{project}-{d}", cache_creation=100,
                        cache_read=size - 110, entrypoint="cli")])
    # A state as ccdrift 0.7.0 left it: no context_rule, and a recorded change (an "up" move,
    # so it isn't deduped against the real "down" one below) that the new rule can't find.
    # It sits on 09-08, past the 8th judged session, which is the earliest day the new rule
    # could report on: a record before that is kept unjudged instead.
    state = new_state()
    del state["context_rule"]
    state["context_changes"] = [{"since": "2026-09-08", "from": 60_000.0, "to": 90_000.0,
                                 "days": ["2026-09-08", "2026-09-10"], "reported_on": "2026-09-11"}]
    state["last_ok"] = "2026-09-01T09:00:00+00:00"
    save_state(tmp_path / "state.json", state)

    check_logs(tmp_path, today=date(2026, 9, 14))
    out = capsys.readouterr().out
    assert sent == ["ccdrift: session start changed"]
    assert ("ccdrift: session start changed: New sessions start with ~64k tokens of context from 2026-09-09, "
            "down from ~130k, in 1 of the 2 projects ccdrift could compare. That project's CLAUDE.md, MCP servers "
            "or skills explain it, not Claude Code.") in out
    assert ("ccdrift: a recorded session-start change was dropped: 2026-09-08, ~60k -> ~90k tokens: judged "
            "against each project's own level, it isn't a change.") in out
    state = load_state(tmp_path / "state.json")
    assert [c["since"] for c in state["context_changes"]] == ["2026-09-09"]
    assert state["context_rule"] == 2

    check_logs(tmp_path, today=date(2026, 9, 15))
    assert sent == ["ccdrift: session start changed"]
    state = load_state(tmp_path / "state.json")
    assert [c["since"] for c in state["context_changes"]] == ["2026-09-09"]
    assert state["context_rule"] == 2


def test_check_warns_within_a_day_when_new_prompts_start_missing_the_cache(tmp_path, sent, capsys):
    # Ten misses in a row on Sep 21 from 10:50 UTC pass h = THRESHOLD = 4.0 at 10:59;
    # the second run, an hour later, stays quiet.
    main_thread_days(tmp_path / "logs", [{}] * 20 + [{"misses": 10}])
    check_logs(tmp_path, today=date(2026, 9, 22))
    check_logs(tmp_path, today=date(2026, 9, 22), now=datetime(2026, 9, 22, 10, 0, tzinfo=timezone.utc))
    assert sent == ["ccdrift: cache misses rising"]
    assert ("ccdrift: cache misses rising: 10 of the last 10 new-prompt turns missed the cache (usually 0.0%), "
            "since 09-21 10:50, on Claude Code 2.1.226 (since 09-01). The daily check confirms or clears it "
            "within a few days.") in capsys.readouterr().out


@pytest.fixture
def loop_settings(monkeypatch):
    """The settings lab gates G8 and G9 could record, whatever they did record."""
    for stream in ("main", "subagent"):
        monkeypatch.setitem(ccdrift.loops.LOOP_SETTINGS, stream, LoopSetting(p1=0.02, h=3, min_sessions=1))


def test_check_warns_when_tool_loop_turns_start_missing_the_cache(tmp_path, sent, capsys, loop_settings):
    # Ten misses in a row from 11:31 on Sep 21 pass h = 3 at the second; an hour later
    # the check stays quiet.
    tool_loop_days(tmp_path / "logs", 21, misses=10)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "changelog.md").write_text(
        "## 2.1.226\n\n- Fixed prompt cache misses after a tool call\n- Added a theme picker\n")
    check_logs(tmp_path, today=date(2026, 9, 22))
    check_logs(tmp_path, today=date(2026, 9, 22), now=datetime(2026, 9, 22, 10, 0, tzinfo=timezone.utc))
    assert sent == ["ccdrift: tool-loop cache misses rising"]
    out = capsys.readouterr().out.splitlines()
    assert ("[check 2026-09-22 09:00] ccdrift: tool-loop cache misses rising: 10 of the last 10 tool-loop turns "
            "missed the cache (usually 0.00%), since 09-21 11:31, in 1 session, rewriting ~110k tokens, on Claude "
            "Code 2.1.226 (since 09-01). `ccdrift report` shows whether it lasts.") in out
    assert "    release notes 2.1.226: Fixed prompt cache misses after a tool call" in out


def test_check_warns_about_subagent_loop_misses_while_a_cache_incident_is_open(tmp_path, sent, capsys,
                                                                               loop_settings):
    # An incident on new prompts doesn't say whether tool loops miss too.
    tool_loop_days(tmp_path / "logs", 21)
    tool_loop_days(tmp_path / "logs", 21, misses=10, subagent=True)
    state = new_state()
    add_incident(state["incidents"], "cache_ratio", "2026-09-21", "2026-09-21", date(2026, 9, 22))
    state["incidents"][0].update(end=None, status="open", source="check")
    save_state(tmp_path / "state.json", state)
    kinds = tmp_path / "kinds.txt"
    check_logs(tmp_path, today=date(2026, 9, 22), exec_command=f'echo "$CCDRIFT_ALERT" >> "{kinds}"')
    assert sent == ["ccdrift: subagent cache misses rising"]
    assert kinds.read_text() == "subagent_loop\n"
    assert [w["stream"] for w in load_state(tmp_path / "state.json")["loop_warnings"]] == ["subagent"]


# Haiku on 12 of 60 responses a day from Sep 15 to 17 on 2.1.233, then none on 2.1.259.
OLD_REGRESSION = [{}] * 14 + [{"haiku": 12, "version": "2.1.233"}] * 3 + [{"version": "2.1.259"}] * 5


def test_the_first_check_replays_its_history_and_says_once_what_it_found(tmp_path, sent, capsys):
    # On Oct 10 the flag from Sep 15 is more than 14 days old, so without the replay a
    # first check would record nothing; an hour later the check stays quiet.
    main_thread_days(tmp_path / "logs", OLD_REGRESSION)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "changelog.md").write_text(
        "## 2.1.259\n\n- Search subagents run on Sonnet again instead of Haiku\n\n"
        "## 2.1.233\n\n- Search subagents now run on the Haiku model\n- Added a theme picker\n")
    kinds = tmp_path / "kinds.txt"
    check_logs(tmp_path, today=date(2026, 10, 10), exec_command=f'echo "$CCDRIFT_ALERT" >> "{kinds}"')
    check_logs(tmp_path, today=date(2026, 10, 10), now=datetime(2026, 10, 10, 10, 0, tzinfo=timezone.utc))
    assert sent == ["ccdrift: past incidents found"]
    assert kinds.read_text() == "history\n"
    out = capsys.readouterr().out.splitlines()
    assert ("[check 2026-10-10 09:00] ccdrift: past incidents found: Replaying your history from 2026-09-01 found "
            "1 incident ccdrift would have followed: Haiku share up 2026-09-15..2026-09-19, back to normal from "
            "2026-09-20, ~36 extra Haiku responses, on Claude Code 2.1.233 (since 09-15). `ccdrift incident list` "
            "has the details; `ccdrift incident dismiss` puts a false alarm's days back in the baseline.") in out
    assert "    release notes 2.1.233: Search subagents now run on the Haiku model" in out
    assert not any(text.startswith("    release notes 2.1.259") for text in out)
    assert [(i["start"], i["status"], i["source"], i["opened_on"], i["closed_on"])
            for i in load_state(tmp_path / "state.json")["incidents"]] == [
        ("2026-09-15", "recovered", "replay", "2026-09-18", "2026-09-23")]


def test_a_first_check_that_finds_a_regression_still_going_flags_it_as_well(tmp_path, sent, capsys):
    # The summary says what the history held; the flag says this is happening now, with the
    # days and z values that opened it: the alert the owner would have had all along.
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"haiku": 12}] * 10)
    check_logs(tmp_path, today=date(2026, 10, 1))
    assert sent == ["ccdrift: past incidents found", "ccdrift flag"]
    out = capsys.readouterr().out.splitlines()
    assert ("[check 2026-10-01 09:00] ccdrift flag: Haiku share on the main thread up from 2026-09-15, on "
            "Claude Code 2.1.226 (since 09-01). ~120 extra Haiku responses so far.") in out
    assert any(line.startswith("    days 2026-09-15, 2026-09-16, 2026-09-17; z = ") for line in out)
    assert short_status(tmp_path / "state.json", datetime(2026, 10, 1, 9, 5, tzinfo=timezone.utc)) == (
        "ccdrift: Haiku share up since 09-15")


def test_a_first_check_whose_incidents_are_all_over_sends_only_the_summary(tmp_path, sent):
    # Nothing is happening now, so nothing is flagged: one alert, as before.
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"haiku": 12}] * 5 + [{}] * 10)
    check_logs(tmp_path, today=date(2026, 10, 10))
    assert sent == ["ccdrift: past incidents found"]


def test_a_first_check_on_a_clean_history_sends_nothing(tmp_path, sent):
    main_thread_days(tmp_path / "logs", [{}] * 20)
    check_logs(tmp_path, today=date(2026, 10, 1))
    assert sent == []


def test_a_check_that_has_run_before_doesnt_replay_an_old_regression(tmp_path, sent):
    main_thread_days(tmp_path / "logs", OLD_REGRESSION)
    ran_before(tmp_path)
    check_logs(tmp_path, today=date(2026, 10, 10))
    assert sent == []
    assert load_state(tmp_path / "state.json")["incidents"] == []


LOCAL = timezone(timedelta(hours=3))


def test_check_sends_the_weekly_summary_on_the_first_run_after_monday_9(tmp_path, sent, capsys):
    main_thread_days(tmp_path / "logs", [{}] * 20)
    for today, hour, minute in ((19, 9, 0), (21, 8, 0), (21, 9, 5), (21, 10, 5)):
        check_logs(tmp_path, today=date(2026, 9, today), now=datetime(2026, 9, today, hour, minute, tzinfo=LOCAL),
                   digest=True)
    assert sent == ["ccdrift: weekly summary"]
    assert ("ccdrift: weekly summary: Week of 09-14: 420 responses on 2.1.226; cache ratio 0.900 (0.0% misses); "
            "no Haiku; no tool-loop turns, no subagent loop turns; no failed requests; no open incidents; "
            "no setting changes; no new fields; "
            "check ran on 1 of 7 days.") in capsys.readouterr().out


def test_check_without_the_digest_leaves_the_week_unsent(tmp_path, sent):
    main_thread_days(tmp_path / "logs", [{}] * 20)
    check_logs(tmp_path, today=date(2026, 9, 19), now=datetime(2026, 9, 19, 9, 0, tzinfo=LOCAL))
    check_logs(tmp_path, today=date(2026, 9, 21), now=datetime(2026, 9, 21, 9, 5, tzinfo=LOCAL))
    assert sent == []
    assert "digest_week" not in load_state(tmp_path / "state.json")


def test_a_failing_check_notifies_at_most_once_in_20_hours(tmp_path, sent, capsys):
    (tmp_path / "logs").mkdir()
    first = datetime(2026, 9, 4, 9, 0, tzinfo=timezone.utc)
    for hours in (0, 1, 20):
        check_logs(tmp_path, now=first + timedelta(hours=hours))
    assert sent == ["ccdrift check failed", "ccdrift check failed"]
    assert capsys.readouterr().out.count("ccdrift check failed: ") == 3


def test_a_new_failure_after_a_successful_run_notifies_again(tmp_path, sent):
    # The 20 hours are between notices about one stretch of failures.
    main_thread_days(tmp_path / "logs", [{}] * 3)
    first = datetime(2026, 9, 4, 9, 0, tzinfo=timezone.utc)
    (tmp_path / "state.json").write_text("not json")
    assert check_logs(tmp_path, now=first) == 1
    (tmp_path / "state.json").unlink()
    assert check_logs(tmp_path, now=first + timedelta(hours=1)) == 0
    (tmp_path / "state.json").write_text("not json")
    assert check_logs(tmp_path, now=first + timedelta(hours=2)) == 1
    assert sent == ["ccdrift check failed", "ccdrift check failed"]


def test_an_incident_dismissed_while_the_check_runs_stays_dismissed(tmp_path, sent, monkeypatch):
    # The check saved the state it had read before its run, over the dismissal.
    main_thread_days(tmp_path / "logs", [{}] * 3)
    state_path = tmp_path / "state.json"
    state = new_state()
    add_incident(state["incidents"], "cache_ratio", "2026-09-01", "2026-09-02", date(2026, 9, 4))
    save_state(state_path, state)
    load_history = ccdrift.check.load_history
    dismissing = []

    def load_history_while_dismissing(*args, **kwargs):
        env = {**os.environ, "PYTHONPATH": str(Path(ccdrift.check.__file__).parents[1])}
        dismissing.append(subprocess.Popen(
            [sys.executable, "-m", "ccdrift", "incident", "dismiss", "cache", "2026-09-01", "--state", str(state_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=env))
        dismissing[0].stderr.readline()  # it waits for the check, or has already saved
        return load_history(*args, **kwargs)

    monkeypatch.setattr(ccdrift.check, "load_history", load_history_while_dismissing)
    assert check_logs(tmp_path) == 0
    assert dismissing[0].wait(timeout=60) == 0
    saved = load_state(state_path)
    assert (saved["incidents"][0]["status"], saved["last_run"]["ok"]) == ("dismissed", True)


def a_new_field_history(tmp_path):
    """14 days on one version, then 2 on a version that carries a field ccdrift
    doesn't read."""
    source, state_path = tmp_path / "logs", tmp_path / "state.json"
    save_state(state_path, new_state())
    main_thread_days(source, [{"version": "2.1.270"}] * 14
                     + [{"version": "2.1.280", "extra": {"advisorModel": "claude-opus-5"}}] * 2)
    return source, state_path


def test_a_field_claude_code_has_started_logging_is_reported_to_the_log(tmp_path, capsys):
    source, state_path = a_new_field_history(tmp_path)
    assert run_check(source, state_path, today=date(2026, 9, 17)) == 0
    assert "logs 1 field ccdrift doesn't read: advisorModel" in capsys.readouterr().out


def test_a_new_field_does_not_notify(tmp_path, sent):
    source, state_path = a_new_field_history(tmp_path)
    run_check(source, state_path, notify_user=True, today=date(2026, 9, 17))
    assert not any("field ccdrift doesn't read" in title for title in sent)


def test_a_new_field_is_recorded_in_the_state(tmp_path):
    source, state_path = a_new_field_history(tmp_path)
    run_check(source, state_path, today=date(2026, 9, 17))
    assert load_state(state_path)["new_fields"][0]["paths"] == ["advisorModel"]
