"""The daily check: new flags once, and alerts when it fails or can't compute the
cache metric."""

from datetime import date, datetime, time, timedelta, timezone

import pytest

import ccdrift.check
from ccdrift.check import run_check
from ccdrift.incidents import add_incident
from ccdrift.state import load_state, new_state, save_state
from ccdrift.status import status_report
from tests.helpers import (DAY, at, busy_days, damage_responses_table, hook_days_logs, line, main_thread_days,
                           prompt, text, write)


@pytest.fixture
def sent(monkeypatch):
    titles = []
    monkeypatch.setattr(ccdrift.check, "notify", lambda title, message: titles.append(title))
    return titles


def check_logs(tmp_path, today=date(2026, 9, 4), **options):
    """Run the check as the schedule would at 09:00 UTC on `today`."""
    options.setdefault("now", datetime.combine(today, time(9, 0), tzinfo=timezone.utc))
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
    assert check_logs(tmp_path, today=date(2026, 9, 18)) == 0
    assert sent == ["ccdrift flag"]
    assert ("ccdrift flag: Haiku share on the main thread up from 2026-09-15, on Claude Code 2.1.233 "
            "(since 09-15). ~36 extra Haiku responses so far.") in capsys.readouterr().out


def test_check_alerts_when_an_incident_is_back_to_normal(tmp_path, sent, capsys):
    days = [{}] * 14 + [{"haiku": 12, "version": "2.1.233"}] * 3 + [{"version": "2.1.259"}] * 5
    main_thread_days(tmp_path / "logs", days)
    check_logs(tmp_path, today=date(2026, 9, 18))
    check_logs(tmp_path, today=date(2026, 9, 23))
    assert sent == ["ccdrift flag", "ccdrift: back to normal"]
    assert ("ccdrift: back to normal: Haiku share on the main thread back to normal from 2026-09-20, on Claude "
            "Code 2.1.259 (since 09-18). The incident from 2026-09-15: ~36 extra Haiku responses.") \
        in capsys.readouterr().out


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
        "no such column: r.ts. Move it aside to rebuild it from the transcripts still on disk.\n")


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
    assert run_check(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 18),
                     exec_command="exit 7") == 0
    assert '--exec failed for "ccdrift flag": exit 7' in capsys.readouterr().out


def test_an_exec_command_that_raises_is_logged_and_every_alert_still_goes_out(tmp_path, sent, capsys, monkeypatch):
    # The state already marks these alerts as sent, so none may be lost.
    def broken(command, kind, title, message):
        raise RuntimeError("boom")

    monkeypatch.setattr(ccdrift.check, "run_exec", broken)
    main_thread_days(tmp_path / "logs", [{}] * 14 + [{"haiku": 12, "tier": "5m"}] * 3)
    assert run_check(tmp_path / "logs", tmp_path / "state.json", notify_user=True, today=date(2026, 9, 18),
                     exec_command="notify-me") == 0
    assert sent == ["ccdrift flag", "ccdrift: setting changed"]
    out = capsys.readouterr().out
    assert '--exec failed for "ccdrift flag": RuntimeError: boom' in out
    assert '--exec failed for "ccdrift: setting changed": RuntimeError: boom' in out


def test_alerts_quote_release_notes_on_their_topic_from_new_versions(tmp_path, sent, capsys):
    logs = tmp_path / "cfg" / "projects"
    main_thread_days(logs, [{}] * 14 + [{"haiku": 12, "version": "2.1.233"}] * 3)
    (tmp_path / "cfg" / "cache").mkdir(parents=True)
    (tmp_path / "cfg" / "cache" / "changelog.md").write_text(
        "## 2.1.233\n\n- Search subagents now run on the Haiku model\n- Added a theme picker\n")
    run_check(logs, tmp_path / "state.json", today=date(2026, 9, 18))
    out = capsys.readouterr().out
    assert "    release notes 2.1.233: Search subagents now run on the Haiku model" in out.splitlines()
    assert "theme picker" not in out


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
        write(tmp_path / "logs" / f"session-{d}.jsonl",
              [prompt(at(d * DAY), sid=f"s{d}"),
               line(f"m{d}", text(40), ts=at(d * DAY), sid=f"s{d}", cache_creation=100, cache_read=tokens - 110,
                    version=version, entrypoint="cli")])
    check_logs(tmp_path, today=date(2026, 9, 12))
    assert sent == ["ccdrift: session start changed"]
    assert ("ccdrift: session start changed: New sessions start with ~54k tokens of context from 2026-09-09, on "
            "Claude Code 2.1.267 (since 09-09), down from ~130k.") in capsys.readouterr().out


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
