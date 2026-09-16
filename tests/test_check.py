"""The daily check: new flags once, and alerts when it fails or can't compute the
cache metric."""

from datetime import date, datetime, timedelta, timezone

import pytest

import ccdrift.check
from ccdrift.check import run_check
from ccdrift.state import load_state
from tests.helpers import busy_days, main_thread_days


@pytest.fixture
def sent(monkeypatch):
    titles = []
    monkeypatch.setattr(ccdrift.check, "notify", lambda title, message: titles.append(title))
    return titles


def check_logs(tmp_path, today=date(2026, 9, 4)):
    return run_check(tmp_path / "logs", tmp_path / "state.json", notify_user=True, today=today)


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
