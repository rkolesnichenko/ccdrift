"""The daily check: new flags once, and alerts when it fails or can't compute the
cache metric."""

from datetime import date

import pytest

import ccdrift.check
from ccdrift.check import check, run_check
from tests.helpers import HAIKU, QUIET, busy_days, daily_turns


def check_days(tmp_path, days, today):
    df = daily_turns(days)
    if "main_thread" not in df:
        df["main_thread"] = True
    flags = check(df, today=today, state_path=tmp_path / "state.json")
    return [(f["metric"], f["onset"]) for f in flags]


def test_check_reports_a_flag_that_starts_in_recent_days(tmp_path):
    days = [QUIET] * 14 + [HAIKU] * 3
    assert check_days(tmp_path, days, date(2026, 9, 18)) == [("haiku_fraction", "2026-09-15")]


def test_check_does_not_repeat_a_flag_it_already_reported(tmp_path):
    days = [QUIET] * 14 + [HAIKU] * 4
    check_days(tmp_path, days[:17], date(2026, 9, 18))
    assert check_days(tmp_path, days, date(2026, 9, 19)) == []


def test_check_waits_for_the_current_day_to_finish(tmp_path):
    # A day still in progress holds only part of its turns.
    days = [QUIET] * 14 + [HAIKU] * 3
    assert check_days(tmp_path, days, date(2026, 9, 17)) == []


def test_check_stays_quiet_about_flags_from_weeks_ago(tmp_path):
    # Installed on Sep 15, the check would otherwise open with the August cache
    # regression, four weeks old by then.
    days = [QUIET] * 14 + [HAIKU] * 3
    assert check_days(tmp_path, days, date(2026, 10, 10)) == []


def test_check_leaves_subagent_haiku_out(tmp_path):
    # Subagent Haiku comes in bursts: across all turns, 3 of 5 synthetic logs
    # were falsely flagged.
    quiet = {"is_haiku": [0.0] * 440, "main_thread": [True] * 400 + [False] * 40}
    burst = {"is_haiku": [0.0] * 400 + [1.0] * 40, "main_thread": [True] * 400 + [False] * 40}
    assert check_days(tmp_path, [quiet] * 14 + [burst] * 3, date(2026, 9, 18)) == []


def test_check_leaves_effort_out(tmp_path):
    # Effort swings more from day to day than a 70% cut in thinking moves it,
    # so an effort flag in one user's logs says more about the work than Claude.
    days = [{"thinking_fraction": [0.5] * 400}] * 14 + [{"thinking_fraction": [0.1] * 400}] * 3
    assert check_days(tmp_path, days, date(2026, 9, 18)) == []


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
