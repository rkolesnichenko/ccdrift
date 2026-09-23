"""The early warning's core: a Bernoulli likelihood-ratio CUSUM on cache misses."""

from datetime import datetime, timedelta, timezone

import pandas as pd

from ccdrift.early import alarm_runs, clamp_rate, early_warning, miss_cusum
from ccdrift.state import new_state
from ccdrift.texts import early_message
from tests.helpers import nth_day


def every(n, total):
    return [(i + 1) % n == 0 for i in range(total)]


def test_misses_as_rare_as_usual_raise_no_alarm():
    assert miss_cusum(every(200, 4000), base_rate=0.005, h=4) == []


def test_misses_ten_times_as_common_raise_an_alarm_within_three_misses():
    # Each miss adds ln(0.05/0.005) = 2.30; each of the 19 hits between misses takes
    # ln(0.95/0.995) = -0.046, so the sum passes 4 at the third miss.
    assert miss_cusum(every(20, 60), base_rate=0.005, h=4) == [59]


def test_the_sum_restarts_after_an_alarm():
    assert miss_cusum(every(20, 120), base_rate=0.005, h=4) == [59, 119]


def test_the_usual_rate_is_kept_within_bounds():
    assert (clamp_rate(0.0), clamp_rate(0.01), clamp_rate(0.2)) == (0.002, 0.01, 0.025)


def test_each_alarm_comes_with_the_turn_its_rise_began():
    # The sum stands at 0 through the hits before each first miss, so each rise begins there.
    assert alarm_runs(every(20, 120), base_rate=0.005, h=4) == [(19, 59), (79, 119)]


NOW = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)


def prompt_frame(misses, days=21, per_day=30, blank=()):
    """CLI main-thread new-prompt turns from Sep 1, every 20 minutes from 08:00 UTC, on
    2.1.270 and, the last day, on 2.1.280; `misses` holds the (day, turn) pairs that miss,
    `blank` those logged without cache token counts, which read as misses too."""
    return pd.DataFrame([
        {"timestamp": pd.Timestamp(f"{nth_day(d)}T08:00:00Z") + pd.Timedelta(minutes=20 * k), "day": nth_day(d),
         "main_thread": True, "entrypoint": "cli", "prompt_within_ttl": True,
         "version": "2.1.280" if d == days - 1 else "2.1.270",
         "cache_read": 0.0 if (d, k) in blank else 900.0, "cache_creation": 0.0 if (d, k) in blank else 100.0,
         "is_miss": (d, k) in misses or (d, k) in blank}
        for d in range(days) for k in range(per_day)])


# Two misses in the 420 turns of the 14 days before the week, then 3 in 7 on Sep 21 from 09:00.
RISING = {(2, 0), (9, 0), (20, 3), (20, 6), (20, 9)}


def test_misses_rising_today_warn_once_saying_how_many_missed():
    state = new_state()
    warning = early_warning(prompt_frame(RISING), [], state, NOW, h=6)
    assert (warning["at"], warning["since"], warning["misses"], warning["turns"], warning["base_rate"]) == (
        "2026-09-21T11:00:00+00:00", "2026-09-21T09:00:00+00:00", 3, 7, 0.0048)
    assert early_message(warning, NOW) == (
        "3 of the last 7 new-prompt turns missed the cache (usually 0.5%), since 09:00, on Claude Code 2.1.280 "
        "(since 09-21). The daily check confirms or clears it within a few days.")
    assert state["early_warnings"] == [warning]
    assert early_warning(prompt_frame(RISING), [], state, NOW, h=6) is None


def test_a_sustained_rise_across_several_alarms_reports_the_whole_rise():
    # Nine misses in a row cross h=6 three times, about every third miss (each miss
    # adds about 2.35): the CUSUM's reset after each alarm is bookkeeping, not the sum
    # standing at 0, so the rise is reported from the first of the nine, not just the
    # last three.
    misses = RISING | {(20, k) for k in range(3, 12)}
    warning = early_warning(prompt_frame(misses), [], new_state(), NOW, h=6)
    assert (warning["at"], warning["since"], warning["misses"], warning["turns"]) == (
        "2026-09-21T11:40:00+00:00", "2026-09-21T09:00:00+00:00", 9, 9)


def test_turns_without_cache_token_counts_dont_count_toward_a_warning():
    # A parser that loses the cache counts makes every turn read as a miss; the
    # blank-cache alert is the one for that, and a warning would block real ones for a week.
    blank = {(20, k) for k in range(3, 13)}
    assert early_warning(prompt_frame({(2, 0), (9, 0)}, blank=blank), [], new_state(), NOW, h=6) is None


def test_no_warning_while_a_cache_incident_is_open_or_over_its_days():
    open_incident = {"metric": "cache_ratio", "start": "2026-09-20", "end": None, "status": "open"}
    known = {"metric": "cache_ratio", "start": "2026-09-21", "end": "2026-09-21", "status": "recovered"}
    assert early_warning(prompt_frame(RISING), [open_incident], new_state(), NOW, h=6) is None
    assert early_warning(prompt_frame(RISING), [known], new_state(), NOW, h=6) is None


def test_an_alarm_more_than_a_day_old_is_not_news():
    assert early_warning(prompt_frame(RISING), [], new_state(), NOW + timedelta(days=1), h=6) is None


def test_too_few_turns_before_the_week_give_no_warning():
    # 14 turns a day: 196 in the 14 days before the week. At h=4 the misses at turns
    # 3, 6 and 9 would otherwise cross it (1.589, then 3.096 and 3.014 after the
    # hits that follow, then 4.604), so this exercises the MIN_BASE_TURNS gate rather
    # than the CUSUM staying quiet on its own.
    assert early_warning(prompt_frame(RISING, per_day=14), [], new_state(), NOW, h=4) is None
