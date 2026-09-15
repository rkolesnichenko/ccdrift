"""The daily detector: robust z-scores and the flag rule."""

import random

from ccdrift.detector import DetectorConfig, bin_metrics, detect, first_flag_bin, flag_onsets
from tests.helpers import HAIKU, MOSTLY_CLEAN_CACHE, QUIET, daily_turns, prompt_turn_days


def test_detector_flags_haiku_appearing_after_days_without_any():
    # Main-thread Haiku share is exactly 0 on every real day, so past days show
    # no spread at all; that must not leave the detector blind.
    days = [{"is_haiku": [0.0] * 400}] * 14 + [{"is_haiku": [1.0] * 20 + [0.0] * 380}] * 6
    det = detect(bin_metrics(daily_turns(days)), DetectorConfig())
    assert first_flag_bin(det, "haiku_fraction") == 14


def test_detector_ignores_one_cache_miss_a_day_against_a_mostly_clean_baseline():
    # Days near 1.0 barely differ from each other, so judged against that
    # spread alone one miss in 40 turns looks like z = -40 (real logs: -12.9).
    days = prompt_turn_days(MOSTLY_CLEAN_CACHE + [1] * 6, random.Random(0))
    det = detect(bin_metrics(daily_turns(days)), DetectorConfig())
    assert first_flag_bin(det, "cache_ratio") is None


def test_detector_flags_a_sustained_rise_in_cache_misses():
    days = prompt_turn_days(MOSTLY_CLEAN_CACHE + [8] * 6, random.Random(0))
    det = detect(bin_metrics(daily_turns(days)), DetectorConfig())
    assert first_flag_bin(det, "cache_ratio") == 14


def test_cache_metric_flags_at_its_own_lower_threshold():
    # A confirmed caching regression in real logs scored z = -4.9, -6.4, -3.1,
    # -3.7 on its first days: 3.5 misses it and 3.0 catches it, but 3.0 on every
    # metric raised a false Haiku flag on clean synthetic logs. Here both
    # metrics shift by the same z = 3.37 for four days.
    ks = [[0, 1, 2][i % 3] for i in range(14)] + [6] * 4
    days = [{"prompt_cache_read_ratio": [0.0] * k + [1.0] * (100 - k),
             "is_haiku": [1.0] * k + [0.0] * (100 - k)} for k in ks]
    det = detect(bin_metrics(daily_turns(days)), DetectorConfig())
    assert first_flag_bin(det, "cache_ratio") == 14
    assert first_flag_bin(det, "haiku_fraction") is None


def test_detector_flags_a_shift_with_one_day_under_the_cutoff():
    # The real caching regression scored z = -3.8, -5.4, -2.8, -3.2 on its first
    # days against a cutoff of 3.0. Counting only days in a row, the one day
    # short of the cutoff restarted the count and hid a three-week incident.
    days = [QUIET] * 14 + [HAIKU, HAIKU, QUIET, HAIKU]
    det = detect(bin_metrics(daily_turns(days)), DetectorConfig())
    assert flag_onsets(det, "haiku_fraction") == [14]


def test_detector_ignores_a_metric_that_crosses_the_cutoff_every_third_day():
    days = [QUIET] * 14 + [HAIKU, QUIET, QUIET] * 3
    det = detect(bin_metrics(daily_turns(days)), DetectorConfig())
    assert flag_onsets(det, "haiku_fraction") == []
