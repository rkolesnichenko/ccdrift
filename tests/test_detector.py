"""The daily detector: robust z-scores and the flag rule."""

import random

import numpy as np
import pandas as pd
import pytest

from ccdrift.detector import (DetectorConfig, _robust_z, baseline_bins, bin_metrics, detect,
                              first_flag_bin, flag_onsets, pooled_z)
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


def test_detector_output_is_unchanged_when_no_day_is_excluded():
    metrics = bin_metrics(daily_turns(prompt_turn_days(MOSTLY_CLEAN_CACHE + [8] * 6, random.Random(0))))
    nothing = {"cache_ratio": [False] * 20, "haiku_fraction": [False] * 20}
    pd.testing.assert_frame_equal(detect(metrics, DetectorConfig()), detect(metrics, DetectorConfig(), nothing))


def test_excluded_days_leave_the_baseline_so_a_long_shift_stays_deviant():
    # A real caching regression kept missing 5-10% of prompt turns for three
    # weeks, but against a rolling baseline its z-scores were back to about 0
    # within 8 days.
    metrics = bin_metrics(daily_turns([QUIET] * 14 + [HAIKU] * 20))
    rolling = detect(metrics, DetectorConfig())
    excluded = detect(metrics, DetectorConfig(), {"haiku_fraction": [False] * 14 + [True] * 20})
    assert abs(rolling["haiku_fraction__z"].iloc[-1]) < 1
    assert excluded["haiku_fraction__z"].iloc[-1] > 3.5


def test_baseline_bins_are_the_latest_days_not_excluded():
    assert baseline_bins(6, np.array([False, True, True, False, False, True]), window=2) == [3, 4]


def test_pooled_z_scores_days_together_weighted_by_their_turns():
    metrics = bin_metrics(daily_turns([QUIET] * 14 + [HAIKU, QUIET]))
    vals, n, var = (metrics[c].to_numpy(dtype=float)
                    for c in ("haiku_fraction", "haiku_fraction__n", "haiku_fraction__var"))
    expected = _robust_z(20 / 800, vals[:14], n[:14], var[:14], 800)
    assert pooled_z(metrics, "haiku_fraction", [14, 15], list(range(14))) == pytest.approx(expected)


def test_pooled_z_leaves_out_days_without_a_value():
    # A day with no new-prompt turns has a NaN cache ratio and 0 turns; it must
    # not turn the whole pooled score into NaN.
    clean = {"prompt_cache_read_ratio": [0.9] * 40}
    blank = {"prompt_cache_read_ratio": [float("nan")] * 400}
    days = [clean] * 5 + [blank] + [clean] * 8 + [clean, blank]
    metrics = bin_metrics(daily_turns(days))
    bins, baseline = [14, 15], list(range(14))
    vals = metrics["cache_ratio"].to_numpy(dtype=float)
    kept_bins = [j for j in bins if not np.isnan(vals[j])]
    kept_baseline = [j for j in baseline if not np.isnan(vals[j])]
    assert kept_bins != bins and kept_baseline != baseline
    scored = pooled_z(metrics, "cache_ratio", bins, baseline)
    assert scored == pytest.approx(pooled_z(metrics, "cache_ratio", kept_bins, kept_baseline))
    assert not np.isnan(scored)
