"""The early warning's core: a Bernoulli likelihood-ratio CUSUM on cache misses."""

from ccdrift.early import clamp_rate, miss_cusum


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
