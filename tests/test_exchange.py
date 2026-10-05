"""What a point of the usage limit costs: the walk over the quota samples, and its prices."""

from datetime import datetime, timezone

from ccdrift.exchange import Step, readings, steps


def sample(stamp, used, resets=1_790_000_000):
    """A quota sample as `ccdrift status --short --stdin` keeps one."""
    return {"at": stamp, "version": "2.1.289", "model": "claude-opus-5",
            "seven_day": {"used_percentage": used, "resets_at": resets}}


def utc(stamp):
    return datetime.fromisoformat(stamp).astimezone(timezone.utc)


def walk(samples):
    return list(steps(readings(samples)))


def test_readings_keep_only_samples_with_a_time_a_7_day_share_and_its_reset_in_time_order():
    samples = [sample("2026-09-05T12:00:00+00:00", 14), {"at": "2026-09-05T11:00:00+00:00"},
               {"at": "2026-09-05T11:30:00+00:00", "seven_day": {"used_percentage": True, "resets_at": 1}},
               {"at": "nonsense", "seven_day": {"used_percentage": 3, "resets_at": 1}},
               sample("2026-09-05T13:00:00+03:00", 12)]
    assert readings(samples) == [(utc("2026-09-05T10:00:00+00:00"), 12.0, 1_790_000_000),
                                 (utc("2026-09-05T12:00:00+00:00"), 14.0, 1_790_000_000)]


def test_a_step_is_credited_only_when_the_windows_previous_sample_is_on_the_same_utc_day():
    found = walk([sample("2026-09-05T10:00:00+00:00", 10), sample("2026-09-05T20:00:00+00:00", 12),
                  sample("2026-09-09T09:00:00+00:00", 40), sample("2026-09-09T20:00:00+00:00", 41)])
    assert [(step.day, step.points) for step in found] == [("2026-09-05", 2.0), ("2026-09-09", 1.0)]


def test_a_steps_points_are_how_far_it_raised_the_highest_share_so_far_through_stale_samples():
    # A session idle since 40% interleaving with a busy one at 45%, then 46%.
    found = walk([sample("2026-09-05T10:00:00+00:00", 40), sample("2026-09-05T10:05:00+00:00", 45),
                  sample("2026-09-05T10:06:00+00:00", 40), sample("2026-09-05T10:07:00+00:00", 46)])
    assert [step.points for step in found] == [5.0, 0.0, 1.0]


def test_a_steps_since_is_when_the_highest_share_was_set_and_never_before_that_days_first_sample():
    found = walk([sample("2026-09-05T22:00:00+00:00", 30),
                  sample("2026-09-06T08:00:00+00:00", 30), sample("2026-09-06T08:30:00+00:00", 31),
                  sample("2026-09-06T08:45:00+00:00", 31), sample("2026-09-06T09:00:00+00:00", 29),
                  sample("2026-09-06T09:10:00+00:00", 33)])
    # 30 was set on Sep 5, so Sep 6's first rise counts from Sep 6's first sample; neither 31
    # again at 08:45 nor the stale 29 at 09:00 moves it, so the rise to 33 counts from 08:30.
    assert [(step.when, step.points, step.since) for step in found] == [
        (utc("2026-09-06T08:30:00+00:00"), 1.0, utc("2026-09-06T08:00:00+00:00")),
        (utc("2026-09-06T08:45:00+00:00"), 0.0, utc("2026-09-06T08:30:00+00:00")),
        (utc("2026-09-06T09:00:00+00:00"), 0.0, utc("2026-09-06T08:30:00+00:00")),
        (utc("2026-09-06T09:10:00+00:00"), 2.0, utc("2026-09-06T08:30:00+00:00"))]


def test_a_reset_day_steps_through_each_window_on_its_own():
    # The old window's 96 after the reset is a stale reading from a session idle since then.
    found = walk([sample("2026-09-06T01:00:00+00:00", 90, resets=1), sample("2026-09-06T05:00:00+00:00", 95, resets=1),
                  sample("2026-09-06T06:10:00+00:00", 2, resets=2), sample("2026-09-06T07:00:00+00:00", 5, resets=2),
                  sample("2026-09-06T07:30:00+00:00", 96, resets=1)])
    assert found == [Step(utc("2026-09-06T05:00:00+00:00"), "2026-09-06", 1, 5.0, utc("2026-09-06T01:00:00+00:00")),
                     Step(utc("2026-09-06T07:00:00+00:00"), "2026-09-06", 2, 3.0, utc("2026-09-06T06:10:00+00:00")),
                     Step(utc("2026-09-06T07:30:00+00:00"), "2026-09-06", 1, 1.0, utc("2026-09-06T05:00:00+00:00"))]
