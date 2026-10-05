"""What a point of the usage limit costs: the 7-day limit's used share from the status line
samples, walked once into the points each sample raised, and priced at list from this
machine's responses in the hours the samples cover.

It judges nothing yet: no rule, threshold or alert turns on any number here. A rule on the
rate needs weeks of samples on both sides of a change, and a gate measured on them."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Iterator, NamedTuple, Sequence


class Step(NamedTuple):
    """One sample whose window's previous sample is on the same UTC day: when, which day and
    window, how far it raised the window's highest share so far, and since when that highest
    share stood, but no earlier than the window's first sample that day."""
    when: datetime
    day: str
    window: Any
    points: float
    since: datetime


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def readings(samples: Sequence[Any]) -> list[tuple[datetime, float, Any]]:
    """Each sample's UTC time, the 7-day window's used share and which window it is (its
    resets_at), for the samples that carry all three, in time order."""
    out = []
    for sample in samples:
        window = sample.get("seven_day") if isinstance(sample, dict) else None
        if not isinstance(window, dict) or not (_finite(window.get("used_percentage"))
                                                and _finite(window.get("resets_at"))):
            continue
        try:
            when = datetime.fromisoformat(sample["at"]).astimezone(timezone.utc)
        except (KeyError, TypeError, ValueError):
            continue
        out.append((when, float(window["used_percentage"]), window["resets_at"]))
    return sorted(out, key=lambda reading: reading[0])


def steps(found: Sequence[tuple[datetime, float, Any]]) -> Iterator[Step]:
    """The steps of `found` (readings, in time order): every sample whose window's sample
    before it is on the same UTC day, with how far it raised that window's highest share so
    far, 0 when it didn't.

    The highest so far, not each rise between neighbours. On the owner's samples (526 of
    them, 2026-09-28 to 2026-10-04) 31 of the 33 falls inside one window were undone by the
    very next sample: a session idle since an older reading interleaving with a busy one.
    Summing every rise counted 42 points on 09-30, where the highest share rose by 20.

    Only rises seen within one day. What was used between a day's last sample and the next
    day's first, on claude.ai or another machine or while this one was off, belongs to no
    day this machine sampled, so it is left out rather than added to whichever day the
    samples resume on. A window's first sample of a day is no step."""
    level: dict[Any, float] = {}
    level_at: dict[Any, datetime] = {}
    last_day: dict[Any, str] = {}
    day_start: dict[Any, datetime] = {}
    for when, used, window in found:
        day = when.date().isoformat()
        if last_day.get(window) == day:
            yield Step(when, day, window, max(0.0, used - level[window]), max(level_at[window], day_start[window]))
        else:
            day_start[window] = when
        if window not in level or used > level[window]:
            level[window], level_at[window] = used, when
        last_day[window] = day
