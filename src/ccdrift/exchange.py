"""What a point of the usage limit costs: the 7-day limit's used share from the status line
samples, walked once into the points each sample raised, and priced at list from this
machine's responses in the hours the samples cover.

It judges nothing yet: no rule, threshold or alert turns on any number here. A rule on the
rate needs weeks of samples on both sides of a change, and a gate measured on them."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any, Iterator, NamedTuple, Sequence

import pandas as pd

from ccdrift.logs import Tables, outside_sdk
from ccdrift.prices import Price
from ccdrift.spend import fitted_prices, priced_in_window, priced_total, response_dollars, unpriced_models

# Days in the window when --days isn't given: two of the 7-day limit's weeks, so a reset
# falls inside it and each day of the week appears twice.
DEFAULT_DAYS = 14


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


def spans(found: Sequence[tuple[datetime, float, Any]]) -> dict[str, list[tuple[datetime, datetime]]]:
    """Each UTC day's sampled spans: for every window sampled that day, its first and last
    sample's time. A reset day has one span per window, and they can overlap, since an idle
    session can still show the old window after the reset."""
    bounds: dict[tuple[str, Any], tuple[datetime, datetime]] = {}
    for when, _, window in found:
        key = (when.date().isoformat(), window)
        bounds[key] = (bounds[key][0] if key in bounds else when, when)
    out: dict[str, list[tuple[datetime, datetime]]] = {}
    for (day, _), span in bounds.items():
        out.setdefault(day, []).append(span)
    return out


def _within(times: pd.Series, start: datetime, end: datetime) -> pd.Series:
    return (times > start) & (times <= end)


def in_spans(responses: pd.DataFrame, day_spans: Sequence[tuple[datetime, datetime]]) -> pd.DataFrame:
    """The responses timestamped inside any of `day_spans`, each once however many of the
    spans it falls in."""
    mask = pd.Series(False, index=responses.index)
    for start, end in day_spans:
        mask |= _within(responses["timestamp"], start, end)
    return responses[mask]


def empty_rises(responses: pd.DataFrame, found: Sequence[Step]) -> list[Step]:
    """The steps that raised a window's highest share while this machine logged no response
    at all since that share was set: use on another surface, or by someone else. Counted by
    responses rather than dollars, so a response with no price is still activity."""
    times = responses["timestamp"]
    return [step for step in found if step.points > 0 and not _within(times, step.since, step.when).any()]


def day_rows(responses: pd.DataFrame, prices: dict[str, Price], found: Sequence[tuple[datetime, float, Any]],
             days: Sequence[str]) -> list[dict[str, Any]]:
    """One row per day of `days`: its points, the list dollars of this machine's responses
    in its sampled spans (none when its unpriced models carry MATERIAL_SHARE of their tokens,
    as `cost` decides), and dollars per point when both exist and points rose."""
    points: dict[str, float] = {}
    for step in steps(found):
        points[step.day] = points.get(step.day, 0.0) + step.points
    day_spans = spans(found)
    rows = []
    for day in days:
        dollars = priced_total(in_spans(responses, day_spans.get(day, [])), prices)
        gained = points.get(day, 0.0)
        rows.append({"day": day, "points": gained, "dollars": dollars,
                     "rate": dollars / gained if dollars is not None and gained > 0 else None})
    return rows


def quota_summary(tables: Tables, samples: Sequence[Any], days: int, today: date) -> dict[str, Any]:
    """Everything `ccdrift quota` reports, as plain values: the window's days, each day's
    points, dollars and rate, the window's dollars over its points on the days that have
    both, the spread of the day rates, the Agent SDK share of the window's dollars, the
    models with no price, the empty rises, and how many samples there were. Every response
    on this machine counts, both threads and Agent SDK sessions alike."""
    found = readings(samples)
    walked = list(steps(found))
    window = sorted({step.day for step in walked if step.day < today.isoformat()})[-days:]
    responses = tables.responses
    prices = fitted_prices(tables)
    rows = day_rows(responses, prices, found, window)
    day_spans = spans(found)
    counted = pd.concat([in_spans(responses, day_spans[day]) for day in window]) if window else responses.iloc[0:0]
    priced = [row for row in rows if row["rate"] is not None]
    points, dollars = sum(row["points"] for row in priced), sum(row["dollars"] for row in priced)
    rates = pd.Series([row["rate"] for row in priced], dtype="float64")
    spent = response_dollars(counted, prices)
    total = float(spent.sum())
    return {"first": window[0] if window else None, "last": window[-1] if window else None, "days": rows,
            "points": points, "dollars": dollars if priced else None, "rate": dollars / points if priced else None,
            "day_rates": ({"median": float(rates.median()), "low": float(rates.min()), "high": float(rates.max())}
                          if priced else None),
            "sdk_share": float(spent[~outside_sdk(counted)].sum()) / total if total > 0 else None,
            "priced_models": sorted(priced_in_window(counted, prices)),
            "unpriced": [{"model": model, "share": share} for model, share in unpriced_models(counted, prices)],
            "empty": [{"when": step.when.isoformat(), "day": step.day, "points": step.points,
                       "minutes": round((step.when - step.since).total_seconds() / 60)}
                      for step in empty_rises(responses, [step for step in walked if step.day in window])],
            "samples": len(samples)}
