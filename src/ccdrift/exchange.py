"""What a point of the usage limit costs: the 7-day limit's used share from the status line
samples, walked once into the points each sample raised, and priced at list from this
machine's responses in the hours the samples cover.

It judges nothing yet: no rule, threshold or alert turns on any number here. A rule on the
rate needs weeks of samples on both sides of a change, and a gate measured on them."""

from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, NamedTuple, Optional, Sequence

import pandas as pd

from ccdrift.history import HistoryError, load_history
from ccdrift.logs import Tables, default_source, outside_sdk
from ccdrift.prices import Price
from ccdrift.quota import quota_path, read_samples
from ccdrift.spend import (MATERIAL_SHARE, fitted_prices, priced_in_window, priced_total, response_dollars,
                           unpriced_models)
from ccdrift.texts import (COMPARE_LINES, QUOTA_LINES, QUOTA_TABLE, empty_rise_line, left_out_line,
                           no_transcripts_message, quota_row, table_header, unpriced_models_text)

# Days in the window when --days isn't given: two of the 7-day limit's weeks, so a reset
# falls inside it and each day of the week appears twice.
DEFAULT_DAYS = 14


class Step(NamedTuple):
    """One sample after its window's day opened (see `steps`): when, which day and window,
    how far it raised the window's highest share so far, and since when that highest share
    stood, but no earlier than the sample the window's day opened at."""
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
    """The steps of `found` (readings, in time order): every sample after its window's day
    opened (below), with how far it raised that window's highest share so far, 0 when it
    didn't.

    The highest so far, not each rise between neighbours. On the owner's samples (526 of
    them, 2026-09-28 to 2026-10-04) 31 of the 33 falls inside one window were undone by the
    very next sample: a session idle since an older reading interleaving with a busy one.
    Summing every rise counted 42 points on 09-30, where the highest share rose by 20.

    Only rises seen within one day. What was used between a day's last sample and the next
    day's first, on claude.ai or another machine or while this one was off, belongs to no
    day this machine sampled, so it is left out rather than added to whichever day the
    samples resume on. A window's day opens at its first sample that day at or above the
    highest share carried in from earlier days, and that sample is no step. A sample below
    it before then is an idle session's older reading and is passed over: opening on it
    would put the use since the day before into the day's first real rise. On the owner's
    samples that happened on none of 8 window-days, 2026-09-28 to 2026-10-04. An older
    reading equal to the carried share still opens the day, since samples carry no session
    to tell it from a current one, so the use from then to the next rise lands there."""
    level: dict[Any, float] = {}
    level_at: dict[Any, datetime] = {}
    opened: dict[Any, tuple[str, datetime]] = {}
    for when, used, window in found:
        day = when.date().isoformat()
        if window in opened and opened[window][0] == day:
            yield Step(when, day, window, max(0.0, used - level[window]), max(level_at[window], opened[window][1]))
        elif window in level and used < level[window]:
            continue
        else:
            opened[window] = (day, when)
        if window not in level or used > level[window]:
            level[window], level_at[window] = used, when


def spans(found: Sequence[tuple[datetime, float, Any]]) -> dict[str, list[tuple[datetime, datetime]]]:
    """Each UTC day's sampled spans: for every window with a step that day, from the sample
    its day opened at to its last step, so the hours priced are the hours its points were
    counted over. A reset day has one span per window, and they can overlap, since an idle
    session can still show the old window after the reset."""
    bounds: dict[tuple[str, Any], tuple[datetime, datetime]] = {}
    for step in steps(found):
        key = (step.day, step.window)
        bounds[key] = (bounds[key][0] if key in bounds else step.since, step.when)
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
    points, dollars and rate, the window's dollars over its points on every day with a
    dollar figure, the days with points and none, the spread of the day rates, the Agent
    SDK share of the window's dollars, the models with no price, how many responses fell in
    the sampled hours, the empty rises, and how many samples there were. Every response on
    this machine counts, both threads and Agent SDK sessions alike.

    A day that gained no point still adds its dollars to the window's: shares are whole
    numbers, so what a quiet day bought shows up in a later day's rise, and leaving its
    dollars out would understate the rate wherever use is light."""
    found = readings(samples)
    walked = list(steps(found))
    window = sorted({step.day for step in walked if step.day < today.isoformat()})[-days:]
    responses = tables.responses
    prices = fitted_prices(tables)
    rows = day_rows(responses, prices, found, window)
    day_spans = spans(found)
    counted = pd.concat([in_spans(responses, day_spans[day]) for day in window]) if window else responses.iloc[0:0]
    dated = [row for row in rows if row["dollars"] is not None]
    left = [row for row in rows if row["dollars"] is None and row["points"] > 0]
    points, dollars = sum(row["points"] for row in dated), sum(row["dollars"] for row in dated)
    rates = pd.Series([row["rate"] for row in rows if row["rate"] is not None], dtype="float64")
    paid = (pd.concat([in_spans(responses, day_spans[row["day"]]) for row in dated]) if dated
            else responses.iloc[0:0])
    spent = response_dollars(paid, prices)
    total = float(spent.sum())
    return {"first": window[0] if window else None, "last": window[-1] if window else None, "days": rows,
            "points": points, "dollars": dollars if dated else None, "rate": dollars / points if points > 0 else None,
            "day_rates": ({"median": float(rates.median()), "low": float(rates.min()), "high": float(rates.max())}
                          if points > 0 else None),
            "left_out": {"days": len(left), "points": sum(row["points"] for row in left)},
            "sdk_share": float(spent[~outside_sdk(paid)].sum()) / total if total > 0 else None,
            "priced_models": sorted(priced_in_window(counted, prices)),
            "unpriced": [{"model": model, "share": share} for model, share in unpriced_models(counted, prices)],
            "empty": [{"when": step.when.isoformat(), "day": step.day, "points": step.points,
                       "minutes": round((step.when - step.since).total_seconds() / 60)}
                      for step in empty_rises(responses, [step for step in walked if step.day in window])],
            "responses": len(counted), "samples": len(samples)}


def quota_lines(summary: dict[str, Any]) -> list[str]:
    """The rates as the terminal shows them: the window's rate and spread, a row a day, the
    empty rises, then what was counted and what the rate can't see."""
    lines = [QUOTA_LINES["head"]]
    if summary["rate"] is None:
        lines.append(QUOTA_LINES["window_unpriced"].format(first=summary["first"], last=summary["last"]))
    else:
        spread = summary["day_rates"]
        lines.append(QUOTA_LINES["window"].format(first=summary["first"], last=summary["last"],
                                                  points=summary["points"], dollars=summary["dollars"],
                                                  rate=summary["rate"], median=spread["median"], low=spread["low"],
                                                  high=spread["high"]))
    if summary["left_out"]["days"]:
        lines.append(left_out_line(summary["left_out"]["days"], summary["left_out"]["points"]))
    lines += ["", table_header(QUOTA_TABLE), *[quota_row(row) for row in summary["days"]], ""]
    if summary["empty"]:
        lines += [QUOTA_LINES["empty"], *[empty_rise_line(rise) for rise in summary["empty"]]]
    else:
        lines.append(QUOTA_LINES["no_empty"])
    if not summary["responses"]:
        lines.append(QUOTA_LINES["nothing_here"])
    elif not summary["priced_models"]:
        lines.append(COMPARE_LINES["no_prices"])
    elif summary["unpriced"]:
        lines.append(COMPARE_LINES["unpriced_one" if len(summary["unpriced"]) == 1 else "unpriced_many"].format(
            models=unpriced_models_text(summary["unpriced"]), cutoff=MATERIAL_SHARE))
    lines.append(QUOTA_LINES["counted_plain"] if summary["sdk_share"] is None
                 else QUOTA_LINES["counted"].format(share=summary["sdk_share"]))
    lines += [QUOTA_LINES["elsewhere"], QUOTA_LINES["models"]]
    if not summary["default_source"]:
        lines.append(QUOTA_LINES["named_source"])
    return lines


def quota_json(summary: dict[str, Any]) -> str:
    """The rates as JSON: the summary, with each empty rise as its day and points only."""
    shared = {**summary, "empty": [{"day": rise["day"], "points": rise["points"]} for rise in summary["empty"]]}
    return json.dumps(shared, indent=1) + "\n"


def run_quota(source: Path, state_path: Path, days: Optional[int] = None, as_json: bool = False,
              today: Optional[date] = None) -> int:
    """Print what a point of the 7-day limit cost on each sampled day. Reads the samples beside
    the state file first, so a machine with none is told how to start before the history is
    read, then the history like `cost`. Saves nothing and judges nothing."""
    today = today or datetime.now(timezone.utc).date()
    samples = read_samples(quota_path(state_path))
    found = readings(samples)
    if not found:
        print(QUOTA_LINES["no_samples"], file=sys.stderr)
        return 2
    if not any(step.day < today.isoformat() for step in steps(found)):
        print(QUOTA_LINES["no_steps"].format(samples=len(samples)), file=sys.stderr)
        return 2
    try:
        tables = load_history(source, state_path, claim=False)
    except HistoryError as exc:
        print(exc, file=sys.stderr)
        return 1
    if tables.responses.empty:
        print(no_transcripts_message(source), file=sys.stderr)
        return 2
    summary = quota_summary(tables, samples, days or DEFAULT_DAYS, today)
    summary["default_source"] = source.expanduser().resolve() == default_source().expanduser().resolve()
    print(quota_json(summary) if as_json else "\n".join(quota_lines(summary)) + "\n", end="")
    return 0
