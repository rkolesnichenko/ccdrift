"""Before and after a change you made on purpose: the complete UTC days on each side of a
date you name, on the numbers a configuration change moves, each side with its own spread
from day to day beside it.

It judges nothing. A difference inside a side's own day-to-day spread is not evidence the
change did anything, and telling a real change from that spread takes a threshold measured
by a gate, which this command does not have, so no rule, verdict or alert turns on any
number here."""

from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from ccdrift.history import HistoryError, load_history
from ccdrift.logs import Tables, outside_sdk
from ccdrift.prices import Price
from ccdrift.quota import quota_path, read_samples
from ccdrift.sessions import session_starts
from ccdrift.spend import (MATERIAL_SHARE, fitted_prices, priced_in_window, priced_total, spend_turns,
                           unpriced_models)
from ccdrift.texts import (COMPARE_LINES, COMPARE_ROWS, COMPARE_TABLE, SIDE_NAMES, compactions_cell,
                           no_transcripts_message, spread_cell, starts_cell, table_header, table_row,
                           unpriced_models_text, version_key, versions_text)

# Days on each side when --days isn't given: a week, so each side holds every weekday once
# and a weekly rhythm in the work lands on both sides rather than on one.
DEFAULT_DAYS = 7

# The metrics each side reports a spread of, in the order they print.
METRICS = ("dollars_per_day", "dollars_per_prompt", "context_per_response", "session_start", "quota_points")


def sides(at: date, days: int, today: date) -> tuple[list[str], list[str]]:
    """The calendar days of each side as ISO dates: the `days` before `at`, and those of the
    `days` after it that are complete, so none on or after `today`. `at` is on neither
    side: the change was made some time that day."""
    before = [(at - timedelta(days=k)).isoformat() for k in range(days, 0, -1)]
    after = [(at + timedelta(days=k)).isoformat() for k in range(1, days + 1) if at + timedelta(days=k) < today]
    return before, after


def _days(frame: pd.DataFrame) -> pd.Series:
    return frame["day"].astype(str)


def daily_dollars(turns: pd.DataFrame, prices: dict[str, Price]) -> pd.Series:
    """What each day cost, priced as `ccdrift cost` prices its window: a day whose models
    with no price carry MATERIAL_SHARE of its tokens has no figure rather than a partial one."""
    totals = {day: priced_total(group, prices) for day, group in turns.groupby(_days(turns), sort=True)}
    return pd.Series({day: total for day, total in totals.items() if total is not None}, dtype="float64")


def daily_dollars_per_prompt(turns: pd.DataFrame, prices: dict[str, Price]) -> pd.Series:
    """Each day's dollars over its main-thread prompts, subagents' spend included, since a
    subagent's spend is charged to the prompts that set it off. No figure on a day without
    a prompt or without dollars."""
    prompts = turns[turns["main_thread"].astype(bool) & turns["new_prompt"].astype(bool)]
    counts = prompts.groupby(_days(prompts)).size()
    return (daily_dollars(turns, prices) / counts).dropna()


def daily_context(turns: pd.DataFrame) -> pd.Series:
    """Each day's median main-thread prompt size: input, cache writes and cache reads
    together, everything the model was sent. A compaction window moves this first."""
    main = turns[turns["main_thread"].astype(bool)]
    size = main["input_tokens"].fillna(0) + main["cache_creation"].fillna(0) + main["cache_read"].fillna(0)
    return size.groupby(_days(main)).median().astype("float64")


def daily_session_start(starts: pd.DataFrame) -> pd.Series:
    """Each day's median session start, from session_starts: the CLI main thread's first
    prompt size per transcript."""
    return starts.groupby("day")["prompt_tokens"].median().astype("float64")


def auto_compactions(compactions: pd.DataFrame) -> pd.DataFrame:
    """The compactions Claude Code started itself on the main thread outside the Agent SDK,
    the ones a compaction window decides, filtered as the report's "compacts at" is."""
    if compactions.empty:
        return compactions
    return compactions[outside_sdk(compactions) & ~compactions["is_sidechain"].astype(bool)
                       & (compactions["trigger"] == "auto")]


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _readings(samples: Sequence[Any]) -> list[tuple[datetime, float, Any]]:
    """Each sample's time, the 7-day window's used share and which window it is (its
    resets_at), for the samples that carry all three."""
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
    return out


def quota_points(samples: Sequence[Any]) -> pd.Series:
    """Quota points used each UTC day, from the 7-day window's used share in the status
    line samples: how far each sample raised the highest share seen so far in its window
    (one resets_at), counted on that sample's day. A day sampled with no rise used 0; a
    window's first sample has nothing to rise from and gives its day no figure.

    The highest so far, not each rise between neighbours. On the owner's samples (526 of
    them, 2026-09-28 to 2026-10-04) 31 of the 33 falls inside one window were undone by the
    very next sample: a session idle since an older reading interleaving with a busy one.
    Summing every rise counted 42 points on 09-30, where the highest share rose by 20."""
    level: dict[Any, float] = {}
    points: dict[str, float] = {}
    for when, used, window in sorted(_readings(samples), key=lambda reading: reading[0]):
        if window in level:
            day = when.date().isoformat()
            points[day] = points.get(day, 0.0) + max(0.0, used - level[window])
            level[window] = max(level[window], used)
        else:
            level[window] = used
    return pd.Series(points, dtype="float64").sort_index(kind="stable")


def spread(daily: pd.Series, days: Sequence[str]) -> dict[str, Any]:
    """A side's median day, its lowest and highest day, and how many of its days had a figure."""
    values = daily[daily.index.isin(list(days))].dropna()
    if values.empty:
        return {"median": None, "low": None, "high": None, "days": 0}
    return {"median": float(values.median()), "low": float(values.min()), "high": float(values.max()),
            "days": int(len(values))}


def compaction_spread(compactions: pd.DataFrame, days: Sequence[str]) -> dict[str, Any]:
    """A side's compactions, counted with the smallest, median and largest context each
    began at. Too few on most days for a daily median, and the one that stands out is the
    one worth seeing, so they are not summarised by day."""
    values = (compactions.loc[_days(compactions).isin(list(days)), "pre_tokens"].dropna().astype(float)
              if not compactions.empty else pd.Series(dtype="float64"))
    if values.empty:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {"count": int(len(values)), "min": float(values.min()), "median": float(values.median()),
            "max": float(values.max())}


def version_shares(turns: pd.DataFrame, days: Sequence[str]) -> dict[str, float]:
    """The Claude Code versions a side's main-thread responses ran, each with its share, in
    version order."""
    main = turns[turns["main_thread"].astype(bool) & _days(turns).isin(list(days))]
    if main.empty:
        return {}
    versions = (main["version"] if "version" in main else pd.Series(None, index=main.index, dtype="object"))
    shares = versions.fillna("unknown").astype(str).value_counts(normalize=True)
    return {version: float(shares[version]) for version in sorted(shares.index, key=version_key)}


def sdk_left_out(responses: pd.DataFrame, days: Sequence[str]) -> int:
    """How many of a side's main-thread responses came from Agent SDK sessions, which every
    metric here leaves out as `ccdrift cost` does."""
    rows = responses["main_thread"].astype(bool) & ~outside_sdk(responses) & _days(responses).isin(list(days))
    return int(rows.sum())


def compare_summary(tables: Tables, at: date, days: int, today: date, samples: Sequence[Any]) -> dict[str, Any]:
    """Everything `ccdrift compare` reports, as plain values: each side's days, versions and
    what it left out, each metric's spread on both sides, the compactions, and the models
    with no price. Aggregates only: no path, session id or project name."""
    before, after = sides(at, days, today)
    responses = tables.responses
    turns = spend_turns(responses, today)
    turns = turns[_days(turns).isin(before + after)]
    prices = fitted_prices(tables)
    starts = session_starts(turns)
    daily = {"dollars_per_day": daily_dollars(turns, prices),
             "dollars_per_prompt": daily_dollars_per_prompt(turns, prices),
             "context_per_response": daily_context(turns),
             "session_start": daily_session_start(starts),
             "quota_points": quota_points(samples)}
    compactions = auto_compactions(tables.compactions)
    present = set(_days(turns))

    def side(window: list[str]) -> dict[str, Any]:
        return {"first": window[0] if window else None, "last": window[-1] if window else None,
                "days_spanned": len(window), "days_with_responses": len(present & set(window)),
                "versions": version_shares(turns, window), "sdk_left_out": sdk_left_out(responses, window),
                "session_starts": int(starts["day"].astype(str).isin(window).sum()),
                "compactions": compaction_spread(compactions, window)}

    return {"at": at.isoformat(), "days": days,
            "before": side(before), "after": side(after),
            "metrics": {name: {"before": spread(daily[name], before), "after": spread(daily[name], after)}
                        for name in METRICS},
            "priced_models": sorted(priced_in_window(turns, prices)),
            "unpriced": [{"model": model, "share": share} for model, share in unpriced_models(turns, prices)]}


def compare_lines(summary: dict[str, Any]) -> list[str]:
    """The comparison as the terminal shows it: each side's days, the table, then what
    frames it: versions, what was left out, the quota rows' reach and models with no price."""
    before, after, metrics = summary["before"], summary["after"], summary["metrics"]
    lines = [COMPARE_LINES["before"].format(at=summary["at"], first=before["first"], last=before["last"],
                                            present=before["days_with_responses"], spanned=before["days_spanned"]),
             COMPARE_LINES["after"].format(first=after["first"], last=after["last"],
                                           present=after["days_with_responses"], spanned=after["days_spanned"]),
             COMPARE_LINES["neither"].format(at=summary["at"]),
             COMPARE_LINES["cells"], "", table_header(COMPARE_TABLE)]
    for metric in ("dollars_per_day", "dollars_per_prompt", "context_per_response"):
        lines.append(table_row(COMPARE_TABLE, [COMPARE_ROWS[metric], spread_cell(metric, metrics[metric]["before"]),
                                               spread_cell(metric, metrics[metric]["after"])]))
    lines.append(table_row(COMPARE_TABLE, [COMPARE_ROWS["session_start"],
                                           starts_cell(metrics["session_start"]["before"], before["session_starts"]),
                                           starts_cell(metrics["session_start"]["after"], after["session_starts"])]))
    lines.append(table_row(COMPARE_TABLE, [COMPARE_ROWS["compactions"], compactions_cell(before["compactions"]),
                                           compactions_cell(after["compactions"])]))
    quota = metrics["quota_points"]
    sampled = quota["before"]["days"] or quota["after"]["days"]
    if sampled:
        lines.append(table_row(COMPARE_TABLE, [COMPARE_ROWS["quota_points"],
                                               spread_cell("quota_points", quota["before"]),
                                               spread_cell("quota_points", quota["after"])]))
    lines.append("")
    lines += [COMPARE_LINES["versions"].format(side=SIDE_NAMES[name],
                                               versions=versions_text(summary[name]["versions"]))
              for name in ("before", "after")]
    lines.append(COMPARE_LINES["sdk"].format(before=before["sdk_left_out"], after=after["sdk_left_out"]))
    lines.append(COMPARE_LINES["quota" if sampled else "no_quota"])
    if not summary["priced_models"]:
        lines.append(COMPARE_LINES["no_prices"])
    elif summary["unpriced"]:
        lines.append(COMPARE_LINES["unpriced"].format(models=unpriced_models_text(summary["unpriced"]),
                                                      cutoff=MATERIAL_SHARE))
    lines.append(COMPARE_LINES["not_evidence"])
    return lines


def compare_json(summary: dict[str, Any]) -> str:
    """The comparison as JSON: the summary itself, which holds aggregates only."""
    return json.dumps(summary, indent=1) + "\n"


def run_compare(source: Path, state_path: Path, at: date, days: Optional[int] = None, as_json: bool = False,
                today: Optional[date] = None) -> int:
    """Print the days before `at` beside the days after it. Reads the history like `cost`
    and the quota samples beside the state file, saves nothing, and judges nothing."""
    today = today or datetime.now(timezone.utc).date()
    days = days or DEFAULT_DAYS
    if at >= today:
        print(COMPARE_LINES["at_not_past"].format(at=at.isoformat()), file=sys.stderr)
        return 2
    try:
        tables = load_history(source, state_path, claim=False)
    except HistoryError as exc:
        print(exc, file=sys.stderr)
        return 1
    if tables.responses.empty:
        print(no_transcripts_message(source), file=sys.stderr)
        return 2
    summary = compare_summary(tables, at, days, today, read_samples(quota_path(state_path)))
    if not summary["after"]["days_with_responses"]:
        print(COMPARE_LINES["no_after"].format(at=at.isoformat()), file=sys.stderr)
        return 2
    if not summary["before"]["days_with_responses"]:
        print(COMPARE_LINES["no_before"].format(days=days, at=at.isoformat()), file=sys.stderr)
        return 2
    print(compare_json(summary) if as_json else "\n".join(compare_lines(summary)) + "\n", end="")
    return 0
