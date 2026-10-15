"""`ccdrift calibrate`: whether the daily verdict works on this history as it did on the logs
it was tuned on. Every cutoff ccdrift ships was measured on one person's logs; this runs the
measurement behind three of them on anyone's: the daily detector's cache ratio, main-thread
Haiku share and subagent read-back, judged on the turns the check judges with the shipped
DetectorConfig.

Each metric is judged as the check judges it: over the days the check reads, with the days of
open and recovered incidents left out of the baselines (incidents.exclusions), and a flag run
that meets a recorded incident left unreported, as the check leaves it (_new_flag_run). The
quiet half: every other flag run on the unplanted history is a false alarm, and one over an
incident the user dismissed counts too, since dismissing it is the user saying so, though the
check would stay silent there. The reach half: a regression is planted from up to STARTS days
whose days meet no incident and fall within RECENT_DAYS, since the check reports only flags
begun that recently, and counts as caught only by a flag run the check would report by
the start's AFTER-th day after it, judged on the history up to that day as the check that day
would judge it, that the unplanted history up to then doesn't raise. Nothing is saved.

The plants are the ones the lab gates measured those settings with. lab/harness.py and
lab/subagent_cache.py import them from here, so a gate and this command can't plant two
different things under one name.

The early warnings are judged by their gates' own measurement, in warning_gates, on the
complete days the check reads: the cache-miss warning (G3) by how often it would warn on
nothing at this history's usual rate and volume and how fast it catches a planted miss rate,
the tool-loop warning (G8) by whether it alarms at the most sensitive usual rate and how
often it catches a planted one."""

from __future__ import annotations

import functools
import json
import math
import random
import statistics
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ccdrift import early, loops
from ccdrift.check import ACTIVE_DAY_RESPONSES, HISTORY_ACTIVE_DAYS, history_start
from ccdrift.detector import SUBAGENT_METRICS, DetectorConfig, bin_metrics, detect
from ccdrift.early import MIN_P0, prompt_turns
from ccdrift.history import HistoryError, load_history
from ccdrift.incidents import EXCLUDING, OPEN_END, RECENT_DAYS, exclusions
from ccdrift.logs import add_ratios, judged_subagent_loops, judged_turns
from ccdrift.loops import LOOP_SETTINGS, STREAMS, loop_turns
from ccdrift.state import load_state
from ccdrift.texts import (CALIBRATE_LINES, CALIBRATE_MOVES, CALIBRATE_NAMES, MOVES, WARNING_LINES, WARNING_NAMES,
                           WARNING_WATCHES, floors_text, no_transcripts_message, planted_text, spans_text)
from ccdrift.warning_gates import (CAUGHT_SHARE, LOOP_MAX_TURNS, LOOP_PLANT, MAX_FALSE_PER_WEEK, PROMPT_MAX_TURNS,
                                   PROMPT_PLANT, alarms_per_turn, loop_alarm_days, loop_base_rate, loop_plants,
                                   prompt_alarm_days, prompt_base_rate, prompt_plants, shift, turns_to_catch, within)

# Starting days for a plant, and the judged days each needs around it: the detector needs
# min_baseline (5) days to judge one, and a flag needs deviant_bins (3) deviant days of
# flag_window (4), so a start has 7 judged days before it and 3 after.
STARTS = 10
BEFORE = 7
AFTER = 3
# Seeds per start for the subagent plant, as G17 (lab/subagent_cache.py) measured it.
SEEDS = 5
# The bar the shipped settings cleared on the owner's logs, judged on the check's own turns
# with the recorded incident (cache, 2026-08-18..09-03) left out: a planted 5% caught from 10
# of 10 starts for the cache ratio and for Haiku share (2026-10-14, 51 days), and for subagent
# read-back in 50 of 50 runs against G17's bar of 90% (2026-09-29).
BAR = 0.05
SUBAGENT_SHARE = 0.9
# The plant sizes: the harness's sweep grid for the main thread, G17's for subagents. Each
# start's floor is the smallest size caught there.
SIZES = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70)
SUBAGENT_SIZES = (0.02, BAR)
# Seed of the Haiku plant, the harness sweep's.
HAIKU_SEED = 7
# Incident statuses: the ones the check follows, whose flag runs aren't false alarms, and
# every one, whose days no plant starts on and whose runs the check never reports.
FOLLOWED = ("open", "recovered", "persistent")
STATUSES = (*FOLLOWED, "dismissed")
# The plant behind each main-thread metric.
KINDS = {"cache_ratio": "cache", "haiku_fraction": "haiku"}
# The alerts with a measured setting this doesn't judge yet, by their keys in the state.
NOT_JUDGED = ("context_changes", "failed_requests", "cut_short", "hook_changes", "thinking_rises", "model_mismatches")


def plant(df: pd.DataFrame, kind: str, magnitude: float, seed: int = 0, start: Optional[int] = None) -> pd.DataFrame:
    """A copy of the turns `df` in time order with a regime change of `magnitude` (0 to 1)
    from turn `start` (default: the midpoint turn) to the end, the per-turn columns
    recomputed by add_ratios. "cache" moves that share of each new-prompt turn's cache
    read within the cache's TTL into cache writes; "haiku" moves that share of the turns
    to a Haiku model, chosen with `seed`."""
    rng = random.Random(seed)
    d = df.sort_values("timestamp", kind="stable").reset_index(drop=True).copy()
    cut = len(d) // 2 if start is None else start
    post = d.index >= cut
    if kind == "haiku":
        post_idx = list(d.index[post])
        k = int(round(len(post_idx) * magnitude))
        d.loc[rng.sample(post_idx, k), "model"] = "claude-haiku-4-5"
    elif kind == "cache":
        mask = post & d["prompt_within_ttl"]
        moved = d.loc[mask, "cache_read"] * magnitude
        d.loc[mask, "cache_read"] = d.loc[mask, "cache_read"] - moved
        d.loc[mask, "cache_creation"] = d.loc[mask, "cache_creation"] + moved
    else:
        raise ValueError(kind)
    return add_ratios(d)


def subagent_openings(responses: pd.DataFrame) -> pd.Series:
    """What each subagent transcript's first response read from the cache, by source file."""
    sub = responses[responses["is_sidechain"].astype(bool)].sort_values("timestamp", kind="stable")
    return sub.groupby("source_file", sort=True)["cache_read"].first()


def plant_subagent(loops: pd.DataFrame, opening: pd.Series, days: list[str], share: float,
                   seed: int) -> pd.DataFrame:
    """`loops` with an extra `share` of each of `days`' turns that read back at least half
    reading only what their transcript's opening turn read, at most what was cached, and
    writing the rest again, the per-turn columns recomputed by add_ratios."""
    planted = loops.copy()
    rng = np.random.default_rng(seed)
    for day in days:
        hits = planted.index[(planted["day"].astype(str) == day) & ~planted["is_loop_miss"].astype(bool)]
        count = int(round(share * int((planted["day"].astype(str) == day).sum())))
        prefix_reads(planted, opening, rng.choice(hits.to_numpy(), size=min(count, len(hits)), replace=False))
    return add_ratios(planted)


def prefix_reads(planted: pd.DataFrame, opening: pd.Series, chosen: np.ndarray) -> None:
    """The `chosen` turns of `planted` read only what their transcript's opening turn read,
    at most what was cached, and write the rest again."""
    before = planted.loc[chosen, "cache_read"]
    read = np.minimum(planted.loc[chosen, "source_file"].map(opening).fillna(0.0).to_numpy(),
                      planted.loc[chosen, "prev_cached"].to_numpy())
    planted.loc[chosen, "cache_read"] = read
    planted.loc[chosen, "cache_creation"] = planted.loc[chosen, "cache_creation"] + before - read


def plant_starts(days: list[str], starts: int = STARTS, avoid: Sequence[tuple[str, Optional[str]]] = (),
                 within: Optional[int] = None) -> list[str]:
    """Up to `starts` days, evenly spread, with BEFORE judged days before and AFTER after,
    and none whose AFTER days after it meet a span of `avoid` or, given `within`, end that
    many calendar days or more after it."""
    eligible = [day for i, day in enumerate(days)
                if BEFORE <= i < len(days) - AFTER and not meets(day, days[i + AFTER], avoid)
                and (within is None or (date.fromisoformat(days[i + AFTER]) - date.fromisoformat(day)).days < within)]
    if len(eligible) <= starts:
        return eligible
    return [eligible[int(i)] for i in np.linspace(0, len(eligible) - 1, starts).round()]


def spans_of(incidents: Sequence[dict], metric: str, statuses: Sequence[str]) -> list[tuple[str, Optional[str]]]:
    """The spans of `metric`'s incidents with one of `statuses`, in order: (start, end), with
    no end while one is open."""
    return sorted((i["start"], i["end"]) for i in incidents if i["metric"] == metric and i["status"] in statuses)


def left_out(incidents: Sequence[dict], metric: str) -> list[tuple[str, Optional[str]]]:
    """The spans whose days are left out of the baselines: `metric`'s open and recovered
    incidents, as the check leaves them out."""
    return spans_of(incidents, metric, EXCLUDING)


def meets(first: str, last: str, spans: Sequence[tuple[str, Optional[str]]]) -> bool:
    """Whether the days `first` to `last` meet a span of `spans`."""
    return any(start <= last and first <= (end or OPEN_END) for start, end in spans)


def _judged(frame: pd.DataFrame, metric: str, incidents: Sequence[dict]) -> tuple[list[str], np.ndarray]:
    """The days the shipped detector judges `metric` on in `frame`, with the days of open and
    recovered incidents out of the baselines as the check leaves them, and which it flags."""
    binned = bin_metrics(frame, metrics=SUBAGENT_METRICS) if metric in SUBAGENT_METRICS else bin_metrics(frame)
    if binned.empty:
        return [], np.zeros(0, dtype=bool)
    days = binned["bin"].astype(str).tolist()
    detected = detect(binned, DetectorConfig(), exclusions(days, incidents), only=[metric])
    return days, detected[f"{metric}__flag"].to_numpy(dtype=bool)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """The first and last index of each run of flagged days."""
    runs = []
    for i in range(len(flags)):
        if flags[i] and (i == 0 or not flags[i - 1]):
            end = i
            while end + 1 < len(flags) and flags[end + 1]:
                end += 1
            runs.append((i, end))
    return runs


def _reported(days: list[str], flags: np.ndarray, clean: np.ndarray, spans: Sequence[tuple[str, Optional[str]]],
              at: int) -> bool:
    """Whether a planted history's `flags` hold a run the check would report, meeting no span of
    `spans` (every incident of the metric), with a day from index `at` on that the unplanted
    `clean` doesn't flag."""
    return any(not meets(days[first], days[last], spans) and any(not clean[i] for i in range(max(first, at), last + 1))
               for first, last in _runs(flags))


def _until(frame: pd.DataFrame, day: str) -> pd.DataFrame:
    """The rows of `frame` up to `day`: the history a check the day after it reads."""
    return frame[frame["day"].astype(str) <= day]


def reach_main(frame: pd.DataFrame, metric: str, days: list[str], incidents: Sequence[dict]) -> dict[str, Any]:
    """Plant KINDS[metric] at each of SIZES from each start's first turn on, as the harness
    sweep does, over the history up to the AFTER-th day after the start: caught counts by size
    and the floor per start."""
    spans = spans_of(incidents, metric, STATUSES)
    base = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    first_turn = base.reset_index().groupby(base["day"].astype(str))["index"].min()
    starts = plant_starts(days, avoid=spans, within=RECENT_DAYS)
    caught = {size: 0 for size in SIZES}
    floors: dict[str, Optional[float]] = {}
    for start in starts:
        at = days.index(start)
        floors[start] = None
        # A prefix in time order, so first_turn's indexes hold in it.
        seen = _until(base, days[at + AFTER])
        unplanted = _judged(seen, metric, incidents)[1]
        for size in SIZES:
            planted = plant(seen, KINDS[metric], size, seed=HAIKU_SEED, start=int(first_turn[start]))
            if _reported(days, _judged(planted, metric, incidents)[1], unplanted, spans, at):
                caught[size] += 1
                floors[start] = size if floors[start] is None else floors[start]
    return {"starts": starts, "runs": len(starts), "caught": caught, "floors": floors}


def reach_subagent(frame: pd.DataFrame, opening: pd.Series, days: list[str],
                   incidents: Sequence[dict]) -> dict[str, Any]:
    """Plant G17's prefix reads at each of SUBAGENT_SIZES over each start and the AFTER days
    after it, with SEEDS seeds, as G17 does: caught counts by size over every run, and the
    floor per start, the smallest size every seed caught there."""
    spans = spans_of(incidents, "subagent_cache", STATUSES)
    starts = plant_starts(days, avoid=spans, within=RECENT_DAYS)
    caught = {size: 0 for size in SUBAGENT_SIZES}
    floors: dict[str, Optional[float]] = {start: None for start in starts}
    for start in starts:
        at = days.index(start)
        window = days[at:at + AFTER + 1]
        seen = _until(frame, window[-1])
        unplanted = _judged(seen, "subagent_cache", incidents)[1]
        for size in SUBAGENT_SIZES:
            hits = 0
            for seed in range(SEEDS):
                planted = plant_subagent(seen, opening, window, size, seed + 1000 * at)
                hits += _reported(days, _judged(planted, "subagent_cache", incidents)[1], unplanted, spans, at)
            caught[size] += hits
            if hits == SEEDS and floors[start] is None:
                floors[start] = size
    return {"starts": starts, "runs": len(starts) * SEEDS, "caught": caught, "floors": floors}


def measure(frame: pd.DataFrame, metric: str, incidents: Sequence[dict],
            opening: Optional[pd.Series] = None) -> dict[str, Any]:
    """One metric's calibration on `frame`, the turns the check judges it on: the days judged,
    the first day of each false alarm and of those the user dismissed, the reach, and the
    verdict ("pass", "fail" or "not_enough_data" when no start fits) with the halves that
    failed."""
    days, clean = _judged(frame, metric, incidents)
    followed = spans_of(incidents, metric, FOLLOWED)
    alarms = [(days[first], days[last]) for first, last in _runs(clean)
              if not meets(days[first], days[last], followed)]
    flagged = [first for first, _ in alarms]
    dismissed = [first for first, last in alarms if meets(first, last, spans_of(incidents, metric, ("dismissed",)))]
    if metric in SUBAGENT_METRICS:
        opening = opening if opening is not None else pd.Series(dtype=float)
        reach = reach_subagent(frame, opening, days, incidents)
        reached = reach["caught"][BAR] >= SUBAGENT_SHARE * reach["runs"]
    else:
        reach = reach_main(frame, metric, days, incidents)
        reached = reach["caught"][BAR] == reach["runs"]
    failed = (["quiet"] if flagged else []) + (["reach"] if reach["runs"] and not reached else [])
    verdict = "not_enough_data" if not reach["runs"] else "fail" if failed else "pass"
    return {"metric": metric, "verdict": verdict, "failed": failed, "days": len(days),
            "first": days[0] if days else None, "last": days[-1] if days else None,
            "left_out": left_out(incidents, metric), "settled": spans_of(incidents, metric, ("persistent",)),
            "flagged": flagged, "dismissed": dismissed, **reach}


# The week the check's cache-miss warning runs over: on fewer judged days a rate a week rests on
# a fraction of one.
MIN_WARNING_DAYS = 7


def _median(runs: Sequence[Optional[int]]) -> Optional[float]:
    """The median turns to a catch, a run never caught counting as endless; None when the
    median is past every catch."""
    median = statistics.median(math.inf if run is None else run for run in runs)
    return None if math.isinf(median) else median


def _halves(quiet: bool, reach: bool) -> dict[str, Any]:
    failed = [half for half, held in (("quiet", quiet), ("reach", reach)) if not held]
    return {"verdict": "fail" if failed else "pass", "failed": failed}


def cache_miss_warning(responses: pd.DataFrame, incidents: Sequence[dict], today: date) -> dict[str, Any]:
    """The cache-miss warning (G3) judged on the new-prompt turns of the complete days
    before `today`, the days of every cache incident not dismissed left out as the check
    leaves them. Quiet: the false warnings a week to expect, each judged day's turns times
    the simulated alarms per turn at its usual rate, over the calendar weeks judged, at most
    MAX_FALSE_PER_WEEK; the days the history itself alarms are reported beside it. Reach: a
    planted miss rate caught within a median of PROMPT_MAX_TURNS turns."""
    h = early.THRESHOLD
    if h is None:
        return {"warning": "cache_miss", "setting": None, "verdict": "none_ships"}
    setting = {"h": h, "p1": early.P1}
    turns = prompt_turns(responses)
    turns = turns[turns["day"] < today.isoformat()].reset_index(drop=True)
    spans = [(i["start"], i["end"] or today.isoformat()) for i in incidents
             if i["metric"] == "cache_ratio" and i["status"] != "dismissed"]
    counts = turns.loc[~within(turns["day"], spans), "day"].value_counts()
    usual = {day: prompt_base_rate(turns, shift(day, -(early.WINDOW_DAYS - 1)), spans) for day in sorted(counts.index)}
    judged = {day: rate for day, rate in usual.items() if rate is not None}
    starts, runs = prompt_plants(turns, spans, h)
    if len(judged) < MIN_WARNING_DAYS or not runs:
        return {"warning": "cache_miss", "setting": setting, "verdict": "not_enough_data", "failed": [],
                "days": len(judged)}
    per_turn = functools.lru_cache(maxsize=None)(functools.partial(alarms_per_turn, h))
    # The weeks the warning was on: a day with new prompts it doesn't judge, a cache incident's or
    # one without a usual rate, is a day it was off, and leaves; a day without any stays, quiet.
    first, last = min(judged), max(judged)
    off = {day for day in turns["day"].unique() if first <= day <= last and day not in judged}
    weeks = ((date.fromisoformat(last) - date.fromisoformat(first)).days + 1 - len(off)) / 7
    false_a_week = sum(per_turn(rate) * counts[day] for day, rate in judged.items()) / weeks
    median = _median(runs)
    return {"warning": "cache_miss", "setting": setting,
            **_halves(false_a_week <= MAX_FALSE_PER_WEEK, median is not None and median <= PROMPT_MAX_TURNS),
            "days": len(judged), "weeks": round(weeks, 1), "turns_a_week": int(round(sum(counts[day] for day in judged) / weeks)),
            "usual_rate": round(statistics.median(judged.values()), 4), "false_a_week": round(float(false_a_week), 3),
            "bar_false_a_week": round(MAX_FALSE_PER_WEEK, 4), "alarms": sorted(set(prompt_alarm_days(turns, spans, h))),
            "starts": starts, "runs": len(runs), "caught": sum(run is not None for run in runs), "median_turns": median,
            "bar_turns": PROMPT_MAX_TURNS}


def tool_loop_warning(responses: pd.DataFrame, stream: str, incidents: Sequence[dict], today: date) -> dict[str, Any]:
    """The tool-loop warning of `stream` (G8 on the main thread) judged on its tool-loop
    turns of the complete days before `today`. Quiet: no qualifying alarm at the most
    sensitive usual rate on a judged day outside the cache incidents ccdrift follows, one in a
    dismissed incident counting. Reach: a planted miss rate caught in CAUGHT_SHARE of runs
    within a median of LOOP_MAX_TURNS turns. None in LOOP_SETTINGS means no setting ships."""
    setting = LOOP_SETTINGS[stream]
    if setting is None:
        return {"warning": "tool_loop", "stream": stream, "setting": None, "verdict": "none_ships"}
    named = {"p1": setting.p1, "h": setting.h, "min_sessions": setting.min_sessions}
    turns = loop_turns(responses, stream)
    turns = turns[turns["day"] < today.isoformat()].reset_index(drop=True)
    usual = functools.lru_cache(maxsize=None)(functools.partial(loop_base_rate, turns))
    followed = spans_of(incidents, "cache_ratio", FOLLOWED)
    judged, alarms = loop_alarm_days(turns, usual, setting)
    judged = [day for day in judged if not meets(day, day, followed)]
    alarms = sorted({day for day in alarms if not meets(day, day, followed)})
    starts, stretches = loop_plants(turns, usual)
    if not judged or not stretches:
        return {"warning": "tool_loop", "stream": stream, "setting": named, "verdict": "not_enough_data",
                "failed": [], "days": 0}
    runs = [turns_to_catch(stretch, p0, setting) for stretch, p0 in stretches]
    caught = sum(run is not None for run in runs)
    median = _median(runs)
    dismissed = spans_of(incidents, "cache_ratio", ("dismissed",))
    return {"warning": "tool_loop", "stream": stream, "setting": named,
            **_halves(not alarms, caught >= CAUGHT_SHARE * len(runs) and median is not None and median <= LOOP_MAX_TURNS),
            "days": len(judged), "alarms": alarms, "dismissed": [day for day in alarms if meets(day, day, dismissed)],
            "starts": starts, "runs": len(runs), "caught": caught, "median_turns": median, "bar_share": CAUGHT_SHARE,
            "bar_turns": LOOP_MAX_TURNS}


def calibration(responses: pd.DataFrame, incidents: Sequence[dict], today: date) -> dict[str, Any]:
    """Every metric's and early warning's calibration on `responses` as the check on `today`
    would judge them, with the span of days judged and how many on the main thread and in
    subagent loops."""
    turns = judged_turns(responses, today)
    loops = judged_subagent_loops(responses, today)
    opening = subagent_openings(responses) if not loops.empty else pd.Series(dtype=float)
    results = [measure(turns, metric, incidents) for metric in KINDS]
    results.append(measure(loops, "subagent_cache", incidents, opening))
    days = sorted(set(turns["day"].astype(str)) | set(loops["day"].astype(str)))
    return {"first": days[0] if days else None, "last": days[-1] if days else None,
            "main_days": int(turns["day"].nunique()), "subagent_days": int(loops["day"].nunique()),
            "metrics": results,
            "warnings": [cache_miss_warning(responses, incidents, today),
                         *(tool_loop_warning(responses, stream, incidents, today) for stream in STREAMS)]}


def metric_lines(result: dict[str, Any]) -> list[str]:
    """One metric's verdict and what it rests on, as the terminal shows them."""
    metric = result["metric"]
    name, move = CALIBRATE_NAMES[metric], CALIBRATE_MOVES[MOVES[metric]]
    bar = planted_text(BAR)
    if result["verdict"] == "not_enough_data":
        need = BEFORE + AFTER + 1
        lines = [CALIBRATE_LINES["not_enough_data" if result["days"] < need else "no_clear_start"].format(
            name=name, days=result["days"], need=need, before=BEFORE, after=AFTER, within=RECENT_DAYS)]
    elif result["verdict"] == "pass":
        lines = [CALIBRATE_LINES["pass"].format(name=name)]
    else:
        why = {("quiet",): "why_quiet", ("reach",): "why_reach"}.get(tuple(result["failed"]), "why_both")
        lines = [CALIBRATE_LINES["fail"].format(name=name, why=CALIBRATE_LINES[why].format(bar=bar, move=move))]
    if result["runs"] and metric == "subagent_cache":
        lines.append(CALIBRATE_LINES["reach_subagent"].format(
            bar=bar, caught=result["caught"][BAR], runs=result["runs"], share=result["caught"][BAR] / result["runs"],
            needed=SUBAGENT_SHARE, small=planted_text(SUBAGENT_SIZES[0]), small_caught=result["caught"][SUBAGENT_SIZES[0]]))
    elif result["runs"]:
        starts = result["starts"]
        lines.append(CALIBRATE_LINES["reach" if len(starts) > 1 else "reach_one"].format(
            bar=bar, move=move, caught=result["caught"][BAR], runs=result["runs"], first=starts[0], last=starts[-1]))
        lines.append(CALIBRATE_LINES["floors"].format(move=move, floors=floors_text(result["floors"], SIZES[-1])))
    unlabelled = [day for day in result["flagged"] if day not in result["dismissed"]]
    if unlabelled:
        lines.append(CALIBRATE_LINES["flagged"].format(days=", ".join(unlabelled)))
    if result["dismissed"]:
        lines.append(CALIBRATE_LINES["dismissed"].format(days=", ".join(result["dismissed"])))
    if not result["flagged"]:
        lines.append(CALIBRATE_LINES["quiet"])
    if result["left_out"]:
        lines.append(CALIBRATE_LINES["left_out"].format(spans=spans_text(result["left_out"])))
    if result["settled"]:
        lines.append(CALIBRATE_LINES["settled"].format(spans=spans_text(result["settled"])))
    return lines


def _starts_text(starts: Sequence[str]) -> tuple[str, str]:
    """How many starts, "10 starts", and over which days, "2026-08-27 to 2026-10-07"."""
    count = CALIBRATE_LINES["starts_one" if len(starts) == 1 else "starts_many"].format(count=len(starts))
    return count, starts[0] if len(starts) == 1 else CALIBRATE_LINES["span"].format(start=starts[0], end=starts[-1])


def warning_lines(result: dict[str, Any]) -> list[str]:
    """One early warning's verdict and what it rests on, as the terminal shows them."""
    key = result.get("stream", result["warning"])
    name, cache = WARNING_NAMES[key], result["warning"] == "cache_miss"
    kind = "cache_" if cache else "loop_"
    if result["verdict"] == "none_ships":
        return [WARNING_LINES["none_ships"].format(name=name, watches=WARNING_WATCHES[key])]
    if result["verdict"] == "not_enough_data" and cache and 0 < result["days"] < MIN_WARNING_DAYS:
        return [WARNING_LINES["cache_few_days"].format(name=name, days=result["days"], need=MIN_WARNING_DAYS)]
    if result["verdict"] == "not_enough_data":
        return [WARNING_LINES[kind + "not_enough_data"].format(
            name=name, turns=early.MIN_BASE_TURNS if cache else loops.MIN_BASE_TURNS, days=early.BASE_DAYS)]
    plant, bar = planted_text(PROMPT_PLANT if cache else LOOP_PLANT), PROMPT_MAX_TURNS if cache else LOOP_MAX_TURNS
    why = WARNING_LINES["and"].join(WARNING_LINES[kind + "why_" + half].format(
        rate=result.get("false_a_week"), plant=plant, turns=bar, share=CAUGHT_SHARE) for half in result["failed"])
    lines = [WARNING_LINES["fail"].format(name=name, why=why) if result["failed"]
             else WARNING_LINES["pass"].format(name=name)]
    starts, span = _starts_text(result["starts"])
    median = result["median_turns"]
    if cache:
        lines.append(WARNING_LINES["cache_quiet"].format(
            rate=result["false_a_week"], usual=result["usual_rate"], turns=result["turns_a_week"],
            weeks=result["weeks"], every=round(1 / MAX_FALSE_PER_WEEK), bar=MAX_FALSE_PER_WEEK))
        lines.append(WARNING_LINES["cache_alarmed"].format(days=", ".join(result["alarms"])) if result["alarms"]
                     else WARNING_LINES["cache_alarmed_none"])
        lines.append(WARNING_LINES["cache_reach" if median is not None else "cache_reach_never"].format(
            plant=plant, median=median, caught=result["caught"], runs=result["runs"], starts=starts, span=span,
            bar=bar))
        return lines
    unlabelled = [day for day in result["alarms"] if day not in result["dismissed"]]
    if unlabelled:
        lines.append(WARNING_LINES["loop_alarms"].format(days=", ".join(unlabelled), floor=MIN_P0))
    if result["dismissed"]:
        lines.append(WARNING_LINES["loop_dismissed"].format(days=", ".join(result["dismissed"])))
    if not result["alarms"]:
        lines.append(WARNING_LINES["loop_quiet"].format(days=result["days"], floor=MIN_P0))
    lines.append(WARNING_LINES["loop_reach" if median is not None else "loop_reach_never"].format(
        plant=plant, caught=result["caught"], runs=result["runs"], share=result["caught"] / result["runs"],
        median=median, bar_share=CAUGHT_SHARE, bar=bar))
    return lines


def calibrate_lines(result: dict[str, Any]) -> list[str]:
    """The calibration as the terminal shows it: what was judged, each metric's verdict, what
    isn't judged yet and what PASS means."""
    head = (CALIBRATE_LINES["head"].format(first=result["first"], last=result["last"], main=result["main_days"],
                                           subagent=result["subagent_days"]) if result["first"]
            else CALIBRATE_LINES["head_empty"])
    lines = [head, ""]
    for metric in result["metrics"]:
        lines += metric_lines(metric)
    lines += ["", WARNING_LINES["head"]]
    for warning in result["warnings"]:
        lines += warning_lines(warning)
    return lines + ["", CALIBRATE_LINES["not_judged"],
                    CALIBRATE_LINES["bar"].format(bar=planted_text(BAR), needed=SUBAGENT_SHARE),
                    WARNING_LINES["bar"].format(every=round(1 / MAX_FALSE_PER_WEEK), prompt_plant=planted_text(PROMPT_PLANT),
                                                prompt_turns=PROMPT_MAX_TURNS, loop_plant=planted_text(LOOP_PLANT),
                                                share=CAUGHT_SHARE, loop_turns=LOOP_MAX_TURNS)]


def calibrate_json(result: dict[str, Any]) -> str:
    """The calibration as JSON: days, counts and sizes, the bar it was judged against and the
    alerts not judged yet. No path, session or project."""
    metrics = [{**metric, "caught": {str(size): count for size, count in metric["caught"].items()},
                "left_out": [list(span) for span in metric["left_out"]]} for metric in result["metrics"]]
    shared = {**result, "metrics": metrics, "bar": BAR, "subagent_share": SUBAGENT_SHARE,
              "not_judged": list(NOT_JUDGED)}
    return json.dumps(shared, indent=1) + "\n"


def run_calibrate(source: Path, state_path: Path, as_json: bool = False, today: Optional[date] = None) -> int:
    """Print whether the daily detector's shipped settings work on the history under `source`
    that the check reads, with the incidents in the state at `state_path` left out. Changes no state and sends no
    alert; like `cost`, it brings a store a check has claimed up to date first."""
    today = today or datetime.now(timezone.utc).date()
    try:
        incidents = load_state(state_path)["incidents"]
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        tables = load_history(source, state_path, claim=False, since=history_start(incidents, today),
                              active_days=HISTORY_ACTIVE_DAYS, active_responses=ACTIVE_DAY_RESPONSES)
    except HistoryError as exc:
        print(exc, file=sys.stderr)
        return 1
    if tables.responses.empty:
        print(no_transcripts_message(source), file=sys.stderr)
        return 2
    # 12.7 s on the owner's store of 51 days (2026-10-14), so it says so first.
    print(CALIBRATE_LINES["measuring"], file=sys.stderr, flush=True)
    result = calibration(tables.responses, incidents, today)
    print(calibrate_json(result) if as_json else "\n".join(calibrate_lines(result)) + "\n", end="")
    return 0
