"""Does a drop in what subagent tool loops read back get caught? (G17)

Subagents carry most of the owner's spend and nothing judged their caching: the daily
verdict reads main-thread prompt turns, and the subagent tool-loop warning ships no setting
since G9 failed. The subagent cache metric is the mean over a UTC day of what each subagent
tool-loop turn reads back of what the response before it in its transcript had cached
(logs.add_ratios' `loop_readback`), judged by the daily detector like the main-thread cache
ratio, with a z cutoff of its own. This module measures that cutoff on the owner's logs.

The August caching regression (Claude Code 2.1.233-2.1.258) never reached subagents: their
tool-loop misses stayed at 0-0.9% a day while main-thread prompt turns missed 4.3%. So the
logs hold no real subagent regression, and every flag on them is a false alarm. Nothing is
left out. Regressions are planted instead: from each of up to STARTS start days, each with
BEFORE judged days before it and AFTER after, for SEEDS seeds each, an extra share of each
day's loop turns from the start through AFTER days later reads only what its transcript's
opening turn read (the system prompt and tools, the shape of the August misses) and writes
the rest again. The planted turns go back through the shipped add_ratios, bin_metrics and
detect. A plant is caught when a day it covers is flagged that the logs alone don't flag.

A cutoff passes with no false alarm and a planted BAR caught in CAUGHT_SHARE of runs, and
the gate needs the cutoff ccdrift ships to pass along with every stricter one in the grid.
Of the passing cutoffs ccdrift ships the strictest that catches every planted SMALL: on
these logs every cutoff tried passed, and the lower ones only left less margin on real days.
The loop-miss rate (as a hit rate) and each turn's share of its input read from the cache
are measured the same way beside it, for comparison only. The last lines are the verdict
on the shipped cutoff and the flags a first check would open with it: those starting
within the last incidents.RECENT_DAYS days. Output is aggregate: days, counts and z scores.

Run from the repo root:

  uv run --group lab python -m lab.subagent_cache
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ccdrift.detector import MIN_BIN_TURNS, SUBAGENT_METRICS, DetectorConfig, bin_metrics, detect, flag_onsets
from ccdrift.incidents import RECENT_DAYS
from ccdrift.logs import add_ratios, default_source, judged_subagent_loops, parse_all

Z_GRID = (2.5, 3.0, 3.5, 4.0, 4.5, 5.0)
BAR = 0.05             # the August main-thread regression missed about 3.5 points more than usual
SMALL = 0.02
PLANT_GRID = (SMALL, BAR)
MIN_TURNS_GRID = (0, 100, 200, 300, 500)
SHIFT_SHARE = 0.5      # one more ordinary miss may move a judged day at most this share of the way to the cutoff
CAUGHT_SHARE = 0.9
STARTS = 10
SEEDS = 5
BEFORE = 7             # judged days before a start: the detector needs 5 to judge a day
AFTER = 3              # judged days after it: a flag needs 3 deviant days of 4
METRIC = "subagent_cache"
COLUMN = SUBAGENT_METRICS[METRIC][0]
# Each measured the same way: a per-turn value in [0, 1] whose drop is the harm.
VARIANTS = {
    "readback": lambda df: df["loop_readback"],
    "hit_rate": lambda df: 1.0 - df["is_loop_miss"].astype(float),
    "read_share": lambda df: df["cache_read"] / (df["cache_read"] + df["cache_creation"]
                                                 + df["input_tokens"]).clip(lower=1),
}
CANDIDATE = "readback"


def openings(responses: pd.DataFrame) -> pd.Series:
    """What each subagent transcript's first response read from the cache, by source file."""
    sub = responses[responses["is_sidechain"].astype(bool)].sort_values("timestamp", kind="stable")
    return sub.groupby("source_file", sort=True)["cache_read"].first()


def plant(loops: pd.DataFrame, opening: pd.Series, days: list[str], share: float, seed: int) -> pd.DataFrame:
    """`loops` with an extra `share` of each of `days`' turns that read back at least half
    reading only what their transcript's opening turn read, at most what was cached, and
    writing the rest again, the per-turn columns recomputed by add_ratios."""
    planted = loops.copy()
    rng = np.random.default_rng(seed)
    for day in days:
        hits = planted.index[(planted["day"].astype(str) == day) & ~planted["is_loop_miss"].astype(bool)]
        count = int(round(share * int((planted["day"].astype(str) == day).sum())))
        _prefix_reads(planted, opening, rng.choice(hits.to_numpy(), size=min(count, len(hits)), replace=False))
    return add_ratios(planted)


def _prefix_reads(planted: pd.DataFrame, opening: pd.Series, chosen: np.ndarray) -> None:
    """The `chosen` turns of `planted` read only what their transcript's opening turn read,
    at most what was cached, and write the rest again."""
    before = planted.loc[chosen, "cache_read"]
    read = np.minimum(planted.loc[chosen, "source_file"].map(opening).fillna(0.0).to_numpy(),
                      planted.loc[chosen, "prev_cached"].to_numpy())
    planted.loc[chosen, "cache_read"] = read
    planted.loc[chosen, "cache_creation"] = planted.loc[chosen, "cache_creation"] + before - read


def one_miss(loops: pd.DataFrame, opening: pd.Series, day: str) -> pd.DataFrame:
    """`loops` with one more ordinary miss on `day`: the middle one of its turns that read
    back at least half reads only what its transcript's opening turn read."""
    missed = loops.copy()
    hits = missed.index[(missed["day"].astype(str) == day) & ~missed["is_loop_miss"].astype(bool)]
    _prefix_reads(missed, opening, hits.to_numpy()[len(hits) // 2:len(hits) // 2 + 1])
    return add_ratios(missed)


def flagged(loops: pd.DataFrame, variant: str, z: float,
            min_turns: int = MIN_BIN_TURNS[METRIC]) -> tuple[list[str], list[str], pd.DataFrame]:
    """The days the detector flags and the days it scores past `z` when `variant` stands in
    for the metric's column, judging days with `min_turns` loop turns or more, with the
    detected bins."""
    frame = loops[["day"]].copy()
    frame[COLUMN] = VARIANTS[variant](loops).astype(float)
    detected = detect(bin_metrics(frame, metrics=SUBAGENT_METRICS, min_turns={METRIC: min_turns}),
                      DetectorConfig(metric_z_thresholds={METRIC: z}), only=[METRIC])
    bins = detected["bin"].astype(str)
    return (bins[detected[f"{METRIC}__flag"]].tolist(), bins[detected[f"{METRIC}__z"] <= -z].tolist(), detected)


def plant_starts(days: list[str], starts: int = STARTS) -> list[str]:
    """Up to `starts` days, evenly spread, with BEFORE judged days before and AFTER after."""
    eligible = days[BEFORE:len(days) - AFTER]
    if len(eligible) <= starts:
        return eligible
    return [eligible[int(i)] for i in np.linspace(0, len(eligible) - 1, starts).round()]


def rows(loops: pd.DataFrame, opening: pd.Series, z_grid=Z_GRID, plants=PLANT_GRID, seeds: int = SEEDS,
         starts: int = STARTS) -> list[dict[str, Any]]:
    """One row per variant and cutoff: the days it flags on the logs (false alarms), the days
    it scores past the cutoff there, and how many runs of each planted share it catches."""
    days = sorted(loops["day"].astype(str).unique())
    firsts = plant_starts(days, starts)
    runs = {(share, first, seed): plant(loops, opening, days[days.index(first):days.index(first) + AFTER + 1], share,
                                        seed + 1000 * days.index(first))
            for share in plants for first in firsts for seed in range(seeds)}
    out = []
    for variant in VARIANTS:
        for z in z_grid:
            alarms, deviant, _ = flagged(loops, variant, z)
            caught = {share: 0 for share in plants}
            for (share, first, _), planted in runs.items():
                window = days[days.index(first):days.index(first) + AFTER + 1]
                new = set(flagged(planted, variant, z)[0]) - set(alarms)
                caught[share] += bool(new.intersection(window))
            total = len(firsts) * seeds
            out.append({"variant": variant, "z": z, "alarms": alarms, "deviant": deviant, "caught": caught,
                        "runs": total,
                        "passes": not alarms and total > 0 and caught.get(BAR, 0) >= CAUGHT_SHARE * total})
    return out


def shifts(loops: pd.DataFrame, opening: pd.Series, z: float, min_turns: int) -> list[tuple[str, int, float]]:
    """Each day judged with a minimum of `min_turns` loop turns, with its turns and how far
    one more ordinary miss (one_miss) moves its z toward the cutoff."""
    before = flagged(loops, CANDIDATE, z, min_turns)[2].set_index("bin")
    moved = []
    for day, row in before.iterrows():
        score = row[f"{METRIC}__z"]
        if np.isnan(score):
            continue
        after = flagged(one_miss(loops, opening, str(day)), CANDIDATE, z, min_turns)[2].set_index("bin")
        moved.append((str(day), int(row[f"{METRIC}__n"]), float(score - after.loc[day, f"{METRIC}__z"])))
    return moved


def light_rows(loops: pd.DataFrame, opening: pd.Series, z: float, grid=MIN_TURNS_GRID) -> list[dict[str, Any]]:
    """One row per minimum of loop turns a day: the days it leaves unjudged, the judged day
    one more ordinary miss moves furthest (day, turns, shift), and whether that stays under
    SHIFT_SHARE of the cutoff `z`."""
    turns = loops.groupby(loops["day"].astype(str)).size()
    out = []
    for minimum in grid:
        largest = max(shifts(loops, opening, z, minimum), key=lambda moved: moved[2], default=None)
        out.append({"minimum": minimum, "unjudged": int((turns < minimum).sum()), "days": len(turns),
                    "largest": largest, "passes": largest is None or largest[2] < SHIFT_SHARE * z})
    return out


def gate(table: list[dict[str, Any]], shipped: Optional[float], light: Optional[list[dict[str, Any]]] = None,
         minimum: Optional[int] = None) -> tuple[bool, list[str]]:
    """Whether the cutoff ccdrift ships passes with every stricter one in the grid and, given
    the `light` rows, whether the minimum of loop turns a day it ships passes too, what they
    did, and the settings that pass when either fails."""
    candidate = [row for row in table if row["variant"] == CANDIDATE]
    passing = [row["z"] for row in candidate if row["passes"]]
    if shipped is None:
        return False, ["ccdrift ships no cutoff", f"passing cutoffs: {', '.join(f'{z:g}' for z in passing) or 'none'}"]
    row = next((r for r in candidate if r["z"] == shipped), None)
    if row is None:
        return False, [f"the shipped cutoff {shipped:g} isn't in the grid"]
    stricter = [r for r in candidate if r["z"] >= shipped]
    notes = [f"ships z={shipped:g}: {len(row['alarms'])} false alarm(s), planted {BAR:g} caught "
             f"{row['caught'].get(BAR, 0)} of {row['runs']}"]
    ok = all(r["passes"] for r in stricter)
    full = [r["z"] for r in candidate if r["passes"] and r["caught"].get(SMALL, 0) == r["runs"]]
    notes.append(f"strictest cutoff catching every planted {SMALL:g}: {max(full):g}" if full
                 else f"no passing cutoff catches every planted {SMALL:g}")
    if not ok:
        notes.append(f"passing cutoffs: {', '.join(f'{z:g}' for z in passing) or 'none'}")
    if light is None:
        return ok, notes
    kept = next((r for r in light if r["minimum"] == minimum), None)
    if kept is None:
        return False, notes + [f"the shipped minimum {minimum} isn't in the grid"]
    moved = f"{kept['largest'][2]:.2f}" if kept["largest"] else "nothing"
    notes.append(f"ships a minimum of {minimum} loop turns a day: {kept['unjudged']} of {kept['days']} days not "
                 f"judged, one more miss moves a judged day at most {moved} z (bar {SHIFT_SHARE * shipped:.2f})")
    if not kept["passes"]:
        notes.append(f"passing minimums: {', '.join(str(r['minimum']) for r in light if r['passes']) or 'none'}")
    return ok and kept["passes"], notes


def first_check(loops: pd.DataFrame, z: float, today: date) -> list[str]:
    """The first day of each flag run at `z` starting within the last RECENT_DAYS days before
    `today`: what a first check on `today` would open, with no subagent incident recorded."""
    _, _, detected = flagged(loops, CANDIDATE, z)
    since = (today - timedelta(days=RECENT_DAYS)).isoformat()
    bins = detected["bin"].astype(str).tolist()
    return [bins[i] for i in flag_onsets(detected, METRIC) if bins[i] >= since]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    args = parser.parse_args(argv)

    responses = parse_all(args.source).responses
    loops = judged_subagent_loops(responses, args.today).reset_index(drop=True)
    if loops.empty:
        print("no subagent tool-loop turns")
        return 1
    days = sorted(loops["day"].astype(str).unique())
    print(f"{len(loops)} subagent tool-loop turns on {len(days)} days, {days[0]}..{days[-1]}, "
          f"{int(loops['is_loop_miss'].sum())} misses")
    _, _, detected = flagged(loops, CANDIDATE, 3.0)
    for row in detected.itertuples(index=False):
        z = getattr(row, f"{METRIC}__z")
        print(f"  {row.bin} {getattr(row, f'{METRIC}__n'):>5} turns, read-back {getattr(row, METRIC):.4f}"
              + ("" if np.isnan(z) else f", z {z:+.2f}"))
    table = rows(loops, openings(responses))
    for row in table:
        print(f"G17 {row['variant']:<10} z={row['z']:<4g} false-alarms={len(row['alarms'])} "
              + " ".join(f"{share:g}={n}/{row['runs']}" for share, n in row["caught"].items())
              + f" deviant-days={len(row['deviant'])}"
              + (f"  alarms: {', '.join(row['alarms'])}" if row["alarms"] else "")
              + ("  PASS" if row["passes"] else ""))
    shipped = DetectorConfig().metric_z_thresholds.get(METRIC)
    light = light_rows(loops, openings(responses), shipped) if shipped is not None else None
    for row in light or []:
        largest = row["largest"]
        print(f"G17 minimum={row['minimum']:<4} unjudged-days={row['unjudged']}/{row['days']} "
              + (f"largest-one-miss-shift={largest[2]:.2f} on {largest[0]} ({largest[1]} turns)" if largest
                 else "no judged day")
              + ("  PASS" if row["passes"] else ""))
    ok, notes = gate(table, shipped, light, MIN_BIN_TURNS[METRIC])
    print(f"G17: {'PASS' if ok else 'FAIL'}")
    for note in notes:
        print(f"  {note}")
    if shipped is not None:
        opened = first_check(loops, shipped, args.today)
        print(f"  a first check on {args.today} opens: {', '.join(opened) if opened else 'nothing'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
