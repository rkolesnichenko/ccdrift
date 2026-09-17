"""Do the failure rules stay quiet on a real history and still catch a burst? (G10, G11)

Failed requests and responses cut short are rare — 13 banners and 2 truncated responses
in the six weeks this was designed on — so neither rule can lean on a usual rate. Each
day is judged against the days before it, and the question is whether the floors are
high enough to stay quiet and low enough to catch a bad release.

Both gates replay the rules day by day over the logs, as the daily check would have run
them, with one state carried along, and then once more for each of the last PLANT_DAYS
judgeable days, over a copy of the counts with failures planted in that one day. Planting
only on the corpus's last day made a verdict turn on how busy that day happened to be: a
quiet Sunday at the end of the logs flipped a PASS to a FAIL with no change to the rule.

- G10 (requests failing) passes when the settings alert at most ALERT_BUDGET times per
  BUDGET_DAYS days of history, and alert on every day planted with PLANTED_REQUESTS
  failures.
- G11 (responses cut short) passes the same way, with PLANTED_SHARE of a day's responses
  stopping at the token limit.

Run from the repo root:

  uv run --group lab python -m lab.failures
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from ccdrift.failures import (ACTIVE_RESPONSES, BEFORE_DAYS, CUT_FLOOR, CUT_SHARE, FAILURE_DAY_COLUMNS,
                              MIN_BEFORE_DAYS, REQUEST_FLOOR, REQUEST_RATIO, cut_short, failing_requests,
                              failure_counts, judged_failures)
from ccdrift.logs import default_source, judged_turns, parse_all
from ccdrift.state import new_state

REQUEST_FLOOR_GRID = (3, 5, 8, 12)
REQUEST_RATIO_GRID = (2, 3, 5)
CUT_FLOOR_GRID = (3, 5, 8)
CUT_SHARE_GRID = (0.002, 0.005, 0.01)

ALERT_BUDGET = 1        # alerts allowed per BUDGET_DAYS days of history
BUDGET_DAYS = 30
PLANTED_REQUESTS = 10   # failed requests a bad release would bring in a day
PLANTED_SHARE = 0.01    # of a day's responses stopping at the token limit
PLANT_DAYS = 5          # days a burst is planted on in turn, one replay each
GATES = {"requests": "G10", "cut": "G11"}


def replay(counts: pd.DataFrame, rule: Callable[[pd.DataFrame, dict[str, Any], date], list[dict[str, Any]]],
           ) -> list[str]:
    """The days the rule alerts on, judging each day as the check would the morning
    after it, with one state carried along."""
    state = new_state()
    alerts = []
    for day in counts["day"].astype(str):
        today = date.fromisoformat(day) + timedelta(days=1)
        seen = counts[counts["day"].astype(str) <= day]
        alerts += [episode["since"] for episode in rule(seen, state, today)]
    return alerts


def plant(counts: pd.DataFrame, day: str, requests: int = 0, share: float = 0.0) -> pd.DataFrame:
    """The counts with `day` made worse: that day carrying `requests` more failed
    requests, or `share` of its responses stopping at the token limit. Planting only ever
    makes the day worse: `share` raises `truncated` to what it should reach, never lowers
    it."""
    planted = counts.copy()
    where = planted.index[planted["day"].astype(str) == str(day)]
    if len(where) != 1:
        raise KeyError(f"{day} is not one of the counted days")
    bad = where[0]
    if requests:
        planted.loc[bad, "overloaded"] = int(planted.loc[bad, "overloaded"]) + requests
        planted.loc[bad, "requests"] = int(planted.loc[bad, "requests"]) + requests
    if share:
        planted.loc[bad, "truncated"] = max(int(planted.loc[bad, "truncated"]),
                                            round(int(planted.loc[bad, "responses"]) * share))
    return planted


def plant_days(counts: pd.DataFrame, how_many: int = PLANT_DAYS) -> list[str]:
    """The last `how_many` active days a rule could judge a burst on — at least
    MIN_BEFORE_DAYS active days within the BEFORE_DAYS before them — oldest first. A gate
    that planted only on the corpus's last day answered a question about that day (how
    busy it was, and how close to a real failure day), not about the rule."""
    days = [str(day) for day in counts[counts["responses"] >= ACTIVE_RESPONSES]["day"].astype(str)]
    judgeable = []
    for i, day in enumerate(days):
        earliest = (date.fromisoformat(day) - timedelta(days=BEFORE_DAYS)).isoformat()
        if sum(1 for before in days[:i] if before >= earliest) >= MIN_BEFORE_DAYS:
            judgeable.append(day)
    return judgeable[-how_many:]


def _rows(counts: pd.DataFrame, days: int, rule, grid, names, planted: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per setting in `grid`: the days it would alert on over the real history,
    and how many of the planted bursts it catches — `planted` is what plant() puts in a
    day, and each of plant_days(counts) carries it in turn, one replay each. A setting
    catches a plant only when the planted day itself alerts."""
    budget = ALERT_BUDGET * max(1, days / BUDGET_DAYS)
    plants = plant_days(counts)
    rows = []
    for setting in grid:
        kwargs = dict(zip(names, setting))
        alerts = replay(counts, lambda c, state, today: rule(c, state, today, **kwargs))
        caught = sum(day in replay(plant(counts, day, **planted),
                                   lambda c, state, today: rule(c, state, today, **kwargs))
                     for day in plants)
        rows.append({**kwargs, "alerts": len(alerts), "days": days, "on": ", ".join(alerts),
                     "caught": caught, "plants": len(plants),
                     "passes": bool(plants) and caught == len(plants) and len(alerts) <= budget})
    return rows


def request_rows(counts: pd.DataFrame, days: int) -> list[dict[str, Any]]:
    grid = [(floor, ratio) for floor in REQUEST_FLOOR_GRID for ratio in REQUEST_RATIO_GRID]
    return _rows(counts, days, failing_requests, grid, ("floor", "ratio"), {"requests": PLANTED_REQUESTS})


def cut_rows(counts: pd.DataFrame, days: int) -> list[dict[str, Any]]:
    grid = [(floor, share) for floor in CUT_FLOOR_GRID for share in CUT_SHARE_GRID]
    return _rows(counts, days, cut_short, grid, ("floor", "share"), {"share": PLANTED_SHARE})


def gate(rows: list[dict[str, Any]], chosen: dict[str, Any]) -> tuple[bool, list[str]]:
    """Whether the settings ccdrift ships pass, and what to say about them."""
    keys = [key for key in chosen]
    row = next((r for r in rows if all(r[key] == chosen[key] for key in keys)), None)
    if row is None:
        return False, ["the settings ccdrift ships aren't in the grid"]
    notes = [f"ships {chosen}: {row['alerts']} alert(s) over {row['days']} days"
             + (f" on {row['on']}" if row["on"] else "")
             + f", planted bursts caught: {row['caught']} of {row['plants']}"]
    if not row["passes"]:
        smaller = [r for r in rows if r["passes"]]
        notes.append(f"passing settings: {[{k: r[k] for k in keys} for r in smaller]}" if smaller
                     else "no setting in the grid passes")
    return bool(row["passes"]), notes


def counts_of(source: Path, today: date) -> tuple[pd.DataFrame, int]:
    tables = parse_all(source)
    turns = judged_turns(tables.responses, today)
    counts = failure_counts(judged_failures(tables.failures, today), turns)
    if counts.empty:
        return pd.DataFrame(columns=FAILURE_DAY_COLUMNS), 0
    days = (date.fromisoformat(str(counts["day"].iloc[-1])) - date.fromisoformat(str(counts["day"].iloc[0]))).days + 1
    return counts, days


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    args = parser.parse_args(argv)

    counts, days = counts_of(args.source, args.today)
    if counts.empty:
        print("no days to judge")
        return 1
    print(f"{len(counts)} days counted over {days} days, "
          f"{int((counts['responses'] >= ACTIVE_RESPONSES).sum())} of them active, "
          f"{int(counts['requests'].sum())} failed requests, "
          f"{int(counts['slept'].sum())} while asleep, "
          f"{int(counts['truncated'].sum() + counts['refused'].sum())} responses cut short")
    requests = request_rows(counts, days)
    cuts = cut_rows(counts, days)
    for row in requests:
        print(f"{GATES['requests']} floor={row['floor']:<3} ratio={row['ratio']:<2} alerts={row['alerts']} "
              f"caught={row['caught']}/{row['plants']}  {row['on']}")
    for row in cuts:
        print(f"{GATES['cut']} floor={row['floor']:<3} share={row['share']:<6} alerts={row['alerts']} "
              f"caught={row['caught']}/{row['plants']}  {row['on']}")
    passed = True
    for name, rows, chosen in (("requests", requests, {"floor": REQUEST_FLOOR, "ratio": REQUEST_RATIO}),
                               ("cut", cuts, {"floor": CUT_FLOOR, "share": CUT_SHARE})):
        ok, notes = gate(rows, chosen)
        passed = passed and ok
        print(f"{GATES[name]}: {'PASS' if ok else 'FAIL'}")
        for note in notes:
            print(f"  {note}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
