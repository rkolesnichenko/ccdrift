"""Does the hook coverage rule find hooks that stop or start running, and nothing else? (G15)

Each project, thread, hook event and tool is a stream of CLI transcripts, each hooked or
not (ccdrift.hookcover). The gate replays the rule day by day, as the daily check would
see each day the morning after, and asks five things of a setting:

- the one real change in the owner's logs is found once: from Claude Code 2.1.261,
  subagent tool calls got hook records they never had before (2026-09-05). Where the
  transcripts before it are gone, the question can't be put and the gate says so;
- nothing else in the logs is reported;
- a planted stop is caught: from each of the last PLANT_DAYS days a stop could be seen
  on, hook records are removed from every CLI transcript, and, separately, from the one
  stream with the most transcripts before that day in each thread. A plant is credited
  only when the planted replay reports a stop the unplanted one didn't. The gate reports
  the lag in days: 0 when the alert comes the morning after the day the hooks stopped;
- work moving from a hooked project to an unhooked one is no change;
- the first check after upgrading, on an empty state on the gate's --today, reports
  nothing: a change the logs already hold isn't news.

The grid tries each window, baseline, agreement and minimum number of calls in GRID. The
gate checks the shipped setting and every setting in the grid at least as strict as it
(window, baseline, agreement and minimum each at least the shipped one's): it passes when
all of them pass, so the setting doesn't rest on a lucky neighbour. Which passing setting
ships is a stated judgement, not the gate's. On the owner's logs every setting at agreement
1.0 passes, at every window, baseline and minimum, while 4 of the 18 at 0.8 raise a false
alarm. Window 3 ships rather than 2 because hooks scoped to an agent type or a skill, which
these logs hold none of, would make short runs look like a change, and window 2 caught only
4 of its 14 plants sooner (baseline 10, agreement 1.0, a minimum of 1 or 2 calls; 3 of 13
at 3): one by two days, the rest planted too late for window 3 to see by the last day.
The output is aggregate: no project, transcript or tool name.

Run from the repo root:

  uv run --group lab python -m lab.hook_coverage
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ccdrift.hookcover import SETTING, HookSetting, STREAM, hook_coverage_alerts, transcript_states
from ccdrift.logs import coverage_frame, default_source, outside_sdk, parse_all
from ccdrift.state import new_state

GATE = "G15"
GRID = [HookSetting(window, baseline, agree, min_calls)
        for window in (2, 3, 4) for baseline in (5, 10) for agree in (0.8, 1.0) for min_calls in (1, 2, 3)]
PLANT_DAYS = 5
# The real change: the thread, direction and first day of the subagent start at 2.1.261,
# and the days an alert for it may name as its first (its window may start a day or two
# after the step when the first transcripts after it have too few calls).
REAL = ("subagent", "started")
REAL_DAYS = ("2026-09-05", "2026-09-08")


def replay(coverage: pd.DataFrame, setting: HookSetting, first: Optional[str] = None) -> list[dict[str, Any]]:
    """The alerts the rule gives, judging each day from `first` (default: the first day
    in `coverage`) as the check would the morning after it, with one state carried along.
    Each alert carries `on`, the day it was given for."""
    if coverage.empty:
        return []
    state = new_state()
    days = sorted(day for day in coverage["day"].astype(str).unique() if first is None or day >= first)
    alerts = []
    for day in days:
        today = date.fromisoformat(day) + timedelta(days=1)
        alerts += [{**alert, "on": day} for alert in hook_coverage_alerts(coverage, state, today, setting)]
    return alerts


def plant_stop(coverage: pd.DataFrame, day: str, stream: Optional[tuple] = None) -> pd.DataFrame:
    """`coverage` with no hook run on any CLI tool call from `day` on, or only on those of
    `stream` (project, thread, event, tool) when given."""
    planted = coverage.copy()
    hit = (planted["day"].astype(str) >= day) & outside_sdk(planted)
    if stream is not None:
        states = transcript_states(planted, date.max, 0)
        files = set(states.loc[(states[STREAM] == pd.Series(dict(zip(STREAM, stream)))).all(axis=1), "source_file"])
        project, thread, event, tool = stream
        hit &= (planted["source_file"].isin(files) & (planted["event"] == event) & (planted["tool"] == tool)
                & (planted["is_sidechain"].astype(bool) == (thread == "subagent")))
    planted.loc[hit, "hooked"] = 0
    return planted


def plant_days(coverage: pd.DataFrame, setting: HookSetting, today: date) -> list[str]:
    """The last PLANT_DAYS days a stop could be seen on: some stream has `baseline`
    hooked transcripts before the day and `window` transcripts from it on."""
    states = transcript_states(coverage, today, setting.min_calls)
    days = sorted(states["day"].astype(str).unique())
    seen = []
    for day in days:
        for _, rows in states.groupby(STREAM, sort=True):
            before = rows[rows["day"].astype(str) < day]
            if before["on"].tail(setting.baseline).sum() >= setting.baseline and \
                    (rows["day"].astype(str) >= day).sum() >= setting.window:
                seen.append(day)
                break
    return seen[-PLANT_DAYS:]


def busiest_streams(coverage: pd.DataFrame, setting: HookSetting, today: date, day: str) -> list[tuple]:
    """In each thread, the stream with the most hooked transcripts before `day` that has
    `window` transcripts from it on; the one-stream plants go there."""
    states = transcript_states(coverage, today, setting.min_calls)
    best: dict[str, tuple[int, tuple]] = {}
    for stream, rows in states.groupby(STREAM, sort=True):
        hooked_before = int(rows.loc[rows["day"].astype(str) < day, "on"].sum())
        if hooked_before >= setting.baseline and (rows["day"].astype(str) >= day).sum() >= setting.window:
            if hooked_before > best.get(stream[1], (-1, ()))[0]:
                best[stream[1]] = (hooked_before, stream)
    return [stream for _, stream in sorted(best.values(), key=lambda item: item[1])]


def move_history() -> pd.DataFrame:
    """Work moving from a hooked project to an unhooked one: fifteen days of subagents in
    -hooked with every Bash call hooked, then eight in -unhooked with none."""
    rows = []
    for i in range(23):
        project, on = ("-hooked", True) if i < 15 else ("-unhooked", False)
        day = (date(2026, 9, 1) + timedelta(days=i)).isoformat()
        rows.append({"source_file": f"{project}/s/subagents/agent-{i}.jsonl", "session_id": f"s{i}", "day": day,
                     "version": "2.1.261", "entrypoint": "cli", "is_sidechain": True, "event": "PreToolUse",
                     "tool": "Bash", "calls": 5, "hooked": 5 if on else 0})
    return coverage_frame(rows)


def judge(coverage: pd.DataFrame, setting: HookSetting, today: date) -> dict[str, Any]:
    """What the five questions answer for one setting."""
    quiet = replay(coverage, setting)
    real = [a for a in quiet if (a["thread"], a["direction"]) == REAL and REAL_DAYS[0] <= a["since"] <= REAL_DAYS[1]]
    others = [a for a in quiet if a not in real]
    states = transcript_states(coverage, today, setting.min_calls)
    before_real = states[(states["thread"] == REAL[0]) & (states["day"].astype(str) < REAL_DAYS[0])]
    measurable = len(before_real) >= setting.baseline
    seen = {(a["thread"], a["direction"], a["since"]) for a in quiet}
    plants, caught, lags = 0, 0, []
    for day in plant_days(coverage, setting, today):
        targets: list[Optional[tuple]] = [None, *busiest_streams(coverage, setting, today, day)]
        for stream in targets:
            plants += 1
            alerts = [a for a in replay(plant_stop(coverage, day, stream), setting, first=day)
                      if a["direction"] == "stopped" and (a["thread"], a["direction"], a["since"]) not in seen
                      and (stream is None or a["thread"] == stream[1])]
            if alerts:
                caught += 1
                lags.append((date.fromisoformat(alerts[0]["on"]) - date.fromisoformat(day)).days)
    moved = replay(move_history(), setting)
    first = hook_coverage_alerts(coverage, new_state(), today, setting)
    passed = ((not measurable or len(real) == 1 and real[0]["new_version"]) and not others and plants > 0
              and caught == plants and not moved and not first)
    return {"setting": setting, "real": len(real), "measurable": measurable, "others": len(others),
            "plants": plants, "caught": caught, "lags": lags, "moved": len(moved), "first": len(first),
            "passed": passed}


def stricter(results: list[dict[str, Any]], setting: HookSetting) -> list[dict[str, Any]]:
    """The results of the settings at least as strict as `setting`, itself among them:
    each of window, baseline, agreement and minimum at least its own."""
    return [r for r in results if all(getattr(r["setting"], name) >= getattr(setting, name)
                                      for name in HookSetting._fields)]


def stricter_pass(results: list[dict[str, Any]], setting: HookSetting) -> bool:
    """Whether `setting` and every setting at least as strict as it pass."""
    return all(r["passed"] for r in stricter(results, setting))


def line(result: dict[str, Any]) -> str:
    s = result["setting"]
    lag = f" lag in days {sorted(result['lags'])}" if result["lags"] else ""
    real = f"real {result['real']}" if result["measurable"] else "real not measurable"
    return (f"window {s.window} baseline {s.baseline} agree {s.agree} min_calls {s.min_calls}: {real}, "
            f"others {result['others']}, planted {result['caught']}/{result['plants']}{lag}, moved {result['moved']}, "
            f"first check {result['first']}{' PASS' if result['passed'] else ''}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    args = parser.parse_args(argv)
    coverage = parse_all(args.source).hook_coverage
    coverage = coverage[coverage["day"].astype(str) < args.today.isoformat()].reset_index(drop=True)
    states = transcript_states(coverage, args.today, 1)
    print(f"{states['source_file'].nunique()} CLI transcripts in {states.groupby(STREAM).ngroups} streams")
    results = [judge(coverage, setting, args.today) for setting in GRID]
    if SETTING not in GRID:
        results.append(judge(coverage, SETTING, args.today))
    for result in results:
        print(f"  {line(result)}")
    shipped = next(r for r in results if r["setting"] == SETTING)
    passed = stricter_pass(results, SETTING)
    at_least = stricter(results, SETTING)
    print(f"{GATE}: {'PASS' if passed else 'FAIL'} at the shipped {line(shipped)}")
    print(f"  settings at least as strict as the shipped one: {sum(r['passed'] for r in at_least)} of {len(at_least)} pass")
    print(f"  the grid: {sum(r['passed'] for r in results)} of {len(results)} pass")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
