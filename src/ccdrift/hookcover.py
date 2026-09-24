"""Hook coverage: whether the hooks configured for tool calls run. A Claude Code update
can stop running them, or start, with nothing else to say so; a hook that quietly stops
is a safety or formatting hook doing nothing. Each project, thread, hook event and tool
is a stream of transcripts, each either hooked or not, and a change is a stream whose
latest transcripts all turned the other way."""

from __future__ import annotations

from collections import namedtuple
from datetime import date, timedelta
from typing import Any, Sequence

import pandas as pd

from ccdrift.logs import outside_sdk
from ccdrift.sessions import project_of

HookSetting = namedtuple("HookSetting", ["window", "baseline", "agree", "min_calls"])

# Measured by lab/hook_coverage.py (G15) on 2026-09-24 over 1,222 CLI transcripts in 354
# streams: the real start at 2.1.261 found once, no other alert, 11 of 11 planted stops
# caught, 7 of them the morning after and the rest within 3 days, and a move to an
# unhooked project left alone. 35 of the 36 settings tried pass; window 2 with baseline 10,
# agreement 0.8 and no call minimum raised one false alarm, and window 3 is the smallest
# at which every one passes.
SETTING = HookSetting(window=3, baseline=10, agree=1.0, min_calls=3)

RECENT_DAYS = 14   # a change must end within this many days, as the other change alerts
MERGE_DAYS = 14    # changes in one thread and direction starting this close are one alert

STREAM = ["project", "thread", "event", "tool"]
STATE_COLUMNS = [*STREAM, "source_file", "day", "versions", "on"]


def transcript_states(coverage: pd.DataFrame, today: date, min_calls: int) -> pd.DataFrame:
    """One row per CLI transcript and stream, from complete UTC days: whether it was
    hooked (a hook ran on at least half its calls of that tool, for that event), its
    first day and its Claude Code versions. A transcript with fewer than `min_calls`
    calls of a tool doesn't count in that tool's streams. Sorted by stream, then day and
    transcript, so two runs over one history read the same."""
    if coverage.empty:
        return pd.DataFrame(columns=STATE_COLUMNS)
    rows = coverage[(coverage["day"].astype(str) < today.isoformat()) & outside_sdk(coverage)]
    if rows.empty:
        return pd.DataFrame(columns=STATE_COLUMNS)
    rows = rows.assign(project=rows["source_file"].astype(str).map(project_of),
                       thread=rows["is_sidechain"].astype(bool).map({True: "subagent", False: "main"}),
                       version=rows["version"].fillna("unknown").astype(str))
    grouped = rows.groupby([*STREAM, "source_file"], sort=True).agg(
        calls=("calls", "sum"), hooked=("hooked", "sum"), day=("day", "min"),
        versions=("version", lambda v: tuple(sorted(set(v))))).reset_index()
    grouped = grouped[grouped["calls"] >= min_calls]
    grouped = grouped.assign(on=2 * grouped["hooked"] >= grouped["calls"])
    return (grouped[STATE_COLUMNS].sort_values([*STREAM, "day", "source_file"], kind="stable")
            .reset_index(drop=True))


def stream_changes(states: pd.DataFrame, setting: HookSetting) -> list[dict[str, Any]]:
    """Every change in every stream: its latest `window` transcripts all in one state,
    while at least `agree` of the `baseline` transcripts before them were in the other.
    A stream needs a full baseline. Several windows after one step qualify; the first of
    each run is kept. `direction` is "stopped" (hooked, then not) or "started"."""
    found = []
    for stream, rows in states.groupby(STREAM, sort=True):
        on, days, versions = rows["on"].tolist(), rows["day"].astype(str).tolist(), rows["versions"].tolist()
        previous = None
        for end in range(setting.baseline + setting.window - 1, len(on)):
            window = range(end - setting.window + 1, end + 1)
            baseline = range(window[0] - setting.baseline, window[0])
            state = on[window[0]]
            other = sum(on[i] != state for i in baseline)
            qualifies = all(on[i] == state for i in window) and other >= setting.agree * setting.baseline
            if qualifies and previous != state:
                arrived = set().union(*(versions[i] for i in window)) - set().union(*(versions[i] for i in baseline))
                found.append({**dict(zip(STREAM, stream)), "direction": "started" if state else "stopped",
                              "since": days[window[0]], "until": days[window[-1]], "window": setting.window,
                              "baseline": setting.baseline, "baseline_other": other,
                              "new_version": bool(arrived - {"unknown"})})
            previous = state if qualifies else None
    return found


def stream_key(change: dict[str, Any]) -> str:
    return "|".join(str(change[name]) for name in STREAM)


def merged(changes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Changes in one thread and direction that start within MERGE_DAYS of the first of
    them, as one: a Claude Code update reaches every stream at once. Each carries the
    projects, events and tools that moved, the first change's counts, and `days`, the
    first and last day of its windows, which the check names versions from."""
    alerts: list[dict[str, Any]] = []
    for change in sorted(changes, key=lambda c: (c["thread"], c["direction"], c["since"], stream_key(c))):
        last = alerts[-1] if alerts else None
        if (last is None or (last["thread"], last["direction"]) != (change["thread"], change["direction"])
                or (date.fromisoformat(change["since"]) - date.fromisoformat(last["since"])).days > MERGE_DAYS):
            last = {"thread": change["thread"], "direction": change["direction"], "since": change["since"],
                    "until": change["until"], "window": change["window"], "baseline": change["baseline"],
                    "baseline_other": change["baseline_other"], "new_version": False, "streams": []}
            alerts.append(last)
        last["until"] = max(last["until"], change["until"])
        last["new_version"] = last["new_version"] or change["new_version"]
        last["streams"].append(change)
    for alert in alerts:
        streams = alert.pop("streams")
        alert.update(projects=sorted({c["project"] for c in streams}), events=sorted({c["event"] for c in streams}),
                     tools=sorted({c["tool"] for c in streams}), days=[alert["since"], alert["until"]])
    return alerts


def _near(one: str, other: str) -> bool:
    return abs((date.fromisoformat(one) - date.fromisoformat(other)).days) <= MERGE_DAYS


def hook_coverage_alerts(coverage: pd.DataFrame, state: dict[str, Any], today: date,
                         setting: HookSetting = SETTING) -> list[dict[str, Any]]:
    """Changes whose window ends within the last RECENT_DAYS days and that aren't
    recorded yet, merged into alerts. Each stream's change is recorded in
    state["hook_changes"]; a stream is reported again only once it turns the other way,
    after the latest change recorded for it. A stream whose window fills after a change
    in the same thread and direction was reported, within MERGE_DAYS of it, is recorded
    without an alert of its own: one update reaches every stream, each on its own day."""
    cutoff = (today - timedelta(days=RECENT_DAYS)).isoformat()
    recorded = state["hook_changes"]
    reported = list(recorded)
    fresh = []
    for change in stream_changes(transcript_states(coverage, today, setting.min_calls), setting):
        if change["until"] < cutoff:
            continue
        key = stream_key(change)
        mine = [r for r in recorded if r["stream"] == key]
        latest = max(mine, key=lambda r: r["since"]) if mine else None
        if latest is not None and (latest["direction"] == change["direction"] or latest["since"] > change["since"]):
            continue
        recorded.append({"stream": key, "thread": change["thread"], "direction": change["direction"],
                         "since": change["since"], "reported_on": today.isoformat()})
        if not any(r.get("thread") == change["thread"] and r["direction"] == change["direction"]
                   and _near(r["since"], change["since"]) for r in reported):
            fresh.append(change)
    return merged(fresh)
