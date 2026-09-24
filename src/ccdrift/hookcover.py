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
# caught, 7 of them the morning after and the rest within 3 days, a move to an unhooked
# project left alone, and the first check after upgrading quiet. The one stricter setting
# tried passes too. 32 of the 36 settings pass: all 18 at agreement 1.0, while 4 of the 18
# at 0.8 raise one false alarm, a subagent stream starting 2026-09-21 on a mostly unhooked
# baseline. Window 3 rather than 2 is a judgement: hooks scoped to an agent type or a
# skill, which these logs hold none of, would make short runs look like a change.
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
    keys = [*STREAM, "source_file"]
    grouped = rows.groupby(keys, sort=True).agg(
        calls=("calls", "sum"), hooked=("hooked", "sum"), day=("day", "min")).reset_index()
    # A transcript's versions can differ between its streams (46 of the 6,244 states on
    # 2026-09-24), so they stay per stream. Nearly every one ran a single version there, so
    # only the rest are gathered a group at a time, which otherwise cost the most here.
    pairs = rows[[*keys, "version"]].drop_duplicates().sort_values([*keys, "version"], kind="stable")
    mixed = pairs.duplicated(keys, keep=False)
    versions = pd.concat([pairs[~mixed].set_index(keys)["version"].map(lambda v: (v,)),
                          pairs[mixed].groupby(keys, sort=True)["version"].agg(tuple)])
    grouped = grouped.join(versions.rename("versions"), on=keys)
    grouped = grouped[grouped["calls"] >= min_calls]
    grouped = grouped.assign(on=2 * grouped["hooked"] >= grouped["calls"])
    return (grouped[STATE_COLUMNS].sort_values([*STREAM, "day", "source_file"], kind="stable")
            .reset_index(drop=True))


def stream_changes(states: pd.DataFrame, setting: HookSetting) -> list[dict[str, Any]]:
    """Every change in every stream: its latest `window` transcripts all in one state,
    while at least `agree` of the `baseline` transcripts before them were in the other.
    A stream needs a full baseline. Several windows after one step qualify; the first of
    each run is kept. `direction` is "stopped" (hooked, then not) or "started". The step
    lies between `after`, the day of the last transcript before the window, and `since`,
    the day of the first in it; `new_version` says whether that first transcript ran a
    Claude Code version the last one before it didn't."""
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
                arrived = set(versions[window[0]]) - set(versions[baseline[-1]])
                found.append({**dict(zip(STREAM, stream)), "direction": "started" if state else "stopped",
                              "after": days[baseline[-1]], "since": days[window[0]], "until": days[window[-1]],
                              "window": setting.window,
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
    """New changes merged into alerts. Every change is recorded in state["hook_changes"],
    each with `after`, `since` and whether it `alerted`; a stream's change is new only
    once the stream turns the other way, after the latest change recorded for it. A new
    change alerts when its window ends within the last RECENT_DAYS days, unless a change
    in the same thread and direction, recorded before, started between its `after` and
    its `since` (the same step, seen first in another stream), or one that alerted
    started within MERGE_DAYS of it (one update reaching each stream on its own day).
    Only changes that alerted fold others in, so silent records don't chain. Changes that
    ended before the last RECENT_DAYS days count as recorded before, whether or not a
    check saw them: on the first check after upgrading, a step's quieter streams are no
    news either."""
    cutoff = (today - timedelta(days=RECENT_DAYS)).isoformat()
    recorded = state["hook_changes"]
    before = list(recorded)
    new = []
    for change in stream_changes(transcript_states(coverage, today, setting.min_calls), setting):
        key = stream_key(change)
        mine = [r for r in recorded if r["stream"] == key]
        latest = max(mine, key=lambda r: r["since"]) if mine else None
        if latest is not None and (latest["direction"] == change["direction"] or latest["since"] > change["since"]):
            continue
        record = {"stream": key, "thread": change["thread"], "direction": change["direction"],
                  "after": change["after"], "since": change["since"], "reported_on": today.isoformat(),
                  "alerted": False}
        recorded.append(record)
        new.append((change, record))
    known = before + [record for change, record in new if change["until"] < cutoff]
    fresh = []
    for change, record in new:
        if change["until"] < cutoff:
            continue
        same = [r for r in known if r.get("thread") == change["thread"] and r["direction"] == change["direction"]]
        if any(change["after"] <= r["since"] <= change["since"] for r in same):
            continue
        if any(r.get("alerted") and _near(r["since"], change["since"]) for r in same):
            continue
        record["alerted"] = True
        fresh.append(change)
    return merged(fresh)
