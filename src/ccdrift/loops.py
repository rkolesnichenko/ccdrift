"""Tool-loop cache misses: a turn-by-turn CUSUM on the turns inside the tool loop,
on the main thread and in subagents, so a regression that breaks caching there shows
within hours. The new-prompt metric can't see one: about 1,500 of the owner's
responses open with a prompt, against about 60,000 tool-loop and subagent turns.

lab/loop_cache.py measured it on the owner's logs (G8 for the main thread, G9 for
subagents) and set LOOP_SETTINGS; None means that stream gets no warning."""

from __future__ import annotations

from collections import namedtuple
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import pandas as pd

from ccdrift.early import alarm_runs, rise_first
from ccdrift.incidents import versions_text
from ccdrift.logs import outside_sdk
from ccdrift.texts import approx, clock_text

STREAMS = ("main", "subagent")
LoopSetting = namedtuple("LoopSetting", "p1 h min_sessions")
LOOP_SETTINGS: dict[str, Optional[LoopSetting]] = {"main": LoopSetting(p1=0.02, h=3.0, min_sessions=1),
                                                   "subagent": LoopSetting(p1=0.05, h=4.0, min_sessions=1)}

LOOP_COLUMNS = ["timestamp", "day", "session_id", "version", "cache_creation", "is_loop_miss"]


def loop_turns(df: pd.DataFrame, stream: str) -> pd.DataFrame:
    """The tool-loop turns of `stream` outside Agent SDK sessions, in time order:
    "main" on the main thread, "subagent" in subagents (whose session_id is their
    parent session's)."""
    if df.empty or "loop_turn" not in df:
        return pd.DataFrame(columns=LOOP_COLUMNS)
    main = df["main_thread"].astype(bool)
    keep = df["loop_turn"].astype(bool) & (main if stream == "main" else ~main) & outside_sdk(df)
    turns = df.loc[keep, LOOP_COLUMNS].copy()
    turns["day"] = turns["day"].astype(str)
    turns["is_loop_miss"] = turns["is_loop_miss"].astype(bool)
    return turns.sort_values("timestamp", kind="stable").reset_index(drop=True)


PREFIXES = {"main": "loop", "subagent": "subagent_loop"}
COUNT_COLUMNS = ["loop_turns", "loop_misses", "subagent_loop_turns", "subagent_loop_misses"]


def loop_counts(df: pd.DataFrame, today: date, by: str = "day", days: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Tool-loop turns and misses of each stream (COUNT_COLUMNS) on complete UTC days
    before `today`, or only on `days`, per day or per Claude Code version ("unknown"
    when not logged)."""
    parts = []
    for stream in STREAMS:
        turns = loop_turns(df, stream)
        keep = turns["day"] < today.isoformat()
        if days is not None:
            keep &= turns["day"].isin(list(days))
        turns = turns[keep]
        key = turns["day"] if by == "day" else turns["version"].fillna("unknown").astype(str)
        grouped = turns["is_loop_miss"].groupby(key)
        prefix = PREFIXES[stream]
        parts.append(pd.DataFrame({f"{prefix}_turns": grouped.size(), f"{prefix}_misses": grouped.sum()}))
    return pd.concat(parts, axis=1).fillna(0).astype(int)


def qualifying_alarms(turns: pd.DataFrame, base_rate: float, setting: LoopSetting) -> list[tuple[int, int]]:
    """(first, alarm) for each alarm of the CUSUM over `turns` (misses at setting.p1
    against `base_rate`, held under half of p1, passing setting.h) whose rise holds
    misses from at least setting.min_sessions sessions. `first` is where the rise
    began, chained back over the alarms before it (see early.rise_first). With 2
    sessions, one long session that keeps missing now and then can't raise an alarm on
    its own."""
    misses = turns["is_loop_miss"].tolist()
    sessions = turns["session_id"].tolist()
    runs = alarm_runs(misses, base_rate, setting.h, p1=setting.p1, high=setting.p1 / 2)
    found = []
    for i, (_, alarm) in enumerate(runs):
        first = rise_first(runs[:i + 1])
        missed = {sessions[k] for k in range(first, alarm + 1) if misses[k]}
        if len(missed) >= setting.min_sessions:
            found.append((first, alarm))
    return found


WINDOW_DAYS = 7      # the stretch the CUSUM runs over, today included
BASE_DAYS = 14       # complete days before it that give the usual miss rate
MIN_BASE_TURNS = 1000
RECENT_HOURS = 24    # an alarm this recent is news
QUIET_DAYS = 7       # at most one warning per stream in this many days


def loop_warning(responses: pd.DataFrame, stream: str, state: dict[str, Any], now: datetime,
                 setting: Optional[LoopSetting] = None) -> Optional[dict[str, Any]]:
    """A warning when the CUSUM on the tool-loop turns of `stream` over the last
    WINDOW_DAYS days (up to `now`) raised a qualifying alarm in the last RECENT_HOURS
    hours, against the miss rate of the BASE_DAYS days before them. `setting` defaults
    to LOOP_SETTINGS[stream]; no warning without one, within QUIET_DAYS of the stream's
    last warning, or with under MIN_BASE_TURNS usual turns. An open cache incident
    doesn't hold it back: one on new prompts doesn't say whether tool loops miss too.
    The warning is also recorded in state["loop_warnings"]."""
    setting = LOOP_SETTINGS[stream] if setting is None else setting
    if setting is None:
        return None
    if any(w["stream"] == stream and now - datetime.fromisoformat(w["at"]) < timedelta(days=QUIET_DAYS)
           for w in state["loop_warnings"]):
        return None
    turns = loop_turns(responses, stream)
    today = now.astimezone(timezone.utc).date()
    first_day = (today - timedelta(days=WINDOW_DAYS - 1)).isoformat()
    base_start = (today - timedelta(days=WINDOW_DAYS - 1 + BASE_DAYS)).isoformat()
    days = turns["day"]
    base = (days >= base_start) & (days < first_day)
    if base.sum() < MIN_BASE_TURNS:
        return None
    base_rate = float(turns.loc[base, "is_loop_miss"].mean())
    stretch = turns[(days >= first_day) & (turns["timestamp"] <= pd.Timestamp(now))].reset_index(drop=True)
    alarms = qualifying_alarms(stretch, base_rate, setting)
    if not alarms:
        return None
    first, alarm = alarms[-1]
    at = stretch["timestamp"][alarm].to_pydatetime()
    if now - at > timedelta(hours=RECENT_HOURS):
        return None
    rise = stretch.iloc[first:alarm + 1]
    missed = rise[rise["is_loop_miss"]]
    main = responses["main_thread"].astype(bool)
    in_stream = responses[(main if stream == "main" else ~main) & outside_sdk(responses)]
    warning = {"stream": stream, "at": at.isoformat(timespec="seconds"),
               "since": rise["timestamp"].iloc[0].to_pydatetime().isoformat(timespec="seconds"),
               "misses": len(missed), "turns": len(rise), "sessions": int(missed["session_id"].nunique()),
               "base_rate": round(base_rate, 4), "tokens": int(missed["cache_creation"].sum()),
               "versions": versions_text(in_stream, sorted(rise["day"].unique())),
               "reported_on": now.date().isoformat()}
    state["loop_warnings"].append(warning)
    return warning


STREAM_TURNS = {"main": "tool-loop turns", "subagent": "subagent tool-loop turns"}


def loop_message(warning: dict[str, Any], now: datetime) -> str:
    on = f", on Claude Code {', '.join(warning['versions'])}" if warning["versions"] else ""
    sessions = f"{warning['sessions']} session{'' if warning['sessions'] == 1 else 's'}"
    return (f"{warning['misses']} of the last {warning['turns']} {STREAM_TURNS[warning['stream']]} missed the cache "
            f"(usually {warning['base_rate']:.2%}), since {clock_text(warning['since'], now)}, in {sessions}, "
            f"rewriting ~{approx(warning['tokens'])} tokens{on}. The weekly summary shows whether it lasts.")
