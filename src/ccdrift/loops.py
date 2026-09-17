"""Tool-loop cache misses: a turn-by-turn CUSUM on the turns inside the tool loop,
on the main thread and in subagents, so a regression that breaks caching there shows
within hours. The new-prompt metric can't see one: about 1,500 of the owner's
responses open with a prompt, against about 60,000 tool-loop and subagent turns.

lab/loop_cache.py measured it on the owner's logs (G8 for the main thread, G9 for
subagents) and set LOOP_SETTINGS; None means that stream gets no warning."""

from __future__ import annotations

from collections import namedtuple
from typing import Optional

import pandas as pd

from ccdrift.early import alarm_runs, rise_first
from ccdrift.logs import outside_sdk

STREAMS = ("main", "subagent")
LoopSetting = namedtuple("LoopSetting", "p1 h min_sessions")
LOOP_SETTINGS: dict[str, Optional[LoopSetting]] = {"main": None, "subagent": None}

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
