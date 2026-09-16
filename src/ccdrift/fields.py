"""Fields Claude Code stops logging. Claude Code's transcript format changes between
versions; when a new version drops a field ccdrift reads, the alerts built on it go
quiet without a word, so the check says so."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

FIELDS = ("version", "entrypoint", "effort", "speed", "service_tier", "thinking_logged", "cache_split")
FIELD_NAMES = {"version": "its version", "entrypoint": "the entrypoint", "effort": "effort",
               "speed": "the speed", "service_tier": "the service tier", "thinking_logged": "thinking token counts",
               "cache_split": "the 1-hour/5-minute cache split"}
CONSEQUENCES = {"version": "Alerts can't name versions",
                "entrypoint": "Agent SDK sessions can't be told apart",
                "effort": "Effort change alerts can't work",
                "speed": "The report can't show the speed",
                "service_tier": "The report can't show the service tier",
                "thinking_logged": "The lab can't compare logged thinking tokens",
                "cache_split": "Cache tier alerts can't work"}
MIN_RESPONSES = 50   # a new version's responses, to judge it
MIN_BEFORE = 200     # responses in the days before it
BEFORE_DAYS = 14
USUAL = 0.9          # the field was on at least this share before
GONE = 0.1           # and is on less than this on the new version
RECENT_DAYS = 14


def _present(turns: pd.DataFrame, field: str) -> pd.Series:
    """Whether each response logged `field`; NaN where it doesn't apply (the cache
    split on responses without cache writes)."""
    if field == "cache_split":
        return ((turns["cache_1h"] + turns["cache_5m"]) > 0).where(turns["cache_creation"] > 0)
    if field not in turns:
        return pd.Series(False, index=turns.index)
    return turns[field].notna()


def _share(mask: pd.Series) -> tuple[float, int]:
    valid = mask.dropna()
    return (float(valid.astype(bool).mean()), len(valid)) if len(valid) else (0.0, 0)


def field_gaps(turns: pd.DataFrame, state: dict[str, Any], today: date) -> list[dict[str, Any]]:
    """Fields that a version first seen in the last RECENT_DAYS days logs on under GONE
    of its responses after they were on USUAL or more in the BEFORE_DAYS days before
    it; each is recorded in state["field_gaps"] and reported once per (field, version).
    Responses without a version form the version "unknown", the only one judged on
    the version field itself."""
    if turns.empty:
        return []
    versions = turns["version"].fillna("unknown").astype(str) if "version" in turns else pd.Series("unknown", index=turns.index)
    days = turns["day"].astype(str)
    since = (today - timedelta(days=RECENT_DAYS)).isoformat()
    new = []
    for version, first in sorted(days.groupby(versions).min().items(), key=lambda item: item[1]):
        if first < since:
            continue
        before = (days < first) & (days >= (date.fromisoformat(first) - timedelta(days=BEFORE_DAYS)).isoformat())
        for field in FIELDS:
            if (field == "version") != (version == "unknown"):
                continue
            if any(g["field"] == field and g["version"] == version for g in state["field_gaps"]):
                continue
            present = _present(turns, field)
            share, responses = _share(present[versions == version])
            share_before, responses_before = _share(present[before])
            if responses < MIN_RESPONSES or responses_before < MIN_BEFORE:
                continue
            if share_before >= USUAL and share < GONE:
                gap = {"field": field, "version": version, "share_before": round(share_before, 3),
                       "share": round(share, 3), "responses": responses, "reported_on": today.isoformat()}
                state["field_gaps"].append(gap)
                new.append(gap)
    return new


def gap_message(gap: dict[str, Any]) -> str:
    where = "Claude Code" if gap["version"] == "unknown" else f"Claude Code {gap['version']}"
    return (f"{where} no longer logs {FIELD_NAMES[gap['field']]} (on {gap['share']:.0%} of {gap['responses']} "
            f"responses, {gap['share_before']:.0%} before). {CONSEQUENCES[gap['field']]} until ccdrift reads it "
            "again; run `ccdrift peek`.")
