"""Fields Claude Code stops logging, and fields it starts logging. Claude Code's
transcript format changes between versions; when a new version drops a field ccdrift
reads, the alerts built on it go quiet without a word, so the check says so. The
opposite direction: a new version can also start carrying a field ccdrift doesn't
read, an opportunity rather than a fault, which the check reports too."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from ccdrift.logs import READ_PATHS, first_days_by_version

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

# A path is new when a new version carries it on USUAL or more of its responses after
# under GONE of the responses before it: the departure rule read backwards, with the
# same two numbers. On one person's logs, 1,843 transcripts over 2026-08-06 to 09-22,
# that reports 8 paths in 5 alerts over 35 days; keyed per path and version, as gaps
# are, the same logs give 13 reports in 7 alerts.
ARRIVED = USUAL


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
    # A version's first day over the whole history, not just the recent history a check reads.
    firsts = {**days.groupby(versions).min(), **first_days_by_version(turns)}
    for version, first in sorted(firsts.items(), key=lambda item: item[1]):
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


def _census_shares(census: pd.DataFrame, mask: pd.Series) -> tuple[dict[str, float], int]:
    """Each key path's share of the responses the census rows under `mask` cover, and how
    many responses that is. The denominator counts each day and version once, since every
    path of that day and version carries the same total."""
    rows = census[mask]
    if rows.empty:
        return {}, 0
    total = int(rows.drop_duplicates(subset=["day", "version"])["day_responses"].sum())
    if total == 0:
        return {}, 0
    counts = rows.groupby("path")["responses"].sum()
    return {str(path): int(count) / total for path, count in counts.items()}, total


def new_fields(census: pd.DataFrame, turns: pd.DataFrame, state: dict[str, Any],
               today: date) -> list[dict[str, Any]]:
    """Key paths a version first seen in the last RECENT_DAYS days carries on ARRIVED or
    more of its responses, after under GONE of the responses in the BEFORE_DAYS days
    before it: fields Claude Code has started logging that ccdrift doesn't read. Every
    path arriving on one version is one record in state["new_fields"], and a path already
    named in one is never reported again, whatever version it turns up on later, since a
    field can arrive, leave and arrive again. A path ccdrift already reads (READ_PATHS)
    is skipped even when it arrives through a deeper leaf than the census walks: ccdrift
    reads message.usage.output_tokens_details.thinking_tokens, which the census records
    as message.usage.output_tokens_details."""
    if census.empty or turns.empty:
        return []
    # Only the days the check judges, which drops the day still in progress: the census
    # is counted at parse time, when a day's completeness isn't known.
    census = census[census["day"].astype(str).isin(set(turns["day"].astype(str)))]
    if census.empty:
        return []
    known = {path for record in state["new_fields"] for path in record["paths"]}
    since = (today - timedelta(days=RECENT_DAYS)).isoformat()
    days = census["day"].astype(str)
    versions = census["version"].astype(str)
    present = set(versions)
    firsts = {**days.groupby(versions).min(), **first_days_by_version(turns)}
    new = []
    for version, first in sorted(firsts.items(), key=lambda item: item[1]):
        if first < since or version not in present:
            continue
        before = (days < first) & (days >= (date.fromisoformat(first) - timedelta(days=BEFORE_DAYS)).isoformat())
        on_version, responses = _census_shares(census, versions == version)
        earlier, responses_before = _census_shares(census, before)
        if responses < MIN_RESPONSES or responses_before < MIN_BEFORE:
            continue
        found = sorted(path for path, share in on_version.items()
                       if share >= ARRIVED and earlier.get(path, 0.0) < GONE and path not in known
                       and path not in READ_PATHS)
        if not found:
            continue
        record = {"paths": found, "version": version,
                  "share": round(min(on_version[path] for path in found), 3),
                  "responses": responses, "reported_on": today.isoformat()}
        state["new_fields"].append(record)
        known.update(found)
        new.append(record)
    return new


def new_fields_message(record: dict[str, Any]) -> str:
    paths = record["paths"]
    count = f"{len(paths)} field{'' if len(paths) == 1 else 's'}"
    subject = "It may be worth reading" if len(paths) == 1 else "They may be worth reading"
    return (f"Claude Code {record['version']} logs {count} ccdrift doesn't read: {', '.join(paths)} "
            f"(on {record['share']:.0%} of {record['responses']:,} responses). {subject}; "
            "please open an issue.")
