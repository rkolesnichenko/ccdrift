"""Stop hooks: how often they run and fail and how long they take, from the summary
Claude Code logs after running them. A Claude Code update that changes what hooks
receive shows up as hooks failing."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional, Sequence

import pandas as pd

HOOK_DAY_COLUMNS = ["day", "runs", "failed", "median_ms"]

ACTIVE_RUNS = 10     # runs for a day to count
FAILING = 0.5        # share of runs with an error
BEFORE_DAYS = 14
MIN_BEFORE_DAYS = 5
RECENT_DAYS = 14


def judged_hook_runs(hook_runs: pd.DataFrame, today: date) -> pd.DataFrame:
    """Main-thread hook runs of complete UTC days, without Agent SDK sessions."""
    if hook_runs.empty:
        return hook_runs
    keep = (hook_runs["day"].astype(str) < today.isoformat()) & ~hook_runs["is_sidechain"].astype(bool)
    if "entrypoint" in hook_runs:
        keep &= ~hook_runs["entrypoint"].fillna("").astype(str).str.startswith("sdk-")
    return hook_runs[keep]


def hook_days(runs: pd.DataFrame) -> pd.DataFrame:
    """Per day: runs, runs with an error, median duration in ms."""
    if runs.empty:
        return pd.DataFrame(columns=HOOK_DAY_COLUMNS)
    by_day = runs.groupby(runs["day"].astype(str), sort=True)
    return pd.DataFrame({"runs": by_day.size(),
                         "failed": by_day["error_count"].agg(lambda s: int((s > 0).sum())),
                         "median_ms": by_day["duration_ms"].median()}).rename_axis("day").reset_index()


def hooks_summary(runs: pd.DataFrame, days: Sequence[str]) -> Optional[dict[str, Any]]:
    """Runs, days with an error and median duration over `days`; None without runs."""
    window = runs[runs["day"].astype(str).isin(list(days))] if not runs.empty else runs
    if window.empty:
        return None
    error_days = int((window.groupby(window["day"].astype(str))["error_count"].sum() > 0).sum())
    median = window["duration_ms"].median()
    return {"runs": len(window), "error_days": error_days,
            "median_duration_ms": None if pd.isna(median) else float(median)}


def hooks_lines(summary: Optional[dict[str, Any]]) -> list[str]:
    """The report's hooks line, starting with a blank line; empty without runs."""
    if summary is None:
        return []
    days = summary["error_days"]
    errors = "no errors" if days == 0 else f"errors on {days} day{'s' if days != 1 else ''}"
    median = "" if summary["median_duration_ms"] is None else f", median {summary['median_duration_ms'] / 1000:.1f} s"
    return ["", f"Hooks over these days: {summary['runs']:,} stop-hook runs, {errors}{median}"]


def hook_failures(runs: pd.DataFrame, state: dict[str, Any], today: date) -> list[dict[str, Any]]:
    """Stop hooks failing on at least FAILING of their runs two active days in a row,
    the first within the last RECENT_DAYS days, after at least MIN_BEFORE_DAYS active
    days without that in the BEFORE_DAYS before; each is recorded in
    state["hook_failures"], once per episode."""
    active = hook_days(runs)
    if active.empty:
        return []
    active = active[active["runs"] >= ACTIVE_RUNS].reset_index(drop=True)
    share = active["failed"] / active["runs"]
    since = (today - timedelta(days=RECENT_DAYS)).isoformat()
    new = []
    for i in range(1, len(active)):
        first, second = str(active["day"][i - 1]), str(active["day"][i])
        if first < since or share[i - 1] < FAILING or share[i] < FAILING:
            continue
        earliest = (date.fromisoformat(first) - timedelta(days=BEFORE_DAYS)).isoformat()
        before = (active["day"].astype(str) < first) & (active["day"].astype(str) >= earliest)
        if before.sum() < MIN_BEFORE_DAYS or bool((share[before] >= FAILING).any()):
            continue
        if any(f["since"] >= earliest for f in state["hook_failures"]):
            continue
        failure = {"since": first, "days": [first, second],
                   "runs": [int(active["runs"][i - 1]), int(active["runs"][i])],
                   "failed": [int(active["failed"][i - 1]), int(active["failed"][i])],
                   "reported_on": today.isoformat()}
        state["hook_failures"].append(failure)
        new.append(failure)
    return new


def failure_message(failure: dict[str, Any], versions: Sequence[str]) -> str:
    on = f", on Claude Code {', '.join(versions)}" if versions else ""
    (first, second), (runs1, runs2), (failed1, failed2) = failure["days"], failure["runs"], failure["failed"]
    return (f"Stop hooks failed on {failed1} of {runs1} runs on {first} and {failed2} of {runs2} on {second}{on}. "
            "Check your hooks; a Claude Code update may have changed their input.")
