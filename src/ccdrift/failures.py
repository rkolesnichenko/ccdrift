"""Failed requests and responses cut short: the error banners Claude Code writes when a
request fails, the retries it logs, and the responses that stop at the token limit. Both
are far too rare for a usual rate, so neither is judged like the cache metric: a day
alerts when it stands well above the days before it."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from ccdrift.logs import outside_sdk
from ccdrift.texts import kinds_text

# Every kind parse_file records, and those a rule counts: a banner blaming the user's
# own Mac for going to sleep is no drift, so it is reported but never alerts.
KINDS = ("overloaded", "stream", "other", "retry", "slept")
COUNTED = ("overloaded", "stream", "other", "retry")
FAILURE_DAY_COLUMNS = ["day", "responses", "requests", "truncated", "refused", *KINDS]

REQUEST_FLOOR = 5      # counted failures for a day to alert
REQUEST_RATIO = 2      # times the busiest of the days before, which counts as at least 1
CUT_FLOOR = 5          # responses that stopped at the token limit or refused
CUT_SHARE = 0.005      # of that day's main-thread responses
CUT_RATIO = 3          # times the worst share of the days before, floored at CUT_USUAL
CUT_USUAL = 0.001      # the share a quiet day is taken to have, so one cut day can't mask the next
ACTIVE_RESPONSES = 50  # main-thread responses for a day to count as active
BEFORE_DAYS = 14
MIN_BEFORE_DAYS = 5
RECENT_DAYS = 14


def judged_failures(failures: pd.DataFrame, today: date) -> pd.DataFrame:
    """Failed requests of complete UTC days, without Agent SDK sessions. Subagent
    failures stay in: the report counts them, and only the rules leave them out."""
    if failures.empty:
        return failures
    return failures[(failures["day"].astype(str) < today.isoformat()) & outside_sdk(failures)]


def failure_counts(failures: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """Per day over the judged turns' days: main-thread responses, failures of each kind,
    the counted ones together, and the responses that stopped at the token limit or
    refused. Failures count wherever they happened — a request a subagent made is one
    Claude Code made — while responses and cut-short responses stay main-thread, as every
    other daily verdict is. A day without responses has no row, so it can neither alert
    nor stand as a quiet day before one."""
    if turns.empty:
        return pd.DataFrame(columns=FAILURE_DAY_COLUMNS)
    day = turns["day"].astype(str)
    stop = turns["stop_reason"].astype("string") if "stop_reason" in turns else pd.Series(pd.NA, index=turns.index)
    counts = pd.DataFrame({"responses": turns.groupby(day, sort=True).size(),
                           "truncated": (stop == "max_tokens").groupby(day).sum(),
                           "refused": (stop == "refusal").groupby(day).sum()})
    for kind in KINDS:
        if failures.empty:
            counts[kind] = 0
            continue
        chosen = failures[failures["kind"].astype(str) == kind]
        counts[kind] = chosen.groupby(chosen["day"].astype(str)).size().reindex(counts.index, fill_value=0)
    counts["requests"] = sum(counts[kind] for kind in COUNTED)
    counts = counts.rename_axis("day").reset_index()
    return counts[FAILURE_DAY_COLUMNS].astype({name: int for name in FAILURE_DAY_COLUMNS[1:]})


def _judged_days(counts: pd.DataFrame, state_key: str, state: dict[str, Any], today: date,
                 hit: Callable[[pd.Series, pd.DataFrame], bool]) -> list[pd.Series]:
    """The active days within RECENT_DAYS that `hit` accepts against the active days
    before them, skipping those a reported episode already covers."""
    if counts.empty:
        return []
    active = counts[counts["responses"] >= ACTIVE_RESPONSES].reset_index(drop=True)
    days = active["day"].astype(str)
    since = (today - timedelta(days=RECENT_DAYS)).isoformat()
    found = []
    for i in range(len(active)):
        day = str(days[i])
        if day < since:
            continue
        earliest = (date.fromisoformat(day) - timedelta(days=BEFORE_DAYS)).isoformat()
        before = active[(days < day) & (days >= earliest)]
        if len(before) < MIN_BEFORE_DAYS or not hit(active.loc[i], before):
            continue
        # One spell of failures alerts once: a day already covered by an episode
        # reported for a day within BEFORE_DAYS before it is left alone.
        if any(episode["since"] >= earliest for episode in state[state_key]):
            continue
        found.append(active.loc[i])
    return found


def failing_requests(counts: pd.DataFrame, state: dict[str, Any], today: date,
                     floor: int = REQUEST_FLOOR, ratio: int = REQUEST_RATIO) -> list[dict[str, Any]]:
    """Days with at least `floor` failed requests, at least `ratio` times the busiest of
    the active days before them; each is recorded in state["failed_requests"], once per
    spell. The lab tries other floors and ratios; the check keeps the defaults."""
    def hit(row: pd.Series, before: pd.DataFrame) -> bool:
        return row["requests"] >= floor and row["requests"] >= ratio * max(1, int(before["requests"].max()))

    new = []
    for row in _judged_days(counts, "failed_requests", state, today, hit):
        day = str(row["day"])
        earliest = (date.fromisoformat(day) - timedelta(days=BEFORE_DAYS)).isoformat()
        days = counts["day"].astype(str)
        before = counts[(days < day) & (days >= earliest)]
        episode = {"since": day, "days": [day], "requests": int(row["requests"]),
                   "kinds": {kind: int(row[kind]) for kind in COUNTED if int(row[kind])},
                   "before": int(before["requests"].max()) if len(before) else 0,
                   "reported_on": today.isoformat()}
        state["failed_requests"].append(episode)
        new.append(episode)
    return new


def cut_short(counts: pd.DataFrame, state: dict[str, Any], today: date,
              floor: int = CUT_FLOOR, share: float = CUT_SHARE) -> list[dict[str, Any]]:
    """Days where at least `floor` responses stopped at the token limit or refused, on at
    least `share` of the day's responses and CUT_RATIO times the worst share of the
    active days before them (a quiet day counting as CUT_USUAL); each is recorded in
    state["cut_short"], once per spell. The lab tries other floors and shares; the check
    keeps the defaults."""
    def hit(row: pd.Series, before: pd.DataFrame) -> bool:
        cut = int(row["truncated"] + row["refused"])
        today_share = cut / int(row["responses"])
        usual = max(CUT_USUAL, ((before["truncated"] + before["refused"]) / before["responses"]).max())
        return cut >= floor and today_share >= share and today_share >= CUT_RATIO * usual

    new = []
    for row in _judged_days(counts, "cut_short", state, today, hit):
        episode = {"since": str(row["day"]), "days": [str(row["day"])],
                   "cut": int(row["truncated"] + row["refused"]), "truncated": int(row["truncated"]),
                   "refused": int(row["refused"]), "responses": int(row["responses"]),
                   "reported_on": today.isoformat()}
        state["cut_short"].append(episode)
        new.append(episode)
    return new


def _on(versions: Sequence[str]) -> str:
    return f", on Claude Code {', '.join(versions)}" if versions else ""


def requests_message(episode: dict[str, Any], versions: Sequence[str]) -> str:
    named = kinds_text(episode["kinds"])
    return (f"{episode['requests']} requests failed on {episode['since']}"
            f"{f' ({named})' if named else ''}, against at most {episode['before']} a day in the "
            f"{BEFORE_DAYS} days before{_on(versions)}. Claude Code retries these itself; a run of them points at "
            "the API or your connection, not your setup.")


def cut_short_message(episode: dict[str, Any], versions: Sequence[str]) -> str:
    what = "stopped at the token limit or refused" if episode["refused"] else "stopped at the token limit"
    share = episode["cut"] / episode["responses"] if episode["responses"] else 0.0
    return (f"{episode['cut']} of {episode['responses']:,} main-thread responses {what} on {episode['since']} "
            f"({share:.2%}), against under {CUT_USUAL:.2%} a day in the {BEFORE_DAYS} days before{_on(versions)}. "
            "A Claude Code update may have changed the output limit.")


def failure_summary(counts: pd.DataFrame, days: Sequence[str]) -> Optional[dict[str, Any]]:
    """Failures over `days`: how many of each kind, and how many responses were cut
    short; None when there were none."""
    window = counts[counts["day"].astype(str).isin(list(days))] if not counts.empty else counts
    if window.empty:
        return None
    kinds = {kind: int(window[kind].sum()) for kind in KINDS if int(window[kind].sum())}
    cut = int(window["truncated"].sum() + window["refused"].sum())
    if not kinds and not cut:
        return None
    return {"kinds": kinds, "cut": cut}


def failure_lines(summary: Optional[dict[str, Any]]) -> list[str]:
    """The report's failures line, starting with a blank line; empty without failures."""
    if summary is None:
        return []
    parts = [kinds_text(summary["kinds"])] if summary["kinds"] else []
    if summary["cut"]:
        parts.append(f"{summary['cut']} response{'' if summary['cut'] == 1 else 's'} cut short")
    return ["", "Failures over these days: " + ", ".join(parts)]


def digest_part(counts: pd.DataFrame, days: Sequence[str]) -> str:
    """The weekly summary's failures part: "no failed requests", or what there was."""
    summary = failure_summary(counts, days)
    counted = sum(count for kind, count in (summary or {"kinds": {}})["kinds"].items() if kind in COUNTED)
    if summary is None or (not counted and not summary["cut"]):
        return "no failed requests"
    parts = [f"{counted} failed request{'' if counted == 1 else 's'}"] if counted else []
    if summary["cut"]:
        parts.append(f"{summary['cut']} response{'' if summary['cut'] == 1 else 's'} cut short")
    return ", ".join(parts)
