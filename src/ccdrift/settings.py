"""Settings Claude Code chooses for the main thread: which prompt cache it writes to
(1 hour or 5 minutes) and the effort level. A change in either alters cost or answers
without a word in the release notes; speed and service tier are shown in the report
only, since the user usually changes those. Each setting is judged per model and for
the main thread as a whole, since a new model can arrive with a setting of its own:
on 2026-09-22 the main thread moved from claude-opus-5 at xhigh effort to
claude-opus-5-5 at high, and judged per model neither had days on both sides."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Iterator, Sequence

import pandas as pd

from ccdrift.texts import NOT_LOGGED

ALERT_SETTINGS = ("cache_tier", "effort")
REPORT_SETTINGS = ("cache_tier", "effort", "speed", "service_tier")
ACTIVE_RESPONSES = 20  # responses with a value, a model's or the main thread's, for a day to count
BASELINE_DAYS = 14
MIN_BASELINE_DAYS = 5
USUAL_SHARE = 0.9      # the usual value is on at least this share of baseline responses
CHANGED_SHARE = 0.5    # a change: the usual value is on under this share, two days in a row
RECENT_DAYS = 14
# Judged for the main thread as a whole with the same numbers, over the owner's 23,580 judged
# turns from 2026-08-06 to 09-27 this adds one alert, effort xhigh -> high from 2026-09-23, and
# no other on either setting; per model there were none. See docs/findings.md.
THREAD = "thread"


def _same_change(record: dict[str, Any], change: dict[str, Any]) -> bool:
    """Whether `record` reported `change` already: the same setting and values, for the same
    model or with either one judged for the main thread, which a change on one model also is."""
    either_thread = THREAD in (record.get("scope"), change.get("scope"))
    return (record["setting"] == change["setting"] and record["from"] == change["from"]
            and record["to"] == change["to"] and (either_thread or record["model"] == change["model"]))


def _already_reported(state: dict[str, Any], change: dict[str, Any]) -> bool:
    earliest = (date.fromisoformat(change["since"]) - timedelta(days=RECENT_DAYS)).isoformat()
    return any(_same_change(c, change) and c["since"] >= earliest for c in state["settings"])


def _changes(group: pd.DataFrame, setting: str, since: str) -> Iterator[dict[str, Any]]:
    """The changes in `setting` on `group`'s judged turns whose first day is `since` or
    later: two active days in a row on which the value usual before them is under
    CHANGED_SHARE, each with the days before them it was judged against."""
    counts = group.groupby([group["day"].astype(str), setting]).size().unstack(fill_value=0)
    active = counts[counts.sum(axis=1) >= ACTIVE_RESPONSES]
    days = [str(d) for d in active.index]
    for i in range(1, len(days)):
        if days[i - 1] < since:
            continue
        baseline = active.iloc[max(0, i - 1 - BASELINE_DAYS):i - 1]
        if len(baseline) < MIN_BASELINE_DAYS:
            continue
        totals = baseline.sum()
        usual = str(totals.idxmax())
        if totals[usual] < USUAL_SHARE * totals.sum():
            continue
        pair = active.iloc[i - 1:i + 1]
        if ((pair[usual] / pair.sum(axis=1)) >= CHANGED_SHARE).any():
            continue
        # Under half is still the most common value when three or more share the day.
        yield {"from": usual, "to": str(pair.iloc[1].drop(usual).idxmax()), "since": days[i - 1],
               "days": days[i - 1:i + 1], "baseline": [str(d) for d in baseline.index]}


def _top_model(group: pd.DataFrame, days: Sequence[str]) -> str:
    """The model with the most of `group`'s responses on `days`, the first by name on a tie."""
    counts = group.loc[group["day"].astype(str).isin(list(days)), "model"].astype(str).value_counts()
    return str(counts.sort_index(kind="stable").sort_values(ascending=False, kind="stable").index[0])


def setting_changes(turns: pd.DataFrame, state: dict[str, Any], today: date) -> list[dict[str, Any]]:
    """New changes in the cache tier or effort a model usually gets on judged turns, then
    the main thread as a whole gets, earliest first; each is also recorded in
    state["settings"]. A main-thread change names the model most of its responses came
    from before it and on its two days, and a change already reported either way, for
    the model or the main thread, isn't reported again."""
    since = (today - timedelta(days=RECENT_DAYS)).isoformat()
    new: list[dict[str, Any]] = []
    for setting in ALERT_SETTINGS:
        if setting not in turns:
            continue
        valued = turns.dropna(subset=[setting])
        found = [({"setting": setting, "model": str(model)}, change)
                 for model, group in valued.groupby("model", sort=True) for change in _changes(group, setting, since)]
        found += [({"setting": setting, "scope": THREAD, "model": None}, change)
                  for change in _changes(valued, setting, since)]
        for key, change in found:
            baseline = change.pop("baseline")
            record = {**key, **change}
            if key.get("scope") == THREAD:
                record["from_model"] = _top_model(valued, baseline)
                record["to_model"] = _top_model(valued, change["days"])
            record["reported_on"] = today.isoformat()
            if _already_reported(state, record):
                continue
            state["settings"].append(record)
            new.append(record)
    return sorted(new, key=lambda c: c["since"])


def settings_summary(turns: pd.DataFrame, days: Sequence[str]) -> list[dict[str, Any]]:
    """For each model on the judged turns of `days`: the share of each value of each
    setting ("not logged" where Claude Code didn't log it; cache tier only over
    responses that wrote to the cache), and the days whose most common value
    differs from the previous active day's."""
    window = turns[turns["day"].astype(str).isin(list(days))]
    summary = []
    for model, group in window.groupby("model", sort=True):
        shares: dict[str, dict[str, float]] = {}
        changes = []
        for setting in REPORT_SETTINGS:
            column = group[setting] if setting in group else pd.Series(None, index=group.index, dtype=object)
            values = column.dropna() if setting == "cache_tier" else column.fillna(NOT_LOGGED)
            shares[setting] = {str(v): round(float(s), 3) for v, s in values.value_counts(normalize=True).items()}
            by_day = values.groupby(group.loc[values.index, "day"].astype(str))
            common = by_day.agg(lambda s: s.value_counts().idxmax())
            active = [d for d in sorted(common.index) if by_day.size()[d] >= ACTIVE_RESPONSES]
            changes += [{"day": cur, "setting": setting, "from": str(common[prev]), "to": str(common[cur])}
                        for prev, cur in zip(active, active[1:]) if common[prev] != common[cur]]
        summary.append({"model": str(model), "responses": len(group), "shares": shares,
                        "changes": sorted(changes, key=lambda c: (c["day"], REPORT_SETTINGS.index(c["setting"])))})
    return summary


def subagent_summary(sub_turns: pd.DataFrame, days: Sequence[str]) -> list[dict[str, Any]]:
    """Each agent type's model shares over the subagent turns of `days`."""
    window = sub_turns[sub_turns["day"].astype(str).isin(list(days))] if not sub_turns.empty else sub_turns
    summary = []
    for agent, group in window.groupby("agent_type", sort=True):
        summary.append({"agent_type": str(agent), "responses": len(group),
                        "models": {str(m): round(float(s), 3) for m, s in group["model"].value_counts(normalize=True).items()}})
    return summary
