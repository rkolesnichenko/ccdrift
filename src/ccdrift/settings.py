"""Settings Claude Code chooses for the main thread, per model: which prompt cache
it writes to (1 hour or 5 minutes) and the effort level. A change in either alters
cost or answers without a word in the release notes; speed and service tier are
shown in the report only, since the user usually changes those."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Sequence

import pandas as pd

from ccdrift.texts import NOT_LOGGED, change_line

ALERT_SETTINGS = ("cache_tier", "effort")
REPORT_SETTINGS = ("cache_tier", "effort", "speed", "service_tier")
ACTIVE_RESPONSES = 20  # a model's responses with a value, for a day to count
BASELINE_DAYS = 14
MIN_BASELINE_DAYS = 5
USUAL_SHARE = 0.9      # the usual value is on at least this share of baseline responses
CHANGED_SHARE = 0.5    # a change: the usual value is on under this share, two days in a row
RECENT_DAYS = 14


def _already_reported(state: dict[str, Any], change: dict[str, Any]) -> bool:
    earliest = (date.fromisoformat(change["since"]) - timedelta(days=RECENT_DAYS)).isoformat()
    return any(c["setting"] == change["setting"] and c["model"] == change["model"]
               and c["from"] == change["from"] and c["to"] == change["to"] and c["since"] >= earliest
               for c in state["settings"])


def setting_changes(turns: pd.DataFrame, state: dict[str, Any], today: date) -> list[dict[str, Any]]:
    """New changes in the cache tier or effort a model usually gets on judged turns,
    earliest first; each is also recorded in state["settings"]."""
    since = (today - timedelta(days=RECENT_DAYS)).isoformat()
    new: list[dict[str, Any]] = []
    for setting in ALERT_SETTINGS:
        if setting not in turns:
            continue
        valued = turns.dropna(subset=[setting])
        for model, group in valued.groupby("model", sort=True):
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
                change = {"setting": setting, "model": str(model), "from": usual,
                          "to": str(pair.iloc[1].drop(usual).idxmax()), "since": days[i - 1],
                          "days": days[i - 1:i + 1], "reported_on": today.isoformat()}
                if _already_reported(state, change):
                    continue
                state["settings"].append(change)
                new.append(change)
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
