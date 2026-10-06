"""Subagent spawns judged: the model each Agent call asked for, the one Claude Code resolved
the request to, and the one the subagent's responses were served by.

A mismatch is a fact, not a statistic, so no threshold turns on anything here: one spawn
served a model other than the one resolved for it is enough to say so. On the owner's logs
(1,560 spawns joined to their responses, 2026-08-15 to 2026-10-06) there was none."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional, Sequence

import pandas as pd

from ccdrift.texts import version_key

# How far back, in days counting today, the hourly check looks for a mismatch it hasn't
# reported: a week, so one from while the machine was off still alerts, and older ones
# are left to the report. Not a measured cutoff: a mismatch needs no baseline.
ALERT_DAYS = 7

# A request that names no model of its own: the subagent takes the parent's.
UNJUDGED = frozenset({"inherit"})

JOINED_COLUMNS = ("agent_id", "requested", "resolved", "version", "day", "timestamp", "served", "responses")
MISMATCH_COLUMNS = ("kind", "requested", "resolved", "served", "version", "day")


def model_name(raw: Any) -> Optional[str]:
    """A model as spawns compare it: lowercase, with the 1M-context marker removed. On the
    owner's logs all 219 spawns resolved to "claude-opus-5[1m]" or "claude-opus-5-5[1m]"
    were served under the plain id."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip().lower().replace("[1m]", "")


def honours(asked: str, model: str) -> bool:
    """Whether `model` is what a request for `asked` (both as model_name gives them) should
    get: a full id ("claude-opus-5") exactly, since it is part of claude-opus-5-5 too; an
    alias ("opus") anywhere in it, so claude-opus-5 and claude-opus-5-5 both honour it."""
    if asked.startswith("claude-"):
        return model == asked
    return asked in model


def judge(requested: Any, resolved: Any, served: Sequence[str]) -> list[str]:
    """The ways one spawn went wrong: "not_honoured" when the call named a model and what
    Claude Code resolved isn't it (or, with nothing resolved logged, what was served
    isn't), "served_differs" when a response was served by a model other than the one
    resolved. A spawn with no served response is not judged at all."""
    if not served:
        return []
    asked, got = model_name(requested), model_name(resolved)
    kinds = []
    if asked is not None and asked not in UNJUDGED:
        if not all(honours(asked, model) for model in ([got] if got is not None else served)):
            kinds.append("not_honoured")
    if got is not None and any(model != got for model in served):
        kinds.append("served_differs")
    return kinds


def spawn_models(spawns: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    """Each spawn with the models it was served (the distinct model_name of the responses
    carrying its agent id, sorted) and how many responses it got. A spawn copied into a
    second transcript is already one row: parse_all and the store keep the copy whose
    path sorts first."""
    if spawns.empty:
        return pd.DataFrame(columns=list(JOINED_COLUMNS))
    served: dict[str, tuple[str, ...]] = {}
    counts: dict[str, int] = {}
    if not responses.empty and "agent_id" in responses:
        linked = responses[responses["agent_id"].notna()]
        for agent, group in linked.groupby("agent_id", sort=True):
            served[agent] = tuple(sorted({name for name in group["model"].map(model_name) if name is not None}))
            counts[agent] = len(group)
    out = spawns[["agent_id", "requested", "resolved", "version", "day", "timestamp"]].copy()
    out["served"] = out["agent_id"].map(lambda agent: served.get(agent, ()))
    out["responses"] = out["agent_id"].map(lambda agent: counts.get(agent, 0)).astype(int)
    return out.sort_values(["timestamp", "agent_id"], kind="stable").reset_index(drop=True)


def mismatches(joined: pd.DataFrame) -> pd.DataFrame:
    """One row per way each spawn of spawn_models went wrong (judge), with what it asked
    for ("none" when nothing), what was resolved and served, its version and day."""
    rows = []
    for spawn in joined.to_dict("records"):
        for kind in judge(spawn["requested"], spawn["resolved"], spawn["served"]):
            # pandas 3 reads a missing version as NaN, which would print as "nan".
            rows.append({"kind": kind, "requested": model_name(spawn["requested"]) or "none",
                         "resolved": model_name(spawn["resolved"]), "served": ", ".join(spawn["served"]),
                         "version": spawn["version"] if isinstance(spawn["version"], str) else None,
                         "day": spawn["day"]})
    return pd.DataFrame(rows, columns=list(MISMATCH_COLUMNS))


def alias_history(joined: pd.DataFrame, days: Sequence[str]) -> dict[str, Any]:
    """The spawns of `days`: per alias asked for ("none" when nothing), each model it
    resolved to with its first and last day, its versions and how many spawns; how many
    spawns there were, how many were judged (had a served response) and how many had
    none; and the mismatches among them."""
    window = joined[joined["day"].astype(str).isin(list(days))] if not joined.empty else joined
    aliases = []
    if not window.empty:
        keyed = window.assign(alias=window["requested"].map(lambda raw: model_name(raw) or "none"),
                              model=window["resolved"].map(lambda raw: model_name(raw) or "unresolved"))
        for alias, group in keyed.groupby("alias", sort=True):
            models = []
            for model, spawned in group.groupby("model", sort=True):
                versions = sorted({str(v) for v in spawned["version"].dropna()}, key=version_key)
                models.append({"model": str(model), "first": str(spawned["day"].min()), "last": str(spawned["day"].max()),
                               "versions": versions, "spawns": len(spawned)})
            aliases.append({"alias": str(alias), "models": sorted(models, key=lambda m: (m["first"], m["model"]))})
    found = mismatches(window)
    answered = int((window["responses"] > 0).sum()) if not window.empty else 0
    return {"aliases": aliases, "spawns": len(window), "judged": answered, "no_response": len(window) - answered,
            "mismatches": found.to_dict("records")}


def model_mismatch_alerts(joined: pd.DataFrame, state: dict[str, Any], today: date) -> list[dict[str, Any]]:
    """The mismatches among spawns of the ALERT_DAYS days ending `today` not yet in
    state["model_mismatches"], which each is added to. One is reported once per kind,
    request, resolved and served model, not once per spawn: a change that hits every spawn
    alerts once, and again only when the resolved or served model moves, not on every Claude
    Code version, since Claude Code ships near-daily and a configured override
    (CLAUDE_CODE_SUBAGENT_MODEL with CLAUDE_CODE_SUBAGENT_MODEL_FORCE) resolves every call
    to one model on purpose. The entry keeps the version it was first seen on."""
    since = (today - timedelta(days=ALERT_DAYS - 1)).isoformat()
    recent = joined[joined["day"].astype(str).between(since, today.isoformat())] if not joined.empty else joined
    reported = state["model_mismatches"]
    known = {(m["kind"], m["requested"], m["resolved"], m["served"]) for m in reported}
    new = []
    for row in mismatches(recent).to_dict("records"):
        key = (row["kind"], row["requested"], row["resolved"], row["served"])
        if key in known:
            continue
        known.add(key)
        entry = {"kind": row["kind"], "requested": row["requested"], "resolved": row["resolved"],
                 "served": row["served"], "version": row["version"], "first_day": str(row["day"]),
                 "reported_on": today.isoformat()}
        reported.append(entry)
        new.append(entry)
    return new
