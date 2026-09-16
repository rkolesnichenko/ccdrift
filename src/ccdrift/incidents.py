"""Incidents: a flag the daily check reported, followed until the metric recovers.

While an incident is open its days stay out of the metric's baseline, so each later
day is still judged against the days before it. In real logs a caching regression
kept missing 5-10% of prompt turns for three weeks, but against a rolling baseline
its z-scores were back to about 0 within 8 days."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ccdrift.detector import DetectorConfig, baseline_bins, bin_metrics, detect, flag_onsets, pooled_z

INCIDENT_METRICS = {"cache_ratio": "Cache read ratio on new prompts",
                    "haiku_fraction": "Haiku share on the main thread"}
SHORT_NAMES = {"cache_ratio": "cache", "haiku_fraction": "haiku"}
MOVES = {"cache_ratio": "down", "haiku_fraction": "up"}
# A first run stays quiet about flags from weeks ago but covers a week or so
# without a run.
RECENT_DAYS = 14
# Days pooled to judge recovery, and pooled windows inside the cutoff in a row that
# close an incident. On the real caching regression, single days closed it at
# Sep 4 "from Sep 2" although Sep 3 still missed; pooling 3 days closed it at
# Sep 6, "from Sep 4", when the fixed versions were in use.
RECOVERY_BINS = 3
PERSISTENT_DAYS = 30
EXCLUDING = ("open", "recovered")
OPEN_END = "9999-12-31"


@dataclass
class Event:
    """A change to an incident: `kind` is "flag", "recovered" or "persistent".
    `days` are the days whose Claude Code versions the alert names; `run` and `z`
    are a flag's days and their z-scores, for the log."""
    kind: str
    incident: dict
    days: list[str]
    run: list[str] = field(default_factory=list)
    z: list[float] = field(default_factory=list)


def exclusions(bins: Sequence[str], incidents: Sequence[dict]) -> dict[str, list[bool]]:
    """For each metric, whether each day in `bins` lies in one of its open or
    recovered incidents."""
    masks = {}
    for metric in INCIDENT_METRICS:
        spans = [(i["start"], i["end"] or OPEN_END) for i in incidents
                 if i["metric"] == metric and i["status"] in EXCLUDING]
        masks[metric] = [any(start <= day <= end for start, end in spans) for day in bins]
    return masks


def open_incident(incidents: Sequence[dict], metric: str) -> Optional[dict]:
    return next((i for i in incidents if i["metric"] == metric and i["status"] == "open"), None)


def _first_bin_from(bins: Sequence[str], day: str) -> int:
    return next((i for i, b in enumerate(bins) if b >= day), len(bins))


def update_incidents(turns: pd.DataFrame, state: dict, today: date, cfg: DetectorConfig) -> list[Event]:
    """Open, recover and expire incidents over judged turns, in day order, and return
    what changed; state["incidents"] is updated in place. After each change the
    detector runs again with the new exclusions, so an incident that opened and
    recovered while the check wasn't running is followed in order."""
    if turns.empty:
        return []
    metrics = bin_metrics(turns)
    bins = metrics["bin"].astype(str).tolist()
    incidents = state["incidents"]
    events: list[Event] = []
    for metric in INCIDENT_METRICS:
        # Each pass opens or closes an incident, each later than the one before.
        for _ in range(2 * len(bins) + 1):
            detected = detect(metrics, cfg, exclusions(bins, incidents))
            incident = open_incident(incidents, metric)
            if incident is not None:
                event = _settle(incident, metrics, bins, incidents, today, cfg)
                if event is None:
                    break
                events.append(event)
                continue
            found = _new_flag_run(detected, metric, bins, incidents, state["reported"], today)
            if found is None:
                break
            onset, end = found
            incident = {"metric": metric, "start": bins[onset], "end": None, "status": "open",
                        "source": "check", "closed_by": None, "recovered_from": None,
                        "opened_on": today.isoformat(), "closed_on": None, "versions": [], "cost": 0}
            incidents.append(incident)
            events.append(Event("flag", incident, days=bins[onset:onset + RECOVERY_BINS],
                                run=bins[onset:end + 1],
                                z=[float(z) for z in detected[f"{metric}__z"].iloc[onset:end + 1]]))
    return events


def _new_flag_run(detected: pd.DataFrame, metric: str, bins: list[str], incidents: Sequence[dict],
                  reported: dict, today: date) -> Optional[tuple[int, int]]:
    """The first and last bin of the earliest flag run to open an incident for: it
    starts within the last RECENT_DAYS days, overlaps no recorded incident of the
    metric, and holds no flag a version 1 state file reported."""
    flags = detected[f"{metric}__flag"].to_numpy(dtype=bool)
    since = (today - timedelta(days=RECENT_DAYS)).isoformat()
    spans = [(i["start"], i["end"] or OPEN_END) for i in incidents if i["metric"] == metric]
    legacy = set(reported.get(metric, []))
    for onset in flag_onsets(detected, metric):
        end = onset
        while end + 1 < len(flags) and flags[end + 1]:
            end += 1
        first, last = bins[onset], bins[end]
        if first < since or legacy.intersection(bins[onset:end + 1]):
            continue
        if any(start <= last and first <= stop for start, stop in spans):
            continue
        return onset, end
    return None


def _settle(incident: dict, metrics: pd.DataFrame, bins: list[str], incidents: Sequence[dict],
            today: date, cfg: DetectorConfig) -> Optional[Event]:
    """Close an open incident once its metric has recovered, or once it has lasted
    PERSISTENT_DAYS days; None while it stays open."""
    metric = incident["metric"]
    start = _first_bin_from(bins, incident["start"])
    vals = metrics[metric].to_numpy(dtype=float)
    mask = np.array(exclusions(bins, incidents)[metric], dtype=bool)
    baseline = [j for j in baseline_bins(start, mask, cfg.baseline_window) if not math.isnan(vals[j])]
    if len(baseline) >= cfg.min_baseline:
        cutoff = cfg.metric_z_thresholds.get(metric, cfg.z_threshold)
        valued = [i for i in range(start, len(bins)) if not math.isnan(vals[i])]
        normal_ends: list[int] = []
        for k in range(RECOVERY_BINS - 1, len(valued)):
            z = pooled_z(metrics, metric, valued[k - RECOVERY_BINS + 1:k + 1], baseline)
            normal = z > -cutoff if MOVES[metric] == "down" else z < cutoff
            normal_ends = normal_ends + [valued[k]] if normal else []
            if len(normal_ends) == RECOVERY_BINS:
                first = normal_ends[0]
                incident.update(status="recovered", closed_by="check", recovered_from=bins[first],
                                end=bins[first - 1], closed_on=today.isoformat())
                return Event("recovered", incident, days=[bins[i] for i in normal_ends])
    if (today - date.fromisoformat(incident["start"])).days >= PERSISTENT_DAYS:
        incident.update(status="persistent", closed_by="check",
                        end=(today - timedelta(days=1)).isoformat(), closed_on=today.isoformat())
        return Event("persistent", incident, days=bins[-RECOVERY_BINS:])
    return None


def versions_text(turns: pd.DataFrame, days: Sequence[str]) -> list[str]:
    """The Claude Code versions behind at least 20% of the judged responses on `days`,
    most responses first, at most 3, each with the day it first appears:
    "2.1.233 (since 08-16)"."""
    if "version" not in turns:
        return []
    known = turns.dropna(subset=["version"])
    on_days = known.loc[known["day"].astype(str).isin(list(days)), "version"]
    if on_days.empty:
        return []
    first_seen = known.groupby("version")["day"].min()
    shares = on_days.value_counts(normalize=True)
    return [f"{version} (since {str(first_seen[version])[5:]})"
            for version, share in shares.items() if share >= 0.2][:3]


def incident_cost(turns: pd.DataFrame, incident: dict, incidents: Sequence[dict], cfg: DetectorConfig) -> float:
    """What an incident cost beyond the days before it: for the cache ratio, the
    cache-creation tokens of missed prompt turns above the usual miss rate; for
    Haiku share, the Haiku responses above the usual share. The usual rate comes
    from the incident's baseline days."""
    if turns.empty:
        return 0.0
    days = turns["day"].astype(str)
    bins = sorted(days.unique())
    metric = incident["metric"]
    mask = np.array(exclusions(bins, incidents)[metric], dtype=bool)
    before = days.isin([bins[j] for j in baseline_bins(_first_bin_from(bins, incident["start"]), mask,
                                                       cfg.baseline_window)])
    during = days.between(incident["start"], incident["end"] or bins[-1])
    if metric == "cache_ratio":
        prompts = turns["prompt_within_ttl"].astype(bool)
        misses = turns["is_miss"].astype(bool)
        rate = misses[prompts & before].sum() / max(int((prompts & before).sum()), 1)
        total = 0.0
        for _, day in turns[prompts & during].groupby(days[prompts & during]):
            missed = day["is_miss"].astype(bool)
            if missed.any():
                writes = float(day.loc[missed, "cache_creation"].sum())
                total += writes * max(0.0, 1 - rate * len(day) / int(missed.sum()))
        return total
    haiku = turns["is_haiku"].astype(float)
    rate = haiku[before].sum() / max(int(before.sum()), 1)
    return float(sum(max(0.0, group.sum() - rate * len(group))
                     for _, group in haiku[during].groupby(days[during])))


def approx(value: float) -> str:
    """Two significant figures with a k, M or B suffix: 17_556_103 -> "18M"."""
    if value <= 0:
        return "0"
    rounded = float(f"{value:.2g}")
    for size, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if rounded >= size:
            return f"{rounded / size:g}{suffix}"
    return f"{rounded:g}"


def cost_text(metric: str, cost: float) -> str:
    """"~18M tokens re-cached" or "~120 extra Haiku responses"."""
    if metric == "cache_ratio":
        return f"~{approx(cost)} tokens re-cached" if cost > 0 else "no tokens re-cached"
    return f"~{approx(cost)} extra Haiku responses" if cost > 0 else "no extra Haiku responses"


def describe(event: Event, turns: pd.DataFrame, incidents: Sequence[dict],
             cfg: DetectorConfig) -> tuple[str, str, str, list[str]]:
    """The alert for an event as (kind, title, message, log lines). Refreshes the
    incident's versions and cost, which `ccdrift status` shows."""
    incident = event.incident
    metric, start = incident["metric"], incident["start"]
    label = INCIDENT_METRICS[metric]
    incident["versions"] = versions_text(turns, event.days)
    incident["cost"] = round(incident_cost(turns, incident, incidents, cfg))
    on = f", on Claude Code {', '.join(incident['versions'])}" if incident["versions"] else ""
    cost = cost_text(metric, incident["cost"])
    if event.kind == "flag":
        z = ", ".join(f"{v:+.1f}" for v in event.z)
        return ("flag", "ccdrift flag",
                f"{label} {MOVES[metric]} from {start}{on}. {cost[0].upper()}{cost[1:]} so far.",
                [f"days {', '.join(event.run)}; z = {z}"])
    if event.kind == "recovered":
        return ("recovered", "ccdrift: back to normal",
                f"{label} back to normal from {incident['recovered_from']}{on}. The incident from {start}: {cost}.",
                [])
    return ("persistent", "ccdrift: change persists",
            f"{label} still {MOVES[metric]} {PERSISTENT_DAYS} days after {start}. ccdrift now treats it as the "
            "new normal; `ccdrift incident list` has the details.", [])
