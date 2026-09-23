"""A draft Claude Code issue about an incident: what happened before, during and after
it, by version, what a missed turn looks like, the release notes that may be related,
the environment and how ccdrift measured it, as aggregates only: no paths, project
names, session ids or prompt text. The owner wrote the August 2026 caching regression
up by hand from the lab; `ccdrift incident draft` prints that write-up for any
incident."""

from __future__ import annotations

import platform
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ccdrift import __version__
from ccdrift.changelog import (TOPIC_OF, changelog_path, days_before, load_changelog, note_versions,
                               release_notes)
from ccdrift.detector import DetectorConfig, baseline_bins
from ccdrift.history import HistoryError, load_history
from ccdrift.incidents import OPEN_END, RECOVERY_BINS, exclusions, incident_cost, incident_versions
from ccdrift.logs import judged_turns, no_transcripts_message
from ccdrift.loops import loop_turns
from ccdrift.report import reason_counts
from ccdrift.state import load_state
from ccdrift.texts import COMMAND_LINES, DRAFT_LINES, DRAFT_SETTINGS, SHORT_NAMES, draft_text, version_key

AFTER_DAYS = 14  # judged days after an incident that the draft compares with
PAUSE_BOUNDS = (60, 300, 900, 3600)  # seconds before the prompt: the upper bound of each bucket
PERIODS = ("before", "during", "after")


def find_incident(incidents: Sequence[dict[str, Any]], metric: str,
                  start: Optional[str] = None) -> Optional[dict[str, Any]]:
    """The incident of `metric` that starts on `start`, one not dismissed first; without
    `start`, the latest one not dismissed. None when there is none."""
    matching = [i for i in incidents if i["metric"] == metric and (start is None or i["start"] == start)]
    kept = [i for i in matching if i["status"] != "dismissed"]
    if start is None:
        return max(kept, key=lambda i: i["start"]) if kept else None
    return (kept or matching or [None])[0]


def draft_periods(turns: pd.DataFrame, incident: dict[str, Any], incidents: Sequence[dict[str, Any]],
                  cfg: DetectorConfig) -> dict[str, list[str]]:
    """The judged days before the incident (the baseline its cost compares with), during
    it (through the last judged day while it is open) and, for one with an end that
    isn't persistent, up to AFTER_DAYS after it and before the start of the next
    incident of the same metric that isn't dismissed; an open or persistent incident
    has none."""
    days = sorted(turns["day"].astype(str).unique()) if not turns.empty else []
    first = next((k for k, day in enumerate(days) if day >= incident["start"]), len(days))
    mask = np.array(exclusions(days, incidents)[incident["metric"]], dtype=bool)
    end = incident["end"] or OPEN_END
    after: list[str] = []
    if incident["end"] and incident["status"] != "persistent":
        later = [i["start"] for i in incidents if i["metric"] == incident["metric"] and i["status"] != "dismissed"
                 and i["start"] > incident["end"]]
        limit = min(later) if later else None
        after = [day for day in days if incident["end"] < day and (limit is None or day < limit)][:AFTER_DAYS]
    return {"before": [days[j] for j in baseline_bins(first, mask, cfg.baseline_window)],
            "during": [day for day in days if incident["start"] <= day <= end],
            "after": after}


def _on_days(frame: pd.DataFrame, days: Sequence[str]) -> pd.DataFrame:
    return frame[frame["day"].astype(str).isin(list(days))]


def _shares(values: pd.Series) -> list[tuple[str, float]]:
    """Each value with its share of `values` (nulls left out), largest first."""
    counts = values.dropna().astype(str).value_counts()
    return [(str(value), count / counts.sum()) for value, count in counts.items()]


TITLE_SHARE = 0.02  # of the turns during, for a version to be named in the title


def title_versions(versions: pd.Series) -> list[str]:
    """The versions the title names, oldest first: those behind at least TITLE_SHARE of the
    turns. A session left open on an old version runs a handful of turns weeks later, and
    naming it would widen the range past what the incident was about. The floor is low on
    purpose: the version an incident starts on may carry only a tenth of its turns, and
    dropping it would understate when the regression began."""
    known = versions.dropna().astype(str)
    if known.empty:
        return []
    shares = known.value_counts(normalize=True)
    named = sorted((version for version, share in shares.items() if share >= TITLE_SHARE), key=version_key)
    if not named:
        named = sorted(shares.index.astype(str), key=version_key)
    return [named[0]] if len(named) == 1 else [named[0], named[-1]]


def _version_rows(by_period: dict[str, pd.DataFrame], column: str) -> list[tuple[str, str, int, int]]:
    """(version, period, rows, rows where `column` holds) for each Claude Code version and
    period that has rows, ordered by version (`version_key`) then period."""
    rows = []
    frames = [frame.assign(period=name) for name, frame in by_period.items() if len(frame)]
    if frames:
        both = pd.concat(frames)
        both = both[both["version"].notna()]
        for version in sorted(both["version"].astype(str).unique(), key=version_key):
            for name in PERIODS:
                group = both[(both["version"].astype(str) == version) & (both["period"] == name)]
                if len(group):
                    rows.append((version, name, len(group), int(group[column].astype(float).sum())))
    return rows


def _cache_facts(responses: pd.DataFrame, turns: pd.DataFrame, periods: dict[str, list[str]]) -> dict[str, Any]:
    prompts = turns[turns["prompt_within_ttl"].astype(bool)]
    by_period = {name: _on_days(prompts, periods[name]) for name in PERIODS}
    during = by_period["during"]
    facts: dict[str, Any] = {
        "counts": {name: (int(rows["is_miss"].astype(bool).sum()), len(rows)) for name, rows in by_period.items()},
        "span": title_versions(during["version"]) if "version" in during else [],
        "ratios": {name: float(rows["cache_read_ratio"].mean()) if len(rows) else None
                   for name, rows in by_period.items()},
        "versions": _version_rows(by_period, "is_miss"),
        "missed": None, "reasons": None, "pauses": None, "loops": []}
    missed = during[during["is_miss"].astype(bool)]
    if len(missed):
        facts["missed"] = {"turns": len(missed),
                           "read": missed["cache_read"].quantile([0.5, 0.25, 0.75]).round().astype(int).tolist(),
                           "wrote": missed["cache_creation"].quantile([0.5, 0.25, 0.75]).round().astype(int).tolist()}
    # Counted over every judged response of the period, not over the new-prompt turns
    # the rest of this draft is about: Claude Code records a reason on any response
    # whose prompt did not match what it had cached.
    over = {name: _on_days(turns, periods[name]) for name in PERIODS}
    reasons = {name: reason_counts(rows) for name, rows in over.items()}
    if reasons["during"]:
        shown = [name for name in PERIODS if periods[name]]
        named = sorted({reason for counts in reasons.values() for reason in counts},
                       key=lambda reason: (-reasons["during"].get(reason, 0), reason))
        facts["reasons"] = {"shown": shown,
                            "rows": [(reason, [(reasons[name].get(reason, 0), len(over[name])) for name in shown])
                                     for reason in named]}
    if len(during):
        pauses, low = [], -1.0
        for high in PAUSE_BOUNDS:
            pause = during[(during["gap_seconds"] > low) & (during["gap_seconds"] <= high)]
            pauses.append((high, len(pause), int(pause["is_miss"].astype(bool).sum())))
            low = float(high)
        facts["pauses"] = pauses
    loops = loop_turns(responses, "main")
    for name in PERIODS:
        rows = _on_days(loops, periods[name])
        if len(rows):
            facts["loops"].append((name, int(rows["is_loop_miss"].sum()), len(rows)))
    return facts


def _haiku_facts(turns: pd.DataFrame, periods: dict[str, list[str]]) -> dict[str, Any]:
    by_period = {name: _on_days(turns, periods[name]) for name in PERIODS}
    during = by_period["during"]
    return {"counts": {name: (int(rows["is_haiku"].sum()), len(rows)) for name, rows in by_period.items()},
            "span": title_versions(during["version"]) if "version" in during else [],
            "versions": _version_rows(by_period, "is_haiku")}


def os_text() -> str:
    """"macOS 26.5.2" on a Mac, else the system and its release."""
    if sys.platform == "darwin" and platform.mac_ver()[0]:
        return DRAFT_LINES["macos"].format(version=platform.mac_ver()[0])
    return DRAFT_LINES["system"].format(system=platform.system(), release=platform.release()).strip()


def _environment_facts(during: pd.DataFrame, os_name: str) -> dict[str, Any]:
    entrypoints = _shares(during["entrypoint"]) if "entrypoint" in during and len(during) else []
    settings = {}
    for column, _, _ in DRAFT_SETTINGS:
        shares = _shares(during[column]) if column in during and len(during) else []
        settings[column] = shares[0] if shares else None
    return {"versions": sorted(during["version"].dropna().astype(str).unique(), key=version_key),
            "entrypoints": [name for name, _ in entrypoints], "models": _shares(during["model"]) if len(during) else [],
            "settings": settings, "os": os_name, "ccdrift": __version__}


def draft_facts(responses: pd.DataFrame, incident: dict[str, Any], incidents: Sequence[dict[str, Any]],
                changelog: dict[str, list[str]], today: date, cfg: DetectorConfig,
                os_name: str) -> Optional[dict[str, Any]]:
    """What the draft says, as numbers (texts.draft_text words them): the incident and its
    periods, the counts before, during and after, by version, and for a cache incident what
    a missed turn looks like, why the cache missed, the pause before the prompt and the
    tool-loop turns, then the release notes, the environment and the rule. None when the
    history holds no judged days during the incident."""
    turns = judged_turns(responses, today)
    periods = draft_periods(turns, incident, incidents, cfg)
    if not periods["during"]:
        return None
    metric = incident["metric"]
    facts = (_cache_facts(responses, turns, periods) if metric == "cache_ratio" else _haiku_facts(turns, periods))
    versions = incident["versions"] or incident_versions(turns, incident)
    first_days = periods["during"][:RECOVERY_BINS]
    quoted = note_versions(turns, versions, days_before(incident["start"], 7),
                           first_days[-1] if first_days else incident["start"])
    facts.update({"metric": metric, "incident": incident, "periods": periods,
                  "cost": incident_cost(turns, incident, incidents, cfg),
                  "notes": release_notes(changelog, quoted, TOPIC_OF[metric]),
                  "environment": _environment_facts(_on_days(turns, periods["during"]), os_name),
                  "method": {"metric": metric, "bins": cfg.deviant_bins, "window": cfg.flag_window,
                             "cutoff": cfg.metric_z_thresholds.get(metric, cfg.z_threshold),
                             "recovery": RECOVERY_BINS, "baseline": cfg.baseline_window}})
    return facts


def draft_markdown(responses: pd.DataFrame, incident: dict[str, Any], incidents: Sequence[dict[str, Any]],
                   changelog: dict[str, list[str]], today: date, cfg: DetectorConfig,
                   os_name: str) -> Optional[str]:
    """The draft: a title line, a blank line and the body's sections. None when the
    history holds no judged days during the incident."""
    facts = draft_facts(responses, incident, incidents, changelog, today, cfg, os_name)
    return None if facts is None else draft_text(facts)


def run_draft(source: Path, state_path: Path, metric: str, start: Optional[str] = None,
              today: Optional[date] = None, cfg: Optional[DetectorConfig] = None,
              os_name: Optional[str] = None) -> int:
    """Print a GitHub issue draft about the incident of `metric` starting on `start`, or
    the latest, from the history. It saves no state and creates or claims no history
    store; like `ccdrift report`, it brings a store the check has claimed up to date.
    Nothing is sent."""
    try:
        incidents = load_state(state_path)["incidents"]
    except (OSError, ValueError) as exc:
        print(COMMAND_LINES["state_unreadable"].format(path=state_path, error=exc), file=sys.stderr)
        return 1
    incident = find_incident(incidents, metric, start)
    if incident is None:
        name = SHORT_NAMES[metric]
        print(DRAFT_LINES["no_incident_on"].format(name=name, start=start) if start
              else DRAFT_LINES["no_incident"].format(name=name), file=sys.stderr)
        return 2
    try:
        responses = load_history(source, state_path, claim=False).responses
    except HistoryError as exc:
        print(exc, file=sys.stderr)
        return 1
    if responses.empty:
        print(no_transcripts_message(source), file=sys.stderr)
        return 2
    today = today or datetime.now(timezone.utc).date()
    text = draft_markdown(responses, incident, incidents, load_changelog(changelog_path(source)), today,
                          cfg or DetectorConfig(), os_name or os_text())
    if text is None:
        print(DRAFT_LINES["no_days"].format(name=SHORT_NAMES[metric], start=incident["start"]), file=sys.stderr)
        return 2
    print(text, end="")
    return 0
