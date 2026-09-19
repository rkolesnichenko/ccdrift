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
from ccdrift.state import load_state
from ccdrift.texts import PERSISTENT_DAYS, SHORT_NAMES, approx, version_key

AFTER_DAYS = 14  # judged days after an incident that the draft compares with
PAUSES = [(60, "≤1 min"), (300, "1–5 min"), (900, "5–15 min"), (3600, "15–60 min")]
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


def _rate(part: float, whole: float) -> str:
    return f"{part / whole:.2%}" if whole else "-"


def _span(days: Sequence[str]) -> str:
    return f"{days[0][5:]}..{days[-1][5:]}"


def _days(count: int) -> str:
    return f"{count} day{'' if count == 1 else 's'}"


def _on_days(frame: pd.DataFrame, days: Sequence[str]) -> pd.DataFrame:
    return frame[frame["day"].astype(str).isin(list(days))]


def _shares(values: pd.Series) -> list[tuple[str, float]]:
    """Each value with its share of `values` (nulls left out), largest first."""
    counts = values.dropna().astype(str).value_counts()
    return [(str(value), count / counts.sum()) for value, count in counts.items()]


TITLE_SHARE = 0.02  # of the turns during, for a version to be named in the title


def _version_span(versions: pd.Series) -> str:
    """The versions the title names: those behind at least TITLE_SHARE of the turns, as a
    range. A session left open on an old version runs a handful of turns weeks later, and
    naming it would widen the range past what the incident was about. The floor is low on
    purpose: the version an incident starts on may carry only a tenth of its turns, and
    dropping it would understate when the regression began."""
    known = versions.dropna().astype(str)
    if known.empty:
        return ""
    shares = known.value_counts(normalize=True)
    named = sorted((version for version, share in shares.items() if share >= TITLE_SHARE), key=version_key)
    if not named:
        named = sorted(shares.index.astype(str), key=version_key)
    return named[0] if len(named) == 1 else f"{named[0]}–{named[-1]}"


def _compared(parts: dict[str, tuple[int, int]], periods: dict[str, list[str]]) -> str:
    """", against 1 of 200 (0.50%) on the 5 days before and 2 of 566 (0.35%) on the 11
    days after", for the periods that have days."""
    clauses = [f"{parts[name][0]:,} of {parts[name][1]:,} ({_rate(*parts[name])}) on the "
               f"{_days(len(periods[name]))} {name}" for name in ("before", "after") if periods[name]]
    return f", against {' and '.join(clauses)}" if clauses else ""


def _lead(incident: dict[str, Any], periods: dict[str, list[str]]) -> str:
    if incident["status"] == "persistent":
        return f"From {incident['start']} to {incident['end']}, still changed after {PERSISTENT_DAYS} days"
    if incident["end"]:
        return f"From {incident['start']} to {incident['end']}"
    as_of = f" as of {periods['during'][-1]}" if periods["during"] else ""
    return f"From {incident['start']}, still going{as_of}"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows])


def _cache_sections(responses: pd.DataFrame, turns: pd.DataFrame, incident: dict[str, Any],
                    periods: dict[str, list[str]], cost: float) -> tuple[str, list[str]]:
    prompts = turns[turns["prompt_within_ttl"].astype(bool)]
    by_period = {name: _on_days(prompts, periods[name]) for name in PERIODS}
    counts = {name: (int(rows["is_miss"].astype(bool).sum()), len(rows)) for name, rows in by_period.items()}
    during = by_period["during"]
    span = _version_span(during["version"]) if "version" in during else ""
    usually = f" (usually {_rate(*counts['before'])})" if counts["before"][1] else ""
    title = (f"New prompts miss the prompt cache {_rate(*counts['during'])} of the time"
             + (f" on Claude Code {span}" if span else "") + usually)
    beyond = (f"~{approx(cost)} tokens were written to the cache again" if cost > 0
              else "no tokens were written to the cache again")
    sections = [
        "### What happened\n\n"
        f"{_lead(incident, periods)}, {counts['during'][0]:,} of {counts['during'][1]:,} main-thread turns that "
        f"open with a new prompt ({_rate(*counts['during'])}) missed the prompt cache"
        f"{_compared(counts, periods)}. ccdrift estimates {beyond} beyond the usual miss rate.",
        "### Before, during and after\n\nBefore is the baseline ccdrift judged the incident against: the days it "
        "compared with, which skip the days of other incidents, except ones dismissed or taken as "
        "the new normal, so they need not run up to the day it started.\n\n"
        + _table(
            ["", "Days", "New-prompt turns", "Misses", "Miss rate", "Cache read ratio"],
            [[f"{name.capitalize()} ({_span(periods[name])})", len(periods[name]), f"{counts[name][1]:,}",
              f"{counts[name][0]:,}", _rate(*counts[name]),
              f"{by_period[name]['cache_read_ratio'].mean():.3f}" if len(by_period[name]) else "-"]
             for name in PERIODS if periods[name]]),
    ]
    versions = _version_table(by_period, "is_miss", ["Turns", "Misses", "Miss rate"])
    if versions:
        sections.append("### By Claude Code version\n\n" + versions)
    missed = during[during["is_miss"].astype(bool)]
    if len(missed):
        read = missed["cache_read"].quantile([0.5, 0.25, 0.75]).round().astype(int).tolist()
        wrote = missed["cache_creation"].quantile([0.5, 0.25, 0.75]).round().astype(int).tolist()
        turns_text = f"{len(missed):,} missed turn{'' if len(missed) == 1 else 's'}"
        sections.append(
            "### What a missed turn looks like\n\n"
            f"The {turns_text} during read a median {read[0]:,} tokens from the cache (middle half "
            f"{read[1]:,}–{read[2]:,}) and wrote a median {wrote[0]:,} (middle half {wrote[1]:,}–{wrote[2]:,}), "
            "so each wrote most of its input to the cache again.")
    if len(during):
        rows, low = [], -1.0
        for high, label in PAUSES:
            pause = during[(during["gap_seconds"] > low) & (during["gap_seconds"] <= high)]
            misses = int(pause["is_miss"].astype(bool).sum())
            rows.append([label, f"{len(pause):,}", f"{misses:,}", _rate(misses, len(pause))])
            low = float(high)
        sections.append("### Pause before the prompt\n\nHow long the turn waited between the previous response and "
                        "the prompt that opened it.\n\n"
                        + _table(["Pause", "Turns", "Misses", "Miss rate"], rows))
    loops = loop_turns(responses, "main")
    loop_parts = []
    for name in PERIODS:
        rows = _on_days(loops, periods[name])
        if len(rows):
            misses = int(rows["is_loop_miss"].sum())
            loop_parts.append(f"{misses:,} of {len(rows):,} ({_rate(misses, len(rows))}) {name}")
    if loop_parts:
        joined = loop_parts[0] if len(loop_parts) == 1 else f"{', '.join(loop_parts[:-1])} and {loop_parts[-1]}"
        sections.append(f"### Tool-loop turns\n\nTurns inside the tool loop on the main thread missed {joined}.")
    return title, sections


def _version_table(by_period: dict[str, pd.DataFrame], column: str, names: Sequence[str]) -> str:
    """One row per Claude Code version and period that has rows, ordered by version
    (`version_key`) then by period order before, during, after: its rows, those where
    `column` holds, and their share; "" when no version is logged."""
    rows = []
    frames = [frame.assign(period=name) for name, frame in by_period.items() if len(frame)]
    if frames:
        both = pd.concat(frames)
        both = both[both["version"].notna()]
        for version in sorted(both["version"].astype(str).unique(), key=version_key):
            for name in PERIODS:
                group = both[(both["version"].astype(str) == version) & (both["period"] == name)]
                if len(group):
                    hits = int(group[column].astype(float).sum())
                    rows.append([version, name, f"{len(group):,}", f"{hits:,}", _rate(hits, len(group))])
    return _table(["Version", "Period", *names], rows) if rows else ""


def _haiku_sections(turns: pd.DataFrame, incident: dict[str, Any], periods: dict[str, list[str]],
                    cost: float) -> tuple[str, list[str]]:
    by_period = {name: _on_days(turns, periods[name]) for name in PERIODS}
    counts = {name: (int(rows["is_haiku"].sum()), len(rows)) for name, rows in by_period.items()}
    during = by_period["during"]
    span = _version_span(during["version"]) if "version" in during else ""
    usually = f" (usually {_rate(*counts['before'])})" if counts["before"][1] else ""
    title = (f"Haiku answers {_rate(*counts['during'])} of main-thread responses"
             + (f" on Claude Code {span}" if span else "") + usually)
    extra = f"~{approx(cost)} extra Haiku responses" if cost > 0 else "no extra Haiku responses"
    sections = [
        "### What happened\n\n"
        f"{_lead(incident, periods)}, Haiku answered {counts['during'][0]:,} of {counts['during'][1]:,} main-thread "
        f"responses ({_rate(*counts['during'])}){_compared(counts, periods)}: {extra} by ccdrift's estimate.",
        "### Before, during and after\n\nBefore is the baseline ccdrift judged the incident against: the days it "
        "compared with, which skip the days of other incidents, except ones dismissed or taken as "
        "the new normal, so they need not run up to the day it started.\n\n"
        + _table(
            ["", "Days", "Responses", "Haiku responses", "Haiku share"],
            [[f"{name.capitalize()} ({_span(periods[name])})", len(periods[name]), f"{counts[name][1]:,}",
              f"{counts[name][0]:,}", _rate(counts[name][0], counts[name][1])]
             for name in PERIODS if periods[name]]),
    ]
    versions = _version_table(by_period, "is_haiku", ["Responses", "Haiku responses", "Haiku share"])
    if versions:
        sections.append("### By Claude Code version\n\n" + versions)
    return title, sections


def os_text() -> str:
    """"macOS 26.5.2" on a Mac, else the system and its release."""
    if sys.platform == "darwin" and platform.mac_ver()[0]:
        return f"macOS {platform.mac_ver()[0]}"
    return f"{platform.system()} {platform.release()}".strip()


def _environment(during: pd.DataFrame, os_name: str) -> str:
    versions = sorted(during["version"].dropna().astype(str).unique(), key=version_key)
    entrypoints = _shares(during["entrypoint"]) if "entrypoint" in during and len(during) else []
    names = ", ".join(name for name, _ in entrypoints)
    entry = f" (entrypoint{'s' if len(entrypoints) > 1 else ''} {names})" if entrypoints else ""
    lines = [f"- Claude Code: {', '.join(versions)}{entry}" if versions else f"- Claude Code: version not logged{entry}"]
    models = _shares(during["model"]) if len(during) else []
    if models:
        lines.append("- Models during: " + ", ".join(f"{model} ({share:.2%} of responses)" for model, share in models))
    settings = []
    for column, name, suffix in (("cache_tier", "cache tier", " that write to the cache"), ("effort", "effort", "")):
        shares = _shares(during[column]) if column in during and len(during) else []
        settings.append(f"{name} {shares[0][0]} on {shares[0][1]:.2%} of responses{suffix}" if shares
                        else f"{name} not logged")
    lines += [f"- Main thread: {', '.join(settings)}", f"- OS: {os_name}",
              f"- Measured with ccdrift {__version__} from local session transcripts (aggregates only)"]
    return "### Environment\n\n" + "\n".join(lines)


def _method(metric: str, cfg: DetectorConfig) -> str:
    cutoff = cfg.metric_z_thresholds.get(metric, cfg.z_threshold)
    rule = (f"an incident opens when {cfg.deviant_bins} of {cfg.flag_window} days in a row fall "
            f"{'below z = −' if metric == 'cache_ratio' else 'above z = +'}{cutoff:.1f} and closes once "
            f"{RECOVERY_BINS} pooled days are back inside the cutoff on {RECOVERY_BINS} days in a row, or after "
            f"{PERSISTENT_DAYS} days, when it takes the change as the new normal.")
    if metric == "cache_ratio":
        counted = ("It counts main-thread turns that open with a new prompt within an hour of the previous response, "
                   "outside Agent SDK sessions and not right after a compaction. A turn misses the cache when it "
                   "reads less than half of its input from it.")
    else:
        counted = ("It counts main-thread responses outside Agent SDK sessions and the share answered by a Haiku "
                   "model.")
    return ("### How this was measured\n\nccdrift reads Claude Code's local session transcripts. " + counted
            + " A day is a UTC day, and only complete ones are judged. Each day is compared with the median of up "
            f"to {cfg.baseline_window} days before it, in units of their spread, which never falls below the noise "
            "a day of that many turns shows anyway; those days skip the days of other incidents, except ones "
            "dismissed or taken as the new normal. Then "
            + rule)


def draft_markdown(responses: pd.DataFrame, incident: dict[str, Any], incidents: Sequence[dict[str, Any]],
                   changelog: dict[str, list[str]], today: date, cfg: DetectorConfig,
                   os_name: str) -> Optional[str]:
    """The draft: a title line, a blank line and the body's sections. None when the
    history holds no judged days during the incident."""
    turns = judged_turns(responses, today)
    periods = draft_periods(turns, incident, incidents, cfg)
    if not periods["during"]:
        return None
    cost = incident_cost(turns, incident, incidents, cfg)
    if incident["metric"] == "cache_ratio":
        title, sections = _cache_sections(responses, turns, incident, periods, cost)
    else:
        title, sections = _haiku_sections(turns, incident, periods, cost)
    versions = incident["versions"] or incident_versions(turns, incident)
    first_days = periods["during"][:RECOVERY_BINS]
    quoted = note_versions(turns, versions, days_before(incident["start"], 7),
                           first_days[-1] if first_days else incident["start"])
    notes = release_notes(changelog, quoted, TOPIC_OF[incident["metric"]])
    if notes:
        sections.append("### Release notes that may be related\n\n"
                        + "\n".join(f"- {version}: {text}" for version, text in notes))
    sections += [_environment(_on_days(turns, periods["during"]), os_name), _method(incident["metric"], cfg)]
    return "\n\n".join([title, *sections]) + "\n"


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
        print(f"Can't read the state file {state_path}: {exc}", file=sys.stderr)
        return 1
    incident = find_incident(incidents, metric, start)
    if incident is None:
        name = SHORT_NAMES[metric]
        print(f"No {name} incident starts on {start}." if start else f"No {name} incident is recorded.",
              file=sys.stderr)
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
        print(f"The history holds no judged days during the {SHORT_NAMES[metric]} incident from {incident['start']}.",
              file=sys.stderr)
        return 2
    print(text, end="")
    return 0
