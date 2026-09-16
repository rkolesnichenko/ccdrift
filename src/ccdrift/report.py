"""The details behind an alert: recent days or Claude Code versions with their
metrics, the incidents ccdrift follows, and the settings Claude Code chose."""

from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ccdrift.changelog import changelog_path, load_changelog, release_notes
from ccdrift.detector import DetectorConfig, bin_metrics, detect
from ccdrift.history import HistoryError, load_history
from ccdrift.hooks import hooks_lines, hooks_summary, judged_hook_runs
from ccdrift.incidents import exclusions, incident_cost
from ccdrift.logs import judged_subagent_turns, judged_turns, no_transcripts_message
from ccdrift.sessions import session_starts
from ccdrift.settings import settings_lines, settings_summary, subagent_lines, subagent_summary
from ccdrift.state import load_state
from ccdrift.texts import INCIDENT_METRICS, SHORT_NAMES, approx, incident_line, version_key

COLUMNS = ["day", "responses", "cache_ratio", "cache_z", "haiku_share", "haiku_z", "flagged"]
VERSION_COLUMNS = ["version", "first_day", "last_day", "responses", "prompt_turns", "cache_ratio",
                   "miss_share", "haiku_share", "session_start", "compacts_at", "release_notes"]
REPORT_TOPICS = ("cache", "haiku", "effort", "context", "hooks", "subagents")
DEFAULT_DAYS = 21
Entry = tuple[dict, float]  # an incident and its cost


def daily_rows(turns: pd.DataFrame, days: int = DEFAULT_DAYS, cfg: Optional[DetectorConfig] = None,
               incidents: Sequence[dict] = ()) -> pd.DataFrame:
    """The last `days` days of judged turns, each judged against the days before it
    with incident days left out: responses, cache ratio and Haiku share with their
    z-scores, and the metrics flagged that day."""
    if turns.empty:
        return pd.DataFrame(columns=COLUMNS)
    metrics = bin_metrics(turns)
    excluded = exclusions(metrics["bin"].astype(str).tolist(), incidents)
    detected = detect(metrics, cfg or DetectorConfig(), excluded).tail(days)
    flagged = [", ".join(short for metric, short in SHORT_NAMES.items() if row[f"{metric}__flag"])
               for _, row in detected.iterrows()]
    return pd.DataFrame({
        "day": detected["bin"].astype(str).to_numpy(),
        "responses": detected["n_turns"].to_numpy(),
        "cache_ratio": detected["cache_ratio"].to_numpy(),
        "cache_z": detected["cache_ratio__z"].to_numpy(),
        "haiku_share": detected["haiku_fraction"].to_numpy(),
        "haiku_z": detected["haiku_fraction__z"].to_numpy(),
        "flagged": flagged,
    }, columns=COLUMNS)


def _median_for(table: Optional[pd.DataFrame], version: str, column: str) -> float:
    if table is None or table.empty:
        return math.nan
    values = table.loc[table["version"].fillna("unknown").astype(str) == version, column].dropna()
    return float(values.median()) if len(values) else math.nan


def _size(value: float) -> str:
    return "-" if math.isnan(value) else approx(value)


def version_rows(turns: pd.DataFrame, changelog: Optional[dict] = None, starts: Optional[pd.DataFrame] = None,
                 compactions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per Claude Code version on judged turns, oldest version first: first and last
    day, responses, new-prompt turns with their cache ratio and share of misses,
    Haiku share, median session-start size and pre-compaction size, and up to 2
    release notes on file for that version."""
    rows = []
    if not turns.empty:
        versions = (turns["version"].fillna("unknown") if "version" in turns
                    else pd.Series("unknown", index=turns.index))
        for version, group in turns.groupby(versions):
            prompts = group[group["prompt_within_ttl"].astype(bool)]
            rows.append({
                "version": str(version), "first_day": str(group["day"].min()), "last_day": str(group["day"].max()),
                "responses": len(group), "prompt_turns": len(prompts),
                "cache_ratio": float(prompts["cache_read_ratio"].mean()) if len(prompts) else math.nan,
                "miss_share": float(prompts["is_miss"].astype(bool).mean()) if len(prompts) else math.nan,
                "haiku_share": float(group["is_haiku"].mean()),
                "session_start": _median_for(starts, str(version), "prompt_tokens"),
                "compacts_at": _median_for(compactions, str(version), "pre_tokens"),
                "release_notes": [text for _, text in release_notes(changelog or {}, [str(version)], REPORT_TOPICS)],
            })
    rows.sort(key=lambda row: version_key(row["version"]))
    return pd.DataFrame(rows, columns=VERSION_COLUMNS)


def _number(value: Any, spec: str) -> str:
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def _cutoffs(cfg: DetectorConfig) -> tuple[float, float]:
    return (cfg.metric_z_thresholds.get("cache_ratio", cfg.z_threshold),
            cfg.metric_z_thresholds.get("haiku_fraction", cfg.z_threshold))


def _tail(entries: Sequence[Entry], reported: dict, summary: list[dict], extra: Sequence[str] = ()) -> list[str]:
    """What follows either table: incidents, flags from before incidents, settings,
    and whatever else the caller adds (hooks and subagents, on the day view)."""
    lines = [""]
    if entries:
        lines += ["Incidents:"] + [f"  {incident_line(incident, cost)}" for incident, cost in entries]
    else:
        lines.append("Incidents: none yet")
    legacy = [f"  {label} from {day}" for metric, label in INCIDENT_METRICS.items() for day in reported.get(metric, [])]
    if legacy:
        lines += ["", "Flags reported before ccdrift followed incidents:"] + legacy
    return lines + settings_lines(summary) + list(extra)


def format_report(rows: pd.DataFrame, entries: Sequence[Entry], reported: dict, summary: list[dict],
                  cfg: Optional[DetectorConfig] = None, extra: Sequence[str] = ()) -> str:
    cache_cutoff, haiku_cutoff = _cutoffs(cfg or DetectorConfig())
    cfg = cfg or DetectorConfig()
    lines = [
        f"Last {len(rows)} complete UTC days with main-thread activity.",
        f"Flagged once {cfg.deviant_bins} of any {cfg.flag_window} days in a row pass the cutoff: "
        f"z <= -{cache_cutoff:.1f} for the cache ratio, z >= +{haiku_cutoff:.1f} for Haiku share.",
        "",
        f"{'day':<10}  {'responses':>9}  {'cache ratio':>11}  {'z':>5}  {'haiku share':>11}  {'z':>5}  flagged",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"{row.day:<10}  {int(row.responses):>9}  {_number(row.cache_ratio, '.3f'):>11}  "
            f"{_number(row.cache_z, '+.1f'):>5}  {_number(row.haiku_share, '.3f'):>11}  "
            f"{_number(row.haiku_z, '+.1f'):>5}  {row.flagged}".rstrip())
    return "\n".join(lines + _tail(entries, reported, summary, extra)) + "\n"


def format_version_report(rows: pd.DataFrame, entries: Sequence[Entry], reported: dict,
                          summary: list[dict]) -> str:
    lines = [
        "Complete UTC days with main-thread activity, by Claude Code version.",
        "A miss is a new-prompt turn that reads less than half its input from the cache.",
        "",
        f"{'version':<11}  {'first day':<10}  {'last day':<10}  {'responses':>9}  {'prompt turns':>12}  "
        f"{'cache ratio':>11}  {'misses':>6}  {'haiku share':>11}"
        f"  {'session start':>13}  {'compacts at':>11}",
    ]
    for row in rows.itertuples(index=False):
        misses = "-" if math.isnan(row.miss_share) else f"{row.miss_share:.1%}"
        lines.append(
            f"{row.version:<11}  {row.first_day:<10}  {row.last_day:<10}  {int(row.responses):>9}  "
            f"{int(row.prompt_turns):>12}  {_number(row.cache_ratio, '.3f'):>11}  {misses:>6}  "
            f"{_number(row.haiku_share, '.3f'):>11}"
            f"  {_size(row.session_start):>13}  {_size(row.compacts_at):>11}")
        lines += [f"    release notes: {text}" for text in row.release_notes]
    return "\n".join(lines + _tail(entries, reported, summary)) + "\n"


def _plain(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    return value


def report_json(view: str, rows: pd.DataFrame, entries: Sequence[Entry], reported: dict,
                summary: list[dict], cfg: DetectorConfig, extra: Optional[dict] = None) -> str:
    """The report as JSON: aggregates only, no paths, session ids or project names."""
    cache_cutoff, haiku_cutoff = _cutoffs(cfg)
    records = [{key: _plain(value) for key, value in record.items()} for record in rows.to_dict("records")]
    if view == "day":
        for record in records:
            record["flagged"] = [name for name in record["flagged"].split(", ") if name]
    payload = {
        "view": view,
        "days" if view == "day" else "versions": records,
        "incidents": [{**incident, "cost": round(cost)} for incident, cost in entries],
        "reported_before_incidents": reported,
        "settings": summary,
        **(extra or {}),
        "cutoffs": {"cache_ratio": -cache_cutoff, "haiku_fraction": haiku_cutoff},
        "flag_rule": {"deviant_days": cfg.deviant_bins, "of_days": cfg.flag_window},
    }
    return json.dumps(payload, indent=1) + "\n"


def run_report(source: Path, state_path: Path, days: Optional[int] = None, by: str = "day",
               as_json: bool = False, today: Optional[date] = None,
               cfg: Optional[DetectorConfig] = None) -> int:
    try:
        state = load_state(state_path)
    except (OSError, ValueError) as exc:
        print(f"Can't read the state file {state_path}: {exc}", file=sys.stderr)
        return 1
    try:
        tables = load_history(source, state_path, claim=False)
    except HistoryError as exc:
        print(exc, file=sys.stderr)
        return 1
    df = tables.responses
    if df.empty:
        print(no_transcripts_message(source), file=sys.stderr)
        return 2
    cfg = cfg or DetectorConfig()
    today = today or datetime.now(timezone.utc).date()
    turns = judged_turns(df, today)
    starts = session_starts(df)
    starts = starts[starts["day"].astype(str) < today.isoformat()]
    compactions = tables.compactions
    if not compactions.empty:
        cli = ~compactions["entrypoint"].fillna("").astype(str).str.startswith("sdk-")
        compactions = compactions[cli & ~compactions["is_sidechain"] & (compactions["trigger"] == "auto")
                                  & (compactions["day"].astype(str) < today.isoformat())]
    incidents = state["incidents"]
    entries = [(incident, incident_cost(turns, incident, incidents, cfg))
               for incident in sorted(incidents, key=lambda i: i["start"], reverse=True)]
    if by == "version":
        if days is not None:
            recent = sorted(turns["day"].astype(str).unique())[-days:]
            turns = turns[turns["day"].astype(str).isin(recent)]
            starts = starts[starts["day"].astype(str).isin(recent)]
            if not compactions.empty:
                compactions = compactions[compactions["day"].astype(str).isin(recent)]
        changelog = load_changelog(changelog_path(source))
        rows = version_rows(turns, changelog, starts, compactions)
        window = sorted(turns["day"].astype(str).unique())
        extra_lines, extra_json = [], {}
    else:
        rows = daily_rows(turns, days or DEFAULT_DAYS, cfg, incidents)
        window = rows["day"].tolist()
        hooks = hooks_summary(judged_hook_runs(tables.hook_runs, today), window)
        subagents = subagent_summary(judged_subagent_turns(df, today), window)
        extra_lines = hooks_lines(hooks) + subagent_lines(subagents)
        extra_json = {"hooks": hooks, "subagents": subagents}
    summary = settings_summary(turns, window)
    if as_json:
        text = report_json(by, rows, entries, state["reported"], summary, cfg, extra_json)
    elif by == "version":
        text = format_version_report(rows, entries, state["reported"], summary)
    else:
        text = format_report(rows, entries, state["reported"], summary, cfg, extra=extra_lines)
    print(text, end="")
    return 0
