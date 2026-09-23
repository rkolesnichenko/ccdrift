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
from ccdrift.failures import failure_counts, failure_lines, failure_summary, judged_failures
from ccdrift.hooks import hooks_lines, hooks_summary, judged_hook_runs
from ccdrift.incidents import exclusions, incident_cost
from ccdrift.logs import judged_subagent_turns, judged_turns, no_transcripts_message, outside_sdk
from ccdrift.loops import COUNT_COLUMNS, loop_counts
from ccdrift.sessions import MIN_SESSIONS, project_lines, project_summary, session_starts
from ccdrift.settings import settings_lines, settings_summary, subagent_lines, subagent_summary
from ccdrift.state import load_state, make_stream_private
from ccdrift.texts import (COMMAND_LINES, DAY_TABLE, INCIDENT_METRICS, REPORT_LINES, SHORT_NAMES, VERSION_TABLE,
                           incident_line, miss_reason_line, misses as _misses, number as _number, size_text,
                           table_header, table_row, version_key)

COLUMNS = ["day", "responses", "cache_ratio", "cache_z", "haiku_share", "haiku_z", *COUNT_COLUMNS, "flagged"]
VERSION_COLUMNS = ["version", "first_day", "last_day", "responses", "prompt_turns", "cache_ratio",
                   "miss_share", *COUNT_COLUMNS, "haiku_share", "session_start", "compacts_at", "miss_reasons",
                   "release_notes"]
REPORT_TOPICS = ("cache", "haiku", "effort", "context", "hooks", "subagents")
DEFAULT_DAYS = 21
Entry = tuple[dict, float]  # an incident and its cost


def _counts_for(loops: Optional[pd.DataFrame], keys: Sequence[str]) -> pd.DataFrame:
    """The tool-loop counts (see loops.loop_counts) for `keys`, 0 where there are none."""
    if loops is None or loops.empty:
        return pd.DataFrame(0, index=list(keys), columns=COUNT_COLUMNS)
    return loops.reindex(list(keys), fill_value=0)


def daily_rows(turns: pd.DataFrame, days: int = DEFAULT_DAYS, cfg: Optional[DetectorConfig] = None,
               incidents: Sequence[dict] = (), loops: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """The last `days` days of judged turns, each judged against the days before it
    with incident days left out: responses, cache ratio and Haiku share with their
    z-scores, tool-loop turns and misses from `loops` (loop_counts by day), and the
    metrics flagged that day."""
    if turns.empty:
        return pd.DataFrame(columns=COLUMNS)
    metrics = bin_metrics(turns)
    excluded = exclusions(metrics["bin"].astype(str).tolist(), incidents)
    detected = detect(metrics, cfg or DetectorConfig(), excluded).tail(days)
    flagged = [", ".join(short for metric, short in SHORT_NAMES.items() if row[f"{metric}__flag"])
               for _, row in detected.iterrows()]
    day_list = detected["bin"].astype(str).tolist()
    counts = _counts_for(loops, day_list)
    return pd.DataFrame({
        "day": day_list,
        "responses": detected["n_turns"].to_numpy(),
        "cache_ratio": detected["cache_ratio"].to_numpy(),
        "cache_z": detected["cache_ratio__z"].to_numpy(),
        "haiku_share": detected["haiku_fraction"].to_numpy(),
        "haiku_z": detected["haiku_fraction__z"].to_numpy(),
        **{column: counts[column].to_numpy() for column in COUNT_COLUMNS},
        "flagged": flagged,
    }, columns=COLUMNS)


def reason_counts(turns: pd.DataFrame) -> dict[str, int]:
    """How many of `turns` carry each cache-miss reason Claude Code recorded, largest
    first and ties broken by name: `value_counts` on tied counts is hash-ordered, not
    name-ordered, and every caller of this dict (the version table, the day and version
    JSON) relies on it already reading the same way on every run rather than re-sorting
    it itself."""
    if turns.empty or "miss_reason" not in turns:
        return {}
    counts = turns["miss_reason"].dropna().astype(str).value_counts()
    ordered = {str(reason): int(n) for reason, n in counts.items()}
    return dict(sorted(ordered.items(), key=lambda item: (-item[1], item[0])))


def reason_summary(turns: pd.DataFrame, days: Sequence[str]) -> Optional[dict[str, int]]:
    """The cache-miss reasons recorded over `days`; None when Claude Code recorded none."""
    window = turns[turns["day"].astype(str).isin(list(days))] if not turns.empty else turns
    counts = reason_counts(window)
    return counts or None


def reason_lines(summary: Optional[dict[str, int]]) -> list[str]:
    """The report's cache-miss reason line, starting with a blank line; empty when
    Claude Code recorded none. ccdrift counts these and does not judge them."""
    if summary is None:
        return []
    return ["", REPORT_LINES["reasons"].format(reasons=miss_reason_line(summary))]


def _median_for(table: Optional[pd.DataFrame], version: str, column: str, least: int = 1) -> float:
    """The median of `column` over the version's rows; NaN with fewer than `least`."""
    if table is None or table.empty:
        return math.nan
    values = table.loc[table["version"].fillna("unknown").astype(str) == version, column].dropna()
    return float(values.median()) if len(values) >= least else math.nan


def version_rows(turns: pd.DataFrame, changelog: Optional[dict] = None, starts: Optional[pd.DataFrame] = None,
                 compactions: Optional[pd.DataFrame] = None, loops: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per Claude Code version on judged turns, oldest version first: first and last
    day, responses, new-prompt turns with their cache ratio and share of misses,
    tool-loop turns and misses from `loops` (loop_counts by version), Haiku share,
    median session-start size (over MIN_SESSIONS or more sessions) and pre-compaction
    size, the cache-miss reasons recorded, and up to 2 release notes on file for that
    version."""
    rows = []
    if not turns.empty:
        versions = (turns["version"].fillna("unknown") if "version" in turns
                    else pd.Series("unknown", index=turns.index))
        counts = _counts_for(loops, [str(version) for version in versions.unique()])
        for version, group in turns.groupby(versions):
            prompts = group[group["prompt_within_ttl"].astype(bool)]
            rows.append({
                "version": str(version), "first_day": str(group["day"].min()), "last_day": str(group["day"].max()),
                "responses": len(group), "prompt_turns": len(prompts),
                "cache_ratio": float(prompts["cache_read_ratio"].mean()) if len(prompts) else math.nan,
                "miss_share": float(prompts["is_miss"].astype(bool).mean()) if len(prompts) else math.nan,
                **{column: int(counts.loc[str(version), column]) for column in COUNT_COLUMNS},
                "haiku_share": float(group["is_haiku"].mean()),
                "session_start": _median_for(starts, str(version), "prompt_tokens", MIN_SESSIONS),
                "compacts_at": _median_for(compactions, str(version), "pre_tokens"),
                "miss_reasons": reason_counts(group),
                "release_notes": [text for _, text in release_notes(changelog or {}, [str(version)], REPORT_TOPICS)],
            })
    rows.sort(key=lambda row: version_key(row["version"]))
    return pd.DataFrame(rows, columns=VERSION_COLUMNS)


def _cutoffs(cfg: DetectorConfig) -> tuple[float, float]:
    return (cfg.metric_z_thresholds.get("cache_ratio", cfg.z_threshold),
            cfg.metric_z_thresholds.get("haiku_fraction", cfg.z_threshold))


def _tail(entries: Sequence[Entry], reported: dict, summary: list[dict], extra: Sequence[str] = ()) -> list[str]:
    """What follows either table: incidents, flags from before incidents, settings,
    and whatever else the caller adds (hooks and subagents, on the day view)."""
    lines = [""]
    if entries:
        lines += [REPORT_LINES["incidents"]] + [f"  {incident_line(incident, cost)}" for incident, cost in entries]
    else:
        lines.append(REPORT_LINES["no_incidents"])
    legacy = [REPORT_LINES["legacy"].format(label=label, day=day)
              for metric, label in INCIDENT_METRICS.items() for day in reported.get(metric, [])]
    if legacy:
        lines += ["", REPORT_LINES["legacy_head"]] + legacy
    return lines + settings_lines(summary) + list(extra)


def format_report(rows: pd.DataFrame, entries: Sequence[Entry], reported: dict, summary: list[dict],
                  cfg: Optional[DetectorConfig] = None, extra: Sequence[str] = ()) -> str:
    cache_cutoff, haiku_cutoff = _cutoffs(cfg or DetectorConfig())
    cfg = cfg or DetectorConfig()
    lines = [
        REPORT_LINES["days"].format(days=len(rows)),
        REPORT_LINES["rule"].format(bins=cfg.deviant_bins, window=cfg.flag_window, cache=cache_cutoff,
                                    haiku=haiku_cutoff),
        "",
        table_header(DAY_TABLE),
    ]
    for row in rows.itertuples(index=False):
        lines.append(table_row(DAY_TABLE, (
            row.day, int(row.responses), _number(row.cache_ratio, ".3f"), _number(row.cache_z, "+.1f"),
            _number(row.haiku_share, ".3f"), _number(row.haiku_z, "+.1f"), _misses(row.loop_misses, row.loop_turns),
            _misses(row.subagent_loop_misses, row.subagent_loop_turns), row.flagged)))
    return "\n".join(lines + _tail(entries, reported, summary, extra)) + "\n"


def format_version_report(rows: pd.DataFrame, entries: Sequence[Entry], reported: dict,
                          summary: list[dict]) -> str:
    lines = [REPORT_LINES["versions"], REPORT_LINES["miss"], REPORT_LINES["loop_miss"], "", table_header(VERSION_TABLE)]
    for row in rows.itertuples(index=False):
        lines.append(table_row(VERSION_TABLE, (
            row.version, row.first_day, row.last_day, int(row.responses), int(row.prompt_turns),
            _number(row.cache_ratio, ".3f"), _number(row.miss_share, ".1%"),
            _misses(row.loop_misses, row.loop_turns, share=True),
            _misses(row.subagent_loop_misses, row.subagent_loop_turns, share=True), _number(row.haiku_share, ".3f"),
            size_text(row.session_start), size_text(row.compacts_at))))
        lines += [REPORT_LINES["release_note"].format(text=text) for text in row.release_notes]
        if row.miss_reasons:
            lines.append(REPORT_LINES["version_reasons"].format(reasons=miss_reason_line(row.miss_reasons)))
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


def _incident_days(entries: Sequence[Entry], days: Sequence[str], metric: str) -> list[str]:
    """The days shown that belong to a recorded incident on `metric`, and only that metric,
    since a Haiku incident accounts for nothing on the cache chart. The page shades them, so
    a dip already accounted for doesn't read as news.

    A dismissed incident shades nothing: ccdrift puts its days back in the baseline and
    scores them like any other, so shading them would tell the reader to discount a dip
    ccdrift itself counts as normal. Open, recovered and persistent incidents all shade:
    those are days ccdrift stands behind, whatever it later decided about them."""
    shaded = set()
    for incident, _ in entries:
        if incident["metric"] != metric or incident["status"] == "dismissed":
            continue
        end = incident["end"] or max(days, default=incident["start"])
        shaded |= {day for day in days if incident["start"] <= day <= end}
    return sorted(shaded)


def run_report(source: Path, state_path: Path, days: Optional[int] = None, by: str = "day",
               as_json: bool = False, today: Optional[date] = None,
               cfg: Optional[DetectorConfig] = None, html_path: Optional[Path] = None) -> int:
    if html_path is not None and by != "day":
        raise ValueError(REPORT_LINES["page_by"].format(by=by))
    if html_path is not None and as_json:
        raise ValueError(REPORT_LINES["page_and_json"])
    try:
        state = load_state(state_path)
    except (OSError, ValueError) as exc:
        print(COMMAND_LINES["state_unreadable"].format(path=state_path, error=exc), file=sys.stderr)
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
        compactions = compactions[outside_sdk(compactions) & ~compactions["is_sidechain"]
                                  & (compactions["trigger"] == "auto")
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
        window = sorted(turns["day"].astype(str).unique())
        rows = version_rows(turns, changelog, starts, compactions, loop_counts(df, today, by="version", days=window))
        extra_lines, extra_json = [], {}
    else:
        rows = daily_rows(turns, days or DEFAULT_DAYS, cfg, incidents, loop_counts(df, today))
        window = rows["day"].tolist()
        hooks = hooks_summary(judged_hook_runs(tables.hook_runs, today), window)
        subagents = subagent_summary(judged_subagent_turns(df, today), window)
        fails = failure_summary(failure_counts(judged_failures(tables.failures, today), turns), window)
        projects = project_summary(starts, window)
        reasons = reason_summary(turns, window)
        extra_lines = (hooks_lines(hooks) + subagent_lines(subagents) + failure_lines(fails)
                      + reason_lines(reasons) + project_lines(projects))
        # The day view names the project folders; --json keeps its promise of holding no
        # paths, so the projects stay out of it. The five reason values name none of
        # those, so unlike the projects, they stay in.
        extra_json = {"hooks": hooks, "subagents": subagents, "failures": fails, "miss_reasons": reasons or {}}
    summary = settings_summary(turns, window)
    if html_path is not None:
        from ccdrift import __version__
        from ccdrift.page import render

        shown = [str(day) for day in rows["day"]] if not rows.empty else []
        page = render(rows, entries, state["reported"], summary, extra_lines, cfg, version=__version__,
                      today=today, source=source,
                      incident_days={metric: _incident_days(entries, shown, metric) for metric in INCIDENT_METRICS})
        try:
            # The page names project folders. Made private through the open file, so a folder
            # given by mistake fails to open before its own permissions are touched.
            with open(html_path, "w", encoding="utf-8") as out:
                make_stream_private(out)
                out.write(page)
        except OSError as exc:
            print(REPORT_LINES["unwritable"].format(path=html_path, error=exc), file=sys.stderr)
            return 1
        print(REPORT_LINES["wrote"].format(path=html_path))
        return 0
    if as_json:
        text = report_json(by, rows, entries, state["reported"], summary, cfg, extra_json)
    elif by == "version":
        text = format_version_report(rows, entries, state["reported"], summary)
    else:
        text = format_report(rows, entries, state["reported"], summary, cfg, extra=extra_lines)
    print(text, end="")
    return 0
