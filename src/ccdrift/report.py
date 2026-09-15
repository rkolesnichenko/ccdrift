"""The details behind an alert: recent daily metrics, their z-scores and flags."""

from __future__ import annotations

import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ccdrift.check import CHECK_METRICS, complete_main_turns, load_state
from ccdrift.detector import DetectorConfig, bin_metrics, detect
from ccdrift.logs import no_transcripts_message, parse_source

SHORT_NAMES = {"cache_ratio": "cache", "haiku_fraction": "haiku"}
COLUMNS = ["day", "responses", "cache_ratio", "cache_z", "haiku_share", "haiku_z", "flagged"]


def daily_rows(df: pd.DataFrame, today: date, days: int = 21,
               cfg: Optional[DetectorConfig] = None) -> pd.DataFrame:
    """The last `days` complete UTC days with main-thread turns, each judged against
    the days before it: responses, cache ratio and Haiku share with their z-scores,
    and the metrics flagged that day."""
    turns = complete_main_turns(df, today)
    if turns.empty:
        return pd.DataFrame(columns=COLUMNS)
    detected = detect(bin_metrics(turns), cfg or DetectorConfig()).tail(days)
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


def _number(value: Any, spec: str) -> str:
    return "-" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def format_report(rows: pd.DataFrame, reported: dict[str, list[str]],
                  cfg: Optional[DetectorConfig] = None) -> str:
    cfg = cfg or DetectorConfig()
    cache_cutoff = cfg.metric_z_thresholds.get("cache_ratio", cfg.z_threshold)
    haiku_cutoff = cfg.metric_z_thresholds.get("haiku_fraction", cfg.z_threshold)
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
    lines.append("")
    flags = [f"  {label} from {day}" for metric, label in CHECK_METRICS.items()
             for day in reported.get(metric, [])]
    if flags:
        lines += ["Flags reported by the daily check:"] + flags
    else:
        lines.append("Flags reported by the daily check: none yet")
    return "\n".join(lines) + "\n"


def run_report(source: Path, state_path: Path, days: int = 21, today: Optional[date] = None) -> int:
    df = parse_source(source)
    if df.empty:
        print(no_transcripts_message(source), file=sys.stderr)
        return 2
    try:
        reported = load_state(state_path).get("reported", {})
    except (OSError, ValueError) as exc:
        print(f"Can't read the state file {state_path}: {exc}", file=sys.stderr)
        return 1
    rows = daily_rows(df, today or datetime.now(timezone.utc).date(), days)
    print(format_report(rows, reported), end="")
    return 0
