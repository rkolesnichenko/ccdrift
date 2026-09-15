"""The daily check: report each new flag once, and alert when the check fails or
can't compute the cache metric."""

from __future__ import annotations

import json
import os
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from ccdrift.detector import DetectorConfig, bin_metrics, detect, flag_onsets
from ccdrift.logs import no_transcripts_message, parse_source
from ccdrift.notify import notify


# ---------------------------------------------------------------------------
# Daily check — report each new flag once (run it from launchd or cron)
# ---------------------------------------------------------------------------

# The daily check leaves effort out: it swings more from day to day than a 70%
# cut in thinking moves it, so its flags in one user's logs track the work.
CHECK_METRICS = {"cache_ratio": "Cache read ratio on new prompts",
                 "haiku_fraction": "Haiku share on the main thread"}
CHECK_RECENT_DAYS = 14


def ccdrift_home(environ: Mapping[str, str] = os.environ) -> Path:
    """Where the daily check keeps its state file and log: $CCDRIFT_HOME when set,
    otherwise ~/.ccdrift."""
    home = environ.get("CCDRIFT_HOME")
    return Path(home).expanduser() if home else Path.home() / ".ccdrift"


def check(df: pd.DataFrame, today: date, state_path: Path,
          cfg: Optional[DetectorConfig] = None,
          recent_days: int = CHECK_RECENT_DAYS) -> list[dict[str, Any]]:
    """Flags on main-thread turns of complete UTC days that start within the last
    `recent_days` days and weren't reported before. Main thread only, because
    subagent Haiku comes in bursts that flag on their own. The recent-days limit
    keeps a first run from reporting incidents from weeks ago while still
    covering a week or so without a run. Reported onsets are saved to
    `state_path`, so each flag is reported once."""
    before_today = df["day"].astype(str) < today.isoformat()
    turns = df[before_today & df["main_thread"].astype(bool)]
    if turns.empty:
        return []
    detected = detect(bin_metrics(turns), cfg or DetectorConfig())
    bins = detected["bin"].astype(str).tolist()
    since = (today - timedelta(days=recent_days)).isoformat()
    state = _load_state(state_path)
    reported = state.setdefault("reported", {})
    new = []
    for metric, label in CHECK_METRICS.items():
        flags = detected[f"{metric}__flag"].to_numpy(dtype=bool)
        for onset in flag_onsets(detected, metric):
            if bins[onset] < since or bins[onset] in reported.get(metric, []):
                continue
            end = onset
            while end + 1 < len(flags) and flags[end + 1]:
                end += 1
            new.append({"metric": metric, "label": label, "onset": bins[onset],
                        "days": bins[onset:end + 1],
                        "z": detected[f"{metric}__z"].iloc[onset:end + 1].round(2).tolist()})
            reported.setdefault(metric, []).append(bins[onset])
    if new:
        _save_state(state_path, state)
    return new


def _load_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=1) + "\n")


# A stretch of active days without usable cache values means the cache metric
# can't be computed, most likely because Claude Code's log format changed: it
# goes blank when prompts aren't recognised, and reads as all misses when cache
# usage isn't read.
CHECK_BLANK_DAYS = 3
CHECK_ACTIVE_RESPONSES = 50  # main-thread responses; the quietest of 29 real days had 81


def blank_cache_stretch(df: pd.DataFrame, today: date, state_path: Path,
                        days: int = CHECK_BLANK_DAYS,
                        active: int = CHECK_ACTIVE_RESPONSES) -> Optional[dict[str, Any]]:
    """The latest run of complete active days (at least `active` main-thread
    responses) on which no new-prompt turn has cache token counts, once it is
    `days` long and wasn't reported before. Every Claude Code response reads or
    writes the prompt cache, so such days mean the parser has lost track of it."""
    turns = df[(df["day"].astype(str) < today.isoformat()) & df["main_thread"].astype(bool)]
    usable = turns["prompt_within_ttl"].astype(bool) & ((turns["cache_read"] + turns["cache_creation"]) > 0)
    per_day = pd.DataFrame({"responses": turns.groupby("day").size(),
                            "usable": usable.groupby(turns["day"]).sum()})
    per_day = per_day[per_day["responses"] >= active]
    blank = (per_day["usable"] == 0).to_numpy()
    if len(blank) < days or not blank[-days:].all():
        return None
    start = len(blank) - days
    while start > 0 and blank[start - 1]:
        start -= 1
    stretch = per_day.iloc[start:]
    first = str(stretch.index[0])
    state = _load_state(state_path)
    if first in state.get("blank_cache", []):
        return None
    state.setdefault("blank_cache", []).append(first)
    _save_state(state_path, state)
    prompts = int(turns.loc[turns["day"].isin(stretch.index), "new_prompt"].sum())
    return {"first": first, "days": len(stretch), "responses": int(stretch["responses"].sum()),
            "prompts": prompts}


def run_check(source: Path, state_path: Path, cfg: Optional[DetectorConfig] = None,
              notify_user: bool = False, today: Optional[date] = None) -> int:
    """Run the daily check and print a line per result. With notify_user, also
    show a notification for each new flag, for days the cache metric can't be
    computed on, and when the check fails: a broken check otherwise looks like a
    quiet week."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def alert(title: str, message: str) -> None:
        print(f"[check {stamp}] {title}: {message}")
        if notify_user:
            notify(title, message)

    try:
        df = parse_source(source)
        if df.empty:
            raise RuntimeError(no_transcripts_message(source))
        today = today or datetime.now(timezone.utc).date()
        flags = check(df, today, state_path, cfg)
        blank = blank_cache_stretch(df, today, state_path)
    except Exception as exc:
        traceback.print_exc()
        alert("ccdrift check failed", f"{type(exc).__name__}: {exc}")
        return 1
    for f in flags:
        alert("ccdrift flag", f"{f['label']} flagged from {f['onset']}")
        z = ", ".join(f"{v:+.1f}" for v in f["z"])
        print(f"    days {', '.join(f['days'])}; z = {z}")
    if blank:
        alert("ccdrift can't compute the cache metric",
              f"no usable cache values on {blank['days']} active days from {blank['first']} "
              f"({blank['responses']} responses, {blank['prompts']} prompts recognised). "
              "Claude Code's log format may have changed; try --schema-peek.")
    if not flags and not blank:
        print(f"[check {stamp}] no new flags")
    return 0
