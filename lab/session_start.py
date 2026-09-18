"""Is the context a Claude Code session starts with steady enough to alert on? (G2)

For each version: how many sessions, their median prompt size and spread; then the steps
the shipped rule finds — `found_changes` over the ratios, pooled and per project, the same
call the check makes — with the versions their sessions ran.

The gate passes when every version with 3+ sessions has a spread (MAD over median) of at
most 0.10. It used to demand that every step come with a version its baseline never ran,
and 0.8.0's whole thesis contradicts that: a project's own CLAUDE.md, skills or MCP servers
step its sessions with no new version at all, which is what G12 measures. The version is
reported beside each step instead of deciding the gate.

Run from the repo root:

  uv run --group lab python -m lab.session_start
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ccdrift.logs import default_source, parse_source
from ccdrift.sessions import ContextChange, MIN_SESSIONS, first_of_each, found_changes, ratio_starts, session_starts

MAX_SPREAD = 0.10


def version_table(starts: pd.DataFrame) -> pd.DataFrame:
    """Per version, in order of first session: sessions, median, lowest, highest and
    spread (1.4826 × MAD over the median)."""
    rows = []
    versions = starts["version"].fillna("unknown")
    for version in dict.fromkeys(versions):
        tokens = starts.loc[versions == version, "prompt_tokens"].astype(float).to_numpy()
        median = float(np.median(tokens))
        spread = 1.4826 * float(np.median(np.abs(tokens - median))) / median if median else math.nan
        rows.append({"version": version, "sessions": len(tokens), "median": median,
                     "low": float(tokens.min()), "high": float(tokens.max()), "spread": spread})
    return pd.DataFrame(rows, columns=["version", "sessions", "median", "low", "high", "spread"])


def step_versions(judged: pd.DataFrame, change: ContextChange) -> list[str]:
    """The versions a step's own sessions ran: the days it spans, and — when it was found
    in one project's own rows — that project alone. Its row positions can't be used, since
    they index the frame it was found in rather than the judged table, and its days alone
    would credit it with every other project's versions on them."""
    days = judged["day"].astype(str)
    rows = judged[(days >= change.since) & (days <= change.until)]
    if change.project is not None:
        rows = rows[rows["project"].astype(str) == change.project]
    return sorted(set(rows["version"].fillna("unknown").astype(str)))


def gate(starts: pd.DataFrame) -> tuple[bool, list[str]]:
    judged = ratio_starts(starts)
    table = version_table(starts)
    steady = table[table["sessions"] >= MIN_SESSIONS]
    spread_ok = bool(len(steady)) and bool((steady["spread"] <= MAX_SPREAD).all())
    kept = first_of_each(found_changes(judged))
    with_new_version = sum(1 for change in kept if change.new_version)
    notes = [f"versions with {MIN_SESSIONS}+ sessions: {len(steady)}; largest spread "
             f"{steady['spread'].max():.3f}" if len(steady) else "no version has 3+ sessions",
             f"steps found: {len(kept)}; with a version new to their baseline: {with_new_version}"]
    return spread_ok, notes


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="G2: session-start size per version and its alert rule")
    ap.add_argument("--source", default=None, help="folder of Claude Code transcripts (default: as ccdrift)")
    args = ap.parse_args(argv)
    source = Path(args.source).expanduser() if args.source else default_source()
    starts = session_starts(parse_source(source))
    print(f"{len(starts)} session starts")
    for row in version_table(starts).itertuples(index=False):
        print(f"  {row.version:<10} sessions {row.sessions:>3}  median {row.median / 1000:>6.1f}k  "
              f"range {row.low / 1000:.1f}-{row.high / 1000:.1f}k  spread {row.spread:.3f}")
    judged = ratio_starts(starts)
    for change in first_of_each(found_changes(judged)):
        print(f"  step from {change.since} to {change.until}: {change.before / 1000:.0f}k -> "
              f"{change.after / 1000:.0f}k; its sessions on {step_versions(judged, change)}, "
              f"{'on a version new to its baseline' if change.new_version else 'on no new version'}")
    passed, notes = gate(starts)
    for note in notes:
        print(note)
    print(f"G2: {'PASS' if passed else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
