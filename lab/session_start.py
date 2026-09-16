"""Is the context a Claude Code session starts with steady enough to alert on? (G2)

For each version: how many sessions, their median prompt size and spread; then the
session-start alert rule replayed over all sessions, with the versions around each
step. The gate passes when every version with 3+ sessions has a spread (MAD over
median) of at most 0.10 and every step comes with a version its baseline never ran.

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
from ccdrift.sessions import MIN_SESSIONS, context_changes_in, first_of_each, session_starts

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


def gate(starts: pd.DataFrame) -> tuple[bool, list[str]]:
    table = version_table(starts)
    steady = table[table["sessions"] >= MIN_SESSIONS]
    spread_ok = bool(len(steady)) and bool((steady["spread"] <= MAX_SPREAD).all())
    versions = starts["version"].fillna("unknown").tolist()
    kept = first_of_each(context_changes_in(starts))
    with_new_version = [bool({versions[i] for i in c.window} - {versions[i] for i in c.baseline}) for c in kept]
    notes = [f"versions with {MIN_SESSIONS}+ sessions: {len(steady)}; largest spread "
             f"{steady['spread'].max():.3f}" if len(steady) else "no version has 3+ sessions",
             f"steps found: {len(kept)}; with a version new to their baseline: {sum(with_new_version)}"]
    return spread_ok and all(with_new_version), notes


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
    versions = starts["version"].fillna("unknown").tolist()
    for change in first_of_each(context_changes_in(starts)):
        print(f"  step from {change.since}: {change.before / 1000:.0f}k -> {change.after / 1000:.0f}k; window on "
              f"{sorted({versions[i] for i in change.window})}, baseline on {sorted({versions[i] for i in change.baseline})}")
    passed, notes = gate(starts)
    for note in notes:
        print(note)
    print(f"G2: {'PASS' if passed else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
