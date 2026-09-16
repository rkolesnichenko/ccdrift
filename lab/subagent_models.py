"""Do Claude Code's built-in subagents keep one model? (G7)

Per agent type other than general-purpose (whose model its caller picks): days with
at least 5 responses, the most common model on those days and its share. The gate
passes when at least one type has 5 such days with one model on 90% or more of them.

Run from the repo root:

  uv run --group lab python -m lab.subagent_models
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import pandas as pd

from ccdrift.logs import default_source, parse_source

ACTIVE_RESPONSES = 5
MIN_ACTIVE_DAYS = 5
USUAL_SHARE = 0.90
CALLER_PICKED = "general-purpose"


def subagent_turns(df: pd.DataFrame) -> pd.DataFrame:
    """Subagent responses outside Agent SDK sessions whose agent type is known."""
    keep = df["is_sidechain"].astype(bool) & df["agent_type"].notna()
    if "entrypoint" in df:
        keep &= ~df["entrypoint"].fillna("").astype(str).str.startswith("sdk-")
    return df[keep]


def agent_table(turns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for agent, group in turns.groupby("agent_type", sort=True):
        per_day = group.groupby(group["day"].astype(str)).size()
        active = group[group["day"].astype(str).isin(per_day[per_day >= ACTIVE_RESPONSES].index)]
        if active.empty:
            rows.append({"agent_type": agent, "responses": len(group), "active_days": 0, "model": "-",
                         "share": math.nan})
            continue
        counts = active["model"].value_counts()
        rows.append({"agent_type": agent, "responses": len(group), "active_days": int((per_day >= ACTIVE_RESPONSES).sum()),
                     "model": str(counts.index[0]), "share": float(counts.iloc[0] / counts.sum())})
    return pd.DataFrame(rows, columns=["agent_type", "responses", "active_days", "model", "share"])


def gate(turns: pd.DataFrame) -> tuple[bool, list[str]]:
    table = agent_table(turns)
    built_in = table[table["agent_type"] != CALLER_PICKED]
    steady = built_in[(built_in["active_days"] >= MIN_ACTIVE_DAYS) & (built_in["share"] >= USUAL_SHARE)]
    notes = [f"agent types other than {CALLER_PICKED}: {len(built_in)}; with {MIN_ACTIVE_DAYS}+ active days and one "
             f"model on {USUAL_SHARE:.0%}+: {', '.join(steady['agent_type']) or 'none'}"]
    return bool(len(steady)), notes


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="G7: do built-in subagents keep one model?")
    ap.add_argument("--source", default=None, help="folder of Claude Code transcripts (default: as ccdrift)")
    args = ap.parse_args(argv)
    source = Path(args.source).expanduser() if args.source else default_source()
    turns = subagent_turns(parse_source(source))
    for row in agent_table(turns).itertuples(index=False):
        share = "-" if math.isnan(row.share) else f"{row.share:.0%}"
        print(f"  {row.agent_type:<28} responses {row.responses:>6}  active days {row.active_days:>3}  "
              f"usual model {row.model} ({share})")
    passed, notes = gate(turns)
    for note in notes:
        print(note)
    print(f"G7: {'PASS' if passed else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
