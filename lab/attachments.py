"""What the attachment arrival rule reports on the owner's logs, across settings.

ccdrift.attachments reports an attachment type a new Claude Code version writes on nearly all
of its main-thread sessions of one entrypoint, after almost none before it. This replays the
shipped rule day by day over the transcripts, as the check would have run it the morning after
each day with one state carried along, for each share cutoff and minimum number of sessions in
the grid, and prints what each setting reports: versions, entrypoints, types and session
counts. It is a census like the new-field rule, not a detector judged against planted changes,
so it has no PASS or FAIL; the shipped setting's reports come last. Output is aggregate.

Run from the repo root:

  uv run --group lab python -m lab.attachments
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ccdrift.attachments import ARRIVED, MIN_TRANSCRIPTS, new_attachments
from ccdrift.logs import default_source, parse_all
from ccdrift.state import new_state

ARRIVED_GRID = (0.5, 0.7, 0.8, 0.9)
MINIMUM_GRID = (3, 5, 10)


def replay(census: pd.DataFrame, arrived: float = ARRIVED, minimum: int = MIN_TRANSCRIPTS) -> list[dict[str, Any]]:
    """Every record the rule reports, judging each day of `census` the morning after it, as the
    check would, with one state carried along."""
    state = new_state()
    found = []
    for day in sorted(set(census["day"].astype(str))):
        found += new_attachments(census, state, date.fromisoformat(day) + timedelta(days=1), arrived, minimum)
    return found


def sweep(census: pd.DataFrame, arrived_grid=ARRIVED_GRID, minimum_grid=MINIMUM_GRID) -> list[dict[str, Any]]:
    """One row per setting, with the records it reports."""
    return [{"arrived": arrived, "minimum": minimum, "reports": replay(census, arrived, minimum)}
            for arrived in arrived_grid for minimum in minimum_grid]


def _reports(records: list[dict[str, Any]]) -> str:
    return "; ".join(f"{r['version']} ({r['entrypoint']}, {r['transcripts']} sessions): {', '.join(r['types'])}"
                     for r in records) or "nothing"


def utc_today() -> date:
    """Today as the census counts days: in UTC, as the check does."""
    return datetime.now(timezone.utc).date()


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--today", type=date.fromisoformat, default=utc_today())
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)

    census = parse_all(args.source).attachment_census
    census = census[census["day"].astype(str) < args.today.isoformat()]
    main_thread = census[~census["is_sidechain"].astype(bool)]
    for entrypoint, rows in main_thread.groupby(main_thread["entrypoint"].fillna("unknown"), sort=True):
        print(f"{entrypoint}: {rows['source_file'].nunique()} sessions, {rows['type'].nunique()} attachment types, "
              f"{rows['version'].nunique()} versions")
    for row in sweep(census):
        types = sum(len(r["types"]) for r in row["reports"])
        print(f"arrived={row['arrived']:g} minimum={row['minimum']}: {len(row['reports'])} report(s), {types} type(s): "
              f"{_reports(row['reports'])}")
    print(f"ships arrived={ARRIVED:g} minimum={MIN_TRANSCRIPTS}: {_reports(replay(census))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
