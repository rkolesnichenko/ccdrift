"""Attachment types Claude Code starts writing. A third of a transcript's lines are
attachment records, and a new version can start writing a new kind on every session without
a word, as 2.1.267 did with `date`, `environment`, `model` and `remote_session_change`. This
reports such a type the way fields.new_fields reports a response field: once, to the log, the
weekly summary and `ccdrift status`, never as a notification.

Most new types follow what the owner does rather than the version (plan mode re-entered, a
hook that blocked), so a type counts only when a new version writes it on nearly all of its
sessions. CLI and Agent SDK sessions are judged apart, since Claude Code writes different
attachments for each."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from ccdrift.fields import GONE, USUAL as ARRIVED
from ccdrift.logs import READ_ATTACHMENTS

# The owner's transcripts on 2026-09-29 (`python -m lab.attachments`), judged per version and
# entrypoint: every ARRIVED from 0.5 to 0.9 reports the same 3 arrivals, 6 types (2.1.234,
# 2.1.267 and 2.1.269, SDK sessions); requiring 10 sessions moves the last to 2.1.272, and
# requiring 3 lets three-session CLI versions through.
MIN_TRANSCRIPTS = 5     # main-thread sessions on the new version, for it to be judged
MIN_BEFORE = 10         # sessions of the same entrypoint in the BEFORE_DAYS before it
BEFORE_DAYS = 14
RECENT_DAYS = 14


def _types_by_transcript(rows: pd.DataFrame) -> dict[str, set[str]]:
    return {str(source): set(group["type"].astype(str)) for source, group in rows.groupby("source_file", sort=True)}


def _shares(transcripts: dict[str, set[str]]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for types in transcripts.values():
        for kind in types:
            counts[kind] = counts.get(kind, 0) + 1
    return {kind: count / len(transcripts) for kind, count in counts.items()}


def new_attachments(census: pd.DataFrame, state: dict[str, Any], today: date, arrived: float = ARRIVED,
                    minimum: int = MIN_TRANSCRIPTS) -> list[dict[str, Any]]:
    """Attachment types a version first seen in the last RECENT_DAYS writes on `arrived` or more
    of its main-thread sessions of one entrypoint, after under GONE of that entrypoint's
    sessions in the BEFORE_DAYS before it, with `minimum` and MIN_BEFORE sessions to judge by
    (lab/attachments.py sweeps the two): types Claude Code has started writing that ccdrift doesn't read. Only complete
    UTC days count. Every type arriving on one version and entrypoint is one record in
    state["new_attachments"], and a type named in any record is never reported again."""
    if census.empty:
        return []
    main = census[~census["is_sidechain"].astype(bool) & (census["day"].astype(str) < today.isoformat())]
    main = main.assign(day=main["day"].astype(str), version=main["version"].fillna("unknown").astype(str),
                       entrypoint=main["entrypoint"].fillna("unknown").astype(str))
    known = {kind for record in state["new_attachments"] for kind in record["types"]}
    since = (today - timedelta(days=RECENT_DAYS)).isoformat()
    new = []
    for entrypoint, rows in main.groupby("entrypoint", sort=True):
        firsts = rows.groupby("version")["day"].min()
        for version, first in sorted(firsts.items(), key=lambda item: (item[1], item[0])):
            if first < since or version == "unknown":
                continue
            on_version = _types_by_transcript(rows[rows["version"] == version])
            start = (date.fromisoformat(first) - timedelta(days=BEFORE_DAYS)).isoformat()
            before = _types_by_transcript(rows[(rows["day"] >= start) & (rows["day"] < first)])
            if len(on_version) < minimum or len(before) < MIN_BEFORE:
                continue
            shares, earlier = _shares(on_version), _shares(before)
            types = sorted(kind for kind, share in shares.items()
                           if share >= arrived and earlier.get(kind, 0.0) < GONE and kind not in known
                           and kind not in READ_ATTACHMENTS)
            if not types:
                continue
            record = {"types": types, "version": version, "entrypoint": entrypoint,
                      "share": round(min(shares[kind] for kind in types), 3), "transcripts": len(on_version),
                      "reported_on": today.isoformat()}
            state["new_attachments"].append(record)
            known.update(types)
            new.append(record)
    return new
