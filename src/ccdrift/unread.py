"""Transcripts the check couldn't read: those the parser failed on, and long ones that
hold no response it recognises. Either leaves days short or missing without failing the
check, as a change in how Claude Code logs would, and every other alert needs the
responses they lost to see anything."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional, Sequence

QUIET_DAYS = 7  # an episode goes on while it is seen again within this many days


def unread_episode(records: list[dict[str, Any]], found: Sequence[tuple[str, Any]],
                   today: date) -> Optional[dict[str, Any]]:
    """A new episode when this run found transcripts it couldn't read (`found`, as
    Tables.skipped or Tables.no_responses) and the last one in `records` wasn't seen
    within QUIET_DAYS; it is appended to `records`. A run that finds them within that
    extends the last episode instead, so a transcript read and failed on every hour
    alerts once."""
    if not found:
        return None
    day = today.isoformat()
    if records and records[-1]["last"] >= (today - timedelta(days=QUIET_DAYS)).isoformat():
        records[-1]["last"] = day
        return None
    episode = {"since": day, "last": day, "transcripts": len(found)}
    records.append(episode)
    return episode
