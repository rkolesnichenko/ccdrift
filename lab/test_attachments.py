"""The attachment arrival sweep: the shipped rule replayed day by day over a grid of settings."""

from datetime import date, datetime, timezone

import pandas as pd

from ccdrift.logs import ATTACHMENT_COLUMNS
import lab.attachments
from lab.attachments import arguments, replay, sweep, utc_today
from tests.helpers import nth_day


def sessions(count, day, version, types, first=0):
    return [{"source_file": f"{version}-{first + i}.jsonl", "session_id": f"s{first + i}", "day": nth_day(day),
             "version": version, "entrypoint": "sdk-py", "is_sidechain": False, "type": kind, "records": 1}
            for i in range(count) for kind in types]


def history():
    """Ten sessions without `date`, then four with it and three without on a new version."""
    rows = (sessions(10, 3, "2.1.266", ["date_change"], first=100) + sessions(4, 10, "2.1.267", ["date_change", "date"])
            + sessions(3, 10, "2.1.267", ["date_change"], first=4))
    return pd.DataFrame(rows, columns=list(ATTACHMENT_COLUMNS))


def test_the_replay_reports_each_arrival_once_judging_each_day_the_morning_after():
    found = replay(history(), arrived=0.5, minimum=5)
    assert [(r["version"], r["types"], r["reported_on"]) for r in found] == [("2.1.267", ["date"], nth_day(11))]


def test_the_sweep_gives_each_setting_its_reports():
    rows = sweep(history(), arrived_grid=(0.5, 0.9), minimum_grid=(5, 8))
    assert [(row["arrived"], row["minimum"], [r["types"] for r in row["reports"]]) for row in rows] == [
        (0.5, 5, [["date"]]), (0.5, 8, []), (0.9, 5, []), (0.9, 8, [])]


def test_the_sweep_counts_its_today_as_a_utc_day_like_the_census(monkeypatch):
    # Found in the 0.21.0 review: the local date keeps a partial UTC day, or drops a complete
    # one, between UTC and local midnight.
    class Clock:
        @staticmethod
        def now(tz=None):
            assert tz is timezone.utc
            return datetime(2030, 1, 1, 0, 30, tzinfo=timezone.utc)

    monkeypatch.setattr(lab.attachments, "datetime", Clock)
    assert utc_today() == date(2030, 1, 1)
    assert arguments([]).today == date(2030, 1, 1)
