"""Fields Claude Code stops logging, per new version."""

from datetime import date

import pandas as pd

from ccdrift.fields import field_gaps, gap_message
from ccdrift.state import new_state
from tests.helpers import nth_day


def responses(days, per_day, **fields):
    base = {"version": "2.1.270", "entrypoint": "cli", "effort": "xhigh", "speed": None, "service_tier": "standard",
            "thinking_logged": 120.0, "cache_creation": 100.0, "cache_1h": 100.0, "cache_5m": 0.0}
    return pd.DataFrame([{**base, **fields, "day": nth_day(d)} for d in days for _ in range(per_day)])


def turns_with(new_version_days=(14, 15), per_day=30, **new_fields):
    fields = {"version": "2.1.280", **new_fields}
    return pd.concat([responses(range(14), 20), responses(new_version_days, per_day, **fields)],
                     ignore_index=True)


def test_a_field_a_new_version_stopped_logging_is_reported_once():
    state = new_state()
    turns = turns_with(effort=None)
    gaps = field_gaps(turns, state, date(2026, 9, 17))
    assert [(g["field"], g["version"], g["share_before"], g["share"], g["responses"]) for g in gaps] == [
        ("effort", "2.1.280", 1.0, 0.0, 60)]
    assert field_gaps(turns, state, date(2026, 9, 17)) == []


def test_cache_writes_without_the_tier_split_are_a_gap():
    gaps = field_gaps(turns_with(cache_1h=0.0), new_state(), date(2026, 9, 17))
    assert [g["field"] for g in gaps] == ["cache_split"]


def test_a_field_never_logged_before_is_not_a_gap():
    assert field_gaps(turns_with(), new_state(), date(2026, 9, 17)) == []


def test_a_version_with_few_responses_is_not_judged_yet():
    assert field_gaps(turns_with(per_day=20, new_version_days=(15,), effort=None), new_state(), date(2026, 9, 17)) == []


def test_responses_without_a_version_are_a_gap_in_the_version_itself():
    gaps = field_gaps(turns_with(version=None), new_state(), date(2026, 9, 17))
    assert [(g["field"], g["version"]) for g in gaps] == [("version", "unknown")]
    assert gap_message(gaps[0]) == ("Claude Code no longer logs its version (on 0% of 60 responses, 100% before). "
                                    "Alerts can't name versions until ccdrift reads it again; run `ccdrift peek`.")
