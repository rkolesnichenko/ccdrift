"""Fields Claude Code stops logging, per new version, and fields it starts logging."""

from datetime import date

import pandas as pd

from ccdrift.fields import (ARRIVED, MIN_BEFORE, MIN_RESPONSES, USUAL, field_gaps, gap_message, new_fields,
                            new_fields_message)
from ccdrift.logs import census_frame
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


def a_census(old_days=range(14), old_per_day=20, new_days=(14, 15), new_per_day=30,
             old_paths=("version", "message.model"),
             new_paths=("version", "message.model", "advisorModel"),
             new_share=1.0, old_version="2.1.270", new_version="2.1.280", census_only_days=()):
    """A census frame and the turns it was counted from: `old_days` on `old_version`
    carrying `old_paths`, then `new_days` on `new_version` carrying `new_paths`. A path
    the old days didn't carry is on `new_share` of the new version's responses. Days in
    `census_only_days` reach the census but not the turns, as the day still in progress
    does."""
    census, days = {}, {}
    for spec_days, per_day, paths, version in (
            (list(old_days), old_per_day, old_paths, old_version),
            (list(new_days) + list(census_only_days), new_per_day, new_paths, new_version)):
        for d in spec_days:
            days[(nth_day(d), version)] = per_day
            for path in paths:
                census[(nth_day(d), version, path)] = (per_day if path in old_paths
                                                       else round(per_day * new_share))
    turns = pd.concat([responses(old_days, old_per_day, version=old_version),
                       responses(new_days, new_per_day, version=new_version)], ignore_index=True)
    return census_frame(census, days), turns


def test_a_field_a_new_version_logs_and_the_days_before_did_not_is_reported():
    census, turns = a_census()
    records = new_fields(census, turns, new_state(), date(2026, 9, 17))
    assert [(r["paths"], r["version"], r["responses"]) for r in records] == [(["advisorModel"], "2.1.280", 60)]


def test_a_field_already_reported_is_not_reported_again_on_a_later_version():
    state = new_state()
    census, turns = a_census()
    assert len(new_fields(census, turns, state, date(2026, 9, 17))) == 1
    later, later_turns = a_census(new_version="2.1.282")
    assert new_fields(later, later_turns, state, date(2026, 9, 17)) == []


def test_every_field_arriving_on_one_version_is_one_record():
    census, turns = a_census(new_paths=("version", "message.model", "advisorModel", "perTurnEffort",
                                        "message.container", "message.context_management"))
    records = new_fields(census, turns, new_state(), date(2026, 9, 17))
    assert len(records) == 1 and len(records[0]["paths"]) == 4


def test_the_paths_of_one_record_are_sorted():
    census, turns = a_census(new_paths=("version", "message.model", "perTurnEffort", "advisorModel"))
    records = new_fields(census, turns, new_state(), date(2026, 9, 17))
    assert records[0]["paths"] == ["advisorModel", "perTurnEffort"]


def test_a_version_with_too_few_responses_is_not_judged():
    census, turns = a_census(new_days=(15,), new_per_day=MIN_RESPONSES - 1)
    assert new_fields(census, turns, new_state(), date(2026, 9, 17)) == []


def test_a_version_with_too_few_responses_before_it_is_not_judged():
    # 14 days of 14 responses is 196, one short of MIN_BEFORE.
    census, turns = a_census(old_per_day=(MIN_BEFORE - 4) // 14)
    assert new_fields(census, turns, new_state(), date(2026, 9, 17)) == []


def test_a_version_first_seen_before_the_recent_window_is_not_judged():
    census, turns = a_census()
    assert new_fields(census, turns, new_state(), date(2026, 10, 20)) == []


def test_a_field_on_exactly_the_share_a_gap_needs_is_reported():
    census, turns = a_census(new_share=ARRIVED)
    assert new_fields(census, turns, new_state(), date(2026, 9, 17))[0]["paths"] == ["advisorModel"]


def test_a_field_just_under_that_share_is_not_reported():
    census, turns = a_census(new_per_day=100, new_share=USUAL - 0.01)
    assert new_fields(census, turns, new_state(), date(2026, 9, 17)) == []


def test_a_field_the_days_before_already_carried_is_not_new():
    census, turns = a_census(old_paths=("version", "message.model", "advisorModel"))
    assert new_fields(census, turns, new_state(), date(2026, 9, 17)) == []


def test_a_day_the_check_hasnt_judged_yet_is_left_out():
    # The day still in progress reaches the census, which is counted at parse time,
    # and not the turns, which the check judges.
    census, turns = a_census(census_only_days=(16,))
    assert new_fields(census, turns, new_state(), date(2026, 9, 17))[0]["responses"] == 60


def test_the_new_fields_message_names_the_version_and_the_paths():
    record = {"paths": ["advisorModel"], "version": "2.1.276", "share": 0.94,
              "responses": 1602, "reported_on": "2026-09-19"}
    assert new_fields_message(record) == (
        "Claude Code 2.1.276 logs 1 field ccdrift doesn't read: advisorModel (on 94% of 1,602 "
        "responses). It may be worth reading; please open an issue.")
