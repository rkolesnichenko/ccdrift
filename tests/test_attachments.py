"""Attachment types a new Claude Code version starts writing on most sessions: what the check
reports, once, to the log, the weekly summary and `ccdrift status`."""

from datetime import date

import pandas as pd

from ccdrift.attachments import new_attachments
from ccdrift.logs import ATTACHMENT_COLUMNS
from ccdrift.state import new_state
from ccdrift.texts import new_attachment_line, new_attachments_message
from tests.helpers import nth_day

BEFORE = ["skill_listing", "date_change"]


def sessions(count, day, version, types, entrypoint="sdk-py", first=0, sidechain=False):
    """`count` transcripts, each holding every type in `types` on `day`."""
    return [{"source_file": f"{entrypoint}-{version}-{first + i}.jsonl", "session_id": f"s{first + i}",
             "day": nth_day(day), "version": version, "entrypoint": entrypoint, "is_sidechain": sidechain,
             "type": kind, "records": 1} for i in range(count) for kind in types]


def census(*groups):
    return pd.DataFrame([row for group in groups for row in group], columns=list(ATTACHMENT_COLUMNS))


def baseline(entrypoint="sdk-py", count=10):
    return sessions(count, 5, "2.1.266", BEFORE, entrypoint, first=100)


def found(frame, today=11, state=None):
    return new_attachments(frame, state if state is not None else new_state(), date.fromisoformat(nth_day(today)))


def test_a_type_on_every_session_of_a_new_version_is_reported_once():
    frame = census(baseline(), sessions(6, 10, "2.1.267", BEFORE + ["date", "environment"]))
    state = new_state()
    assert found(frame, state=state) == [{"types": ["date", "environment"], "version": "2.1.267", "entrypoint": "sdk-py",
                                          "share": 1.0, "transcripts": 6, "reported_on": nth_day(11)}]
    assert found(frame, state=state) == []
    assert len(state["new_attachments"]) == 1


def test_a_type_on_fewer_than_nine_in_ten_sessions_is_not_reported():
    frame = census(baseline(), sessions(6, 10, "2.1.267", BEFORE + ["date"]), sessions(1, 10, "2.1.267", BEFORE, first=6))
    assert found(frame) == []


def test_a_type_already_on_a_tenth_of_the_sessions_before_is_not_new():
    frame = census(baseline(count=9), sessions(1, 5, "2.1.266", ["date"], first=9),
                   sessions(6, 10, "2.1.267", BEFORE + ["date"]))
    assert found(frame) == []


def test_too_few_sessions_on_the_version_or_before_it_are_not_judged():
    assert found(census(baseline(), sessions(4, 10, "2.1.267", ["date"]))) == []
    assert found(census(baseline(count=9), sessions(6, 10, "2.1.267", ["date"]))) == []


def test_a_type_ccdrift_reads_is_not_reported():
    assert found(census(baseline(), sessions(6, 10, "2.1.267", ["prompt_snapshot"]))) == []


def test_cli_and_agent_sdk_sessions_are_judged_apart():
    # credential_org is written on every CLI session of 2.1.281 and on none of its many SDK sessions.
    frame = census(baseline(), baseline("cli"), sessions(30, 10, "2.1.281", BEFORE),
                   sessions(6, 10, "2.1.281", BEFORE + ["credential_org"], entrypoint="cli"))
    assert [(r["entrypoint"], r["types"]) for r in found(frame)] == [("cli", ["credential_org"])]


def test_a_version_first_seen_more_than_two_weeks_ago_is_not_judged():
    frame = census(baseline(), sessions(6, 10, "2.1.267", ["date"]))
    assert found(frame, today=25) == []
    assert found(frame, today=24) != []


def test_a_type_is_reported_once_whatever_version_it_turns_up_on_later():
    state = new_state()
    found(census(baseline(), sessions(6, 10, "2.1.267", ["date"])), state=state)
    later = census(sessions(10, 12, "2.1.268", BEFORE, first=200), sessions(6, 14, "2.1.269", ["date"], first=300))
    assert found(later, today=15, state=state) == []


def test_only_complete_days_and_main_thread_sessions_count():
    assert found(census(baseline(), sessions(6, 10, "2.1.267", ["date"])), today=10) == []
    assert found(census(baseline(), sessions(6, 10, "2.1.267", ["date"], sidechain=True))) == []


def test_rows_with_no_version_are_never_judged_as_a_new_version():
    frame = census(baseline(), sessions(6, 10, "2.1.267", ["date"]))
    frame.loc[frame["version"] == "2.1.267", "version"] = None
    assert found(frame) == []


def test_sessions_with_no_entrypoint_are_judged_on_their_own_not_as_cli():
    # An entrypoint Claude Code didn't log isn't known to be the CLI's: its sessions form
    # their own population, so they can neither dilute nor fill out the CLI's.
    frame = census(baseline(), baseline(entrypoint=None), sessions(6, 10, "2.1.267", ["date"], entrypoint=None),
                   sessions(3, 10, "2.1.267", BEFORE, entrypoint="cli"))
    assert [(r["entrypoint"], r["types"]) for r in found(frame)] == [("unknown", ["date"])]


def test_the_message_and_status_line_name_the_types_version_and_sessions():
    record = {"types": ["date", "environment"], "version": "2.1.267", "entrypoint": "sdk-py", "share": 1.0,
              "transcripts": 27, "reported_on": "2026-09-11"}
    assert new_attachments_message(record) == (
        "Claude Code 2.1.267 writes 2 attachment types ccdrift doesn't read on its sdk-py sessions: date, environment "
        "(on 100% of 27 sessions). They may be worth reading; please open an issue.")
    assert new_attachment_line(record) == "2 new attachment types on 2.1.267 (sdk-py): date, environment"
    one = {**record, "types": ["credential_org"], "entrypoint": "cli", "share": 0.9, "transcripts": 10}
    assert new_attachment_line(one) == "1 new attachment type on 2.1.267 (cli): credential_org"
