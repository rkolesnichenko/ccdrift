"""Failed requests and responses cut short: what the parser keeps, what a day's counts
say, and when the check alerts."""

from datetime import date

import pandas as pd
import pytest

from ccdrift.failures import (cut_short, cut_short_message, failing_requests, failure_counts, judged_failures,
                              requests_message)
from ccdrift.history import History, load_history
from ccdrift.logs import judged_turns, parse_all, parse_source
from ccdrift.state import new_state, save_state
from ccdrift.texts import cut_short_line, failure_line
from tests.helpers import api_error, at, failure_days, line, no_response_stub, retry_record, text, write


def parsed_failures(tmp_path, records):
    write(tmp_path / "logs" / "s1.jsonl", records)
    return parse_all(tmp_path / "logs").failures


@pytest.mark.parametrize("kind, status", [("overloaded", 529.0), ("slept", None), ("stream", None), ("other", None)])
def test_an_error_banner_is_kept_as_its_kind_and_status_without_its_text(tmp_path, kind, status):
    failures = parsed_failures(tmp_path, [api_error(at(0), kind=kind, version="2.1.226")])
    assert len(failures) == 1
    row = failures.iloc[0]
    assert (row["kind"], row["version"], row["day"]) == (kind, "2.1.226", "2026-09-01")
    assert (None if pd.isna(row["status"]) else row["status"]) == status
    assert not any("API Error" in str(value) for value in row.values)


def test_a_retry_record_counts_and_the_no_response_stub_doesnt(tmp_path):
    failures = parsed_failures(tmp_path, [retry_record(at(0), version="2.1.226"), no_response_stub(at(60))])
    assert list(failures["kind"]) == ["retry"]


def test_a_banner_is_not_a_response_and_a_response_keeps_its_stop_reason(tmp_path):
    records = [line("m1", text(10), ts=at(0), stop_reason=None),
               line("m1", text(10), ts=at(1), stop_reason="max_tokens"),
               api_error(at(60))]
    write(tmp_path / "logs" / "s1.jsonl", records)
    responses = parse_source(tmp_path / "logs")
    assert list(responses["stop_reason"]) == ["max_tokens"]


def test_the_store_keeps_failures_and_an_older_store_is_upgraded(tmp_path):
    failure_days(tmp_path / "logs", [{"errors": 2, "retries": 1}])
    save_state(tmp_path / "state.json", new_state())
    tables = load_history(tmp_path / "logs", tmp_path / "state.json", claim=True)
    assert sorted(tables.failures["kind"]) == ["overloaded", "overloaded", "retry"]
    with History(tmp_path / "history.sqlite") as history:
        assert history.meta["schema_version"] == "4"
        assert {row[1] for row in history.db.execute("PRAGMA table_info(responses)")} >= {"stop_reason"}


def counts_of(tmp_path, days, today=date(2026, 10, 1)):
    failure_days(tmp_path / "logs", days)
    tables = parse_all(tmp_path / "logs")
    return failure_counts(judged_failures(tables.failures, today), judged_turns(tables.responses, today))


def test_a_days_counts_hold_its_responses_failures_by_kind_and_responses_cut_short(tmp_path):
    counts = counts_of(tmp_path, [{}, {"errors": 2, "slept": 1, "retries": 1, "truncated": 3}])
    assert list(counts["day"]) == ["2026-09-01", "2026-09-02"]
    second = counts.iloc[1]
    assert (int(second["responses"]), int(second["requests"]), int(second["overloaded"]), int(second["retry"]),
            int(second["slept"]), int(second["truncated"])) == (60, 3, 2, 1, 1, 3)
    assert int(counts.iloc[0]["requests"]) == 0


def state_with(**keys):
    return {**new_state(), **keys}


def test_a_day_alerts_when_its_failed_requests_pass_the_floor_and_the_days_before_it(tmp_path):
    counts = counts_of(tmp_path, [{}] * 6 + [{"errors": 5}])
    state = state_with()
    episodes = failing_requests(counts, state, date(2026, 9, 8))
    assert [(e["since"], e["requests"], e["kinds"], e["before"]) for e in episodes] == [
        ("2026-09-07", 5, {"overloaded": 5}, 0)]
    assert state["failed_requests"] == episodes


def test_failed_requests_need_the_floor_quiet_days_before_and_days_to_compare_with(tmp_path):
    under = counts_of(tmp_path, [{}] * 6 + [{"errors": 4}])
    assert failing_requests(under, state_with(), date(2026, 9, 8)) == []
    busy = counts_of(tmp_path, [{"errors": 3}] * 6 + [{"errors": 5}])
    assert failing_requests(busy, state_with(), date(2026, 9, 8)) == []
    short = counts_of(tmp_path, [{}] * 3 + [{"errors": 9}])
    assert failing_requests(short, state_with(), date(2026, 9, 5)) == []


def test_one_spell_of_failures_alerts_once_and_an_old_day_never_does(tmp_path):
    counts = counts_of(tmp_path, [{}] * 6 + [{"errors": 5}, {"errors": 6}])
    state = state_with()
    first = failing_requests(counts, state, date(2026, 9, 9))
    assert [e["since"] for e in first] == ["2026-09-07"]
    assert failing_requests(counts, state, date(2026, 9, 9)) == []
    assert failing_requests(counts, state_with(), date(2026, 9, 30)) == []


def test_a_day_alerts_when_responses_stop_at_the_token_limit_far_more_than_before(tmp_path):
    counts = counts_of(tmp_path, [{}] * 6 + [{"truncated": 5}])
    state = state_with()
    episodes = cut_short(counts, state, date(2026, 9, 8))
    assert [(e["since"], e["cut"], e["truncated"], e["refused"], e["responses"]) for e in episodes] == [
        ("2026-09-07", 5, 5, 0, 60)]
    assert state["cut_short"] == episodes
    few = counts_of(tmp_path, [{}] * 6 + [{"truncated": 4}])
    assert cut_short(few, state_with(), date(2026, 9, 8)) == []


def test_the_messages_name_the_kinds_the_counts_and_the_version():
    episode = {"since": "2026-09-20", "days": ["2026-09-20"], "requests": 9,
               "kinds": {"overloaded": 7, "retry": 2}, "before": 1, "reported_on": "2026-09-21"}
    assert requests_message(episode, ["2.1.280"]) == (
        "9 requests failed on 2026-09-20 (7 overloaded, 2 retried), against at most 1 a day in the 14 days before, "
        "on Claude Code 2.1.280. Claude Code retries these itself; a run of them points at the API or your "
        "connection, not your setup.")
    cut = {"since": "2026-09-20", "days": ["2026-09-20"], "cut": 8, "truncated": 8, "refused": 0,
           "responses": 640, "reported_on": "2026-09-21"}
    assert cut_short_message(cut, []) == (
        "8 of 640 main-thread responses stopped at the token limit on 2026-09-20 (1.25%), against under 0.10% a day "
        "in the 14 days before. A Claude Code update may have changed the output limit.")
    assert failure_line(episode) == (
        "requests failing on 2026-09-20: 9 (7 overloaded, 2 retried), at most 1 a day before")
    assert cut_short_line(cut) == "responses cut short on 2026-09-20: 8 of 640 main-thread responses"
