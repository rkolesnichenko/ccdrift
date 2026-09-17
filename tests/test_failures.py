"""Failed requests and responses cut short: what the parser keeps, what a day's counts
say, and when the check alerts."""

import pandas as pd
import pytest

from ccdrift.history import History, load_history
from ccdrift.logs import parse_all, parse_source
from ccdrift.state import new_state, save_state
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
