"""Failed requests and responses cut short: what the parser keeps, what a day's counts
say, and when the check alerts."""

from datetime import date, datetime, timezone

import pandas as pd
import pytest

from ccdrift.check import run_check
from ccdrift.failures import (cut_short, cut_short_message, digest_part, failing_requests, failure_counts,
                              failure_lines, failure_summary, judged_failures, requests_message)
from ccdrift.history import History, load_history
from ccdrift.logs import judged_turns, parse_all, parse_source
from ccdrift.state import load_state, new_state, save_state
from ccdrift.texts import cut_short_line, failure_line
from tests.helpers import DAY, api_error, at, failure_days, line, no_response_stub, prompt, retry_record, text, write


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
    columns = ["kind", "status", "version", "entrypoint", "is_sidechain", "day"]
    parsed = parse_all(tmp_path / "logs")
    pd.testing.assert_frame_equal(tables.failures[columns].sort_values(columns).reset_index(drop=True),
                                  parsed.failures[columns].sort_values(columns).reset_index(drop=True))
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


def test_a_worse_burst_a_fortnight_later_alerts_again(tmp_path):
    # 2026-09-07 alerts on its own. 2026-09-21, a fortnight later, more than doubles it
    # and alerts again: the suppression window is now the 3-day spell, not the 14-day
    # comparison window, so a burst outside the spell is judged fresh. 2026-09-28
    # doesn't clear the ratio against 2026-09-21's burst and stays silent.
    counts = counts_of(tmp_path, [{}] * 6 + [{"errors": 5}] + [{}] * 13 + [{"errors": 10}]
                       + [{}] * 6 + [{"errors": 15}])
    state = state_with()
    first = failing_requests(counts[counts["day"] < "2026-09-08"], state, date(2026, 9, 8))
    assert [e["since"] for e in first] == ["2026-09-07"]
    second = failing_requests(counts[counts["day"] < "2026-09-22"], state, date(2026, 9, 22))
    assert [e["since"] for e in second] == ["2026-09-21"]
    third = failing_requests(counts[counts["day"] < "2026-09-29"], state, date(2026, 9, 29))
    assert third == []


def test_a_day_alerts_when_responses_stop_at_the_token_limit_far_more_than_before(tmp_path):
    counts = counts_of(tmp_path, [{}] * 6 + [{"truncated": 5}])
    state = state_with()
    episodes = cut_short(counts, state, date(2026, 9, 8))
    assert [(e["since"], e["cut"], e["truncated"], e["refused"], e["responses"], e["before_share"])
            for e in episodes] == [("2026-09-07", 5, 5, 0, 60, 0.0)]
    assert state["cut_short"] == episodes
    few = counts_of(tmp_path, [{}] * 6 + [{"truncated": 4}])
    assert cut_short(few, state_with(), date(2026, 9, 8)) == []


def test_the_days_an_alert_compares_with_are_the_active_ones(tmp_path):
    # 5 quiet active days, then the alerting day; a day within the 14 days before it has
    # far more failures than any of them but too few responses to count, and must be
    # ignored both for whether the alert fires and for the "before" it reports.
    failure_days(tmp_path / "logs", [{}] * 5 + [{"errors": 5}])
    quiet = []
    for k in range(10):
        ts = at(-2 * DAY + 60 * k)
        quiet.append(prompt(ts, sid="quiet"))
        quiet.append(line(f"quiet-{k}", text(40), ts=ts, sid="quiet", version="2.1.226", entrypoint="cli"))
    quiet += [api_error(at(-2 * DAY + 600 + j), sid="quiet", version="2.1.226") for j in range(20)]
    write(tmp_path / "logs" / "quiet.jsonl", quiet)

    today = date(2026, 10, 1)
    tables = parse_all(tmp_path / "logs")
    counts = failure_counts(judged_failures(tables.failures, today), judged_turns(tables.responses, today))
    episodes = failing_requests(counts, state_with(), date(2026, 9, 7))
    assert [(e["since"], e["requests"], e["before"]) for e in episodes] == [("2026-09-06", 5, 0)]


def test_the_messages_name_the_kinds_the_counts_and_the_version():
    episode = {"since": "2026-09-20", "days": ["2026-09-20"], "requests": 9,
               "kinds": {"overloaded": 7, "retry": 2}, "before": 1, "reported_on": "2026-09-21"}
    assert requests_message(episode, ["2.1.280"]) == (
        "9 requests failed on 2026-09-20 (7 overloaded, 2 retried), against at most 1 a day in the 14 days before, "
        "on Claude Code 2.1.280. Claude Code retries these itself; a run of them points at the API or your "
        "connection, not your setup.")
    clean = {**episode, "before": 0}
    assert requests_message(clean, []) == (
        "9 requests failed on 2026-09-20 (7 overloaded, 2 retried), against none in the 14 days before. Claude "
        "Code retries these itself; a run of them points at the API or your connection, not your setup.")
    cut = {"since": "2026-09-20", "days": ["2026-09-20"], "cut": 8, "truncated": 8, "refused": 0,
           "responses": 640, "before_share": 0.0014, "reported_on": "2026-09-21"}
    assert cut_short_message(cut, []) == (
        "8 of 640 main-thread responses stopped at the token limit on 2026-09-20 (1.25%), against at most 0.14% a "
        "day in the 14 days before. A Claude Code update may have changed the output limit.")
    assert failure_line(episode) == (
        "requests failing on 2026-09-20: 9 (7 overloaded, 2 retried), at most 1 a day before")
    assert cut_short_line(cut) == "responses cut short on 2026-09-20: 8 of 640 main-thread responses"
    cut_clean = {**cut, "before_share": 0.0}
    assert cut_short_message(cut_clean, []) == (
        "8 of 640 main-thread responses stopped at the token limit on 2026-09-20 (1.25%), against none in the 14 "
        "days before. A Claude Code update may have changed the output limit.")


def test_the_report_line_and_the_weekly_part_say_what_the_days_held(tmp_path):
    counts = counts_of(tmp_path, [{"errors": 2, "slept": 1, "truncated": 1}, {"retries": 1}])
    days = ["2026-09-01", "2026-09-02"]
    assert failure_lines(failure_summary(counts, days)) == [
        "", "Failures over these days: 2 overloaded, 1 retried, 1 while the Mac slept, 1 response cut short"]
    assert digest_part(counts, days) == "3 failed requests, 1 response cut short"
    quiet = counts_of(tmp_path, [{}])
    assert failure_lines(failure_summary(quiet, ["2026-09-01"])) == []
    assert digest_part(quiet, ["2026-09-01"]) == "no failed requests"
    # The Mac sleeping mid-response is not Claude Code drift: the report names it, but
    # the weekly summary's failed-request count leaves it out.
    slept_only = counts_of(tmp_path, [{"slept": 3}])
    assert failure_lines(failure_summary(slept_only, ["2026-09-01"])) == [
        "", "Failures over these days: 3 while the Mac slept"]
    assert digest_part(slept_only, ["2026-09-01"]) == "no failed requests"


def test_the_check_alerts_on_a_day_of_failed_requests_and_records_it(tmp_path, capsys):
    failure_days(tmp_path / "logs", [{}] * 6 + [{"errors": 5, "version": "2.1.280"}])
    save_state(tmp_path / "state.json", new_state())
    assert run_check(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 8),
                     now=datetime(2026, 9, 8, 9, 0, tzinfo=timezone.utc), digest=False) == 0
    out = capsys.readouterr().out
    assert "ccdrift: requests failing: 5 requests failed on 2026-09-07 (5 overloaded)" in out
    assert "on Claude Code 2.1.280" in out
    state = load_state(tmp_path / "state.json")
    assert [e["since"] for e in state["failed_requests"]] == ["2026-09-07"]
    assert state["cut_short"] == []


def test_the_check_alerts_when_responses_are_cut_short(tmp_path, capsys):
    failure_days(tmp_path / "logs", [{}] * 6 + [{"truncated": 6}])
    save_state(tmp_path / "state.json", new_state())
    assert run_check(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 8),
                     now=datetime(2026, 9, 8, 9, 0, tzinfo=timezone.utc), digest=False) == 0
    assert ("ccdrift: responses cut short: 6 of 60 main-thread responses stopped at the token limit on 2026-09-07 "
            "(10.00%)") in capsys.readouterr().out


def test_a_quiet_history_alerts_about_nothing(tmp_path, capsys):
    failure_days(tmp_path / "logs", [{}] * 7)
    save_state(tmp_path / "state.json", new_state())
    assert run_check(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 8),
                     now=datetime(2026, 9, 8, 9, 0, tzinfo=timezone.utc), digest=False) == 0
    assert "no alerts" in capsys.readouterr().out
