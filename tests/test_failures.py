"""Failed requests and responses cut short: what the parser keeps, what a day's counts
say, and when the check alerts."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from ccdrift.check import run_check
from ccdrift.failures import (FAILURE_DAY_COLUMNS, _judged_days, cut_short, digest_part, failing_requests,
                              failure_counts, failure_lines, failure_summary, judged_failures)
from ccdrift.history import History, load_history
from ccdrift.logs import judged_turns, parse_all, parse_source
from ccdrift.state import load_state, new_state, save_state
from ccdrift.texts import cut_short_line, cut_short_message, failure_line, requests_message
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


@pytest.mark.parametrize("status", [float("inf"), float("nan"), 1e300, 2**70, 10**400])
def test_an_error_status_no_http_status_can_be_reads_as_none(tmp_path, status):
    # JSON parsing accepts Infinity, NaN and integers of any size; SQLite stores none of them.
    banner = api_error(at(0), kind="overloaded", version="2.1.226")
    banner["apiErrorStatus"] = status
    failures = parsed_failures(tmp_path, [banner])
    assert pd.isna(failures.iloc[0]["status"])


def test_a_retry_record_counts_and_the_no_response_stub_doesnt(tmp_path):
    failures = parsed_failures(tmp_path, [retry_record(at(0), version="2.1.226"), no_response_stub(at(60))])
    assert list(failures["kind"]) == ["retry"]


def test_one_request_retried_several_times_is_one_failure(tmp_path):
    # Claude Code writes a record per attempt, so a flaky request retried twice logs
    # three: only the first attempt is a failed request. A transcript old enough to log
    # no attempt number still counts.
    records = [retry_record(at(k), attempt=k + 1, version="2.1.226") for k in range(3)]
    assert list(parsed_failures(tmp_path, records)["kind"]) == ["retry"]
    older = [retry_record(at(0), attempt=None, version="2.1.226")]
    assert list(parsed_failures(tmp_path, older)["kind"]) == ["retry"]
    # Only a later attempt is dropped. An attempt number ccdrift can't read, or a Claude
    # Code that numbered them from 0, still counts: losing a real failure is the worse
    # mistake, and a first attempt is a request either way.
    odd = [retry_record(at(0), attempt="first", version="2.1.226"), retry_record(at(60), attempt=0)]
    assert list(parsed_failures(tmp_path, odd)["kind"]) == ["retry", "retry"]


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
        assert history.meta["schema_version"] == "6"
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


def reported(state, key):
    """The episodes a state holds, without the day they were reported on: a run judging
    a fortnight at once reports on its own day, a run a day judges on the next."""
    return [{name: value for name, value in episode.items() if name != "reported_on"} for episode in state[key]]


def test_one_run_alerts_once_for_a_spell_just_as_daily_runs_would(tmp_path):
    # A first check after an upgrade, a fresh install or days with the Mac off judges a
    # whole fortnight in one call. 2026-09-08 doubles 2026-09-07 and would alert on its
    # own, but it belongs to the same spell: the run must suppress it as a check that had
    # run the morning before would have.
    counts = counts_of(tmp_path, [{}] * 6 + [{"errors": 5}, {"errors": 10}])
    once = state_with()
    assert [e["since"] for e in failing_requests(counts, once, date(2026, 9, 9))] == ["2026-09-07"]
    daily = state_with()
    for today in (date(2026, 9, 8), date(2026, 9, 9)):
        failing_requests(counts[counts["day"] < today.isoformat()], daily, today)
    assert reported(once, "failed_requests") == reported(daily, "failed_requests")


def test_one_run_alerts_once_for_a_spell_of_responses_cut_short(tmp_path):
    # The second day worsens (10 -> 15 of 60) but stays under CUT_RATIO times the first
    # day's share, so it carries the run on rather than escalating it: this test is about
    # the spell suppressing a repeat, not about magnitude, which test_a_run_that_gets_
    # three_times_worse_is_reported_again and its neighbours cover on their own.
    counts = counts_of(tmp_path, [{}] * 6 + [{"truncated": 10}, {"truncated": 15}])
    once = state_with()
    assert [e["since"] for e in cut_short(counts, once, date(2026, 9, 9))] == ["2026-09-07"]
    daily = state_with()
    for today in (date(2026, 9, 8), date(2026, 9, 9)):
        cut_short(counts[counts["day"] < today.isoformat()], daily, today)
    assert reported(once, "cut_short") == reported(daily, "cut_short")


def test_a_run_that_escalates_says_the_same_in_one_call_as_day_by_day(tmp_path):
    # The escalation records its episode in the middle of the run, and the day after is
    # judged against it. A fortnight judged in one call (a first check after an upgrade,
    # or after days with the Mac off) must reach the same two words a check running each
    # morning would: this is what _judged_days being a generator is for, and an escalation
    # is the second way its state grows while it runs.
    counts = counts_of(tmp_path, [{}] * 6 + [{"truncated": 10}, {"truncated": 40}])
    once = state_with()
    assert [e["since"] for e in cut_short(counts, once, date(2026, 9, 9))] == ["2026-09-07", "2026-09-08"]
    daily = state_with()
    for today in (date(2026, 9, 8), date(2026, 9, 9)):
        cut_short(counts[counts["day"] < today.isoformat()], daily, today)
    assert reported(once, "cut_short") == reported(daily, "cut_short")


def test_a_day_alerts_when_responses_stop_at_the_token_limit_far_more_than_before(tmp_path):
    counts = counts_of(tmp_path, [{}] * 6 + [{"truncated": 5}])
    state = state_with()
    episodes = cut_short(counts, state, date(2026, 9, 8))
    assert [(e["since"], e["cut"], e["truncated"], e["refused"], e["responses"], e["before_share"])
            for e in episodes] == [("2026-09-07", 5, 5, 0, 60, 0.0)]
    assert state["cut_short"] == episodes
    few = counts_of(tmp_path, [{}] * 6 + [{"truncated": 4}])
    assert cut_short(few, state_with(), date(2026, 9, 8)) == []


def cut_days(tmp_path, days, today=date(2026, 10, 1)):
    """One CLI main-thread session a day from Sep 1, each day `(responses, cut)`: that
    many main-thread responses a minute apart, the first `cut` of them stopping at the
    token limit. Days differ in size here, which is the whole point: a share of a small
    day is not the same evidence as a share of a big one."""
    for d, (responses, cut) in enumerate(days):
        records = []
        for k in range(responses):
            ts = at(d * DAY + 60 * k)
            records += [prompt(ts, sid=f"c{d}"),
                        line(f"c{d}-{k}", text(40), ts=ts, sid=f"c{d}", version="2.1.226", entrypoint="cli",
                             stop_reason="max_tokens" if k < cut else "end_turn")]
        write(tmp_path / "logs" / f"c{d}.jsonl", records)
    tables = parse_all(tmp_path / "logs")
    return failure_counts(judged_failures(tables.failures, today), judged_turns(tables.responses, today))


def test_a_run_that_starts_too_quietly_alerts_on_its_first_day_over_the_floor(tmp_path):
    # The shape the owner's logs would have taken: a release truncates about 1% of
    # responses from 2026-09-06, but that day holds only 356 responses, so its 4 cut short
    # are under the floor and it can't be reported. 2026-09-07 is the first day that can
    # be, and only because the day before it belongs to the same run and is left out of
    # the usual level it is compared with. Judged against 1.12%, its 1.04% would have to
    # reach 3.37%, and the regression would never be reported at all.
    counts = cut_days(tmp_path, [(60, 0)] * 5 + [(356, 4), (579, 6)])
    episodes = cut_short(counts, state_with(), date(2026, 9, 8))
    assert [(e["since"], e["cut"], e["responses"], e["before_share"], e["run_days"]) for e in episodes] == [
        ("2026-09-07", 6, 579, 0.0, 1)]
    # The baseline was clean only because one day was left out of it, and the alert says so.
    assert cut_short_message(episodes[0], []) == (
        "6 of 579 main-thread responses stopped at the token limit on 2026-09-07 (1.04%), against none on the days "
        "judged in the 14 before, leaving out the 1 day of this run. A Claude Code update may have changed the "
        "output limit.")


def test_a_judged_day_comes_with_the_episodes_whose_word_covers_it(tmp_path):
    # An episode covers the days within SPELL_DAYS of it; the caller gets them so it can
    # ask whether the day is worse than what they said. This call passes no `ongoing`, so
    # the spell is the only cover and 09-10 is past it. A covered day is otherwise
    # suppressed (yielding nothing to inspect), so `louder` is given as always-true here
    # purely to let every candidate through and expose what covers each of them.
    counts = cut_days(tmp_path, [(60, 0)] * 5 + [(200, 5)] * 5)
    state = state_with()
    assert [e["since"] for e in cut_short(counts[counts["day"] <= "2026-09-06"], state, date(2026, 9, 7))] == [
        "2026-09-06"]
    covered = [(str(row["day"]), [e["since"] for e in covering])
               for row, _, covering in _judged_days(counts, "cut_short", state, date(2026, 9, 11),
                                                    lambda row, before: True,
                                                    louder=lambda episode, row: True)]
    assert covered == [("2026-09-06", ["2026-09-06"]), ("2026-09-07", ["2026-09-06"]),
                       ("2026-09-08", ["2026-09-06"]), ("2026-09-09", ["2026-09-06"]),
                       ("2026-09-10", [])]


def test_the_days_that_carry_a_reported_run_on_stay_silent(tmp_path):
    # 2026-09-07 is reported; 09-08 to 09-12 hold the same regression at the same level,
    # and the spell window covers only the first three of them. Every one of them is a day
    # the run carries on, so none is reported again.
    counts = cut_days(tmp_path, [(60, 0)] * 5 + [(200, 2)] + [(200, 5)] * 6)
    state = state_with()
    assert [e["since"] for e in cut_short(counts, state, date(2026, 9, 13))] == ["2026-09-07"]
    assert [e["since"] for e in state["cut_short"]] == ["2026-09-07"]
    assert cut_short(counts, state, date(2026, 9, 13)) == []


def test_an_episode_that_has_left_the_window_silences_nothing(tmp_path):
    # A regression 20 days long is reported when it starts and again once the reported day
    # has fallen out of the 14 days a day is judged against: an episode that is no longer
    # among the days judged says nothing about today, and a run that outlives the window
    # must not go unreported for as long as it lasts.
    counts = cut_days(tmp_path, [(60, 0)] * 5 + [(200, 5)] * 20)
    state = state_with()
    said = []
    for day in counts["day"].astype(str):
        today = date.fromisoformat(day) + timedelta(days=1)
        said += [e["since"] for e in cut_short(counts[counts["day"] <= day], state, today)]
    assert said == ["2026-09-06", "2026-09-21"]


def test_a_fresh_run_alerts_again_once_the_level_has_been_back_to_normal(tmp_path):
    # The check reports 2026-09-06 the morning after it; the fortnight that follows is
    # clean, so 2026-09-21 is a new regression rather than the old one carrying on, and
    # the run it starts is reported too.
    counts = cut_days(tmp_path, [(60, 0)] * 5 + [(200, 5)] + [(60, 0)] * 14 + [(200, 5)])
    state = state_with()
    first = cut_short(counts[counts["day"] < "2026-09-07"], state, date(2026, 9, 7))
    assert [e["since"] for e in first] == ["2026-09-06"]
    assert [e["since"] for e in cut_short(counts, state, date(2026, 9, 22))] == ["2026-09-21"]


def test_a_run_that_gets_three_times_worse_is_reported_again(tmp_path):
    # 2026-09-06 opens the run at 2.50% and is reported. 09-07 to 09-09 hold it there and
    # stay silent. 09-10 cuts 8.00%, over three times what ccdrift said, so it says so,
    # naming the level it was last told about rather than the baseline before the run.
    counts = cut_days(tmp_path, [(60, 0)] * 5 + [(200, 5)] * 4 + [(200, 16)])
    state = state_with()
    episodes = cut_short(counts, state, date(2026, 9, 11))
    assert [(e["since"], e.get("worse_than")) for e in episodes] == [
        ("2026-09-06", None), ("2026-09-10", {"share": 0.025, "since": "2026-09-06"})]


def test_a_run_that_worsens_by_less_than_three_times_stays_silent(tmp_path):
    # The same run rising from 2.50% to 6.00%: worse, but not the three times a first alert
    # needs over its baseline, so it is the same regression carrying on and says nothing.
    counts = cut_days(tmp_path, [(60, 0)] * 5 + [(200, 5)] * 4 + [(200, 12)])
    assert [e["since"] for e in cut_short(counts, state_with(), date(2026, 9, 11))] == ["2026-09-06"]


def test_an_escalation_speaks_inside_the_spell_window(tmp_path):
    # The day after the one reported is three times worse. SPELL_DAYS would have swallowed
    # it: the whole point is that a regression deepening is not one burst reported twice.
    counts = cut_days(tmp_path, [(60, 0)] * 5 + [(200, 5), (200, 16)])
    episodes = cut_short(counts, state_with(), date(2026, 9, 8))
    assert [(e["since"], e.get("worse_than")) for e in episodes] == [
        ("2026-09-06", None), ("2026-09-07", {"share": 0.025, "since": "2026-09-06"})]


def test_the_next_word_is_owed_three_times_the_last_one_not_the_first(tmp_path):
    # 09-06 at 2.50% is reported, 09-07 at 8.00% escalates. 09-08 at 10.00% is over three
    # times the first level but not the second, so it stays quiet; 09-09 at 25.00% clears
    # the second and speaks. Each word is judged against the last one said, which is why an
    # escalation records its own episode instead of amending the old one.
    counts = cut_days(tmp_path, [(60, 0)] * 5 + [(200, 5), (200, 16), (200, 20), (200, 50)])
    episodes = cut_short(counts, state_with(), date(2026, 9, 10))
    assert [(e["since"], e.get("worse_than")) for e in episodes] == [
        ("2026-09-06", None), ("2026-09-07", {"share": 0.025, "since": "2026-09-06"}),
        ("2026-09-09", {"share": 0.08, "since": "2026-09-07"})]


def test_a_second_alert_names_the_level_it_escalated_from(tmp_path):
    episode = {"since": "2026-09-20", "days": ["2026-09-20"], "cut": 60, "truncated": 60, "refused": 0,
               "responses": 400, "before_share": 0.0, "run_days": 3, "reported_on": "2026-09-21",
               "worse_than": {"share": 0.006, "since": "2026-09-03"}}
    assert cut_short_message(episode, ["2.1.281"]) == (
        "60 of 400 main-thread responses stopped at the token limit on 2026-09-20 (15.00%), against the 0.60% "
        "reported on 2026-09-03, on Claude Code 2.1.281. A Claude Code update may have changed the output limit.")
    assert cut_short_line(episode) == (
        "responses cut short on 2026-09-20: 60 of 400 main-thread responses, against the 0.60% reported on "
        "2026-09-03")


def test_an_episode_recorded_before_escalations_still_reads(tmp_path):
    # An episode written by 0.7.0 to 0.9.0 has no "worse_than" and must read as it always
    # did: the state file is not rewritten on upgrade.
    episode = {"since": "2026-09-07", "days": ["2026-09-07"], "cut": 6, "truncated": 6, "refused": 0,
               "responses": 579, "before_share": 0.0, "run_days": 1, "reported_on": "2026-09-08"}
    assert cut_short_message(episode, []) == (
        "6 of 579 main-thread responses stopped at the token limit on 2026-09-07 (1.04%), against none on the days "
        "judged in the 14 before, leaving out the 1 day of this run. A Claude Code update may have changed the "
        "output limit.")
    assert cut_short_line(episode) == (
        "responses cut short on 2026-09-07: 6 of 579 main-thread responses, leaving out the 1 day of this run")


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


def test_a_day_too_quiet_to_be_active_is_judged_for_its_failed_requests(tmp_path):
    # The harder the API fails, the fewer responses a day holds: 8 failed requests on a
    # day with 10 responses is the day the rule is for, so it alerts although it is too
    # quiet ever to be a baseline. The cut-short rule is a share of a day's responses,
    # which 10 responses can't support, so it leaves that day alone.
    failure_days(tmp_path / "logs", [{}] * 5)
    records = []
    for k in range(10):
        ts = at(5 * DAY + 60 * k)
        records += [prompt(ts, sid="quiet"),
                    line(f"quiet-{k}", text(40), ts=ts, sid="quiet", version="2.1.226", entrypoint="cli",
                         stop_reason="max_tokens" if k < 8 else "end_turn")]
    records += [api_error(at(5 * DAY + 900 + j), sid="quiet", version="2.1.226") for j in range(8)]
    write(tmp_path / "logs" / "quiet.jsonl", records)

    today = date(2026, 9, 7)
    tables = parse_all(tmp_path / "logs")
    counts = failure_counts(judged_failures(tables.failures, today), judged_turns(tables.responses, today))
    day = counts[counts["day"] == "2026-09-06"].iloc[0]
    assert (int(day["responses"]), int(day["requests"]), int(day["truncated"])) == (10, 8, 8)
    assert [(e["since"], e["requests"], e["before"]) for e in failing_requests(counts, state_with(), today)] == [
        ("2026-09-06", 8, 0)]
    assert cut_short(counts, state_with(), today) == []


def test_a_day_whose_requests_all_failed_still_gets_a_row(tmp_path):
    # Nothing was answered, so the day holds no turns at all. Without a row of its own
    # the counts would hide the worst day of an outage from the rule that watches for it.
    failure_days(tmp_path / "logs", [{}])
    write(tmp_path / "logs" / "down.jsonl", [api_error(at(DAY + j), sid="down", version="2.1.226") for j in range(7)])
    today = date(2026, 9, 3)
    tables = parse_all(tmp_path / "logs")
    counts = failure_counts(judged_failures(tables.failures, today), judged_turns(tables.responses, today))
    assert list(counts["day"]) == ["2026-09-01", "2026-09-02"]
    down = counts.iloc[1]
    assert (int(down["responses"]), int(down["requests"]), int(down["overloaded"]), int(down["truncated"]),
            int(down["refused"])) == (0, 7, 7, 0, 0)
    assert all(counts[name].dtype == int for name in FAILURE_DAY_COLUMNS[1:])


def test_the_messages_name_the_kinds_the_counts_and_the_version():
    episode = {"since": "2026-09-20", "days": ["2026-09-20"], "requests": 9,
               "kinds": {"overloaded": 7, "retry": 2}, "before": 1, "reported_on": "2026-09-21"}
    assert requests_message(episode, ["2.1.280"]) == (
        "9 requests failed on 2026-09-20 (7 overloaded, 2 retried), against at most 1 a day on the days judged in "
        "the 14 before, on Claude Code 2.1.280. Claude Code retries these itself; a run of them points at the API "
        "or your connection, not your setup.")
    clean = {**episode, "before": 0}
    assert requests_message(clean, []) == (
        "9 requests failed on 2026-09-20 (7 overloaded, 2 retried), against none on the days judged in the 14 "
        "before. Claude Code retries these itself; a run of them points at the API or your connection, not your "
        "setup.")
    cut = {"since": "2026-09-20", "days": ["2026-09-20"], "cut": 8, "truncated": 8, "refused": 0,
           "responses": 640, "before_share": 0.0014, "run_days": 0, "reported_on": "2026-09-21"}
    assert cut_short_message(cut, []) == (
        "8 of 640 main-thread responses stopped at the token limit on 2026-09-20 (1.25%), against at most 0.14% a "
        "day on the days judged in the 14 before. A Claude Code update may have changed the output limit.")
    # The days of the day's own run are left out of that comparison, so when there were
    # any, both the message and the status line say how many rather than let "against
    # none" stand for a fortnight that wasn't clean.
    carried = {**cut, "before_share": 0.0, "run_days": 3}
    assert cut_short_message(carried, ["2.1.280"]) == (
        "8 of 640 main-thread responses stopped at the token limit on 2026-09-20 (1.25%), against none on the days "
        "judged in the 14 before, leaving out the 3 days of this run, on Claude Code 2.1.280. A Claude Code update "
        "may have changed the output limit.")
    assert cut_short_line(carried) == (
        "responses cut short on 2026-09-20: 8 of 640 main-thread responses, leaving out the 3 days of this run")
    assert cut_short_line({**carried, "run_days": 1}).endswith("leaving out the 1 day of this run")
    assert failure_line(episode) == (
        "requests failing on 2026-09-20: 9 (7 overloaded, 2 retried), against at most 1 a day on the days judged "
        "in the 14 before")
    assert failure_line(clean) == (
        "requests failing on 2026-09-20: 9 (7 overloaded, 2 retried), against none on the days judged in the 14 "
        "before")
    assert cut_short_line(cut) == "responses cut short on 2026-09-20: 8 of 640 main-thread responses"
    cut_clean = {**cut, "before_share": 0.0}
    assert cut_short_message(cut_clean, []) == (
        "8 of 640 main-thread responses stopped at the token limit on 2026-09-20 (1.25%), against none on the days "
        "judged in the 14 before. A Claude Code update may have changed the output limit.")
    # An episode recorded before 0.7.0 knew about runs still reads.
    older = {name: value for name, value in cut_clean.items() if name != "run_days"}
    assert cut_short_message(older, []) == cut_short_message(cut_clean, [])
    assert cut_short_line(older) == cut_short_line(cut_clean)


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


def test_the_check_alerts_twice_when_a_run_escalates_and_the_escalation_survives_state(tmp_path, capsys):
    # 2026-09-06 opens the run at 2.50% and is reported; 2026-09-07 triples it to 8.00%,
    # which speaks again in the same check run, naming the level it escalated from.
    failure_days(tmp_path / "logs", [{}] * 5 + [{"truncated": 5}, {"truncated": 16}], per_day=200)
    save_state(tmp_path / "state.json", new_state())
    assert run_check(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 8),
                     now=datetime(2026, 9, 8, 9, 0, tzinfo=timezone.utc), digest=False) == 0
    out = capsys.readouterr().out
    assert ("ccdrift: responses cut short: 5 of 200 main-thread responses stopped at the token limit on 2026-09-06 "
            "(2.50%), against none on the days judged in the 14 before, on Claude Code 2.1.226") in out
    assert ("ccdrift: responses cut short: 16 of 200 main-thread responses stopped at the token limit on 2026-09-07 "
            "(8.00%), against the 2.50% reported on 2026-09-06, on Claude Code 2.1.226") in out
    # The escalation episode round-trips through save_state/load_state as JSON, worse_than intact.
    state = load_state(tmp_path / "state.json")
    assert [(e["since"], e.get("worse_than")) for e in state["cut_short"]] == [
        ("2026-09-06", None), ("2026-09-07", {"share": 0.025, "since": "2026-09-06"})]


def test_a_quiet_history_alerts_about_nothing(tmp_path, capsys):
    failure_days(tmp_path / "logs", [{}] * 7)
    save_state(tmp_path / "state.json", new_state())
    assert run_check(tmp_path / "logs", tmp_path / "state.json", today=date(2026, 9, 8),
                     now=datetime(2026, 9, 8, 9, 0, tzinfo=timezone.utc), digest=False) == 0
    assert "no alerts" in capsys.readouterr().out
