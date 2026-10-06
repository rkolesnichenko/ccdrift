"""What a point of the usage limit costs: the walk over the quota samples, and its prices."""

import json
from datetime import date, datetime, timezone

import pytest

from ccdrift import exchange
from ccdrift.cli import main
from ccdrift.exchange import Step, day_rows, quota_summary, readings, run_quota, steps
from ccdrift.texts import empty_rise_line
from ccdrift.logs import parse_all
from ccdrift.prices import Price
from tests.helpers import line, text, write


def sample(stamp, used, resets=1_790_000_000):
    """A quota sample as `ccdrift status --short --stdin` keeps one."""
    return {"at": stamp, "version": "2.1.289", "model": "claude-opus-5",
            "seven_day": {"used_percentage": used, "resets_at": resets}}


def utc(stamp):
    return datetime.fromisoformat(stamp).astimezone(timezone.utc)


def walk(samples):
    return list(steps(readings(samples)))


def test_readings_keep_only_samples_with_a_time_a_7_day_share_and_its_reset_in_time_order():
    samples = [sample("2026-09-05T12:00:00+00:00", 14), {"at": "2026-09-05T11:00:00+00:00"},
               {"at": "2026-09-05T11:30:00+00:00", "seven_day": {"used_percentage": True, "resets_at": 1}},
               {"at": "nonsense", "seven_day": {"used_percentage": 3, "resets_at": 1}},
               sample("2026-09-05T13:00:00+03:00", 12)]
    assert readings(samples) == [(utc("2026-09-05T10:00:00+00:00"), 12.0, 1_790_000_000),
                                 (utc("2026-09-05T12:00:00+00:00"), 14.0, 1_790_000_000)]


def test_a_step_is_credited_only_when_the_windows_previous_sample_is_on_the_same_utc_day():
    found = walk([sample("2026-09-05T10:00:00+00:00", 10), sample("2026-09-05T20:00:00+00:00", 12),
                  sample("2026-09-09T09:00:00+00:00", 40), sample("2026-09-09T20:00:00+00:00", 41)])
    assert [(step.day, step.points) for step in found] == [("2026-09-05", 2.0), ("2026-09-09", 1.0)]


def test_a_steps_points_are_how_far_it_raised_the_highest_share_so_far_through_stale_samples():
    # A session idle since 40% interleaving with a busy one at 45%, then 46%.
    found = walk([sample("2026-09-05T10:00:00+00:00", 40), sample("2026-09-05T10:05:00+00:00", 45),
                  sample("2026-09-05T10:06:00+00:00", 40), sample("2026-09-05T10:07:00+00:00", 46)])
    assert [step.points for step in found] == [5.0, 0.0, 1.0]


def test_a_steps_since_is_when_the_highest_share_was_set_and_never_before_that_days_first_sample():
    found = walk([sample("2026-09-05T22:00:00+00:00", 30),
                  sample("2026-09-06T08:00:00+00:00", 30), sample("2026-09-06T08:30:00+00:00", 31),
                  sample("2026-09-06T08:45:00+00:00", 31), sample("2026-09-06T09:00:00+00:00", 29),
                  sample("2026-09-06T09:10:00+00:00", 33)])
    # 30 was set on Sep 5, so Sep 6's first rise counts from Sep 6's first sample; neither 31
    # again at 08:45 nor the stale 29 at 09:00 moves it, so the rise to 33 counts from 08:30.
    assert [(step.when, step.points, step.since) for step in found] == [
        (utc("2026-09-06T08:30:00+00:00"), 1.0, utc("2026-09-06T08:00:00+00:00")),
        (utc("2026-09-06T08:45:00+00:00"), 0.0, utc("2026-09-06T08:30:00+00:00")),
        (utc("2026-09-06T09:00:00+00:00"), 0.0, utc("2026-09-06T08:30:00+00:00")),
        (utc("2026-09-06T09:10:00+00:00"), 2.0, utc("2026-09-06T08:30:00+00:00"))]


def test_a_reset_day_steps_through_each_window_on_its_own():
    # The old window's 96 after the reset is a stale reading from a session idle since then.
    found = walk([sample("2026-09-06T01:00:00+00:00", 90, resets=1), sample("2026-09-06T05:00:00+00:00", 95, resets=1),
                  sample("2026-09-06T06:10:00+00:00", 2, resets=2), sample("2026-09-06T07:00:00+00:00", 5, resets=2),
                  sample("2026-09-06T07:30:00+00:00", 96, resets=1)])
    assert found == [Step(utc("2026-09-06T05:00:00+00:00"), "2026-09-06", 1, 5.0, utc("2026-09-06T01:00:00+00:00")),
                     Step(utc("2026-09-06T07:00:00+00:00"), "2026-09-06", 2, 3.0, utc("2026-09-06T06:10:00+00:00")),
                     Step(utc("2026-09-06T07:30:00+00:00"), "2026-09-06", 1, 1.0, utc("2026-09-06T05:00:00+00:00"))]


TODAY = date(2026, 9, 10)
OPUS = {"claude-opus-5": Price(input_rate=5e-6, cache_read_rate=5e-7, output_rate=25e-6, web_search_rate=0.0,
                               residual=0.0, rows=9)}
# One response at OPUS: 10 input and a million output tokens.
ONE = 25.00005


def answer(mid, stamp, **kw):
    """A response at `stamp` (UTC, written as Claude Code writes it) costing ONE at OPUS."""
    return line(mid, text(40), ts=stamp.replace("+00:00", "Z"), out=1_000_000, **{"entrypoint": "cli", **kw})


def summary_of(tmp_path, monkeypatch, records, samples, days=14, today=TODAY):
    write(tmp_path / "p" / "s1.jsonl", records)
    monkeypatch.setattr(exchange, "fitted_prices", lambda tables: OPUS)
    return quota_summary(parse_all(tmp_path), samples, days, today)


def test_a_days_dollars_are_this_machines_responses_in_its_sampled_hours_both_threads_and_sdk_included(
        tmp_path, monkeypatch):
    samples = [sample("2026-09-01T10:00:00+00:00", 10), sample("2026-09-01T11:00:00+00:00", 12),
               sample("2026-09-01T12:00:00+00:00", 14)]
    records = [answer("m0", "2026-09-01T09:30:00+00:00"), answer("m1", "2026-09-01T10:30:00+00:00"),
               answer("a1", "2026-09-01T10:40:00+00:00", sidechain=True),
               answer("x1", "2026-09-01T11:30:00+00:00", entrypoint="sdk-py"),
               answer("m2", "2026-09-01T12:30:00+00:00")]
    summary = summary_of(tmp_path, monkeypatch, records, samples)
    assert summary["days"] == [pytest.approx({"day": "2026-09-01", "points": 4.0, "dollars": 3 * ONE,
                                              "rate": 3 * ONE / 4})]
    assert summary["sdk_share"] == pytest.approx(1 / 3)


def test_a_day_whose_unpriced_models_are_material_has_no_dollars_and_no_rate(tmp_path):
    samples = [sample("2026-09-01T10:00:00+00:00", 10), sample("2026-09-01T11:00:00+00:00", 12)]
    write(tmp_path / "p" / "s1.jsonl", [answer("m1", "2026-09-01T10:30:00+00:00"),
                                         answer("f1", "2026-09-01T10:40:00+00:00", model="claude-fable-5-1")])
    rows = day_rows(parse_all(tmp_path).responses, OPUS, readings(samples), ["2026-09-01"])
    assert rows == [{"day": "2026-09-01", "points": 2.0, "dollars": None, "rate": None}]


def test_a_day_that_gained_no_points_keeps_its_dollars_and_has_no_rate(tmp_path):
    samples = [sample("2026-09-01T10:00:00+00:00", 10), sample("2026-09-01T11:00:00+00:00", 10)]
    write(tmp_path / "p" / "s1.jsonl", [answer("m1", "2026-09-01T10:30:00+00:00")])
    rows = day_rows(parse_all(tmp_path).responses, OPUS, readings(samples), ["2026-09-01"])
    assert rows == [pytest.approx({"day": "2026-09-01", "points": 0.0, "dollars": ONE, "rate": None})]


def test_the_windows_rate_is_its_dollars_over_its_points_and_not_the_median_of_its_days(tmp_path, monkeypatch):
    # Sep 3 gains 5 points with no response on this machine, so it has no dollars, and its
    # points stay out of the window's rate.
    samples = [sample("2026-09-01T10:00:00+00:00", 10), sample("2026-09-01T12:00:00+00:00", 20),
               sample("2026-09-02T10:00:00+00:00", 20), sample("2026-09-02T12:00:00+00:00", 21),
               sample("2026-09-03T10:00:00+00:00", 30), sample("2026-09-03T12:00:00+00:00", 35)]
    records = [answer(f"m{i}", f"2026-09-01T11:0{i}:00+00:00") for i in range(4)]
    records += [answer(f"n{i}", f"2026-09-02T11:0{i}:00+00:00") for i in range(2)]
    summary = summary_of(tmp_path, monkeypatch, records, samples)
    assert summary["days"][2] == {"day": "2026-09-03", "points": 5.0, "dollars": None, "rate": None}
    assert (summary["points"], summary["dollars"]) == (11.0, pytest.approx(6 * ONE))
    assert summary["rate"] == pytest.approx(6 * ONE / 11)
    assert summary["day_rates"] == pytest.approx({"median": 1.2 * ONE, "low": 0.4 * ONE, "high": 2 * ONE})


def test_a_rise_with_no_response_since_the_highest_share_was_set_is_empty_though_a_stale_sample_came_between(
        tmp_path, monkeypatch):
    # The response at 10:10 bought the rise to 42, though the stale 38 at 10:20 sits between
    # them; nothing on this machine bought the rise to 43 in the hour before 11:30.
    samples = [sample("2026-09-01T10:00:00+00:00", 40), sample("2026-09-01T10:20:00+00:00", 38),
               sample("2026-09-01T10:30:00+00:00", 42), sample("2026-09-01T11:30:00+00:00", 43),
               sample("2026-09-01T11:45:00+00:00", 43)]
    # The 11:45 sample raised nothing, so though nothing ran before it, it is no empty rise.
    summary = summary_of(tmp_path, monkeypatch, [answer("m1", "2026-09-01T10:10:00+00:00")], samples)
    assert summary["empty"] == [{"when": "2026-09-01T11:30:00+00:00", "day": "2026-09-01", "points": 1.0,
                                 "minutes": 60}]


def test_a_response_with_no_price_still_counts_as_activity_for_an_empty_rise(tmp_path, monkeypatch):
    samples = [sample("2026-09-01T10:00:00+00:00", 40), sample("2026-09-01T11:00:00+00:00", 41)]
    records = [answer("f1", "2026-09-01T10:30:00+00:00", model="claude-fable-5-1")]
    assert summary_of(tmp_path, monkeypatch, records, samples)["empty"] == []


def test_the_window_is_the_last_days_with_a_step_and_today_is_left_out(tmp_path, monkeypatch):
    samples = [sample(f"2026-09-0{d}T10:00:00+00:00", 10 * d) for d in (1, 2, 3, 4)]
    samples += [sample(f"2026-09-0{d}T11:00:00+00:00", 10 * d + 1) for d in (1, 2, 3, 4)]
    summary = summary_of(tmp_path, monkeypatch, [answer("m1", "2026-09-02T10:30:00+00:00")], samples, days=2,
                         today=date(2026, 9, 4))
    assert (summary["first"], summary["last"], [row["day"] for row in summary["days"]]) == (
        "2026-09-02", "2026-09-03", ["2026-09-02", "2026-09-03"])
    # Sep 1 and Sep 4 rose with nothing spent too, but they are outside the window.
    assert [rise["day"] for rise in summary["empty"]] == ["2026-09-03"]


def test_a_response_inside_two_overlapping_spans_of_a_reset_day_is_counted_once(tmp_path, monkeypatch):
    # After the 06:00 reset an idle session still shows the old window, at 07:30, so the old
    # window's span (01:00-07:30) covers the new window's (06:10-07:00).
    samples = [sample("2026-09-02T01:00:00+00:00", 90, resets=1), sample("2026-09-02T05:00:00+00:00", 95, resets=1),
               sample("2026-09-02T06:10:00+00:00", 2, resets=2), sample("2026-09-02T07:00:00+00:00", 5, resets=2),
               sample("2026-09-02T07:30:00+00:00", 96, resets=1)]
    summary = summary_of(tmp_path, monkeypatch, [answer("m1", "2026-09-02T06:30:00+00:00")], samples)
    assert summary["days"] == [pytest.approx({"day": "2026-09-02", "points": 9.0, "dollars": ONE, "rate": ONE / 9})]


def test_the_summary_names_the_priced_and_unpriced_models_and_counts_the_samples(tmp_path, monkeypatch):
    samples = [sample("2026-09-01T10:00:00+00:00", 10), sample("2026-09-01T11:00:00+00:00", 12)]
    records = [answer("m1", "2026-09-01T10:30:00+00:00"),
               line("f1", text(40), ts="2026-09-01T10:40:00Z", entrypoint="cli", model="claude-fable-5-1")]
    summary = summary_of(tmp_path, monkeypatch, records, samples)
    assert summary["priced_models"] == ["claude-opus-5"]
    assert [model["model"] for model in summary["unpriced"]] == ["claude-fable-5-1"]
    assert summary["samples"] == 2


def two_days(root, project="p"):
    """Sep 1: 4 points over 10:00-12:00 while a main-thread, a subagent and an Agent SDK
    response ran in it. Sep 2: 1 point bought by a response at 10:15, then 2 more by 11:30
    with nothing on this machine since 10:30."""
    write(root / project / "s1.jsonl", [
        answer("m0", "2026-09-01T09:30:00+00:00"), answer("m1", "2026-09-01T10:30:00+00:00"),
        answer("a1", "2026-09-01T10:40:00+00:00", sidechain=True),
        answer("x1", "2026-09-01T11:30:00+00:00", entrypoint="sdk-py"),
        answer("m2", "2026-09-02T10:15:00+00:00")])
    return [sample("2026-09-01T10:00:00+00:00", 10), sample("2026-09-01T11:00:00+00:00", 12),
            sample("2026-09-01T12:00:00+00:00", 14), sample("2026-09-02T10:00:00+00:00", 14),
            sample("2026-09-02T10:30:00+00:00", 15), sample("2026-09-02T11:30:00+00:00", 17)]


def samples_beside(state, samples):
    """The quota samples file `ccdrift status --short --stdin` keeps beside the state file."""
    state.parent.mkdir(parents=True, exist_ok=True)
    (state.parent / "quota.jsonl").write_text("".join(json.dumps(one) + "\n" for one in samples))


EXPECTED = """\
Quota points of the 7-day limit, priced at list from this machine's responses in the hours the status line sampled.
2026-09-01 to 2026-09-02: 7 points, $100.00 list, $14.29 a point (days: median $13.54, lowest $8.33, highest $18.75).

day           points      list $    $ per point
2026-09-01         4          75          18.75
2026-09-02         3          25           8.33

Points rose while this machine spent nothing (UTC):
  09-02 11:30, 2 points, nothing on this machine in the 60 minutes before
Counted: every response on this machine, both threads, Agent SDK sessions included (25% of these dollars).
This machine only: use on claude.ai or another device raises points with no dollars here.
Per-model limits aren't in the status line, so the rate is for all models together.
"""


def test_the_command_prints_the_windows_rate_a_row_a_day_the_empty_rises_and_what_it_counted(
        tmp_path, capsys, monkeypatch):
    state = tmp_path / "home" / "state.json"
    samples_beside(state, two_days(tmp_path / "logs"))
    monkeypatch.setattr(exchange, "fitted_prices", lambda tables: OPUS)
    monkeypatch.setattr(exchange, "default_source", lambda: tmp_path / "logs")
    assert run_quota(tmp_path / "logs", state, today=TODAY) == 0
    assert capsys.readouterr().out == EXPECTED


def test_with_no_empty_rise_the_command_says_none_rose_and_names_a_model_with_no_price(
        tmp_path, capsys, monkeypatch):
    state = tmp_path / "home" / "state.json"
    samples_beside(state, [sample("2026-09-01T10:00:00+00:00", 10), sample("2026-09-01T11:00:00+00:00", 12)])
    write(tmp_path / "logs" / "p" / "s1.jsonl", [
        answer("m1", "2026-09-01T10:30:00+00:00"),
        line("f1", text(40), ts="2026-09-01T10:40:00Z", entrypoint="cli", model="claude-fable-5-1")])
    monkeypatch.setattr(exchange, "fitted_prices", lambda tables: OPUS)
    assert run_quota(tmp_path / "logs", state, today=TODAY) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "No point rose while this machine spent nothing." in lines
    assert ("No price for claude-fable-5-1 (<0.1%): its spend is left out of the dollars on a day where it stays "
            "under 1% of that day's tokens, and a day where it reaches 1% has no dollar figure.") in lines


def test_with_no_cost_records_every_figure_is_a_dash_and_the_window_says_no_day_has_one(tmp_path, capsys):
    state = tmp_path / "home" / "state.json"
    samples_beside(state, two_days(tmp_path / "logs"))
    assert run_quota(tmp_path / "logs", state, today=TODAY) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "2026-09-01 to 2026-09-02: no day has both points and a dollar figure."
    assert lines[2] == ("The window's figures leave out 2 days that have points and no dollar figure, and their "
                        "7 points.")
    assert lines[5].split() == ["2026-09-01", "4", "-", "-"]
    assert "No cost records price these days, so they have no dollar figures." in lines
    assert "Counted: every response on this machine, both threads, Agent SDK sessions included." in lines


def test_the_json_is_the_summary_with_empty_rises_as_day_and_points_and_names_no_path_or_session(
        tmp_path, capsys, monkeypatch):
    state = tmp_path / "home" / "state.json"
    samples_beside(state, two_days(tmp_path / "logs", project="secret-client"))
    monkeypatch.setattr(exchange, "fitted_prices", lambda tables: OPUS)
    assert run_quota(tmp_path / "logs", state, as_json=True, today=TODAY) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert set(payload) == {"first", "last", "days", "points", "dollars", "rate", "day_rates", "left_out",
                            "sdk_share", "priced_models", "unpriced", "responses", "empty", "samples",
                            "default_source"}
    assert payload["empty"] == [{"day": "2026-09-02", "points": 2.0}]
    assert payload["rate"] == pytest.approx(4 * ONE / 7)
    assert "secret-client" not in out and str(tmp_path) not in out and '"s1"' not in out and "11:30" not in out


def test_with_no_seven_day_sample_it_says_how_to_start_before_reading_the_history(tmp_path, capsys):
    state = tmp_path / "home" / "state.json"
    for samples in ([], [{"at": "2026-09-01T10:00:00+00:00", "five_hour": {"used_percentage": 5, "resets_at": 1}}]):
        samples_beside(state, samples)
        assert run_quota(tmp_path / "nowhere", state, today=TODAY) == 2
        assert capsys.readouterr().err == (
            "No status line sample carries the 7-day limit yet: ccdrift keeps them once your status line runs "
            "`ccdrift status --short --stdin`, on Pro and Max plans.\n")
    assert not (state.parent / "history.sqlite").exists()


def test_samples_that_never_show_the_limit_twice_in_one_past_day_are_refused_with_their_count(tmp_path, capsys):
    state = tmp_path / "home" / "state.json"
    samples_beside(state, [sample("2026-09-01T10:00:00+00:00", 10), sample("2026-09-02T10:00:00+00:00", 12),
                           sample("2026-09-10T08:00:00+00:00", 13), sample("2026-09-10T09:00:00+00:00", 14)])
    assert run_quota(tmp_path / "nowhere", state, today=TODAY) == 2
    assert capsys.readouterr().err == ("None of the status line samples on file (4) shows the 7-day limit twice in "
                                       "one day before today, so there is no point to price yet.\n")


def test_an_unusable_store_exits_1_and_a_source_with_no_transcripts_exits_2(tmp_path, capsys):
    state = tmp_path / "home" / "state.json"
    samples_beside(state, two_days(tmp_path / "logs"))
    (state.parent / "history.sqlite").write_text("not a database")
    assert run_quota(tmp_path / "logs", state, today=TODAY) == 1
    assert "Move it aside" in capsys.readouterr().err
    other = tmp_path / "other" / "state.json"
    samples_beside(other, two_days(tmp_path / "unused"))
    (tmp_path / "empty").mkdir()
    assert run_quota(tmp_path / "empty", other, today=TODAY) == 2
    assert "No Claude Code transcripts found" in capsys.readouterr().err


def test_the_command_line_passes_its_days_and_json_through(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(exchange, "run_quota", lambda *args, **kwargs: calls.append(kwargs) or 0)
    paths = ["--source", str(tmp_path / "logs"), "--state", str(tmp_path / "state.json")]
    assert main(["quota", "--days", "3", "--json", *paths]) == 0
    assert main(["quota", *paths]) == 0
    assert calls == [{"days": 3, "as_json": True}, {"days": None, "as_json": False}]
    with pytest.raises(SystemExit):
        main(["quota", "--days", "0", *paths])


def test_a_rise_of_one_point_after_one_minute_is_said_in_the_singular():
    assert empty_rise_line({"when": "2026-09-02T11:30:00+00:00", "day": "2026-09-02", "points": 1.0,
                            "minutes": 1}) == "  09-02 11:30, 1 point, nothing on this machine in the 1 minute before"


def test_a_quiet_days_dollars_count_in_the_windows_rate_though_its_share_rose_on_a_later_day(tmp_path, monkeypatch):
    # Shares are whole numbers: Sep 1's response bought part of the point that shows on Sep 2.
    samples = [sample("2026-09-01T10:00:00+00:00", 10), sample("2026-09-01T12:00:00+00:00", 10),
               sample("2026-09-02T10:00:00+00:00", 10), sample("2026-09-02T12:00:00+00:00", 11)]
    records = [answer("m1", "2026-09-01T11:00:00+00:00"), answer("n1", "2026-09-02T11:00:00+00:00")]
    summary = summary_of(tmp_path, monkeypatch, records, samples)
    assert (summary["points"], summary["dollars"], summary["rate"]) == (1.0, pytest.approx(2 * ONE),
                                                                         pytest.approx(2 * ONE))
    assert summary["day_rates"] == pytest.approx({"median": ONE, "low": ONE, "high": ONE})


def test_the_sdk_share_is_taken_over_the_same_days_dollars_as_the_window(tmp_path, monkeypatch):
    # Sep 1 has no dollar figure (its unpriced model carries most of its tokens), so its
    # Agent SDK response is in neither the window's dollars nor their SDK share.
    samples = [sample("2026-09-01T10:00:00+00:00", 10), sample("2026-09-01T12:00:00+00:00", 12),
               sample("2026-09-02T10:00:00+00:00", 20), sample("2026-09-02T12:00:00+00:00", 21)]
    records = [answer("m1", "2026-09-01T10:30:00+00:00"), answer("f1", "2026-09-01T10:40:00+00:00",
                                                                    model="claude-fable-5-1"),
               answer("x1", "2026-09-01T11:00:00+00:00", entrypoint="sdk-py"),
               answer("n1", "2026-09-02T11:00:00+00:00")]
    summary = summary_of(tmp_path, monkeypatch, records, samples)
    assert summary["days"][0]["dollars"] is None
    assert summary["sdk_share"] == 0.0


def test_a_day_with_points_and_no_dollar_figure_is_named_as_left_out_of_the_windows_figures(
        tmp_path, capsys, monkeypatch):
    state = tmp_path / "home" / "state.json"
    # Sep 3 has no dollar figure either, but gained no point, so there is nothing to leave out.
    samples_beside(state, [sample("2026-09-01T10:00:00+00:00", 10), sample("2026-09-01T12:00:00+00:00", 12),
                           sample("2026-09-02T10:00:00+00:00", 20), sample("2026-09-02T12:00:00+00:00", 25),
                           sample("2026-09-03T10:00:00+00:00", 25), sample("2026-09-03T12:00:00+00:00", 25)])
    write(tmp_path / "logs" / "p" / "s1.jsonl", [answer("m1", "2026-09-01T11:00:00+00:00")])
    monkeypatch.setattr(exchange, "fitted_prices", lambda tables: OPUS)
    assert run_quota(tmp_path / "logs", state, today=TODAY) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].startswith("2026-09-01 to 2026-09-03: 2 points, ")
    assert lines[2] == "The window's figures leave out 1 day that has points and no dollar figure, and its 5 points."


def test_with_no_response_in_any_sampled_hour_it_says_so_rather_than_that_there_are_no_cost_records(
        tmp_path, capsys, monkeypatch):
    state = tmp_path / "home" / "state.json"
    samples_beside(state, [sample("2026-09-01T10:00:00+00:00", 10), sample("2026-09-01T12:00:00+00:00", 12)])
    write(tmp_path / "logs" / "p" / "s1.jsonl", [answer("m1", "2026-09-01T09:30:00+00:00")])
    monkeypatch.setattr(exchange, "fitted_prices", lambda tables: OPUS)
    assert run_quota(tmp_path / "logs", state, today=TODAY) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "No response on this machine fell in the hours the status line sampled." in lines
    assert "No cost records price these days, so they have no dollar figures." not in lines


def test_a_windows_day_opens_at_its_first_sample_at_or_above_the_share_carried_in_and_passes_over_stale_ones():
    # 42 was the highest by Sep 1's end. Sep 2's 40 at 08:00 is an idle session's older
    # reading, so the day opens at 08:30's 46, and the 4 points from 42 to 46, used overnight
    # or before 08:30, belong to no step. Every Sep 3 sample is stale, so Sep 3 has none.
    found = walk([sample("2026-09-01T22:00:00+00:00", 42), sample("2026-09-02T08:00:00+00:00", 40),
                  sample("2026-09-02T08:30:00+00:00", 46), sample("2026-09-02T09:00:00+00:00", 47),
                  sample("2026-09-03T10:00:00+00:00", 45), sample("2026-09-03T11:00:00+00:00", 46)])
    assert found == [Step(utc("2026-09-02T09:00:00+00:00"), "2026-09-02", 1_790_000_000, 1.0,
                          utc("2026-09-02T08:30:00+00:00"))]


def test_a_days_span_starts_where_its_window_opened_and_not_at_a_stale_sample_before(tmp_path, monkeypatch):
    samples = [sample("2026-09-01T22:00:00+00:00", 42), sample("2026-09-02T08:00:00+00:00", 40),
               sample("2026-09-02T08:30:00+00:00", 46), sample("2026-09-02T09:00:00+00:00", 47)]
    records = [answer("m1", "2026-09-02T08:15:00+00:00"), answer("m2", "2026-09-02T08:45:00+00:00")]
    summary = summary_of(tmp_path, monkeypatch, records, samples)
    assert summary["days"] == [pytest.approx({"day": "2026-09-02", "points": 1.0, "dollars": ONE, "rate": ONE})]


def test_a_source_other_than_this_machines_transcript_folder_is_said_to_narrow_the_dollars_and_not_the_points(
        tmp_path, capsys, monkeypatch):
    state = tmp_path / "home" / "state.json"
    samples_beside(state, two_days(tmp_path / "logs"))
    monkeypatch.setattr(exchange, "fitted_prices", lambda tables: OPUS)
    note = ("The dollars come only from the transcripts under --source, while the points count everything on this "
            "account: a source missing any of this machine's projects lowers the rate and lists the rises they "
            "bought as empty.")
    assert run_quota(tmp_path / "logs", state, today=TODAY) == 0
    assert capsys.readouterr().out.splitlines()[-1] == note
    assert run_quota(tmp_path / "logs", state, as_json=True, today=TODAY) == 0
    assert json.loads(capsys.readouterr().out)["default_source"] is False
    monkeypatch.setattr(exchange, "default_source", lambda: tmp_path / "logs")
    assert run_quota(tmp_path / "logs", state, today=TODAY) == 0
    assert note not in capsys.readouterr().out.splitlines()
    assert run_quota(tmp_path / "logs", state, as_json=True, today=TODAY) == 0
    assert json.loads(capsys.readouterr().out)["default_source"] is True
    # The same folder by another path, through a link and a `..`, is still the default.
    (tmp_path / "link").symlink_to(tmp_path / "logs")
    assert run_quota(tmp_path / "link" / ".." / "link", state, today=TODAY) == 0
    assert note not in capsys.readouterr().out.splitlines()
