"""Before and after a date the user names: each side's days and spread, on the numbers a
configuration change moves."""

import json
from datetime import date

import pandas as pd
import pytest

from ccdrift import compare
from ccdrift.cli import main
from ccdrift.compare import (auto_compactions, compaction_spread, compare_summary, daily_context, daily_dollars,
                             daily_dollars_per_prompt, quota_points, run_compare, sides, spread)
from ccdrift.logs import parse_all
from ccdrift.prices import Price
from ccdrift.quota import read_samples
from ccdrift.spend import spend_turns
from ccdrift.texts import compactions_cell, versions_text
from tests.helpers import DAY, at, compact_boundary, line, nth_day, prompt, text, write

# Sep 8, with three days a side: before is Sep 5-7, after is Sep 9-11, and Sep 12 is today.
AT = date(2026, 9, 8)
TODAY = date(2026, 9, 12)
OPUS = {"claude-opus-5": Price(input_rate=5e-6, cache_read_rate=5e-7, output_rate=25e-6, web_search_rate=0.0,
                               residual=0.0, rows=9)}
# One response at OPUS: 10 input and a million output tokens.
ONE = 25.00005


def day_at(i, seconds=0):
    """A timestamp on day i after Sep 1, an hour or so into it."""
    return at(i * DAY + seconds)


def turns_of(tmp_path, today=TODAY):
    return spend_turns(parse_all(tmp_path).responses, today)


def sample(stamp, used, resets=1_790_000_000):
    """A quota sample as `ccdrift status --short --stdin` keeps one."""
    return {"at": stamp, "version": "2.1.289", "model": "claude-opus-5",
            "seven_day": {"used_percentage": used, "resets_at": resets}}


def test_the_date_itself_is_on_neither_side_and_each_side_spans_the_days_asked_for():
    before, after = sides(AT, 3, TODAY)
    assert before == ["2026-09-05", "2026-09-06", "2026-09-07"]
    assert after == ["2026-09-09", "2026-09-10", "2026-09-11"]


def test_the_after_side_stops_before_today_since_today_is_still_in_progress():
    assert sides(AT, 3, date(2026, 9, 10)) == (["2026-09-05", "2026-09-06", "2026-09-07"], ["2026-09-09"])
    assert sides(AT, 3, date(2026, 9, 9))[1] == []


def test_a_day_costs_what_its_responses_cost_and_has_no_figure_when_an_unpriced_model_is_material(tmp_path):
    write(tmp_path / "p" / "s1.jsonl", [
        line("m1", text(40), ts=day_at(4), entrypoint="cli", out=1_000_000),
        line("m2", text(40), ts=day_at(4, 60), entrypoint="cli", out=1_000_000),
        line("m3", text(40), ts=day_at(5), entrypoint="cli", out=1_000_000),
        line("m4", text(40), ts=day_at(5, 60), entrypoint="cli", out=1_000_000, model="claude-fable-5-1"),
    ])
    dollars = daily_dollars(turns_of(tmp_path), OPUS)
    assert dollars.to_dict() == pytest.approx({nth_day(4): 2 * ONE})


def test_dollars_per_prompt_charge_a_subagents_spend_to_the_main_thread_prompts_of_its_day(tmp_path):
    write(tmp_path / "p" / "s1.jsonl", [
        prompt(day_at(4)),
        line("m1", text(40), ts=day_at(4, 1), entrypoint="cli", out=1_000_000),
        prompt(day_at(4, 2), sidechain=True),
        line("a1", text(40), ts=day_at(4, 3), entrypoint="cli", out=1_000_000, sidechain=True),
        line("a2", text(40), ts=day_at(4, 4), entrypoint="cli", out=1_000_000, sidechain=True),
        line("m1b", text(40), ts=day_at(4, 5), entrypoint="cli", out=1_000_000),
        prompt(day_at(4, 60)),
        line("m2", text(40), ts=day_at(4, 61), entrypoint="cli", out=1_000_000),
    ])
    # Five responses' dollars over the main thread's two prompts: the subagent's own prompt
    # and the main thread's tool-loop response m1b are not prompts of the main thread.
    assert daily_dollars_per_prompt(turns_of(tmp_path), OPUS).to_dict() == pytest.approx({nth_day(4): 5 * ONE / 2})


def test_a_day_with_no_prompt_has_no_dollars_per_prompt(tmp_path):
    write(tmp_path / "p" / "s1.jsonl", [
        prompt(day_at(4)),
        line("m1", text(40), ts=day_at(4, 1), entrypoint="cli", out=1_000_000),
        line("a1", text(40), ts=day_at(5), entrypoint="cli", out=1_000_000, sidechain=True),
    ])
    assert list(daily_dollars_per_prompt(turns_of(tmp_path), OPUS).index) == [nth_day(4)]


def test_context_per_response_is_the_days_median_main_thread_prompt_size_and_leaves_subagents_out(tmp_path):
    write(tmp_path / "p" / "s1.jsonl", [
        line("m1", text(40), ts=day_at(4), entrypoint="cli", cache_read=1_000),
        line("m2", text(40), ts=day_at(4, 60), entrypoint="cli", cache_read=3_000, cache_creation=500),
        line("m3", text(40), ts=day_at(4, 120), entrypoint="cli", cache_read=9_000),
        line("a1", text(40), ts=day_at(4, 180), entrypoint="cli", cache_read=900_000, sidechain=True),
    ])
    assert daily_context(turns_of(tmp_path)).to_dict() == {nth_day(4): 10 + 3_000 + 500}


def test_only_main_thread_auto_compactions_outside_the_sdk_count_with_their_smallest_median_and_largest(tmp_path):
    write(tmp_path / "p" / "s1.jsonl", [
        line("m1", text(40), ts=day_at(4), entrypoint="cli", version="2.1.280"),
        compact_boundary(day_at(4, 60), trigger="auto", pre_tokens=968_000, version="2.1.280"),
        compact_boundary(day_at(5, 60), trigger="auto", pre_tokens=970_000, version="2.1.280"),
        compact_boundary(day_at(6, 60), trigger="auto", pre_tokens=969_000, version="2.1.280"),
        compact_boundary(day_at(6, 120), trigger="manual", pre_tokens=100_000, version="2.1.280"),
        {**compact_boundary(day_at(6, 180), trigger="auto", pre_tokens=300_000, version="2.1.280"),
         "isSidechain": True},
        compact_boundary(day_at(9, 60), trigger="auto", pre_tokens=670_000, version="2.1.280"),
    ])
    write(tmp_path / "p" / "s2.jsonl", [
        line("m2", text(40), ts=day_at(5), entrypoint="sdk-py", sid="s2", version="2.1.280"),
        {**compact_boundary(day_at(5, 60), sid="s2", trigger="auto", pre_tokens=200_000, version="2.1.280"),
         "entrypoint": "sdk-py"},
    ])
    compactions = auto_compactions(parse_all(tmp_path).compactions)
    before, after = sides(AT, 3, TODAY)
    assert compaction_spread(compactions, before) == {"count": 3, "min": 968_000, "median": 969_000,
                                                      "max": 970_000}
    assert compaction_spread(compactions, after) == {"count": 1, "min": 670_000, "median": 670_000,
                                                     "max": 670_000}
    assert compaction_spread(compactions.iloc[0:0], after) == {"count": 0, "min": None, "median": None, "max": None}


def test_quota_points_count_how_far_each_sample_raised_the_highest_share_so_far_not_every_rise():
    # A session idle since 40% interleaving with a busy one at 45%, then 46%: 6 points were
    # used, though the rises between neighbours add up to 12.
    samples = [sample("2026-09-05T10:00:00+00:00", 40), sample("2026-09-05T10:05:00+00:00", 45),
               sample("2026-09-05T10:06:00+00:00", 40), sample("2026-09-05T10:07:00+00:00", 46)]
    assert quota_points(samples).to_dict() == {"2026-09-05": 6.0}


def test_a_reset_starts_a_new_window_whose_first_sample_rises_from_nothing():
    # The old window's 96 after the reset is a stale reading from a session idle since then.
    samples = [sample("2026-09-05T10:00:00+00:00", 90, resets=1), sample("2026-09-05T23:00:00+00:00", 95, resets=1),
               sample("2026-09-06T01:00:00+00:00", 2, resets=2), sample("2026-09-06T05:00:00+00:00", 5, resets=2),
               sample("2026-09-06T06:00:00+00:00", 96, resets=1)]
    assert quota_points(samples).to_dict() == {"2026-09-05": 5.0, "2026-09-06": 3.0}


def test_a_day_sampled_with_no_rise_used_no_points_and_a_day_counts_by_its_samples_utc_date():
    samples = [sample("2026-09-05T22:00:00+00:00", 10), sample("2026-09-06T01:30:00+03:00", 12),
               sample("2026-09-06T10:00:00+00:00", 12), sample("2026-09-06T11:00:00+00:00", 12)]
    assert quota_points(samples).to_dict() == {"2026-09-05": 2.0, "2026-09-06": 0.0}


def test_points_used_while_nothing_was_sampled_are_not_added_to_the_next_sampled_day():
    # Nothing sampled from Sep 6 to 8: the 28 points used then, on claude.ai or another
    # machine, belong to no day this machine saw, least of all Sep 9.
    samples = [sample("2026-09-05T10:00:00+00:00", 10), sample("2026-09-05T20:00:00+00:00", 12),
               sample("2026-09-09T09:00:00+00:00", 40), sample("2026-09-09T20:00:00+00:00", 41)]
    assert quota_points(samples).to_dict() == {"2026-09-05": 2.0, "2026-09-09": 1.0}


def test_samples_without_a_usable_seven_day_window_or_time_are_skipped():
    samples = [{"at": "2026-09-05T09:00:00+00:00", "seven_day": {"used_percentage": True, "resets_at": 1_790_000_000}},
               sample("2026-09-05T10:00:00+00:00", 10), {"at": "2026-09-05T11:00:00+00:00"},
               {"at": "not a time", "seven_day": {"used_percentage": 99, "resets_at": 1_790_000_000}},
               sample("2026-09-05T12:00:00+00:00", 13)]
    assert quota_points(samples).to_dict() == {"2026-09-05": 3.0}


def test_a_missing_samples_file_reads_as_no_samples(tmp_path):
    assert read_samples(tmp_path / "quota.jsonl") == []
    assert quota_points([]).empty


def test_a_sides_spread_is_its_median_lowest_and_highest_day_and_how_many_of_its_days_had_one():
    daily = pd.Series({"2026-09-05": 3.0, "2026-09-06": 1.0, "2026-09-07": 8.0, "2026-09-09": 100.0})
    assert spread(daily, ["2026-09-05", "2026-09-06", "2026-09-07"]) == {"median": 3.0, "low": 1.0, "high": 8.0,
                                                                         "days": 3}
    assert spread(daily, ["2026-09-10"]) == {"median": None, "low": None, "high": None, "days": 0}


# 4 points on Sep 6, then 6 on Sep 9 and 1 on Sep 10, each seen rising within its own day.
SIDES_SAMPLES = [sample("2026-09-06T10:00:00+00:00", 10), sample("2026-09-06T12:00:00+00:00", 14),
                 sample("2026-09-09T10:00:00+00:00", 20), sample("2026-09-09T12:00:00+00:00", 26),
                 sample("2026-09-10T10:00:00+00:00", 26), sample("2026-09-10T12:00:00+00:00", 27)]


def two_sides(tmp_path, project="p"):
    """A main thread on Sep 5 and 6 on 2.1.280 and on Sep 9 on 2.1.289 and 2.1.1000, a subagent
    on Sep 6 on 2.1.281, an Agent SDK session on Sep 6 and 9 with a subagent of its own, and a
    response on Sep 8 itself, on a model with no cost records, which is on neither side."""
    write(tmp_path / project / "s1.jsonl", [
        prompt(day_at(4)),
        line("m1", text(40), ts=day_at(4, 1), entrypoint="cli", version="2.1.280", out=1_000_000),
        prompt(day_at(5)),
        line("m2", text(40), ts=day_at(5, 1), entrypoint="cli", version="2.1.280", out=1_000_000),
        line("a1", text(40), ts=day_at(5, 2), entrypoint="cli", version="2.1.281", out=1_000_000, sidechain=True),
        line("m3", text(40), ts=day_at(7, 1), entrypoint="cli", version="2.1.285", out=1_000_000,
             model="claude-fable-5-1"),
    ])
    write(tmp_path / project / "s2.jsonl", [
        prompt(day_at(8), sid="s2"),
        line("m4", text(40), ts=day_at(8, 1), sid="s2", entrypoint="cli", version="2.1.289", out=1_000_000),
        line("m5", text(40), ts=day_at(8, 2), sid="s2", entrypoint="cli", version="2.1.1000", out=1_000_000),
    ])
    write(tmp_path / project / "s3.jsonl", [
        line("x1", text(40), ts=day_at(5, 5), sid="s3", entrypoint="sdk-py", version="2.1.280", out=1_000_000),
        line("x2", text(40), ts=day_at(8, 5), sid="s3", entrypoint="sdk-py", version="2.1.289", out=1_000_000),
        line("x3", text(40), ts=day_at(8, 6), sid="s3", entrypoint="sdk-py", version="2.1.289", out=1_000_000),
        line("x4", text(40), ts=day_at(8, 7), sid="s3", entrypoint="sdk-py", version="2.1.289", out=1_000_000,
             sidechain=True),
    ])
    return parse_all(tmp_path)


def test_the_summary_prices_each_side_from_the_historys_own_cost_records(tmp_path, monkeypatch):
    tables = two_sides(tmp_path)
    monkeypatch.setattr(compare, "fitted_prices", lambda given: OPUS if given is tables else {})
    summary = compare_summary(tables, AT, 3, TODAY, [])
    assert summary["metrics"]["dollars_per_day"]["before"] == pytest.approx(
        {"median": 1.5 * ONE, "low": ONE, "high": 2 * ONE, "days": 2})
    assert summary["metrics"]["dollars_per_prompt"]["after"] == pytest.approx(
        {"median": 2 * ONE, "low": 2 * ONE, "high": 2 * ONE, "days": 1})


def test_agent_sdk_responses_are_left_out_of_every_metric_and_counted_on_their_side(tmp_path, monkeypatch):
    tables = two_sides(tmp_path)
    monkeypatch.setattr(compare, "fitted_prices", lambda given: OPUS)
    summary = compare_summary(tables, AT, 3, TODAY, [])
    assert (summary["before"]["sdk_left_out"], summary["after"]["sdk_left_out"]) == (1, 2)
    assert summary["metrics"]["dollars_per_day"]["after"]["median"] == pytest.approx(2 * ONE)


def test_each_side_names_its_days_and_the_versions_its_main_thread_ran_in_version_order(tmp_path):
    summary = compare_summary(two_sides(tmp_path), AT, 3, TODAY, [])
    assert {key: summary["before"][key] for key in ("first", "last", "days_spanned", "days_with_responses")} == {
        "first": "2026-09-05", "last": "2026-09-07", "days_spanned": 3, "days_with_responses": 2}
    assert summary["before"]["versions"] == {"2.1.280": 1.0}
    assert list(summary["after"]["versions"].items()) == [("2.1.289", 0.5), ("2.1.1000", 0.5)]
    assert summary["after"]["days_with_responses"] == 1


def test_a_response_on_the_date_itself_counts_on_neither_side(tmp_path):
    summary = compare_summary(two_sides(tmp_path), AT, 3, TODAY, [])
    assert "2.1.285" not in summary["before"]["versions"] and "2.1.285" not in summary["after"]["versions"]


def test_each_side_counts_its_session_starts_and_reports_their_median_size(tmp_path):
    summary = compare_summary(two_sides(tmp_path), AT, 3, TODAY, [])
    assert (summary["before"]["session_starts"], summary["after"]["session_starts"]) == (1, 1)
    assert summary["metrics"]["session_start"]["after"] == {"median": 10.0, "low": 10.0, "high": 10.0, "days": 1}


def test_the_quota_row_reads_the_samples_it_is_given_on_each_sides_days(tmp_path):
    summary = compare_summary(two_sides(tmp_path), AT, 3, TODAY, SIDES_SAMPLES)
    assert summary["metrics"]["quota_points"]["before"] == {"median": 4.0, "low": 4.0, "high": 4.0, "days": 1}
    assert summary["metrics"]["quota_points"]["after"] == {"median": 3.5, "low": 1.0, "high": 6.0, "days": 2}


def test_models_with_no_price_are_named_with_their_share_rather_than_dropped(tmp_path):
    summary = compare_summary(two_sides(tmp_path), AT, 3, TODAY, [])
    assert summary["unpriced"] == [{"model": "claude-opus-5", "share": 1.0}]
    assert summary["metrics"]["dollars_per_day"]["before"]["days"] == 0


def quota_beside(state, samples):
    """The quota samples file `ccdrift status --short --stdin` keeps beside the state file."""
    state.parent.mkdir(parents=True, exist_ok=True)
    (state.parent / "quota.jsonl").write_text("".join(json.dumps(one) + "\n" for one in samples))



EXPECTED = """\
Before 2026-09-08: 2026-09-05 to 2026-09-07, 2 of 3 days with responses.
After: 2026-09-09 to 2026-09-11, 1 of 3 days with responses.
2026-09-08 itself is on neither side.
Each cell: the median day (lowest-highest day, days with a figure).

                        before                                after
dollars per day         $37.50 ($25.00-$50.00, 2d)            $50.00 ($50.00-$50.00, 1d)
dollars per prompt      $37.50 ($25.00-$50.00, 2d)            $50.00 ($50.00-$50.00, 1d)
context per response    10 (10-10, 2d)                        10 (10-10, 1d)
session start           10 (10-10, 1d), 1 start               10 (10-10, 1d), 1 start
auto-compactions        none                                  none
quota points per day    4 (4-4, 1d)                           3.5 (1-6, 2d)

Versions before: 2.1.280 100%.
Versions after: 2.1.289 50%, 2.1.1000 50%.
Left out, as `ccdrift cost` leaves them out: main-thread responses in Agent SDK sessions, 1 before and 2 after.
Quota points count every surface on the account, claude.ai included, on the days the status line sampled them.
A difference inside either side's range is not evidence the change did anything.
"""


def test_the_command_prints_each_sides_days_the_table_and_what_frames_it(tmp_path, capsys, monkeypatch):
    two_sides(tmp_path / "logs")
    monkeypatch.setattr(compare, "fitted_prices", lambda tables: OPUS)
    state = tmp_path / "home" / "state.json"
    quota_beside(state, SIDES_SAMPLES)
    assert run_compare(tmp_path / "logs", state, AT, days=3, today=TODAY) == 0
    assert capsys.readouterr().out == EXPECTED


def test_rows_with_no_figure_show_a_dash_and_the_lines_below_say_why(tmp_path, capsys):
    two_sides(tmp_path / "logs")
    assert run_compare(tmp_path / "logs", tmp_path / "home" / "state.json", AT, days=3, today=TODAY) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[6].split() == ["dollars", "per", "day", "-", "-"]
    assert not any(line.startswith("quota points per day") for line in lines)
    assert lines[-3:] == [
        "No quota samples on these days: ccdrift keeps them once your status line runs `ccdrift status --short "
        "--stdin`.",
        "No cost records price these days, so they have no dollar figures.",
        "A difference inside either side's range is not evidence the change did anything."]


def test_a_model_with_no_price_beside_priced_ones_is_named_with_its_share(tmp_path, capsys, monkeypatch):
    two_sides(tmp_path / "logs")
    write(tmp_path / "logs" / "p" / "s4.jsonl", [
        line("f1", text(40), ts=day_at(5, 30), sid="s4", entrypoint="cli", out=20_000, model="claude-fable-5-1")])
    monkeypatch.setattr(compare, "fitted_prices", lambda tables: OPUS)
    assert run_compare(tmp_path / "logs", tmp_path / "home" / "state.json", AT, days=3, today=TODAY) == 0
    lines = capsys.readouterr().out.splitlines()
    assert ("No price for claude-fable-5-1 (0.4%): its spend is left out of the dollars on a day where unpriced "
            "models stay under 1% of its tokens, and a day where they reach 1% has no dollar figure.") in lines
    assert lines[6].startswith("dollars per day         $37.50 (")


def test_the_json_is_the_summary_and_names_no_path_session_or_project(tmp_path, capsys, monkeypatch):
    two_sides(tmp_path / "logs", project="secret-client")
    monkeypatch.setattr(compare, "fitted_prices", lambda tables: OPUS)
    state = tmp_path / "home" / "state.json"
    quota_beside(state, SIDES_SAMPLES)
    assert run_compare(tmp_path / "logs", state, AT, days=3, as_json=True, today=TODAY) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert set(payload) == {"at", "days", "before", "after", "metrics", "priced_models", "unpriced"}
    assert payload["metrics"]["quota_points"]["after"] == {"median": 3.5, "low": 1.0, "high": 6.0, "days": 2}
    assert payload["priced_models"] == ["claude-opus-5"]
    assert "secret-client" not in out and str(tmp_path) not in out
    assert not any(f'"{sid}"' in out for sid in ("s1", "s2", "s3"))


def test_a_day_that_is_not_past_yet_is_refused_before_the_history_is_read(tmp_path, capsys):
    for day in (TODAY, date(2026, 9, 20)):
        assert run_compare(tmp_path / "nowhere", tmp_path / "state.json", day, days=3, today=TODAY) == 2
        assert capsys.readouterr().err == (f"--at {day.isoformat()} is not a past day: compare needs complete days "
                                           "after it.\n")
    assert not (tmp_path / "history.sqlite").exists()


def test_a_side_with_no_responses_outside_the_sdk_is_refused_and_says_which(tmp_path, capsys):
    two_sides(tmp_path / "logs")
    state = tmp_path / "home" / "state.json"
    assert run_compare(tmp_path / "logs", state, date(2026, 9, 11), days=3, today=TODAY) == 2
    assert capsys.readouterr().err == ("2026-09-11 was yesterday: compare needs a complete day after it. Run it "
                                       "again tomorrow.\n")
    assert run_compare(tmp_path / "logs", state, date(2026, 9, 9), days=3, today=TODAY) == 2
    assert capsys.readouterr().err.startswith("No complete day after 2026-09-09 ")
    assert run_compare(tmp_path / "logs", state, date(2026, 9, 5), days=3, today=TODAY) == 2
    assert capsys.readouterr().err == ("None of the 3 days before 2026-09-05 holds responses outside Agent SDK "
                                       "sessions.\n")


def test_an_unusable_store_exits_1_and_a_source_with_no_transcripts_exits_2(tmp_path, capsys):
    two_sides(tmp_path / "logs")
    (tmp_path / "home").mkdir()
    (tmp_path / "home" / "history.sqlite").write_text("not a database")
    assert run_compare(tmp_path / "logs", tmp_path / "home" / "state.json", AT, days=3, today=TODAY) == 1
    assert "Move it aside" in capsys.readouterr().err
    (tmp_path / "empty").mkdir()
    assert run_compare(tmp_path / "empty", tmp_path / "other" / "state.json", AT, days=3, today=TODAY) == 2
    assert "No Claude Code transcripts found" in capsys.readouterr().err


def test_with_no_days_given_each_side_is_a_week(tmp_path, capsys):
    two_sides(tmp_path / "logs")
    assert run_compare(tmp_path / "logs", tmp_path / "home" / "state.json", AT, as_json=True,
                       today=date(2026, 9, 30)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert (payload["days"], payload["before"]["first"], payload["after"]["last"]) == (7, "2026-09-01", "2026-09-15")


def test_the_command_line_passes_its_date_days_and_json_through(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(compare, "run_compare", lambda *args, **kwargs: calls.append((args, kwargs)) or 0)
    paths = ["--source", str(tmp_path / "logs"), "--state", str(tmp_path / "state.json")]
    assert main(["compare", "--at", "2026-09-08", "--days", "3", "--json", *paths]) == 0
    assert main(["compare", "--at", "2026-09-08", *paths]) == 0
    assert [(args[2], kwargs) for args, kwargs in calls] == [
        (date(2026, 9, 8), {"days": 3, "as_json": True}), (date(2026, 9, 8), {"days": None, "as_json": False})]
    for bad in (["compare", *paths], ["compare", "--at", "2026-13-01", *paths],
                ["compare", "--at", "2026-09-08", "--days", "0", *paths]):
        with pytest.raises(SystemExit):
            main(bad)


def test_a_sides_compactions_print_as_a_count_and_their_range_and_a_side_without_versions_says_so():
    # Two significant figures, as the version table prints token counts.
    assert compactions_cell({"count": 3, "min": 820_000, "median": 904_000, "max": 968_000}) == (
        "3: 820k-970k, median 900k")
    assert compactions_cell({"count": 3, "min": 968_000, "median": 969_000, "max": 970_000}) == "3 at 970k"
    assert compactions_cell({"count": 1, "min": 670_000, "median": 670_000, "max": 670_000}) == "1 at 670k"
    assert compactions_cell({"count": 0, "min": None, "median": None, "max": None}) == "none"
    assert versions_text({}) == "none"


def test_a_version_that_ran_at_all_shows_as_under_1_percent_rather_than_as_0():
    assert versions_text({"2.1.276": 0.012, "2.1.278": 0.004, "2.1.280": 0.984}) == (
        "2.1.276 1%, 2.1.278 <1%, 2.1.280 98%")


def test_counts_of_left_out_responses_print_with_thousands_separators(tmp_path):
    summary = compare_summary(two_sides(tmp_path), AT, 3, TODAY, [])
    summary["before"]["sdk_left_out"], summary["after"]["sdk_left_out"] = 1925, 395
    assert ("Left out, as `ccdrift cost` leaves them out: main-thread responses in Agent SDK sessions, 1,925 "
            "before and 395 after.") in compare.compare_lines(summary)
