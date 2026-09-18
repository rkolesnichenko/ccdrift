"""Session starts: the context a new Claude Code session sends with its first request."""

from datetime import date

import pandas as pd
import pytest

from ccdrift.logs import parse_source
from ccdrift.sessions import (ContextChange, context_alerts, context_changes_in, context_message, first_of_each,
                              project_lines, project_of, project_summary, ratio_starts, rejudged, session_starts)
from ccdrift.texts import context_change_line, project_path
from ccdrift.state import new_state
from tests.helpers import DAY, at, line, nth_day, prompt, text, write


def test_a_session_start_is_the_first_main_thread_cli_response_of_each_transcript(tmp_path):
    write(tmp_path / "a.jsonl", [prompt(at(0)), line("a1", text(40), ts=at(0), cache_creation=90_000, version="2.1.261"),
                                 prompt(at(60)), line("a2", text(40), ts=at(60), cache_read=95_000)])
    write(tmp_path / "b.jsonl", [prompt(at(DAY)), line("b1", text(40), ts=at(DAY), cache_read=40_000,
                                                       cache_creation=14_000, version="2.1.267")])
    write(tmp_path / "a" / "subagents" / "agent-x.jsonl",
          [line("x1", text(40), ts=at(-60), sidechain=True, cache_creation=5_000)])
    write(tmp_path / "sdk.jsonl", [line("s1", text(40), ts=at(30), entrypoint="sdk-py", cache_creation=7_000)])
    starts = session_starts(parse_source(tmp_path))
    assert starts[["source_file", "day", "version", "prompt_tokens"]].values.tolist() == [
        ["a.jsonl", "2026-09-01", "2.1.261", 90_010.0], ["b.jsonl", "2026-09-02", "2.1.267", 54_010.0]]


def test_a_first_response_without_token_counts_is_not_a_session_start(tmp_path):
    # If Claude Code stops logging usage, every start reads as 0 tokens: not a change in context.
    unlogged = line("b1", text(40), ts=at(DAY), version="2.1.280")
    del unlogged["message"]["usage"]
    write(tmp_path / "a.jsonl", [prompt(at(0)), line("a1", text(40), ts=at(0), cache_creation=90_000)])
    write(tmp_path / "b.jsonl", [prompt(at(DAY)), unlogged])
    assert session_starts(parse_source(tmp_path))["source_file"].tolist() == ["a.jsonl"]


def resumed(folder, original, copy):
    """A session of two prompts, resumed a day later into a second transcript that
    begins with copies of both responses."""
    session = [prompt(at(0)), line("m1", text(40), ts=at(0), cache_creation=20_000),
               prompt(at(60)), line("m2", text(40), ts=at(60), cache_read=20_000, cache_creation=500)]
    write(folder / original, session)
    write(folder / copy, session + [prompt(at(DAY)), line("m3", text(40), ts=at(DAY), cache_read=170_000,
                                                        cache_creation=500)])


@pytest.mark.parametrize("original, copy", [("a.jsonl", "z.jsonl"), ("z.jsonl", "a.jsonl")])
def test_a_resumed_session_is_not_a_new_session_start(tmp_path, original, copy):
    # When the original transcript owns the copied responses, the resumed one's first
    # own response carries the whole resumed context: 170k tokens here.
    resumed(tmp_path, original, copy)
    assert session_starts(parse_source(tmp_path))["prompt_tokens"].tolist() == [20_010.0]


def starts_of(tokens, versions=None, projects=None):
    """Sessions one a day, as ratio_starts hands them to the detector: each already
    measured against its own project's level. The detector compares medians, so the
    nominal level only has to be the same for every row."""
    return pd.DataFrame({"source_file": [f"p/{i}.jsonl" for i in range(len(tokens))],
                         "project": projects or ["p"] * len(tokens),
                         "timestamp": pd.to_datetime([f"{nth_day(i)}T10:00:00Z" for i in range(len(tokens))]),
                         "day": [nth_day(i) for i in range(len(tokens))],
                         "version": versions or ["2.1.261"] * len(tokens),
                         "prompt_tokens": [float(t) for t in tokens],
                         "level": [100_000.0] * len(tokens),
                         "ratio": [float(t) / 100_000 for t in tokens]})


def test_a_step_in_session_start_size_is_found_once():
    starts = starts_of([128_000] * 8 + [54_000] * 5)
    changes = context_changes_in(starts)
    assert [(c.since, c.before, c.after) for c in changes[:1]] == [("2026-09-09", 128_000.0, 54_000.0)]
    assert [(c.since, c.up) for c in first_of_each(changes)] == [("2026-09-09", False)]


def test_one_outlier_in_the_window_is_not_a_step():
    # Two small sessions and one normal one: the median moved, but not every session did.
    assert context_changes_in(starts_of([128_000] * 8 + [60_000, 60_000, 122_000])) == []


def test_a_step_already_recorded_is_not_found_again():
    changes = context_changes_in(starts_of([128_000] * 8 + [54_000] * 3))
    recorded = [{"since": "2026-09-05", "from": 130_000.0, "to": 50_000.0}]
    assert first_of_each(changes, recorded) == []


def test_a_stale_recorded_step_re_detected_by_sparse_sessions_is_not_found_again():
    # With under one session a day, the same step can still be showing up in windows
    # more than DEDUPE_DAYS after it was first recorded, as the baseline slowly refills.
    change = ContextChange("2026-09-30", "2026-10-02", 128_000.0, 54_000.0)
    recorded = [{"since": "2026-09-10", "from": 128_000.0, "to": 54_000.0}]
    assert first_of_each([change], recorded) == []


def test_a_later_genuine_step_in_the_same_direction_is_still_found():
    change = ContextChange("2026-09-30", "2026-10-02", 54_000.0, 30_000.0)
    recorded = [{"since": "2026-09-10", "from": 128_000.0, "to": 54_000.0}]
    assert first_of_each([change], recorded) == [change]


def test_a_step_recorded_from_or_to_0_tokens_is_compared_by_date_only():
    # A check that counted 0-token starts could record one; dividing by it failed every later check.
    down = ContextChange("2026-09-30", "2026-10-02", 54_000.0, 30_000.0)
    up = ContextChange("2026-09-30", "2026-10-02", 54_000.0, 130_000.0)
    recorded = [{"since": "2026-09-10", "from": 128_000.0, "to": 0.0},
                {"since": "2026-09-12", "from": 0.0, "to": 54_000.0}]
    assert first_of_each([down, up], recorded) == [down, up]


def test_a_session_start_step_is_alerted_once_with_its_sizes():
    state = new_state()
    starts = starts_of([128_000] * 8 + [54_000] * 3)
    changes = context_alerts(starts, state, date(2026, 9, 12))
    assert [(c["since"], c["from"], c["to"], c["days"]) for c in changes] == [
        ("2026-09-09", 128_000.0, 54_000.0, ["2026-09-09", "2026-09-11"])]
    assert context_alerts(starts, state, date(2026, 9, 12)) == []
    assert context_message(changes[0], ["2.1.267 (since 09-09)"]) == (
        "New sessions start with ~54k tokens of context from 2026-09-09, on Claude Code 2.1.267 (since 09-09), "
        "down from ~130k, in the one project ccdrift could compare with itself, so its CLAUDE.md, MCP servers or "
        "skills explain it as readily as Claude Code does.")


def test_steps_from_weeks_ago_and_on_the_current_day_are_not_alerted():
    starts = starts_of([128_000] * 8 + [54_000] * 3)
    assert context_alerts(starts, new_state(), date(2026, 10, 10)) == []
    assert context_alerts(starts, new_state(), date(2026, 9, 11)) == []


def sessions(root, project, tokens, first_day=0, version="2.1.261"):
    """One session a day in `project`, each starting with `tokens` of context."""
    for i, size in enumerate(tokens):
        d = first_day + i
        write(root / project / f"{project}-{d}.jsonl",
              [prompt(at(d * DAY), sid=f"{project}-{d}"),
               line(f"m{project}-{d}", text(40), ts=at(d * DAY), sid=f"{project}-{d}", cache_creation=100,
                    cache_read=size - 110, version=version, entrypoint="cli")])


def starts_in(tmp_path):
    return session_starts(parse_source(tmp_path))


@pytest.mark.parametrize("folder, path", [
    ("-Users-me-dev-app", "/Users/me/dev/app"),
    ("-Users-me-dev-my-app", "/Users/me/dev/my/app"),  # a dash in the directory's own name reads back as a slash
])
def test_a_transcripts_project_is_its_folder_read_back_as_a_path(folder, path):
    assert project_of(f"{folder}/abc.jsonl") == folder
    assert project_path(folder) == path
    assert project_of("loose.jsonl") == "loose.jsonl"


def test_a_session_is_judged_against_its_own_projects_earlier_level(tmp_path):
    sessions(tmp_path, "-a", [100_000, 100_000, 100_000, 120_000])
    judged = ratio_starts(starts_in(tmp_path))
    # The first three set the level; only the fourth is judged, against their median.
    assert judged["day"].tolist() == ["2026-09-04"]
    assert (judged["level"].iloc[0], round(float(judged["ratio"].iloc[0]), 2)) == (100_000.0, 1.2)


def test_moving_to_a_smaller_project_is_not_a_change(tmp_path):
    # The owner's own history in miniature: one project steady, then work moves to a new
    # one that starts smaller. Pooled, this reads as context halving; per project, nothing
    # moved, and the new project has too few sessions to be judged at all.
    sessions(tmp_path, "-big", [128_000] * 12)
    sessions(tmp_path, "-small", [54_000] * 3, first_day=12)
    state = new_state()
    assert context_alerts(starts_in(tmp_path), state, date(2026, 9, 20)) == []
    assert state["context_changes"] == []


def test_a_step_in_every_project_says_so_and_names_the_version_when_one_arrived(tmp_path):
    sessions(tmp_path, "-a", [128_000] * 8)
    sessions(tmp_path, "-b", [64_000] * 8)
    sessions(tmp_path, "-a", [64_000] * 3, first_day=8, version="2.1.267")
    sessions(tmp_path, "-b", [32_000] * 3, first_day=8, version="2.1.267")
    changes = context_alerts(starts_in(tmp_path), new_state(), date(2026, 9, 14))
    assert [(c["since"], sorted(c["projects"]), c["of_projects"]) for c in changes] == [
        ("2026-09-09", ["-a", "-b"], 2)]
    # The sessions after the step run a version none of the baseline sessions ran.
    assert changes[0]["new_version"] is True
    assert context_message(changes[0], ["2.1.267 (since 09-09)"]).endswith(
        "in every project you used (2 of 2), on a Claude Code version none of the sessions before it ran — "
        "the likeliest cause.")
    assert context_message({**changes[0], "new_version": False}, []).endswith(
        "in every project you used (2 of 2), with no new Claude Code version, so look at your global "
        "configuration in ~/.claude.")


def test_a_step_in_one_project_of_several_blames_that_projects_own_files(tmp_path):
    sessions(tmp_path, "-a", [128_000] * 8)
    sessions(tmp_path, "-b", [64_000] * 8)
    sessions(tmp_path, "-a", [64_000] * 3, first_day=8)
    sessions(tmp_path, "-b", [64_000] * 3, first_day=8)
    changes = context_alerts(starts_in(tmp_path), new_state(), date(2026, 9, 14))
    assert [(c["projects"], c["of_projects"]) for c in changes] == [(["-a"], 2)]
    assert context_message(changes[0], []).endswith(
        "in 1 of 2 projects you used. That project's CLAUDE.md, MCP servers or skills explain it, not Claude Code.")


def test_the_status_line_names_the_projects_and_the_alert_never_does():
    change = {"since": "2026-09-09", "from": 128_000.0, "to": 64_000.0, "days": ["2026-09-09", "2026-09-11"],
              "projects": ["-Users-me-dev-app"], "of_projects": 2, "reported_on": "2026-09-12"}
    assert context_change_line(change) == (
        "session start ~130k -> ~64k tokens from 2026-09-09 in /Users/me/dev/app of 2")
    assert "/Users" not in context_message(change, ["2.1.267"])


def test_an_older_states_recorded_changes_are_rejudged_once(tmp_path):
    sessions(tmp_path, "-big", [128_000] * 12)
    sessions(tmp_path, "-small", [54_000] * 3, first_day=12)
    state = new_state()
    state["context_changes"] = [
        {"since": "2026-09-13", "from": 128_000.0, "to": 54_000.0, "days": ["2026-09-13", "2026-09-15"],
         "reported_on": "2026-09-16"},
        {"since": "2026-08-01", "from": 90_000.0, "to": 60_000.0, "days": ["2026-08-01", "2026-08-03"],
         "reported_on": "2026-08-04"},
    ]
    dropped = rejudged(starts_in(tmp_path), state, date(2026, 9, 20))
    # The pooled rule's project switch goes; a change older than the sessions read stays.
    assert [d["since"] for d in dropped] == ["2026-09-13"]
    assert [c["since"] for c in state["context_changes"]] == ["2026-08-01"]


def test_the_report_names_each_projects_typical_session_start(tmp_path):
    sessions(tmp_path, "-small", [54_000] * 2)
    sessions(tmp_path, "-big", [128_000] * 3, first_day=2)
    starts = starts_in(tmp_path)
    days = sorted(starts["day"].astype(str).unique())
    assert project_lines(project_summary(starts, days)) == [
        "", "Session starts by project over these days: /big ~130k (3 sessions), /small ~54k (2 sessions)"]
    assert project_lines(project_summary(starts, ["2026-10-01"])) == []
