"""The hook coverage rule: when the hooks configured for tool calls stop or start running."""

from datetime import date

import pandas as pd

from ccdrift.hookcover import (HookSetting, hook_coverage_alerts, merged, stream_changes, transcript_states)
from ccdrift.logs import COVERAGE_COLUMNS, coverage_frame
from ccdrift.state import new_state
from tests.helpers import nth_day

EVEN = HookSetting(window=3, baseline=10, agree=0.8, min_calls=2)


def transcripts(project, states, first_day=0, thread="subagent", tools=("Bash",), events=("PreToolUse",),
                version="2.1.247", versions=None, calls=4, entrypoint="cli"):
    """Coverage rows of one transcript a day in `project`, from `first_day`: each hooked
    on every call when its entry in `states` is true, on none when false. `versions`
    gives each transcript its own version."""
    rows = []
    for i, on in enumerate(states):
        day = nth_day(first_day + i)
        for tool in tools:
            for event in events:
                rows.append({"source_file": f"{project}/s/subagents/agent-{first_day + i}.jsonl" if thread == "subagent"
                             else f"{project}/s{first_day + i}.jsonl",
                             "session_id": f"s{first_day + i}", "day": day,
                             "version": versions[i] if versions else version, "entrypoint": entrypoint,
                             "is_sidechain": thread == "subagent", "event": event, "tool": tool, "calls": calls,
                             "hooked": calls if on else 0})
    return rows


def coverage(*groups):
    return coverage_frame([row for group in groups for row in group])


def changes(frame, today=date(2026, 10, 30), setting=EVEN):
    return stream_changes(transcript_states(frame, today, setting.min_calls), setting)


def test_a_transcript_is_hooked_when_at_least_half_its_calls_got_a_hook():
    rows = transcripts("-p", [True, True])
    rows[0]["hooked"], rows[1]["hooked"] = 2, 1
    states = transcript_states(coverage_frame(rows), date(2026, 10, 30), 2)
    assert states["on"].tolist() == [True, False]


def test_a_transcript_with_too_few_calls_of_a_tool_doesnt_count():
    states = transcript_states(coverage(transcripts("-p", [True], calls=1), transcripts("-p", [True], 1, calls=2)),
                               date(2026, 10, 30), 2)
    assert states["day"].tolist() == [nth_day(1)]


def test_agent_sdk_sessions_and_the_current_day_are_left_out():
    frame = coverage(transcripts("-p", [True] * 3), transcripts("-q", [True], entrypoint="sdk-py"))
    states = transcript_states(frame, date(2026, 9, 3), 2)
    assert states["day"].tolist() == [nth_day(0), nth_day(1)] and set(states["project"]) == {"-p"}


def test_hooks_that_stop_running_in_a_stream_are_found_once():
    found = changes(coverage(transcripts("-p", [True] * 10 + [False] * 6)))
    assert [(c["direction"], c["since"], c["thread"], c["tool"]) for c in found] == [
        ("stopped", nth_day(10), "subagent", "Bash")]
    assert found[0]["baseline_other"] == 10 and found[0]["window"] == 3


def test_hooks_that_start_running_are_found_too():
    found = changes(coverage(transcripts("-p", [False] * 10 + [True] * 3)))
    assert [(c["direction"], c["since"]) for c in found] == [("started", nth_day(10))]


def test_a_window_that_isnt_all_one_state_is_no_change():
    assert changes(coverage(transcripts("-p", [True] * 10 + [False, False, True]))) == []


def test_a_baseline_short_of_its_agreement_is_no_change():
    assert changes(coverage(transcripts("-p", [True] * 7 + [False] * 3 + [False] * 3))) == []
    assert len(changes(coverage(transcripts("-p", [True] * 8 + [False] * 2 + [False] * 3)))) == 1


def test_a_stream_needs_a_full_baseline():
    assert changes(coverage(transcripts("-p", [True] * 9 + [False] * 3))) == []


def test_work_moving_to_an_unhooked_project_is_no_change():
    frame = coverage(transcripts("-hooked", [True] * 12), transcripts("-unhooked", [False] * 6, first_day=12))
    assert changes(frame) == []


def test_each_thread_event_and_tool_is_a_stream_of_its_own():
    frame = coverage(transcripts("-p", [True] * 13, thread="main"),
                     transcripts("-p", [True] * 10 + [False] * 3, tools=("Read",)))
    assert [(c["thread"], c["tool"]) for c in changes(frame)] == [("subagent", "Read")]


def test_a_change_says_whether_a_version_new_to_its_baseline_arrived():
    same = changes(coverage(transcripts("-p", [False] * 10 + [True] * 3)))
    new = changes(coverage(transcripts("-p", [False] * 10 + [True] * 3,
                                       versions=["2.1.247"] * 10 + ["2.1.261"] * 3)))
    assert same[0]["new_version"] is False and new[0]["new_version"] is True


def test_changes_in_one_thread_and_direction_within_two_weeks_are_one_alert():
    frame = coverage(transcripts("-a", [False] * 10 + [True] * 3, tools=("Bash", "Read"),
                                 events=("PreToolUse", "PostToolUse")),
                     transcripts("-b", [False] * 10 + [True] * 3, first_day=5),
                     transcripts("-a", [True] * 10 + [False] * 3, thread="main"))
    alerts = merged(changes(frame))
    assert [(a["thread"], a["direction"]) for a in alerts] == [("main", "stopped"), ("subagent", "started")]
    started = alerts[1]
    assert started["projects"] == ["-a", "-b"] and started["tools"] == ["Bash", "Read"]
    assert started["events"] == ["PostToolUse", "PreToolUse"]
    assert started["since"] == nth_day(10) and started["days"] == [nth_day(10), started["until"]]
    assert started["until"] == nth_day(17)


def test_the_main_thread_and_subagents_are_separate_alerts_even_moving_together():
    frame = coverage(transcripts("-a", [False] * 10 + [True] * 3),
                     transcripts("-a", [False] * 10 + [True] * 3, thread="main"))
    assert [(a["thread"], a["direction"]) for a in merged(changes(frame))] == [("main", "started"),
                                                                               ("subagent", "started")]


def test_changes_more_than_two_weeks_apart_are_separate_alerts():
    frame = coverage(transcripts("-a", [False] * 10 + [True] * 3), transcripts("-b", [False] * 10 + [True] * 3,
                                                                                first_day=15))
    assert [a["since"] for a in merged(changes(frame))] == [nth_day(10), nth_day(25)]


def alerts_on(frame, state, today):
    return hook_coverage_alerts(frame, state, today, EVEN)


def test_a_change_is_reported_once_across_runs():
    frame = coverage(transcripts("-p", [True] * 10 + [False] * 5))
    state = new_state()
    first = alerts_on(frame, state, date(2026, 9, 16))
    assert [a["direction"] for a in first] == ["stopped"]
    assert alerts_on(frame, state, date(2026, 9, 17)) == []
    assert [r["direction"] for r in state["hook_changes"]] == ["stopped"]


def test_a_stream_that_flips_back_is_reported_again():
    state = new_state()
    alerts_on(coverage(transcripts("-p", [True] * 10 + [False] * 10)), state, date(2026, 9, 21))
    back = alerts_on(coverage(transcripts("-p", [True] * 10 + [False] * 10 + [True] * 3)), state, date(2026, 9, 24))
    assert [a["direction"] for a in back] == ["started"]


def test_a_change_that_ended_more_than_two_weeks_ago_is_not_reported():
    frame = coverage(transcripts("-p", [True] * 10 + [False] * 3))
    assert alerts_on(frame, new_state(), date(2026, 9, 30)) == []
    assert len(alerts_on(frame, new_state(), date(2026, 9, 27))) == 1


def test_no_coverage_at_all_is_no_change():
    empty = pd.DataFrame(columns=list(COVERAGE_COLUMNS))
    assert alerts_on(empty, new_state(), date(2026, 9, 30)) == []


def test_a_stream_joining_a_change_already_reported_is_recorded_without_a_second_alert():
    # A Claude Code update reaches every stream, but each stream's window fills on its own
    # day: -b's transcripts after the step arrive two days after -a's.
    state = new_state()
    early = coverage(transcripts("-a", [False] * 10 + [True] * 3), transcripts("-b", [False] * 10, first_day=2))
    assert len(alerts_on(early, state, date(2026, 9, 14))) == 1
    later = coverage(transcripts("-a", [False] * 10 + [True] * 5), transcripts("-b", [False] * 10 + [True] * 3, first_day=2))
    assert alerts_on(later, state, date(2026, 9, 16)) == []
    assert sorted(r["stream"].split("|")[0] for r in state["hook_changes"]) == ["-a", "-b"]
