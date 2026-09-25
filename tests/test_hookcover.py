"""The hook coverage rule: when the hooks configured for tool calls stop or start running."""

from datetime import date, timedelta

import pandas as pd

from ccdrift.hookcover import (HookSetting, hook_coverage_alerts, merged, stream_changes, transcript_states)
from ccdrift.logs import COVERAGE_COLUMNS, coverage_frame
from ccdrift.state import new_state
from ccdrift.texts import hook_coverage_lines, hook_coverage_message
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


def test_a_transcripts_versions_in_a_stream_are_those_its_calls_of_that_tool_ran_on():
    # Resumed after an upgrade, the transcript called Bash on both versions but Read only on the new one.
    rows = (transcripts("-p", [True], tools=("Bash", "Read"), version="2.1.261")
            + transcripts("-p", [True], version="2.1.247"))
    states = transcript_states(coverage_frame(rows), date(2026, 10, 30), 2)
    assert states["versions"].tolist() == [("2.1.247", "2.1.261"), ("2.1.261",)]


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


def test_a_project_used_again_long_after_a_reported_change_doesnt_report_it_again():
    # -b sat idle from before the step and is used again 20 days after -a reported it: its
    # window fills late, but the step lies between its last transcript before and its first after.
    frame = coverage(transcripts("-a", [False] * 10 + [True] * 3), transcripts("-b", [False] * 10),
                     transcripts("-b", [True] * 3, first_day=30))
    state = new_state()
    assert [a["since"] for a in alerts_on(frame, state, date(2026, 9, 14))] == [nth_day(10)]
    assert alerts_on(frame, state, date(2026, 10, 4)) == []
    late = [r for r in state["hook_changes"] if r["stream"].startswith("-b|")]
    assert [(r["after"], r["since"], r["alerted"]) for r in late] == [(nth_day(9), nth_day(30), False)]


def test_the_first_check_after_upgrading_doesnt_report_a_step_whose_busiest_stream_ended_long_ago():
    # -a's change ended 16 days before the check; quieter -b's window, over the same step,
    # filled within the last two weeks. Neither is news on a first check.
    frame = coverage(transcripts("-a", [False] * 10 + [True] * 3), transcripts("-b", [False] * 10),
                     transcripts("-b", [True] * 3, first_day=20))
    state = new_state()
    assert alerts_on(frame, state, date(2026, 9, 29)) == []
    assert sorted((r["stream"].split("|")[0], r["alerted"]) for r in state["hook_changes"]) == [("-a", False),
                                                                                                ("-b", False)]


def test_a_quieter_stream_that_saw_the_step_first_is_still_the_same_step():
    # -b's first transcript after the step came a day before -a's, then -b sat idle until its
    # window filled two weeks later. -a's first transcript after the step falls outside -b's
    # gap, but the two gaps overlap: one step, and no news on a first check.
    frame = coverage(transcripts("-a", [False] * 10 + [True] * 3, first_day=1),
                     transcripts("-b", [False] * 10 + [True]), transcripts("-b", [True] * 2, first_day=25))
    state = new_state()
    assert alerts_on(frame, state, date(2026, 9, 29)) == []
    gaps = sorted((r["stream"].split("|")[0], r["after"], r["since"]) for r in state["hook_changes"])
    assert gaps == [("-a", nth_day(10), nth_day(11)), ("-b", nth_day(9), nth_day(10))]


def test_a_separate_earlier_step_seen_late_is_still_news():
    # -b stopped around day 10 and sat idle; -a stopped at day 30 and was reported. -b's window
    # fills after that, but its gap ended 20 days before -a's began: a different step.
    frame = coverage(transcripts("-a", [True] * 30 + [False] * 3),
                     transcripts("-b", [True] * 10 + [False]), transcripts("-b", [False] * 2, first_day=33))
    state = new_state()
    assert [a["since"] for a in alerts_on(frame, state, date(2026, 10, 4))] == [nth_day(30)]
    assert [a["since"] for a in alerts_on(frame, state, date(2026, 10, 6))] == [nth_day(10)]


def test_a_quiet_streams_wide_gap_doesnt_hide_a_separate_later_step():
    # -s stops around day 10 and is reported. -q sat idle from day 9 to day 30, so its stop is
    # only known to lie somewhere in those three weeks: too wide to place a step. -p stops
    # between days 24 and 25, a separate step two weeks after -s's, and must still be reported.
    frame = coverage(transcripts("-s", [True] * 10 + [False] * 3),
                     transcripts("-q", [True] * 10), transcripts("-q", [False] * 3, first_day=30),
                     transcripts("-p", [True] * 10, first_day=15), transcripts("-p", [False], first_day=25),
                     transcripts("-p", [False] * 2, first_day=34))
    state = new_state()
    found = [alert["since"] for day in range(12, 45)
             for alert in alerts_on(frame, state, date(2026, 9, 1) + timedelta(days=day))]
    assert found == [nth_day(10), nth_day(25)]


def test_a_stream_that_flips_back_doesnt_bring_its_old_change_back():
    state = new_state()
    stopped = coverage(transcripts("-p", [True] * 10 + [False] * 10))
    assert [a["direction"] for a in alerts_on(stopped, state, date(2026, 9, 21))] == ["stopped"]
    back = coverage(transcripts("-p", [True] * 10 + [False] * 10 + [True] * 3))
    assert [a["direction"] for a in alerts_on(back, state, date(2026, 9, 24))] == ["started"]
    assert alerts_on(back, state, date(2026, 9, 25)) == []
    assert [r["direction"] for r in state["hook_changes"]] == ["stopped", "started"]


def test_a_change_in_the_main_thread_isnt_folded_into_one_reported_for_subagents():
    frame = coverage(transcripts("-p", [True] * 10 + [False] * 3),
                     transcripts("-p", [True] * 10 + [False] * 3, first_day=5, thread="main"))
    state = new_state()
    assert [a["thread"] for a in alerts_on(frame, state, date(2026, 9, 14))] == ["subagent"]
    assert [(a["thread"], a["since"]) for a in alerts_on(frame, state, date(2026, 9, 19))] == [("main", nth_day(15))]


def test_a_change_folds_only_into_one_that_alerted_so_silent_records_dont_chain():
    # -b starts ten days after -a and is folded into -a's alert; -c starts ten days after
    # -b, twenty after -a, and is news of its own.
    frame = coverage(transcripts("-a", [False] * 10 + [True] * 3), transcripts("-b", [False] * 10 + [True] * 3, 10),
                     transcripts("-c", [False] * 10 + [True] * 3, 20))
    state = new_state()
    assert [a["since"] for a in alerts_on(frame, state, date(2026, 9, 14))] == [nth_day(10)]
    assert alerts_on(frame, state, date(2026, 9, 24)) == []
    assert [a["since"] for a in alerts_on(frame, state, date(2026, 10, 4))] == [nth_day(30)]


def test_a_version_that_arrives_only_within_the_window_isnt_new_to_the_change():
    # The step lies between the last transcript before the window and the first in it.
    found = changes(coverage(transcripts("-p", [False] * 10 + [True] * 3,
                                         versions=["2.1.247"] * 11 + ["2.1.261"] * 2)))
    assert found[0]["new_version"] is False and found[0]["after"] == nth_day(9)


STARTED = {"thread": "subagent", "direction": "started", "since": "2026-09-05", "until": "2026-09-07",
           "days": ["2026-09-05", "2026-09-07"], "window": 3, "baseline": 10, "baseline_other": 10,
           "new_version": True, "projects": ["-Users-me-app"], "events": ["PostToolUse", "PreToolUse"],
           "tools": ["Bash", "Read", "mcp__tracker"]}


def test_a_start_says_what_started_where_and_that_the_logs_cant_tell_running_from_logging():
    assert hook_coverage_message(STARTED, ["2.1.261 (since 09-05)"]) == (
        "Hooks started running on Bash, Read and an MCP server's tools in subagents from 2026-09-05, on Claude Code "
        "2.1.261 (since 09-05): all of the last 3 subagents had PreToolUse and PostToolUse hooks on those calls, "
        "where 10 of the 10 before had none, in 1 project. Claude Code may have started running them, or started "
        "logging them. If you didn't change your hooks, check its release notes.")


def test_a_stop_with_no_new_version_points_at_the_owners_own_settings():
    stopped = {**STARTED, "thread": "main", "direction": "stopped", "new_version": False, "events": ["PreToolUse"],
               "tools": ["Bash"], "projects": ["-Users-me-a", "-Users-me-b"], "baseline_other": 9}
    assert hook_coverage_message(stopped, []) == (
        "Hooks stopped running on Bash in main-thread sessions from 2026-09-05: none of the last 3 main-thread "
        "sessions had a PreToolUse hook on those calls, where 9 of the 10 before did, in 2 projects. Claude Code "
        "may have stopped running them, or stopped logging them. No Claude Code version arrived with it, so your "
        "own hook settings are the likelier cause.")


def test_many_tools_and_mcp_servers_are_counted_not_listed():
    many = {**STARTED, "tools": ["Agent", "Bash", "Edit", "Read", "Write", "mcp__one", "mcp__two"]}
    assert hook_coverage_message(many, []).startswith(
        "Hooks started running on Agent, Bash, Edit, 2 more tools and 2 MCP servers' tools in subagents from")


def test_the_message_names_no_mcp_server_or_project_and_the_log_lines_do():
    message = hook_coverage_message(STARTED, [])
    assert "tracker" not in message and "Users" not in message and "me-app" not in message
    assert hook_coverage_lines(STARTED) == ["projects: /Users/me/app", "tools: Bash, Read, mcp__tracker"]
