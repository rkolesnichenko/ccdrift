"""The G15 gate: how the hook coverage rule's settings are measured."""

from datetime import date, timedelta

from ccdrift.hookcover import HookSetting
from ccdrift.logs import coverage_frame
from lab.hook_coverage import GRID, choose, judge, move_history, plant_stop, replay

EVEN = HookSetting(window=3, baseline=10, agree=0.8, min_calls=2)
TODAY = date(2026, 9, 24)


def stream(project, thread, states, first=date(2026, 8, 15), versions=None, tool="Bash"):
    """Coverage rows of one transcript a day from `first`, hooked on every PreToolUse call
    when its entry in `states` is true. `versions` gives each its version."""
    rows = []
    for i, on in enumerate(states):
        day = (first + timedelta(days=i)).isoformat()
        rows.append({"source_file": f"{project}/s{i}/subagents/agent-a.jsonl" if thread == "subagent"
                     else f"{project}/s{i}.jsonl", "session_id": f"s{i}", "day": day,
                     "version": versions[i] if versions else "2.1.261", "entrypoint": "cli",
                     "is_sidechain": thread == "subagent", "event": "PreToolUse", "tool": tool, "calls": 4,
                     "hooked": 4 if on else 0})
    return rows


def real_shape():
    """The owner's logs as they were on 2026-09-24: the main thread hooked throughout,
    subagents unhooked on 2.1.247 until 2026-09-04 and hooked on 2.1.261 from 09-05."""
    main = stream("-p", "main", [True] * 40)
    subagents = stream("-p", "subagent", [False] * 21 + [True] * 19, versions=["2.1.247"] * 21 + ["2.1.261"] * 19)
    return coverage_frame(main + subagents)


def test_the_real_changes_shape_is_found_once_and_names_a_new_version():
    alerts = replay(real_shape(), EVEN)
    assert [(a["thread"], a["direction"], a["since"], a["new_version"]) for a in alerts] == [
        ("subagent", "started", "2026-09-05", True)]


def test_a_setting_that_finds_the_real_change_and_catches_every_plant_passes():
    result = judge(real_shape(), EVEN, TODAY)
    assert result["real"] == 1 and result["others"] == 0 and result["moved"] == 0
    assert result["plants"] > 0 and result["caught"] == result["plants"] and result["passed"]


def test_a_stop_the_logs_already_hold_is_a_false_alarm_that_fails_the_gate():
    frame = coverage_frame(stream("-p", "main", [True] * 40) + stream("-q", "main", [True] * 20 + [False] * 20)
                           + stream("-p", "subagent", [False] * 21 + [True] * 19,
                                    versions=["2.1.247"] * 21 + ["2.1.261"] * 19))
    result = judge(frame, EVEN, TODAY)
    assert result["others"] == 1 and not result["passed"]


def test_the_real_change_counts_as_not_measurable_once_the_transcripts_before_it_are_gone():
    frame = coverage_frame(stream("-p", "main", [True] * 40)
                           + stream("-p", "subagent", [True] * 19, first=date(2026, 9, 5)))
    result = judge(frame, EVEN, TODAY)
    assert result["measurable"] is False and result["real"] == 0 and result["passed"]


def test_a_plant_removes_hooks_from_its_day_on_in_every_stream_or_in_one():
    frame = real_shape()
    everywhere = plant_stop(frame, "2026-09-20")
    assert everywhere.loc[everywhere["day"] >= "2026-09-20", "hooked"].sum() == 0
    assert everywhere.loc[everywhere["day"] < "2026-09-20", "hooked"].sum() > 0
    one = plant_stop(frame, "2026-09-20", ("-p", "main", "PreToolUse", "Bash"))
    late = one["day"] >= "2026-09-20"
    assert one.loc[late & ~one["is_sidechain"], "hooked"].sum() == 0
    assert one.loc[late & one["is_sidechain"], "hooked"].sum() > 0


def test_every_setting_leaves_a_move_to_an_unhooked_project_alone():
    assert all(replay(move_history(), setting) == [] for setting in GRID)


def test_the_gate_ships_the_smallest_window_where_every_setting_passes_then_the_largest_of_the_rest():
    one_miss = [{"setting": s, "passed": s != HookSetting(2, 10, 0.8, 1)} for s in GRID]
    assert choose(one_miss) == HookSetting(window=3, baseline=10, agree=1.0, min_calls=3)
    all_pass = [{"setting": s, "passed": True} for s in GRID]
    assert choose(all_pass) == HookSetting(window=2, baseline=10, agree=1.0, min_calls=3)
    assert choose([{"setting": s, "passed": s.window != 4} for s in GRID]) == HookSetting(2, 10, 1.0, 3)
    assert choose([{"setting": s, "passed": False} for s in GRID]) is None
