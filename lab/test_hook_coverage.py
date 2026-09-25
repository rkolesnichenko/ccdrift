"""The G15 gate: how the hook coverage rule's settings are measured."""

from datetime import date, timedelta

from ccdrift.hookcover import HookSetting
from ccdrift.logs import coverage_frame
from lab import hook_coverage
from lab.hook_coverage import GRID, judge, move_history, plant_stop, replay, stricter_pass

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


def test_the_real_shapes_first_check_after_upgrading_is_quiet():
    result = judge(real_shape(), EVEN, TODAY)
    assert result["first"] == 0 and result["passed"]


def test_a_setting_whose_first_check_after_upgrading_alerts_fails_the_gate(monkeypatch):
    # Only the first check, on an empty state over the real shape, alerts: the replays,
    # which carry their state from their first day, are left as they are.
    frame, shipped = real_shape(), hook_coverage.hook_coverage_alerts

    def rereported(coverage, state, today, setting):
        first = coverage is frame and today == TODAY and not state["hook_changes"]
        alerts = shipped(coverage, state, today, setting)
        return alerts + [{"thread": "subagent", "direction": "started", "since": "2026-09-05"}] if first else alerts

    monkeypatch.setattr(hook_coverage, "hook_coverage_alerts", rereported)
    result = judge(frame, EVEN, TODAY)
    assert result["first"] == 1 and not result["passed"]
    assert (result["real"], result["others"], result["caught"], result["moved"]) == (1, 0, result["plants"], 0)


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


SHIPPED = HookSetting(window=3, baseline=10, agree=1.0, min_calls=3)


def test_the_gate_passes_when_the_shipped_setting_and_every_stricter_one_pass():
    stricter_ones = [s for s in GRID if s.window >= 3 and s.baseline >= 10 and s.agree >= 1.0 and s.min_calls >= 3]
    assert len(stricter_ones) == 2 and SHIPPED in stricter_ones
    assert stricter_pass([{"setting": s, "passed": s in stricter_ones} for s in GRID], SHIPPED)


def test_one_stricter_setting_failing_fails_the_gate():
    stricter_failing = HookSetting(window=4, baseline=10, agree=1.0, min_calls=3)
    assert not stricter_pass([{"setting": s, "passed": s != stricter_failing} for s in GRID], SHIPPED)
    assert not stricter_pass([{"setting": s, "passed": s != SHIPPED} for s in GRID], SHIPPED)


def test_a_looser_setting_failing_doesnt_matter():
    looser = [HookSetting(2, 10, 1.0, 3), HookSetting(3, 5, 1.0, 3), HookSetting(3, 10, 0.8, 3), HookSetting(3, 10, 1.0, 2),
              HookSetting(4, 10, 0.8, 3)]
    assert stricter_pass([{"setting": s, "passed": s not in looser} for s in GRID], SHIPPED)
