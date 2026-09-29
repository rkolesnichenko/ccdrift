"""The subagent cache metric: what subagent tool-loop turns read back, judged daily."""

import json
from datetime import date

import numpy as np
import pandas as pd

from ccdrift.cli import main
from ccdrift.detector import METRICS, SUBAGENT_METRICS, DetectorConfig, bin_metrics, detect
from ccdrift.incidents import describe, update_incidents
from ccdrift.logs import judged_subagent_loops, judged_turns, parse_source
from ccdrift.replay import run_replay
from ccdrift.report import run_report
from ccdrift.state import load_state, new_state
from tests.helpers import at, line, main_thread_days, nth_day, prompt, subagent_history, text, tool_result, write


def subagent_turns(path, reads, sid="s1", entrypoint="cli", agent_type=None, day=0):
    """A subagent transcript: a prompt, a response writing 1000 tokens, then one tool-loop
    turn per entry of `reads`, each reading that many tokens and writing 100, a minute apart."""
    records = [prompt(at(day * 86400), sid=sid, sidechain=True),
               line(f"{sid}-0", text(40), ts=at(day * 86400), sid=sid, sidechain=True, cache_creation=1000,
                    entrypoint=entrypoint, agent_type=agent_type)]
    for k, read in enumerate(reads, start=1):
        ts = at(day * 86400 + 60 * k)
        records += [tool_result(ts, sid=sid, sidechain=True),
                    line(f"{sid}-{k}", text(40), ts=ts, sid=sid, sidechain=True, cache_read=read, cache_creation=100,
                         entrypoint=entrypoint, agent_type=agent_type)]
    write(path / sid / "subagents" / "agent-a.jsonl", records)


def test_a_loop_turn_reads_back_its_share_of_what_the_response_before_it_had_cached(tmp_path):
    # The first loop turn reads all 1000 the opening response wrote; the second reads 550 of the 1100 cached.
    subagent_turns(tmp_path, [1000, 550])
    df = parse_source(tmp_path).sort_values("timestamp", kind="stable")
    assert df["loop_readback"].tolist()[1:] == [1.0, 0.5]


def test_a_loop_turn_that_reads_more_than_was_cached_reads_back_all_of_it(tmp_path):
    subagent_turns(tmp_path, [1500])
    df = parse_source(tmp_path).sort_values("timestamp", kind="stable")
    assert df["cache_read"].iloc[1] > df["prev_cached"].iloc[1]
    assert df["loop_readback"].iloc[1] == 1.0


def test_a_turn_outside_the_tool_loop_has_no_read_back(tmp_path):
    # The second response opens with a prompt, so it is no loop turn, though it reads what the first had cached.
    write(tmp_path / "s1.jsonl", [prompt(at(0)), line("m0", text(40), ts=at(0), cache_creation=1000),
                                  prompt(at(60)), line("m1", text(40), ts=at(60), cache_read=1000)])
    df = parse_source(tmp_path).sort_values("timestamp", kind="stable")
    assert df["prev_cached"].iloc[1] == 1000 and not df["loop_turn"].iloc[1]
    assert df["loop_readback"].isna().all()


def test_the_subagent_loops_judged_are_those_of_complete_days_outside_the_sdk_with_or_without_an_agent_type(tmp_path):
    subagent_turns(tmp_path, [1000, 1000], sid="typed", agent_type="Explore")
    subagent_turns(tmp_path, [1000, 1000], sid="untyped")
    subagent_turns(tmp_path, [1000, 1000], sid="sdk", entrypoint="sdk-py")
    subagent_turns(tmp_path, [1000, 1000], sid="today", day=1)
    write(tmp_path / "main.jsonl", [prompt(at(0), sid="main"), line("m0", text(40), ts=at(0), sid="main", cache_creation=100),
                                    tool_result(at(60), sid="main"),
                                    line("m1", text(40), ts=at(60), sid="main", cache_read=100)])
    df = parse_source(tmp_path)
    assert df.loc[~df["is_sidechain"], "loop_turn"].any()
    judged = judged_subagent_loops(df, date(2026, 9, 2))
    assert sorted(judged["session_id"].unique()) == ["typed", "untyped"]
    assert len(judged) == 4


def test_a_table_without_loop_turns_has_no_subagent_loops_to_judge():
    assert judged_subagent_loops(pd.DataFrame(), date(2026, 9, 2)).empty
    no_loops = pd.DataFrame({"day": ["2026-09-01"], "is_sidechain": [True]})
    assert judged_subagent_loops(no_loops, date(2026, 9, 2)).empty


def test_the_subagent_metric_is_binned_over_its_own_days_only_when_asked_for():
    turns = pd.DataFrame({"day": ["2026-09-01", "2026-09-01", "2026-09-03"],
                          "loop_readback": [1.0, 0.5, 1.0]})
    binned = bin_metrics(turns, metrics=SUBAGENT_METRICS)
    assert binned["bin"].tolist() == ["2026-09-01", "2026-09-03"]
    assert binned["subagent_cache"].tolist() == [0.75, 1.0]
    assert binned["subagent_cache__n"].tolist() == [2, 1]
    assert not any(name in binned for name in METRICS)


def test_the_main_thread_bins_carry_no_subagent_metric_and_are_judged_without_one():
    turns = pd.DataFrame({"day": ["2026-09-01"], "thinking_fraction": [0.5], "prompt_cache_read_ratio": [1.0],
                          "is_haiku": [0.0], "loop_readback": [0.1]})
    detected = detect(bin_metrics(turns), DetectorConfig())
    assert not any(column.startswith("subagent_cache") for column in detected.columns)


def test_a_drop_in_subagent_read_back_flags_the_subagent_metric():
    rng = np.random.default_rng(0)
    days = [f"2026-09-{d:02d}" for d in range(1, 21)]
    reads = [np.where(rng.random(1000) < (0.002 if d < 15 else 0.05), 0.09, 1.0) for d in range(20)]
    turns = pd.DataFrame({"day": np.repeat(days, 1000), "loop_readback": np.concatenate(reads)})
    detected = detect(bin_metrics(turns, metrics=SUBAGENT_METRICS),
                      DetectorConfig(metric_z_thresholds={"subagent_cache": 3.0}))
    assert detected["subagent_cache__flag"].tolist() == [False] * 15 + [True] * 5


# Incidents on the subagent cache metric.

def judged(path, today):
    df = parse_source(path)
    return judged_turns(df, today), judged_subagent_loops(df, today)


def test_a_drop_in_subagent_read_back_opens_a_subagent_cache_incident_named_by_the_subagents_versions(tmp_path):
    subagent_history(tmp_path, 19, miss_days=[16, 17, 18], version_from=16)
    turns, loops = judged(tmp_path, date.fromisoformat(nth_day(19)))
    state = new_state()
    events = update_incidents(turns, state, date.fromisoformat(nth_day(19)), DetectorConfig(), loops)
    assert [(e.kind, e.incident["metric"], e.incident["start"]) for e in events] == [
        ("flag", "subagent_cache", nth_day(16))]
    kind, _, message, _, named = describe(events[0], turns, state["incidents"], DetectorConfig(), loops)
    assert named == ["2.1.300 (since 09-17)"]
    missed = loops[loops["is_loop_miss"]]
    assert state["incidents"][0]["cost"] == missed["cache_creation"].sum() > 0
    assert message.startswith("Cache read-back in subagent tool loops down from 2026-09-17, on Claude Code 2.1.300")


def test_without_a_cutoff_of_its_own_the_subagent_metric_is_left_alone(tmp_path):
    subagent_history(tmp_path, 19, miss_days=[16, 17, 18])
    turns, loops = judged(tmp_path, date.fromisoformat(nth_day(19)))
    cfg = DetectorConfig(metric_z_thresholds={"cache_ratio": 3.0})
    assert not cfg.judges("subagent_cache") and cfg.judges("cache_ratio")
    state = new_state()
    assert update_incidents(turns, state, date.fromisoformat(nth_day(19)), cfg, loops) == []
    opened = {"metric": "subagent_cache", "start": nth_day(0), "end": None, "status": "open", "source": "check",
              "closed_by": None, "recovered_from": None, "opened_on": nth_day(1), "closed_on": None,
              "versions": [], "cost": 0}
    state["incidents"].append(opened)
    assert update_incidents(turns, state, date.fromisoformat(nth_day(40)), cfg, loops) == []
    assert opened["status"] == "open"


def test_a_subagent_regression_leaves_the_main_thread_metrics_alone(tmp_path):
    subagent_history(tmp_path, 19, miss_days=[16, 17, 18])
    turns, loops = judged(tmp_path, date.fromisoformat(nth_day(19)))
    state = new_state()
    update_incidents(turns, state, date.fromisoformat(nth_day(19)), DetectorConfig())
    assert state["incidents"] == []


def test_a_subagent_cache_incident_recovers_once_read_back_is_usual_again(tmp_path):
    subagent_history(tmp_path, 25, miss_days=[16, 17, 18])
    turns, loops = judged(tmp_path, date.fromisoformat(nth_day(25)))
    state = new_state()
    events = update_incidents(turns, state, date.fromisoformat(nth_day(25)), DetectorConfig(), loops)
    assert [e.kind for e in events] == ["flag", "recovered"]
    # The first 3 days pooled without a miss end on 09-22; three such windows in a row close it from there.
    assert state["incidents"][0]["recovered_from"] == nth_day(21)


def test_an_open_subagent_cache_incident_persists_after_30_days_without_any_subagent(tmp_path):
    main_thread_days(tmp_path, [{}] * 5, first_day=28)
    turns, loops = judged(tmp_path, date.fromisoformat(nth_day(33)))
    assert loops.empty
    state = new_state()
    state["incidents"].append({"metric": "subagent_cache", "start": nth_day(2), "end": None, "status": "open",
                               "source": "check", "closed_by": None, "recovered_from": None, "opened_on": nth_day(5),
                               "closed_on": None, "versions": [], "cost": 0})
    events = update_incidents(turns, state, date.fromisoformat(nth_day(33)), DetectorConfig(), loops)
    assert [e.kind for e in events] == ["persistent"]


def test_a_first_check_replays_a_subagent_cache_incident(tmp_path, capsys):
    subagent_history(tmp_path / "logs", 19, miss_days=[16, 17, 18])
    assert run_replay(tmp_path / "logs", tmp_path / "state.json", today=date.fromisoformat(nth_day(19))) == 0
    assert "  subagent cache  2026-09-17..now" in capsys.readouterr().out


def test_an_incident_on_subagent_cache_can_be_added_by_hand_and_listed_with_its_cost(tmp_path, capsys):
    subagent_history(tmp_path / "logs", 19, miss_days=[16, 17, 18])
    state = tmp_path / "state.json"
    assert main(["incident", "add", "subagent-cache", "2026-09-17..2026-09-19", "--state", str(state)]) == 0
    assert [i["metric"] for i in load_state(state)["incidents"]] == ["subagent_cache"]
    capsys.readouterr()
    assert main(["incident", "list", "--source", str(tmp_path / "logs"), "--state", str(state)]) == 0
    assert "~600k tokens re-cached in subagents" in capsys.readouterr().out


def test_a_version_2_state_file_loads_as_version_3_with_its_incidents(tmp_path):
    incident = {"metric": "cache_ratio", "start": "2026-08-18", "end": "2026-09-03", "status": "recovered"}
    (tmp_path / "state.json").write_text(json.dumps({"version": 2, "incidents": [incident]}))
    state = load_state(tmp_path / "state.json")
    assert state["version"] == 3 and state["incidents"] == [incident]


def test_the_report_shows_subagent_read_back_and_flags_it(tmp_path, capsys):
    subagent_history(tmp_path / "logs", 19, miss_days=[16, 17, 18])
    assert run_report(tmp_path / "logs", tmp_path / "state.json", days=3, today=date.fromisoformat(nth_day(19))) == 0
    assert capsys.readouterr().out.splitlines()[4:7] == [
        "2026-09-17         60        0.900   +0.0        0.000   +0.0            -            20/99"
        "              0.7980  -3923.5  subagent cache",
        "2026-09-18         60        0.900   +0.0        0.000   +0.0            -            20/99"
        "              0.7980  -18.6  subagent cache",
        "2026-09-19         60        0.900   +0.0        0.000   +0.0            -            20/99"
        "              0.7980  -13.2  subagent cache",
    ]


def test_the_report_says_when_subagent_read_back_is_not_judged(tmp_path, capsys):
    subagent_history(tmp_path / "logs", 19, miss_days=[16, 17, 18])
    cfg = DetectorConfig(metric_z_thresholds={"cache_ratio": 3.0})
    run_report(tmp_path / "logs", tmp_path / "state.json", days=3, today=date.fromisoformat(nth_day(19)), cfg=cfg)
    out = capsys.readouterr().out.splitlines()
    assert out[1].endswith("z >= +3.5 for Haiku share, subagent read-back not judged.")
    assert all(row.endswith("-      -") for row in out[4:7])
