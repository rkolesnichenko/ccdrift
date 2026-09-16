"""The G7 spike on hand-made subagent responses."""

import pandas as pd

from lab.subagent_models import agent_table, gate, subagent_turns
from tests.helpers import nth_day


def responses(rows):
    """rows: (agent type, model, day index, count)."""
    records = [{"is_sidechain": True, "entrypoint": "cli", "agent_type": agent, "model": model, "day": nth_day(d)}
               for agent, model, d, n in rows for _ in range(n)]
    return pd.DataFrame(records)


def test_subagent_turns_leave_out_the_main_thread_sdk_sessions_and_unnamed_agents():
    df = pd.DataFrame({"is_sidechain": [True, False, True, True], "entrypoint": ["cli", "cli", "sdk-py", "cli"],
                       "agent_type": ["Plan", "Plan", "Plan", None], "model": ["m"] * 4, "day": [nth_day(0)] * 4})
    assert len(subagent_turns(df)) == 1


def test_agent_table_counts_active_days_and_the_usual_models_share():
    turns = responses([("Plan", "claude-opus-5", d, 6) for d in range(5)] + [("Plan", "claude-sonnet-5", 5, 6),
                                                                             ("Plan", "claude-opus-5", 6, 2)])
    row = agent_table(turns).iloc[0]
    assert (row["agent_type"], row["active_days"], row["model"], round(row["share"], 3)) == \
        ("Plan", 6, "claude-opus-5", 0.833)


def test_gate_passes_when_one_built_in_agent_keeps_its_model():
    turns = responses([("Plan", "claude-opus-5", d, 6) for d in range(5)]
                      + [("general-purpose", m, d, 6) for d in range(5) for m in ("claude-opus-5", "claude-sonnet-5")])
    passed, _ = gate(turns)
    assert passed


def test_gate_fails_when_only_general_purpose_is_steady():
    turns = responses([("general-purpose", "claude-sonnet-5", d, 6) for d in range(9)]
                      + [("Explore", m, d, 3) for d in range(9) for m in ("claude-opus-5", "claude-sonnet-5")])
    passed, _ = gate(turns)
    assert not passed
