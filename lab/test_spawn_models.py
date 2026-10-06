"""G18: served vs requested model per subagent spawn, and the gate that judges the rule."""

import pandas as pd

import lab.spawn_models
from ccdrift.spawns import JOINED_COLUMNS
from lab.spawn_models import PLANTED, gate, one_million_alarms, plants, shorter_id


def spawn(agent, requested, resolved, served, responses=1):
    return {"agent_id": agent, "requested": requested, "resolved": resolved, "version": "2.1.289",
            "day": "2026-09-01", "timestamp": pd.Timestamp("2026-09-01T10:00:00Z"), "served": tuple(served),
            "responses": responses if served else 0}


def clean():
    return pd.DataFrame([spawn("a1", "opus", "claude-opus-5-5[1m]", ["claude-opus-5-5"]),
                         spawn("a2", "sonnet", "claude-sonnet-5", ["claude-sonnet-5"]),
                         spawn("a3", None, "claude-opus-5", ["claude-opus-5"]),
                         spawn("a4", "haiku", "claude-haiku-4-5", [])], columns=list(JOINED_COLUMNS))


def test_the_gate_passes_with_no_mismatch_and_every_plant_judged_as_expected():
    assert gate(clean()) == (True, ["false alarms: 0", "plants judged as expected: 11 of 11"])


def test_one_mismatch_on_the_real_logs_fails_the_gate():
    joined = pd.concat([clean(), pd.DataFrame([spawn("a5", "sonnet", "claude-opus-5", ["claude-opus-5"])])],
                       ignore_index=True)
    ok, lines = gate(joined)
    assert not ok and lines[0] == "false alarms: 1 (not_honoured 1)"


def test_a_rule_that_misses_a_plant_fails_the_gate_and_names_it(monkeypatch):
    monkeypatch.setattr(lab.spawn_models, "judge", lambda requested, resolved, served: [])
    ok, lines = gate(clean())
    assert not ok and lines[1] == "plants judged as expected: 2 of 11"
    assert "opus: served another model judged [], expected ['served_differs']" in lines


def test_the_plants_come_from_the_first_answered_spawn_of_each_alias_with_a_full_id_only_where_a_shorter_one_names_a_model():
    found = plants(clean())
    assert [(alias, name) for alias, name, _, _ in found] == [
        ("opus", "served another model"), ("opus", "served two, one wrong"), ("opus", "resolved to another family"),
        ("opus", "nothing resolved, served another family"), ("opus", "[1m] served under its plain id"),
        ("opus", "full id resolved to a longer one"),
        ("sonnet", "served another model"), ("sonnet", "served two, one wrong"),
        ("sonnet", "resolved to another family"), ("sonnet", "nothing resolved, served another family"),
        ("sonnet", "[1m] served under its plain id")]
    assert found[1][2]["served"] == ("claude-opus-5-5", PLANTED)
    assert found[5][2]["requested"] == "claude-opus-5"
    assert shorter_id("claude-sonnet-5") is None


def test_the_one_million_count_takes_only_answered_spawns_resolved_to_a_1m_model():
    joined = pd.concat([clean(), pd.DataFrame([spawn("a6", "opus", "claude-opus-5[1M]", [])])], ignore_index=True)
    assert one_million_alarms(joined) == 1


def test_logs_where_no_spawn_gives_a_plant_fail_the_gate_rather_than_pass_it_untested():
    unanswered = pd.DataFrame([spawn("a1", "opus", "claude-opus-5", []),
                               spawn("a2", None, "claude-opus-5", ["claude-opus-5"]),
                               spawn("a3", "opus", None, ["claude-opus-5"])], columns=list(JOINED_COLUMNS))
    assert gate(unanswered) == (False, ["false alarms: 0", "plants judged as expected: 0 of 0"])
