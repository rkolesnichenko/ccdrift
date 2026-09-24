"""The G14 gate: whether logged characters can be put in tokens."""

import numpy as np
import pandas as pd

import lab.components
from ccdrift.logs import parse_all
from lab.components import (MIN_VARIED, PARTS, gate, half_factors, part_factors, pooled_factor,
                            starts_with_components)
from tests.helpers import (agent_listing, at, attachment, deferred_tools, line, prompt, skill_listing, text,
                           write)


def constructed(factors, sessions=240, groups=6, seed=1, vary=PARTS, noise=50.0):
    """Session starts in `groups` groups, each with a level of its own, whose tokens are
    that level plus each part's characters times its factor in `factors`. Only the parts
    in `vary` differ between sessions of a group; the others hold one size per group."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(sessions):
        group = i % groups
        sizes = {part: float(rng.integers(500, 5000)) if part in vary else 1000.0 * (group + 1) for part in PARTS}
        tokens = 20_000 * (group + 1) + sum(factors[part] * sizes[part] for part in PARTS) + rng.normal(0, noise)
        rows.append({"group": f"g{group}", "prompt_tokens": tokens, **sizes})
    starts = pd.DataFrame(rows)
    starts["chars"] = starts[list(PARTS)].sum(axis=1)
    return starts


EVEN = {part: 0.3 for part in PARTS}


def test_the_fit_recovers_a_planted_factor_through_each_groups_own_level():
    assert abs(pooled_factor(constructed(EVEN)) - 0.3) < 0.005


def test_a_level_that_rises_with_characters_across_groups_doesnt_bias_the_factor():
    # Between groups, characters and tokens rise together far faster than 0.3 a character:
    # a fit across groups would read that as the factor.
    starts = constructed(EVEN)
    starts["prompt_tokens"] += starts["group"].str[1:].astype(int) * 1_000_000
    starts["chars"] += starts["group"].str[1:].astype(int) * 1_000
    assert abs(pooled_factor(starts) - 0.3) < 0.005


def test_every_part_measured_and_agreeing_passes():
    passed, notes = gate(constructed(EVEN))
    assert passed, notes


def test_tokens_that_dont_follow_characters_fail_the_split_half_rule():
    starts = constructed({part: 0.0 for part in PARTS}, sessions=60, noise=5_000.0)
    passed, notes = gate(starts)
    assert not passed
    assert any("not all within" in note for note in notes) or "no factor" in notes[-1]


def test_a_part_with_a_factor_of_its_own_fails_the_gate_and_is_named():
    passed, notes = gate(constructed({**EVEN, "skills listing": 0.05}))
    assert not passed
    assert any(note.startswith("skills listing:") and "more than" in note for note in notes)


def test_a_part_too_few_sessions_vary_is_not_measurable_and_fails_the_gate():
    starts = constructed(EVEN, vary=[part for part in PARTS if part != "tool definitions"])
    factors = part_factors(starts)
    assert factors["tool definitions"] == (None, 0)
    passed, notes = gate(starts)
    assert not passed and any(note.startswith("tool definitions: not measurable") for note in notes)


def varied_in(sessions):
    """Constructed starts whose tool definitions hold one size per group except in
    `sessions` sessions of the first group, half of them larger and half smaller by the
    same amount, so the group's mean doesn't move and exactly those sessions vary."""
    starts = constructed(EVEN, vary=[part for part in PARTS if part != "tool definitions"])
    first = starts.index[starts["group"] == "g0"][:sessions]
    for sign, rows in ((1, first[::2]), (-1, first[1::2])):
        starts.loc[rows, "tool definitions"] += sign * 500.0
        starts.loc[rows, "prompt_tokens"] += sign * 150.0
    starts["chars"] = starts[list(PARTS)].sum(axis=1)
    return starts


def test_a_part_varied_in_just_enough_sessions_is_measured():
    factor, varied = part_factors(varied_in(MIN_VARIED))["tool definitions"]
    assert varied == MIN_VARIED and abs(factor - 0.3) < 0.01


def test_a_part_varied_in_fewer_sessions_than_that_is_not():
    assert part_factors(varied_in(MIN_VARIED - 2))["tool definitions"] == (None, MIN_VARIED - 2)


def test_halves_that_disagree_fail_the_gate_even_when_every_part_agrees(monkeypatch):
    # One session far larger than the rest with no more tokens drags whichever half draws
    # it; every part is made to agree with the pooled factor, so only the halves can fail.
    starts = constructed(EVEN)
    starts.loc[starts.index[0], "chars"] += 100_000.0
    pooled = pooled_factor(starts)
    monkeypatch.setattr(lab.components, "part_factors", lambda _: {part: (pooled, MIN_VARIED) for part in PARTS})
    passed, notes = gate(starts)
    assert not passed
    assert any(note.startswith("halves:") and "not all within" in note for note in notes)


def test_the_splits_are_the_same_every_run():
    starts = constructed(EVEN)
    assert half_factors(starts) == half_factors(starts)


def test_session_starts_carry_their_parts_and_leave_out_the_agent_sdk_unless_asked(tmp_path):
    for sid, entrypoint in (("a", "cli"), ("b", "sdk-py")):
        write(tmp_path / "p" / f"{sid}.jsonl", [
            attachment(at(0), skill_listing(["review"], chars=2000), sid=sid),
            attachment(at(0), deferred_tools(["Read"], line_chars=20), sid=sid),
            attachment(at(0), agent_listing(["Plan"], line_chars=30), sid=sid),
            prompt(at(1), sid=sid),
            line(f"m-{sid}", text(40), ts=at(2), sid=sid, entrypoint=entrypoint, version="2.1.250",
                 cache_creation=5000)])
    tables = parse_all(tmp_path)
    cli = starts_with_components(tables)
    assert cli["source_file"].tolist() == ["p/a.jsonl"]
    assert cli.loc[0, "skills listing"] == 2000 and cli.loc[0, "the rest"] == 50
    assert cli.loc[0, "first message"] == len("next request") and cli.loc[0, "tool definitions"] == 0
    assert cli.loc[0, "chars"] == 2000 + 50 + len("next request")
    both = starts_with_components(tables, with_sdk=True)
    assert both["source_file"].tolist() == ["p/a.jsonl", "p/b.jsonl"]
    assert both["group"].str.contains("sdk").tolist() == [False, True]
