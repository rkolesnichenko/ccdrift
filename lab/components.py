"""Can what Claude Code logs about a session's start be put in tokens? (G14)

A session-start alert says how many tokens a step added. What Claude Code logs about the
start (skills, deferred tools, agent types, MCP instructions, CLAUDE.md, the system
prompt, from 2.1.267 the tool definitions, and the first message) comes in characters.
Saying what share of a step the logged changes explain needs tokens per logged character,
and this gate asks whether the logs can measure one.

Each session start's prompt tokens are fitted as `level + factor * characters`, the level
taken per project, Claude Code version, entrypoint and set of parts logged. That level
absorbs whatever those share and nothing logs, so the factor comes only from how sessions
in one group differ from each other. G14 passes when:

- the factor fitted on each half of the sessions, split at random (seeded, SPLITS times),
  is within AGREE of the pooled one, and
- each part the alert would put in tokens (PARTS) has a factor of its own, fitted on the
  same groups, within AGREE of the pooled one.

A part that fewer than MIN_VARIED sessions vary within their group can't be fitted at all.
The gate fails on it rather than skipping it: the alert would apply the factor to every
part, and one it was never measured on is a claim with nothing behind it.

The alert judges CLI session starts only, so those are what the gate fits by default.
`--with-sdk` fits over Agent SDK session starts too, which carry the same records. Only
sessions of days before `--today` (default: today) count, so a re-run on the same day
reads the same sessions.

Run from the repo root:

  uv run --group lab python -m lab.components
  uv run --group lab python -m lab.components --with-sdk
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ccdrift.logs import Tables, default_source, outside_sdk, parse_all
from ccdrift.sessions import session_starts

# A judgement, as G3's bar is: how far a factor fitted on part of the data may sit from
# the pooled one before the pooled one isn't taken to describe it.
AGREE = 0.20
# A judgement: how many sessions have to differ from their group in a part before that
# part's own factor counts as measured.
MIN_VARIED = 30
SPLITS = 5
SEED = 14
# The parts the alert would put in tokens, by the component sizes each sums. The four
# small ones go together: no one of them varies enough on its own.
PARTS = {"tool definitions": ("tools_chars",), "skills listing": ("skills_chars",),
         "CLAUDE.md": ("claude_md_chars",), "first message": ("message_chars",),
         "the rest": ("deferred_chars", "agents_chars", "mcp_chars", "system_chars")}
SIZES = tuple(column for columns in PARTS.values() for column in columns)
GATE = "G14"


def starts_with_components(tables: Tables, with_sdk: bool = False) -> pd.DataFrame:
    """One row per session start with its prompt tokens, its group and the characters of
    each part, a part Claude Code didn't log counting as 0 and setting the group apart.
    CLI starts only, or Agent SDK ones too with `with_sdk`."""
    responses = tables.responses
    sdk_files = set(responses.loc[~outside_sdk(responses), "source_file"].astype(str)) if not responses.empty else set()
    if with_sdk and not responses.empty:
        # session_starts keeps CLI transcripts only; SDK ones are told apart by the group instead.
        responses = responses.assign(entrypoint="cli")
    starts = session_starts(responses).merge(tables.components, on="source_file", how="inner")
    logged = starts[list(SIZES)].notna()
    pattern = logged.apply(lambda row: "".join("1" if known else "0" for known in row), axis=1) \
        if not starts.empty else pd.Series(dtype=str)
    starts["group"] = (starts["project"].astype(str) + "|" + starts["version"].fillna("unknown").astype(str) + "|"
                       + starts["source_file"].astype(str).isin(sdk_files).map({True: "sdk", False: "cli"}) + "|"
                       + pattern)
    for part, columns in PARTS.items():
        starts[part] = starts[list(columns)].fillna(0.0).sum(axis=1)
    starts["chars"] = starts[list(PARTS)].sum(axis=1)
    return starts.sort_values(["timestamp", "source_file"], kind="stable").reset_index(drop=True)


def _within(values: pd.Series, groups: pd.Series) -> pd.Series:
    """`values` less their group's mean: what sets a session apart from its own group."""
    return values - values.groupby(groups).transform("mean")


def pooled_factor(starts: pd.DataFrame) -> Optional[float]:
    """Tokens per logged character, fitted within groups; None when no group varies."""
    x = _within(starts["chars"].astype(float), starts["group"])
    y = _within(starts["prompt_tokens"].astype(float), starts["group"])
    spread = float((x * x).sum())
    return float((x * y).sum()) / spread if spread > 0 else None


def half_factors(starts: pd.DataFrame, splits: int = SPLITS, seed: int = SEED) -> list[tuple[Optional[float], ...]]:
    """The pooled factor on each half of the sessions, for `splits` seeded random splits."""
    rng = np.random.default_rng(seed)
    halves = []
    for _ in range(splits):
        first = rng.random(len(starts)) < 0.5
        halves.append((pooled_factor(starts[first]), pooled_factor(starts[~first])))
    return halves


def part_factors(starts: pd.DataFrame) -> dict[str, tuple[Optional[float], int]]:
    """Each part's own factor, fitted together within groups, and how many sessions
    differ from their group in it. The factor is None when fewer than MIN_VARIED do."""
    x = pd.DataFrame({part: _within(starts[part].astype(float), starts["group"]) for part in PARTS})
    y = _within(starts["prompt_tokens"].astype(float), starts["group"])
    varied = {part: int((x[part].abs() > 0.5).sum()) for part in PARTS}
    measured = [part for part in PARTS if varied[part] >= MIN_VARIED]
    fitted: dict[str, float] = {}
    if measured:
        coef, *_ = np.linalg.lstsq(x[measured].to_numpy(), y.to_numpy(), rcond=None)
        fitted = dict(zip(measured, (float(c) for c in coef)))
    return {part: (fitted.get(part), varied[part]) for part in PARTS}


def _agrees(value: Optional[float], pooled: float) -> bool:
    return value is not None and abs(value / pooled - 1) <= AGREE


def _per_token(factor: Optional[float]) -> str:
    return "none" if factor is None else f"{factor:.3f} ({1 / factor:.2f} characters a token)" if factor > 0 \
        else f"{factor:.3f}"


def gate(starts: pd.DataFrame) -> tuple[bool, list[str]]:
    """Whether G14 passes on `starts`, and what to say about it."""
    pooled = pooled_factor(starts)
    groups = starts["group"].nunique() if not starts.empty else 0
    notes = [f"{len(starts)} session starts in {groups} groups; pooled factor {_per_token(pooled)}"]
    if pooled is None or pooled <= 0:
        return False, notes + ["no factor to test: nothing varies within a group, or tokens fall as characters rise"]
    halves = half_factors(starts)
    halves_agree = all(_agrees(one, pooled) for pair in halves for one in pair)
    notes.append("halves: " + ", ".join("/".join("none" if one is None else f"{one:.3f}" for one in pair)
                                        for pair in halves)
                 + (f", all within {AGREE:.0%}" if halves_agree else f", not all within {AGREE:.0%}"))
    parts = part_factors(starts)
    for part, (factor, varied) in parts.items():
        if factor is None:
            notes.append(f"{part}: not measurable, {varied} sessions vary it within their group (needs {MIN_VARIED})")
        else:
            notes.append(f"{part}: {factor:.3f} over {varied} sessions"
                         + ("" if _agrees(factor, pooled) else f", more than {AGREE:.0%} from the pooled factor"))
    parts_agree = all(_agrees(factor, pooled) for factor, _ in parts.values())
    return halves_agree and parts_agree, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument("--with-sdk", action="store_true", help="fit over Agent SDK session starts too")
    parser.add_argument("--today", type=date.fromisoformat, default=date.today())
    args = parser.parse_args(argv)
    starts = starts_with_components(parse_all(args.source), with_sdk=args.with_sdk)
    starts = starts[starts["day"].astype(str) < args.today.isoformat()].reset_index(drop=True)
    passed, notes = gate(starts)
    print(f"{GATE}: {'PASS' if passed else 'FAIL'}")
    for note in notes:
        print(f"  {note}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
