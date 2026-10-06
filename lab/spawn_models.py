"""Is a subagent served the model it asked for? (G18)

Each Agent call names the model it wants (an alias such as "sonnet", a full id, or none),
its result names the model Claude Code resolved that to, and the subagent's own responses
name the model that served them. ccdrift ships a rule (ccdrift.spawns.judge) that calls a
spawn wrong when what was resolved isn't what was asked for, or when what served it isn't
what was resolved; the hourly check alerts once per kind, models and version.

A mismatch is a fact, not a statistic, so there is no cutoff to sweep. The gate asks two
things of the rule as shipped, on the owner's logs:
- false alarms: every joined spawn is judged, and any mismatch found is one, since none was
  seen there; it also counts the spawns that would alarm if "[1m]" weren't dropped first;
- catches: from the first answered spawn of each alias asked for, copies with the resolved
  or served model changed in each way the rule must catch (served another model, served two
  of which one is wrong, resolved to another family, a full id resolved to a longer one,
  nothing resolved and served another family), each caught as its kind, and a copy served
  the resolved "[1m]" model under its plain id, which must not alarm.

The rule passes with no false alarm and every plant judged as expected.
Output is aggregate: aliases, model ids and counts.

Run from the repo root:

  uv run --group lab python -m lab.spawn_models
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ccdrift.logs import default_source, parse_all
from ccdrift.spawns import judge, mismatches, model_name, spawn_models

# A model no spawn is ever resolved to or served, for the plants.
PLANTED = "claude-planted-model-1"


def one_million_alarms(joined: pd.DataFrame) -> int:
    """How many answered spawns would alarm if "[1m]" weren't dropped: those resolved to a
    "[1m]" model, which is always served under its plain id."""
    return sum(1 for spawn in joined.to_dict("records")
               if spawn["responses"] and isinstance(spawn["resolved"], str) and "[1m]" in spawn["resolved"].lower())


def shorter_id(resolved: str) -> Optional[str]:
    """A full id one part shorter than `resolved` that still names a model ("claude-opus-5"
    from "claude-opus-5-5"): a request for it must not be honoured by the longer one."""
    parts = model_name(resolved).split("-")
    return "-".join(parts[:-1]) if len(parts) > 3 else None


def plants(joined: pd.DataFrame) -> list[tuple[str, str, dict[str, Any], list[str]]]:
    """(alias, plant, spawn, the kinds judge must give) for every plant on the first answered
    spawn of each alias asked for."""
    answered = joined[(joined["responses"] > 0) & joined["requested"].notna() & joined["resolved"].notna()]
    out = []
    for alias, group in answered.groupby(answered["requested"].map(model_name), sort=True):
        spawn = group.to_dict("records")[0]
        served = model_name(spawn["resolved"])
        out += [(alias, "served another model", {**spawn, "served": (PLANTED,)}, ["served_differs"]),
                (alias, "served two, one wrong", {**spawn, "served": tuple(sorted((served, PLANTED)))},
                 ["served_differs"]),
                (alias, "resolved to another family", {**spawn, "resolved": PLANTED},
                 ["not_honoured", "served_differs"]),
                (alias, "nothing resolved, served another family", {**spawn, "resolved": None, "served": (PLANTED,)},
                 ["not_honoured"]),
                (alias, "[1m] served under its plain id", {**spawn, "resolved": served + "[1m]", "served": (served,)},
                 [])]
        shorter = shorter_id(spawn["resolved"])
        if shorter is not None:
            out.append((alias, "full id resolved to a longer one", {**spawn, "requested": shorter}, ["not_honoured"]))
    return out


def gate(joined: pd.DataFrame) -> tuple[bool, list[str]]:
    """The verdict on the shipped rule and the lines explaining it."""
    false = mismatches(joined)
    planted = plants(joined)
    missed = [f"{alias}: {name} judged {judge(s['requested'], s['resolved'], s['served'])}, expected {kinds}"
              for alias, name, s, kinds in planted if judge(s["requested"], s["resolved"], s["served"]) != kinds]
    lines = [f"false alarms: {len(false)}" + ("" if false.empty else " (" + ", ".join(
                 f"{kind} {count}" for kind, count in sorted(Counter(false['kind']).items())) + ")"),
             f"plants judged as expected: {len(planted) - len(missed)} of {len(planted)}", *missed]
    return false.empty and bool(planted) and not missed, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=default_source())
    args = parser.parse_args(argv)
    tables = parse_all(args.source)
    joined = spawn_models(tables.spawns, tables.responses)
    if joined.empty:
        print("no spawns found")
        return 1
    answered = int((joined["responses"] > 0).sum())
    print(f"spawns {len(joined)}, answered {answered}, days {joined['day'].min()}..{joined['day'].max()}")
    asked = Counter(joined["requested"].map(lambda raw: model_name(raw) or "none"))
    print("asked for: " + ", ".join(f"{alias} {count}" for alias, count in sorted(asked.items())))
    print(f"served under the plain id of a resolved [1m] model: {one_million_alarms(joined)}")
    ok, lines = gate(joined)
    print("\n".join(lines))
    print(f"G18: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
