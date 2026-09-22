"""Where the tokens went: one partition of the history per dimension.

ccdrift's other reports ask whether something changed. This one asks where the spend
went, so it counts both threads rather than the main thread alone, and it judges
nothing: no rule, no threshold and no alert turns on any number here."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ccdrift.history import HistoryError, load_history
from ccdrift.logs import no_transcripts_message, outside_sdk
from ccdrift.prices import CACHE_READ_RATE, CACHE_WRITE_RATE, Price, fit_prices
from ccdrift.sessions import project_of
from ccdrift.texts import DIMENSION_NAMES, approx, project_path, spend_line

DIMENSIONS = ("thread", "agent", "skill", "plugin", "mcp", "model", "project", "branch")
TOKEN_COLUMNS = ("input_tokens", "output_tokens", "cache_creation", "cache_read")

# Dimensions whose buckets are your folders and your branch names. `ccdrift cost` prints
# them; the JSON withholds them, as report --json already withholds the projects, because
# a branch like "worktree-better-auth-sso" names a plan as surely as a folder names a
# client.
PRIVATE_DIMENSIONS = ("project", "branch")

# The column each dimension groups by, and what a response the dimension does not name
# is called. Every response falls in exactly one bucket, so a dimension's shares sum to 1.
DIMENSION_COLUMNS = {"agent": ("agent_type", "no agent"), "skill": ("attribution_skill", "no skill"),
                     "plugin": ("attribution_plugin", "no plugin"), "mcp": ("attribution_mcp", "no MCP server"),
                     "model": ("model", "unknown model"), "branch": ("git_branch", "no branch")}

# A model's share of the window's tokens below which leaving it unpriced does not
# withhold the total: a rounding error should not silence a figure, a real model should.
MATERIAL_SHARE = 0.01


def spend_turns(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """The responses the breakdown counts: complete UTC days, outside Agent SDK sessions,
    on both threads. This is judged_turns without its main-thread test, since subagents
    are most of the spend and dropping them would discard most of the answer."""
    if df.empty:
        return df
    return df[(df["day"].astype(str) < today.isoformat()) & outside_sdk(df)]


def total_tokens(turns: pd.DataFrame) -> float:
    """Every token the window spent, input, output and both cache halves."""
    if turns.empty:
        return 0.0
    return float(sum(turns[column].fillna(0).sum() for column in TOKEN_COLUMNS))


def buckets(turns: pd.DataFrame, dimension: str) -> pd.Series:
    """Which bucket each response falls in for `dimension`, one each and never none."""
    if dimension == "thread":
        return turns["is_sidechain"].astype(bool).map({True: "subagent", False: "main thread"})
    if dimension == "project":
        # Read back the way report.py's project_lines does, so the same folder reads
        # the same in both commands rather than as its raw, dash-encoded form here.
        return turns["source_file"].astype(str).map(project_of).map(project_path)
    column, absent = DIMENSION_COLUMNS[dimension]
    values = turns[column] if column in turns else pd.Series(None, index=turns.index, dtype="object")
    return values.where(values.notna() & (values.astype(str) != ""), absent).astype(str)


def response_dollars(turns: pd.DataFrame, prices: dict[str, Price]) -> pd.Series:
    """What each response cost, NaN where its model has no price, so a bucket holding one
    unpriced response reports no dollars rather than a total that quietly omits it."""
    out = pd.Series(float("nan"), index=turns.index, dtype="float64")
    if turns.empty:
        return out
    models = turns["model"].astype(str)
    for model, price in prices.items():
        rows = models == model
        if not rows.any():
            continue
        out.loc[rows] = ((turns.loc[rows, "input_tokens"].fillna(0)
                          + CACHE_WRITE_RATE * turns.loc[rows, "cache_creation"].fillna(0)
                          + CACHE_READ_RATE * turns.loc[rows, "cache_read"].fillna(0)) * price.input_rate
                         + turns.loc[rows, "output_tokens"].fillna(0) * price.output_rate)
    return out


def spend_rows(turns: pd.DataFrame, dimension: str, prices: dict[str, Price]) -> pd.DataFrame:
    """One row per bucket of `dimension`: responses, tokens, share of the window's tokens,
    and dollars when every response in the bucket has a priced model. Largest first, ties
    by name, so two runs over one history read the same."""
    columns = ["bucket", "responses", "tokens", "share", "dollars"]
    if turns.empty:
        return pd.DataFrame(columns=columns)
    total = total_tokens(turns)
    frame = pd.DataFrame({
        "bucket": buckets(turns, dimension).to_numpy(),
        "tokens": sum(turns[column].fillna(0) for column in TOKEN_COLUMNS).to_numpy(),
        "dollars": response_dollars(turns, prices).to_numpy(),
    })
    rows = []
    for bucket, group in frame.groupby("bucket", sort=True):
        tokens = float(group["tokens"].sum())
        rows.append({"bucket": str(bucket), "responses": len(group), "tokens": tokens,
                     "share": tokens / total if total else 0.0,
                     "dollars": float(group["dollars"].sum()) if group["dollars"].notna().all() else float("nan")})
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values(["tokens", "bucket"], ascending=[False, True],
                           kind="stable").reset_index(drop=True)


def priced_total(turns: pd.DataFrame, prices: dict[str, Price]) -> Optional[float]:
    """What the window cost, or None when a model carrying at least MATERIAL_SHARE of its
    tokens has no price. A partial total is how a tool becomes quietly wrong."""
    if turns.empty:
        return None
    by_model = spend_rows(turns, "model", prices)
    unpriced = by_model[by_model["dollars"].isna() & (by_model["share"] >= MATERIAL_SHARE)]
    if len(unpriced):
        return None
    return float(by_model["dollars"].fillna(0).sum())


def spend_json(turns: pd.DataFrame, dimensions: Sequence[str], prices: dict[str, Price],
               window: Sequence[str]) -> str:
    """The breakdown as JSON: aggregates only, and no bucket that names a folder or a
    branch. The withheld dimensions are listed rather than silently dropped."""
    shown = [d for d in dimensions if d not in PRIVATE_DIMENSIONS]
    withheld = [d for d in dimensions if d in PRIVATE_DIMENSIONS]
    total = priced_total(turns, prices)
    payload = {
        "days": len(window),
        "tokens": total_tokens(turns),
        "dollars": total,
        "priced_models": sorted(prices),
        "dimensions": {d: [{"bucket": row.bucket, "responses": int(row.responses),
                            "tokens": float(row.tokens), "share": float(row.share),
                            "dollars": None if pd.isna(row.dollars) else float(row.dollars)}
                           for row in spend_rows(turns, d, prices).itertuples(index=False)]
                       for d in shown},
        "withheld": withheld,
    }
    return json.dumps(payload, indent=1) + "\n"


DEFAULT_DAYS = 30
DEFAULT_ORDER = ("thread", "agent", "skill", "plugin", "mcp", "model")


def spend_lines(turns: pd.DataFrame, dimension: str, prices: dict[str, Price]) -> list[str]:
    """One dimension's section, starting with a blank line."""
    rows = spend_rows(turns, dimension, prices)
    if rows.empty:
        return []
    lines = ["", f"By {DIMENSION_NAMES[dimension]}"]
    for row in rows.itertuples(index=False):
        dollars = None if pd.isna(row.dollars) else float(row.dollars)
        lines.append(spend_line(row.bucket, int(row.responses), float(row.tokens), float(row.share), dollars))
    return lines


def run_spend(source: Path, state_path: Path, days: Optional[int] = None, by: Optional[str] = None,
             as_json: bool = False, today: Optional[date] = None) -> int:
    """Print where the window's tokens went. Reads the history like `report`, saves no
    state, and judges nothing."""
    try:
        tables = load_history(source, state_path, claim=False)
    except HistoryError as exc:
        print(exc, file=sys.stderr)
        return 1
    if tables.responses.empty:
        print(no_transcripts_message(source), file=sys.stderr)
        return 2
    today = today or datetime.now(timezone.utc).date()
    turns = spend_turns(tables.responses, today)
    window = sorted(turns["day"].astype(str).unique())[-(days or DEFAULT_DAYS):]
    turns = turns[turns["day"].astype(str).isin(window)]
    prices = fit_prices(tables.model_usage)
    dimensions = [by] if by else list(DEFAULT_ORDER)
    if as_json:
        print(spend_json(turns, dimensions, prices, window), end="")
        return 0
    total = priced_total(turns, prices)
    money = "" if total is None else f", {'$' + format(total, ',.2f')}"
    lines = [f"{len(window)} complete UTC days, {approx(total_tokens(turns))} tokens{money}.",
             "Every section below accounts for all of them; a response can appear in more than one section."]
    for dimension in dimensions:
        lines += spend_lines(turns, dimension, prices)
    print("\n".join(lines) + "\n", end="")
    return 0
