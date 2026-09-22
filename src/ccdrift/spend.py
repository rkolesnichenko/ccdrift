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
from ccdrift.prices import Price, billed_tokens, fit_prices
from ccdrift.sessions import SOURCE_PROJECT, project_of
from ccdrift.texts import DIMENSION_NAMES, approx, project_path, spend_line, unpriced_line

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

# The share of the window's tokens that, left unpriced, withholds the dollar total. This
# is a display rule and not a detection cutoff: nothing judges or alerts on it, it decides
# only whether one figure prints. A rounding-error model should not silence a total and a
# real one should, so the test is against the unpriced models' summed share: five models
# at 0.9% each are 4.5% of the window counted as zero dollars, which is the partial total
# the rule exists to refuse.
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
        files = turns["source_file"].astype(str)
        # Pointing --source at one project's own folder leaves every transcript directly
        # under it, so project_of would make a project of each session and name it with
        # that session's id, dashes read back as slashes. Nothing in ccdrift prints a
        # session id. This is sessions.session_starts' `loose` guard: with no project
        # folder anywhere in the source, the source itself is the one project.
        if not files.str.contains("/").any():
            return pd.Series(project_path(SOURCE_PROJECT), index=turns.index, dtype="object")
        return files.map(project_of).map(project_path)
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
        # The join is exact string equality, and stays exact. Measured over the owner's
        # corpus on 2026-09-22: every one of 158,246 responses carries a message.model
        # with a cost-state counterpart, dated aliases included, so normalising the two
        # key spaces would buy nothing and could only join a response to a price that is
        # not its own. The divergence runs the other way and cannot be repaired from here:
        # cost-state keys claude-opus-5[1m] apart from claude-opus-5, a tier message.model
        # never records, so 1m-context responses price at the plain rate. That
        # understatement is real and is not measurable from the response side.
        rows = models == model
        if not rows.any():
            continue
        out.loc[rows] = (billed_tokens(turns.loc[rows, "input_tokens"].fillna(0),
                                       turns.loc[rows, "cache_creation"].fillna(0),
                                       turns.loc[rows, "cache_read"].fillna(0)) * price.input_rate
                         + turns.loc[rows, "output_tokens"].fillna(0) * price.output_rate)
    return out


def spend_rows(turns: pd.DataFrame, dimension: str, prices: dict[str, Price]) -> pd.DataFrame:
    """One row per bucket of `dimension`: responses, tokens, share of the window's tokens,
    dollars from the models it could price while the ones it could not stay under
    MATERIAL_SHARE of the bucket's own tokens, and the models with no price, named so a
    blank money column can say why it is blank. A bucket keeps naming them once priced,
    since the JSON carries both and a reader is owed the reason the figure is a floor.
    Largest first, ties by name, so two runs over one history read the same."""
    columns = ["bucket", "responses", "tokens", "share", "dollars", "unpriced"]
    if turns.empty:
        return pd.DataFrame(columns=columns)
    total = total_tokens(turns)
    frame = pd.DataFrame({
        "bucket": buckets(turns, dimension).to_numpy(),
        "model": buckets(turns, "model").to_numpy(),
        "tokens": sum(turns[column].fillna(0) for column in TOKEN_COLUMNS).to_numpy(),
        "dollars": response_dollars(turns, prices).to_numpy(),
    })
    rows = []
    for bucket, group in frame.groupby("bucket", sort=True):
        tokens, priced = float(group["tokens"].sum()), group["dollars"].notna()
        missing = float(group.loc[~priced, "tokens"].sum())
        # The window's own materiality rule, applied at the level it was always about. On
        # the owner's corpus on 2026-09-22, all-or-nothing meant 11 responses of a model
        # with no price, 0.009% of the window's tokens, blanked the top row of five of the
        # six public dimensions, 60.5% to 95.5% of the window each, while the total one
        # line above printed because that same 0.009% cleared this same cutoff. The
        # denominator is the bucket and not the window: measured against the window, a
        # small bucket that is entirely unpriced would pass, which is the one case to catch.
        material = (missing / tokens >= MATERIAL_SHARE) if tokens else bool((~priced).any())
        rows.append({"bucket": str(bucket), "responses": len(group), "tokens": tokens,
                     "share": tokens / total if total else 0.0,
                     "dollars": float("nan") if material else float(group.loc[priced, "dollars"].sum()),
                     "unpriced": tuple(sorted(set(group.loc[~priced, "model"])))})
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values(["tokens", "bucket"], ascending=[False, True],
                           kind="stable").reset_index(drop=True)


def priced_in_window(turns: pd.DataFrame, prices: dict[str, Price]) -> dict[str, Price]:
    """The fitted prices for models the window actually spent on. A price for a model that
    never ran in the window explains nothing about the breakdown beside it."""
    if turns.empty:
        return {}
    ran = set(buckets(turns, "model"))
    return {model: price for model, price in prices.items() if model in ran}


def unpriced_models(turns: pd.DataFrame, prices: dict[str, Price]) -> list[tuple[str, float]]:
    """The window's models with no fitted price and the share of its tokens each carries,
    largest first. These are what withholds a dollar figure, so the output can name them
    rather than leaving a reader to work out why the money column went blank."""
    if turns.empty:
        return []
    by_model = spend_rows(turns, "model", prices)
    return [(str(row.bucket), float(row.share))
            for row in by_model.itertuples(index=False) if pd.isna(row.dollars)]


def priced_total(turns: pd.DataFrame, prices: dict[str, Price]) -> Optional[float]:
    """What the window cost, or None when the models with no price carry MATERIAL_SHARE
    of its tokens between them. A partial total is how a tool becomes quietly wrong, and
    the share is summed rather than tested model by model because several small unpriced
    models add up to the same silently missing money as one large one."""
    if turns.empty:
        return None
    by_model = spend_rows(turns, "model", prices)
    if float(by_model.loc[by_model["dollars"].isna(), "share"].sum()) >= MATERIAL_SHARE:
        return None
    return float(by_model["dollars"].fillna(0).sum())


def spend_json(turns: pd.DataFrame, dimensions: Sequence[str], prices: dict[str, Price],
               window: Sequence[str]) -> str:
    """The breakdown as JSON: aggregates only, and no bucket that names a folder or a
    branch. The withheld dimensions are listed rather than silently dropped."""
    shown = [d for d in dimensions if d not in PRIVATE_DIMENSIONS]
    withheld = [d for d in dimensions if d in PRIVATE_DIMENSIONS]
    fitted = priced_in_window(turns, prices)
    payload = {
        "days": len(window),
        "tokens": total_tokens(turns),
        "dollars": priced_total(turns, prices),
        # Scoped to the models the window spent on, and carrying what each price rests on,
        # so a `dollars` of null can be read against the fit rather than guessed at.
        "priced_models": [{"model": model, "residual": fitted[model].residual, "rows": fitted[model].rows}
                          for model in sorted(fitted)],
        "unpriced_models": [{"model": model, "share": share} for model, share in unpriced_models(turns, prices)],
        "dimensions": {d: [{"bucket": row.bucket, "responses": int(row.responses),
                            "tokens": float(row.tokens), "share": float(row.share),
                            "dollars": None if pd.isna(row.dollars) else float(row.dollars),
                            "unpriced": list(row.unpriced)}
                           for row in spend_rows(turns, d, prices).itertuples(index=False)]
                       for d in shown},
        "withheld": withheld,
    }
    return json.dumps(payload, indent=1) + "\n"


DEFAULT_DAYS = 30
DEFAULT_ORDER = ("thread", "agent", "skill", "plugin", "mcp", "model")


def spend_lines(turns: pd.DataFrame, dimension: str, prices: dict[str, Price]) -> list[str]:
    """One dimension's section, starting with a blank line. A bucket with no dollars names
    the models that have none, unless nothing in the window is priced at all: a history
    with no cost record prices nothing, which is the common case and reads as normal, so
    "no price" on every line there would be noise rather than an explanation."""
    rows = spend_rows(turns, dimension, prices)
    if rows.empty:
        return []
    explain = bool(priced_in_window(turns, prices))
    lines = ["", f"By {DIMENSION_NAMES[dimension]}"]
    for row in rows.itertuples(index=False):
        dollars = None if pd.isna(row.dollars) else float(row.dollars)
        lines.append(spend_line(row.bucket, int(row.responses), float(row.tokens), float(row.share), dollars,
                                row.unpriced if explain else ()))
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
    unpriced = unpriced_models(turns, prices)
    # Said once, at the top: which models have no price, and whether that was enough to
    # withhold the total. Missing money that says nothing reads as broken arithmetic.
    if unpriced and priced_in_window(turns, prices):
        lines.append(unpriced_line(unpriced, total is None, MATERIAL_SHARE))
    for dimension in dimensions:
        lines += spend_lines(turns, dimension, prices)
    print("\n".join(lines) + "\n", end="")
    return 0
