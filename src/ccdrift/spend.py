"""Where the tokens went: one partition of the history per dimension.

ccdrift's other reports ask whether something changed. This one asks where the spend
went, so it counts both threads rather than the main thread alone, and it judges
nothing: no rule, no threshold and no alert turns on any number here."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from ccdrift.history import HistoryError, load_history
from ccdrift.logs import outside_sdk
from ccdrift.prices import Price, fit_prices
from ccdrift.sessions import project_of
from ccdrift.texts import (ABSENT_NAMES, COST_LINES, DETACHED_NAME, DIMENSION_NAMES, THREAD_NAMES, approx,
                           no_transcripts_message, project_path, spend_line, unpriced_line)

DIMENSIONS = ("thread", "agent", "skill", "plugin", "mcp", "model", "project", "branch")
TOKEN_COLUMNS = ("input_tokens", "output_tokens", "cache_creation", "cache_read")

# Dimensions whose buckets are your folders and your branch names. `ccdrift cost` prints
# them; the JSON withholds them, as report --json already withholds the projects, because
# a branch like "worktree-better-auth-sso" names a plan as surely as a folder names a
# client.
PRIVATE_DIMENSIONS = ("project", "branch")

# The column each dimension groups by. A response the dimension does not name falls in the
# bucket texts.ABSENT_NAMES calls it, so every response is in exactly one bucket and a
# dimension's shares sum to 1.
DIMENSION_COLUMNS = {"agent": "agent_type", "skill": "attribution_skill", "plugin": "attribution_plugin",
                     "mcp": "attribution_mcp", "model": "model", "branch": "git_branch"}

# What Claude Code writes to gitBranch when no branch is checked out; ccdrift calls it
# texts.DETACHED_NAME. `git rev-parse --abbrev-ref HEAD` answers "HEAD" in that state. Confirmed on
# 2026-09-22 over the owner's own corpus rather than synthetically: gitBranch is live
# per-line state and not a session constant, changing within one transcript and back
# (tool-loop-cache -> HEAD -> tool-loop-cache), and the one session that detached its own
# checkout records gitBranch "HEAD" on the next line carrying one. It is kept apart from
# the "no branch" bucket, which means the field was absent altogether: a Claude Code
# version fact rather than a git one. On the same corpus this was 860 responses over four
# project folders, 1.5% of the window, while "no branch" was empty, since gitBranch is on
# every response.
DETACHED_VALUE = "HEAD"

# The share of the window's tokens that, left unpriced, withholds the dollar total. This
# is a display rule and not a detection cutoff: nothing judges or alerts on it, it decides
# only whether one figure prints. A rounding-error model should not silence a total and a
# real one should, so the test is against the unpriced models' summed share: five models
# at 0.9% each are 4.5% of the window counted as zero dollars, which is the partial total
# the rule exists to refuse. The same cutoff also decides each bucket's own figure, against
# that bucket's own tokens rather than the window's; see spend_rows for the measurement.
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
        return turns["is_sidechain"].astype(bool).map(THREAD_NAMES)
    if dimension == "project":
        # Read back the way report.py's project_lines does, so the same folder reads
        # the same in both commands rather than as its raw, dash-encoded form here.
        return turns["source_file"].astype(str).map(project_of).map(project_path)
    column = DIMENSION_COLUMNS[dimension]
    values = turns[column] if column in turns else pd.Series(None, index=turns.index, dtype="object")
    named = values.where(values.notna() & (values.astype(str) != ""), ABSENT_NAMES[dimension]).astype(str)
    return named.replace(DETACHED_VALUE, DETACHED_NAME) if dimension == "branch" else named


def branch_projects(turns: pd.DataFrame) -> dict[str, int]:
    """How many project folders each branch bucket drew on. A branch name is the only
    bucket key whose meaning is scoped to a project: two repositories both have a `main`,
    and one row holding both names two places at once. Measured over the owner's corpus on
    2026-09-22: 3 of 191 branch names spanned more than one repository and carried 18.0%
    of the window between them, `main` alone 16.6% over five repositories whose largest
    share was 11.7%. Counting is all ccdrift can honestly do here. It reads a project
    folder, which is a cwd, not a git repository root, so naming the split would take a
    repository identity it does not have."""
    if turns.empty:
        return {}
    frame = pd.DataFrame({"branch": buckets(turns, "branch").to_numpy(),
                          "project": buckets(turns, "project").to_numpy()})
    return {str(name): int(count) for name, count in frame.groupby("branch")["project"].nunique().items()}


# The counts Price.charge takes, in its order, and the one it takes by keyword.
CHARGED_COUNTS = ("input_tokens", "cache_creation", "cache_read", "output_tokens")
HOUR_WRITES = "cache_1h"

# The suffix cost-state puts on a model's 1M-context tier, a tier message.model never records.
LONG_CONTEXT = "[1m]"


def joined_prices(prices: dict[str, Price], recorded: Iterable[str]) -> dict[str, Price]:
    """`prices` keyed as responses name their model. A cost-state key is the model's own
    name, except that its 1M-context tier is keyed apart with LONG_CONTEXT, which
    message.model never carries. A model whose cost records name only that tier is joined to
    it: claude-opus-5-5, whose first record on 2026-09-24 was keyed claude-opus-5-5[1m] while
    all 3,001 of its responses said claude-opus-5-5, so no response could ever be priced. A
    model with a plain key in `recorded` (every model name the cost records carry) keeps the
    plain price, or none when its fit was refused: its 1M tier is a price of its own, and a
    response can't say which tier it ran on."""
    names = set(recorded)
    joined = dict(prices)
    for key, price in prices.items():
        plain = key[:-len(LONG_CONTEXT)]
        if key.endswith(LONG_CONTEXT) and plain not in names:
            joined[plain] = price
    return joined


def response_dollars(turns: pd.DataFrame, prices: dict[str, Price]) -> pd.Series:
    """What each response cost, NaN where its model has no price, so a bucket holding one
    unpriced response reports no dollars rather than a total that quietly omits it."""
    out = pd.Series(float("nan"), index=turns.index, dtype="float64")
    if turns.empty:
        return out
    models = turns["model"].astype(str)
    for model, price in prices.items():
        # The join is exact string equality on `prices` as joined_prices keys it. Measured
        # over the owner's corpus on 2026-09-22: every one of 158,246 responses carried a
        # message.model with a cost-state counterpart, dated aliases included, so any wider
        # normalising could only join a response to a price that is not its own. The one
        # exception arrived on 2026-09-24, a model keyed only by its 1M tier, and
        # joined_prices handles it. A model with both keys still prices its 1M-context
        # responses at the plain rate: message.model never records the tier, so that
        # understatement is real and is not measurable from the response side.
        rows = models == model
        if not rows.any():
            continue
        # The parser reads a tier Claude Code did not log as zero, so a response from before
        # the tiers were logged is charged at five minutes, as every write was billed before.
        out.loc[rows] = price.charge(*(turns.loc[rows, name].fillna(0) for name in CHARGED_COUNTS),
                                     cache_1h=turns.loc[rows, HOUR_WRITES].fillna(0))
    return out


def record_write_tiers(usage: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    """`usage` (History.model_usage) with `cache_1h`, how many of each cost record's cache
    writes were at the one-hour tier. A record logs only the total; its session's responses
    of the same model log the tiers, so the record takes the one-hour share of all their
    writes, counting any that logged no tier at five minutes, as response_dollars charges
    them. A record whose session wrote nothing for its model is taken to have written at five
    minutes, as every record was fitted before. Its 1M-tier key is its model's plain name to
    a response, as in joined_prices, so a session holding records under both keys gives both
    one pooled share: a record can't say which thread it ran on. On 2026-09-25 one session
    of the owner's did, and taking its 1M tier as the main thread's instead fitted worse."""
    if usage.empty:
        return usage.assign(cache_1h=pd.Series(dtype="float64"))
    tiers = (responses.assign(model=responses["model"].astype(str))
             .groupby(["session_id", "model"])[[HOUR_WRITES, "cache_creation"]].sum())
    share = (tiers[HOUR_WRITES] / tiers["cache_creation"]).rename("share")
    model = usage["model"].astype(str)
    plain = model.where(~model.str.endswith(LONG_CONTEXT), model.str[:-len(LONG_CONTEXT)])
    keys = pd.MultiIndex.from_arrays([usage["session_id"], plain])
    found = share.reindex(keys).fillna(0).to_numpy()
    return usage.assign(cache_1h=usage["cache_creation"].to_numpy(dtype=float) * found)


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


def _read_ratio(price: Price) -> Optional[float]:
    """What a price charges for a cache read as a share of its input rate, or None when that
    share isn't a discount. Every model priced so far reads the cache below its input rate,
    so a ratio above 1 means the fit's input rate collapsed toward zero by near-cancellation
    (5.4e-21 per token reads as reads costing 9e13 times input), not that reads cost more."""
    if price.input_rate <= 0:
        return None
    ratio = price.cache_read_rate / price.input_rate
    return round(ratio, 3) if ratio <= 1 else None


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
        # so a `dollars` of null can be read against the fit rather than guessed at. The
        # cache-read ratio is part of that evidence since models stopped sharing one:
        # claude-opus-5-5 reads at 0.05x its input rate where the models before it read at
        # 0.1x. A model whose records carry no input at all is refused as rank-deficient; an
        # input rate of exactly zero, which a fit reaches only by an exact cancellation and a
        # price built by hand can carry, has no ratio, and says so rather than crashing.
        "priced_models": [{"model": model, "residual": fitted[model].residual, "rows": fitted[model].rows,
                           "cache_read_ratio": _read_ratio(fitted[model])}
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
    "no price" on every line there would be noise rather than an explanation.

    A branch row also says how many project folders it drew on, when that is more than
    one. No other dimension does: see branch_projects for why the count means something
    there and nothing anywhere else."""
    rows = spend_rows(turns, dimension, prices)
    if rows.empty:
        return []
    explain = bool(priced_in_window(turns, prices))
    # Only the branch dimension: a project count beside `general-purpose` would say that
    # the reader works in more than one place, which is not what its row is about.
    pooled = branch_projects(turns) if dimension == "branch" else {}
    lines = ["", COST_LINES["by"].format(name=DIMENSION_NAMES[dimension])]
    for row in rows.itertuples(index=False):
        dollars = None if pd.isna(row.dollars) else float(row.dollars)
        lines.append(spend_line(row.bucket, int(row.responses), float(row.tokens), float(row.share), dollars,
                                row.unpriced if explain else (), pooled.get(row.bucket, 1)))
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
    usage = record_write_tiers(tables.model_usage, tables.responses)
    prices = joined_prices(fit_prices(usage), usage["model"].astype(str) if "model" in usage else ())
    dimensions = [by] if by else list(DEFAULT_ORDER)
    if as_json:
        # The default view's DEFAULT_ORDER never asks for project or branch, but the JSON
        # contract promises to name every dimension it withholds rather than saying nothing
        # about one nobody asked for: with no --by, the private dimensions still go in.
        print(spend_json(turns, dimensions if by else list(DIMENSIONS), prices, window), end="")
        return 0
    total = priced_total(turns, prices)
    money = "" if total is None else COST_LINES["money"].format(total=total)
    lines = [COST_LINES["window"].format(days=len(window), tokens=approx(total_tokens(turns)), money=money),
             COST_LINES["every"]]
    unpriced = unpriced_models(turns, prices)
    # Said once, at the top: which models have no price, and whether that was enough to
    # withhold the total. Missing money that says nothing reads as broken arithmetic.
    if unpriced and priced_in_window(turns, prices):
        lines.append(unpriced_line(unpriced, total is None, MATERIAL_SHARE))
    for dimension in dimensions:
        lines += spend_lines(turns, dimension, prices)
    print("\n".join(lines) + "\n", end="")
    return 0
