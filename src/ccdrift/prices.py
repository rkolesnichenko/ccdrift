"""Dollars per token, fitted from the cost records Claude Code writes itself.

ccdrift ships no price list. Claude Code records what a session cost, per model, and a
least-squares fit over those records recovers the rates, so nothing here goes stale when
a price changes and nothing reaches the network. The fit checks its own work: a correct
one reproduces the recorded costs exactly, so a model whose residual is large is one
whose cost has a component this expression does not model, and it is left unpriced
rather than guessed at."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Cache writes cost 1.25x input. It is the one ratio still fixed rather than fitted, and it
# is measured: on the owner's corpus on 2026-09-23 a free fit, solving for write and read
# rates beside input and output, recovers exactly 1.25x on every model whose free fit is
# well determined, claude-opus-4-7 at 6.25/5.00 per Mtok over 168 records, claude-sonnet-5
# at 2.50/2.00 over 10 and claude-haiku-4-5-20251001 at 1.25/1.00 over 197. Fixing it is
# what keeps the fit well posed: free, it returns -$11.58 per Mtok of input for
# claude-opus-5 over 6 records and -$15.05 for claude-opus-5[1m] over 24. A model that
# charges some other write ratio is not priced wrong by this; its residual crosses
# MAX_RESIDUAL and it is refused, as a wrong cache-read ratio was measured to be.
CACHE_WRITE_RATE = 1.25

# A fit must reproduce the costs it was fitted to this closely to be trusted. This is not a
# detection cutoff, and it is not fitted to a corpus's noise: Claude Code recorded the answer,
# so the bound asks whether every cost component has been modelled rather than whether a
# signal beat noise. The value sits between measurements taken on the owner's corpus on
# 2026-09-23. It admits the worst real fit with room, claude-opus-5[1m] at 0.70%, the five
# models priced spanning 0.00% to 0.70%. It rejects every known-broken ablation: cache reads
# fixed at 0.05x input, whose nearest miss is claude-sonnet-5 at 1.35%, and
# claude-haiku-4-5-20251001 with web search left out of the expression, at 7.01% against
# 0.00% with it in.
MAX_RESIDUAL = 0.01

# Records a model needs beyond its free parameters before its fit means anything. At none of
# them, three records against the three columns of a model that never searched are exactly
# determined: the fit is perfect by construction, its residual measures nothing, and the
# model would be priced on no evidence.
MIN_EXTRA_ROWS = 1


def input_billed_tokens(input_tokens: pd.Series, cache_creation: pd.Series) -> pd.Series:
    """The tokens of a response or a cost record charged at the input rate: its input, and
    its cache writes at CACHE_WRITE_RATE. Cache reads are not among them, since they carry a
    rate of their own. The one place this expression lives, so what the fit solves for and
    what Price.charge bills can never drift apart."""
    return input_tokens + CACHE_WRITE_RATE * cache_creation


@dataclass(frozen=True, kw_only=True)
class Price:
    """What one model costs, fitted from Claude Code's own records, with the evidence.
    Built by keyword only: five of its fields are floats of a similar size, and a positional
    call that shifted one into another's place would price every response wrong without an
    error."""
    input_rate: float        # per token, also charged on cache writes at CACHE_WRITE_RATE
    cache_read_rate: float   # per token read from the cache, fitted rather than a fixed share of input
    output_rate: float       # per token
    # Per request, 0.0 when the model never searched. Nothing charges a response with it:
    # web searches are billed per request and ccdrift keeps no per-response count of them.
    # It is fitted, and kept, because leaving the term out of the expression is what makes
    # claude-haiku-4-5-20251001 fit at 7.01% instead of 0.00% and lose its token rates with it.
    web_search_rate: float
    residual: float          # relative, against the costs this was fitted to
    rows: int

    def charge(self, input_tokens: pd.Series, cache_creation: pd.Series, cache_read: pd.Series,
               output_tokens: pd.Series) -> pd.Series:
        """What responses or records with these counts cost at this price: the one place a
        count becomes money. Web searches are left out, since a response carries no count of
        its own. The counts must have no gaps: a NaN count makes a NaN charge."""
        return (input_billed_tokens(input_tokens, cache_creation) * self.input_rate
                + cache_read * self.cache_read_rate + output_tokens * self.output_rate)


def fit_prices(usage: pd.DataFrame) -> dict[str, Price]:
    """A price per model, from the per-model cost records in `usage` (History.model_usage).
    A model with too few records, a rank-deficient fit, a negative rate or a residual over
    MAX_RESIDUAL is left out: an absent price reads as "not priced", a wrong one reads as
    money. A model whose records never read the cache is rank-deficient on that column and
    left out with them: its read rate cannot be told from nothing, and charging its reads at
    zero would be a wrong price rather than an absent one. So is one whose reads rest on a
    single record, which would set the read rate on no evidence at all."""
    if usage.empty or "cost_usd" not in usage:
        return {}
    prices: dict[str, Price] = {}
    for model, group in usage.groupby(usage["model"].astype(str), sort=True):
        rows = group.dropna(subset=["cost_usd"])
        # Cache reads get a rate of their own rather than a fixed share of the input rate.
        # They were fixed at 0.1x until claude-opus-5-5, which Claude Code 2.1.280 made the
        # default Opus model at $4.00 per Mtok in and $0.20 read, 0.05x. Measured on the
        # owner's corpus on 2026-09-23, fixing 0.05x in place of 0.1x put every priced model
        # over MAX_RESIDUAL, so no fixed ratio could price both; solved for, the ratio comes
        # back 0.100x on claude-opus-4-7, claude-sonnet-5 and claude-haiku-4-5-20251001.
        searches = rows["web_searches"].to_numpy(dtype=float)
        columns = [input_billed_tokens(rows["input_tokens"], rows["cache_creation"]).to_numpy(dtype=float),
                   rows["cache_read"].to_numpy(dtype=float), rows["output_tokens"].to_numpy(dtype=float)]
        # An all-zero column is rank-deficient, and most models never search.
        if searches.any():
            columns.append(searches)
        if len(rows) < len(columns) + MIN_EXTRA_ROWS:
            continue
        # The read rate needs as much evidence as the fit as a whole: a column that only one
        # record reads is exactly determined by that record, which prices whatever its cost
        # holds beyond its other counts as reads, at a residual of zero by construction. A
        # column no record reads is left to the rank check below, which refuses it.
        reading = int((rows["cache_read"] > 0).sum())
        if 0 < reading < 1 + MIN_EXTRA_ROWS:
            continue
        design = np.column_stack(columns)
        costs = rows["cost_usd"].to_numpy(dtype=float)
        if costs.sum() <= 0 or np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        fitted, *_ = np.linalg.lstsq(design, costs, rcond=None)
        if float(fitted.min()) < 0:
            continue
        residual = float(np.abs(design @ fitted - costs).sum() / costs.sum())
        if residual > MAX_RESIDUAL:
            continue
        prices[str(model)] = Price(input_rate=float(fitted[0]), cache_read_rate=float(fitted[1]),
                                   output_rate=float(fitted[2]),
                                   web_search_rate=float(fitted[3]) if len(columns) == 4 else 0.0,
                                   residual=residual, rows=len(rows))
    return prices
