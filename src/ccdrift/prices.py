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

# Cache writes cost 1.25x input and cache reads 0.1x. Those ratios are not assumed: they
# fall out of the two models whose free four-parameter fit is exact on the owner's corpus,
# claude-opus-4-7 at 5.00/25.00/6.25/0.50 per Mtok over 99 records and claude-sonnet-5 at
# 2.00/10.00/2.50/0.20 over 12. Constraining to them is what makes the fit well posed:
# free, it returns -$10.09 per Mtok of input for claude-opus-5 over its 8 records.
CACHE_WRITE_RATE = 1.25
CACHE_READ_RATE = 0.1

# A fit must reproduce the costs it was fitted to this closely to be trusted. This is not
# a detection cutoff and needs no corpus to measure: Claude Code already recorded the
# answer, so the bound asks whether every cost component has been modelled. The worked
# example is claude-haiku-4-5, which fits at 4.72% with web search left out of the
# expression and at 0.00% with it in. On the owner's corpus the five models measured sit
# between 0.00% and 0.47%.
MAX_RESIDUAL = 0.01

# Records a model needs beyond its free parameters before its fit means anything.
MIN_EXTRA_ROWS = 1


@dataclass(frozen=True)
class Price:
    """What one model costs, fitted from Claude Code's own records, with the evidence."""
    input_rate: float        # per token, also charged on cache writes and reads at the ratios above
    output_rate: float       # per token
    web_search_rate: float   # per request, 0.0 when the model never searched
    residual: float          # relative, against the costs this was fitted to
    rows: int

    def dollars(self, input_tokens: float, output_tokens: float,
                cache_creation: float, cache_read: float) -> float:
        """What a response's tokens cost. Web searches are not counted: they are billed
        per request and ccdrift does not keep a per-response count of them. They are
        fitted only so the token rates come out right."""
        billed = input_tokens + CACHE_WRITE_RATE * cache_creation + CACHE_READ_RATE * cache_read
        return billed * self.input_rate + output_tokens * self.output_rate


def fit_prices(usage: pd.DataFrame) -> dict[str, Price]:
    """A price per model, from the per-model cost records in `usage` (History.model_usage).
    A model with too few records, a rank-deficient fit, a negative rate or a residual over
    MAX_RESIDUAL is left out: an absent price reads as "not priced", a wrong one reads as
    money."""
    if usage.empty or "cost_usd" not in usage:
        return {}
    prices: dict[str, Price] = {}
    for model, group in usage.groupby(usage["model"].astype(str), sort=True):
        rows = group.dropna(subset=["cost_usd"])
        billed = (rows["input_tokens"] + CACHE_WRITE_RATE * rows["cache_creation"]
                  + CACHE_READ_RATE * rows["cache_read"]).to_numpy(dtype=float)
        searches = rows["web_searches"].to_numpy(dtype=float)
        columns = [billed, rows["output_tokens"].to_numpy(dtype=float)]
        # An all-zero column is rank-deficient, and most models never search.
        if searches.any():
            columns.append(searches)
        if len(rows) < len(columns) + MIN_EXTRA_ROWS:
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
        prices[str(model)] = Price(input_rate=float(fitted[0]), output_rate=float(fitted[1]),
                                   web_search_rate=float(fitted[2]) if len(columns) == 3 else 0.0,
                                   residual=residual, rows=len(rows))
    return prices
