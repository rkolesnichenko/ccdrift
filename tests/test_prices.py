"""Fitting dollars per token from the cost records Claude Code writes itself."""

import pandas as pd

from ccdrift.prices import MAX_RESIDUAL, billed_tokens, fit_prices


def usage(rows, model="claude-opus-5"):
    """Cost records as the store returns them: one dict per record, counts and cost."""
    base = {"input_tokens": 0.0, "output_tokens": 0.0, "cache_creation": 0.0, "cache_read": 0.0,
            "thinking_tokens": 0.0, "web_searches": 0.0, "cost_usd": 0.0, "model": model}
    return pd.DataFrame([{**base, **row} for row in rows])


def priced(rows, rate_in=5e-6, rate_out=25e-6, rate_web=0.0):
    """`rows` with each cost filled in at the given rates, as Claude Code would record."""
    out = []
    for row in rows:
        billed = row.get("input_tokens", 0) + 1.25 * row.get("cache_creation", 0) + 0.1 * row.get("cache_read", 0)
        out.append({**row, "cost_usd": billed * rate_in + row.get("output_tokens", 0) * rate_out
                    + row.get("web_searches", 0) * rate_web})
    return out


def test_a_models_rates_are_recovered_from_its_own_cost_records():
    rows = priced([{"input_tokens": 1000, "output_tokens": 50, "cache_creation": 200, "cache_read": 9000},
                   {"input_tokens": 3000, "output_tokens": 700, "cache_creation": 10, "cache_read": 100},
                   {"input_tokens": 50, "output_tokens": 5000, "cache_creation": 900, "cache_read": 40},
                   {"input_tokens": 7000, "output_tokens": 20, "cache_creation": 5, "cache_read": 60000}])
    price = fit_prices(usage(rows))["claude-opus-5"]
    assert round(price.input_rate * 1e6, 3) == 5.0
    assert round(price.output_rate * 1e6, 3) == 25.0
    assert price.residual < 1e-9


def test_a_model_billed_per_web_search_is_still_recovered_exactly():
    rows = priced([{"input_tokens": 1000, "output_tokens": 50, "web_searches": 3},
                   {"input_tokens": 3000, "output_tokens": 700, "web_searches": 0},
                   {"input_tokens": 50, "output_tokens": 5000, "web_searches": 11},
                   {"input_tokens": 7000, "output_tokens": 20, "web_searches": 1}],
                  rate_in=1e-6, rate_out=5e-6, rate_web=0.01)
    price = fit_prices(usage(rows, model="claude-haiku-4-5"))["claude-haiku-4-5"]
    assert round(price.input_rate * 1e6, 3) == 1.0 and round(price.web_search_rate, 4) == 0.01


def test_a_model_with_no_web_searches_is_still_fitted():
    # An all-zero web column makes the design rank-deficient if it is kept, which would
    # refuse every model that never searched: that is most of them.
    rows = priced([{"input_tokens": 1000, "output_tokens": 50}, {"input_tokens": 3000, "output_tokens": 700},
                   {"input_tokens": 50, "output_tokens": 5000}, {"input_tokens": 7000, "output_tokens": 20}])
    price = fit_prices(usage(rows))["claude-opus-5"]
    assert price.web_search_rate == 0.0 and price.residual < 1e-9


def test_a_model_with_too_few_records_is_not_priced():
    assert fit_prices(usage(priced([{"input_tokens": 1000, "output_tokens": 50}]))) == {}


def test_a_model_is_priced_one_record_past_its_free_parameters_and_not_at_them():
    # Two records against the two columns of a model that never searched are exactly
    # determined: the fit is perfect by construction, so its 0.00% residual is evidence of
    # nothing. MIN_EXTRA_ROWS is the whole of what refuses it, and the third record is
    # what makes the residual an observation rather than an identity.
    rows = priced([{"input_tokens": 1000, "output_tokens": 50}, {"input_tokens": 3000, "output_tokens": 700},
                   {"input_tokens": 50, "output_tokens": 5000}])
    assert fit_prices(usage(rows[:2])) == {}
    assert round(fit_prices(usage(rows))["claude-opus-5"].input_rate * 1e6, 3) == 5.0


def test_a_model_whose_costs_do_not_add_up_is_not_priced():
    rows = priced([{"input_tokens": 1000, "output_tokens": 50}, {"input_tokens": 3000, "output_tokens": 700},
                   {"input_tokens": 50, "output_tokens": 5000}, {"input_tokens": 7000, "output_tokens": 20}])
    rows[0]["cost_usd"] *= 3          # one record Claude Code priced by another rule
    assert fit_prices(usage(rows)) == {}


def test_a_fit_that_wants_a_negative_price_is_refused():
    # The real failure this guard exists for: on the owner's corpus a free four-parameter
    # fit returned -$10.09 per Mtok of input for claude-opus-5.
    # Costs that fall as input rises, so the fitted input rate is negative while the
    # total stays positive: the negative-rate guard must catch it, not the sum guard.
    rows = [{"input_tokens": 1000, "output_tokens": 100, "cost_usd": 0.10},
            {"input_tokens": 2000, "output_tokens": 100, "cost_usd": 0.09},
            {"input_tokens": 3000, "output_tokens": 100, "cost_usd": 0.08},
            {"input_tokens": 4000, "output_tokens": 100, "cost_usd": 0.07}]
    assert sum(r["cost_usd"] for r in rows) > 0
    assert fit_prices(usage(rows)) == {}


def test_a_record_with_no_cost_drops_from_the_fit_rather_than_the_model():
    rows = priced([{"input_tokens": 1000, "output_tokens": 50}, {"input_tokens": 3000, "output_tokens": 700},
                   {"input_tokens": 50, "output_tokens": 5000}, {"input_tokens": 7000, "output_tokens": 20}])
    rows.append({"input_tokens": 999, "output_tokens": 9, "cost_usd": None})
    price = fit_prices(usage(rows))["claude-opus-5"]
    assert price.rows == 4 and round(price.input_rate * 1e6, 3) == 5.0


def test_a_fit_reports_how_far_off_it_was():
    rows = priced([{"input_tokens": 1000, "output_tokens": 50}, {"input_tokens": 3000, "output_tokens": 700},
                   {"input_tokens": 50, "output_tokens": 5000}, {"input_tokens": 7000, "output_tokens": 20}])
    assert fit_prices(usage(rows))["claude-opus-5"].residual <= MAX_RESIDUAL


def test_a_cache_write_is_billed_more_than_an_input_token_and_a_cache_read_less():
    # The formula the fit solves for and the breakdown charges a response with, in the one
    # place it lives: 1000 input + 1.25 * 400 written + 0.1 * 10000 read = 2500 billed.
    billed = billed_tokens(pd.Series([1000.0]), pd.Series([400.0]), pd.Series([10000.0]))
    assert billed.tolist() == [2500.0]


def test_an_empty_history_prices_nothing():
    assert fit_prices(pd.DataFrame()) == {}
