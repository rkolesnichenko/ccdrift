"""Fitting dollars per token from the cost records Claude Code writes itself."""

import pandas as pd
import pytest

from ccdrift.prices import MAX_RESIDUAL, Price, fit_prices, input_billed_tokens

# Counts of five cost records, varied so that input, cache reads and output are independent
# columns. Every record reads the cache: a model whose records never do is refused by the
# rank check, which would make a test about some other refusal pass for the wrong reason.
RECORDS = [{"input_tokens": 1000, "output_tokens": 50, "cache_creation": 200, "cache_read": 9000},
           {"input_tokens": 3000, "output_tokens": 700, "cache_creation": 10, "cache_read": 100},
           {"input_tokens": 50, "output_tokens": 5000, "cache_creation": 900, "cache_read": 40},
           {"input_tokens": 7000, "output_tokens": 20, "cache_creation": 5, "cache_read": 60000},
           {"input_tokens": 400, "output_tokens": 300, "cache_creation": 60, "cache_read": 2500}]


def usage(rows, model="claude-opus-5"):
    """Cost records as the store returns them: one dict per record, counts and cost."""
    base = {"input_tokens": 0.0, "output_tokens": 0.0, "cache_creation": 0.0, "cache_read": 0.0,
            "thinking_tokens": 0.0, "web_searches": 0.0, "cost_usd": 0.0, "model": model}
    return pd.DataFrame([{**base, **row} for row in rows])


def priced(rows, rate_in=5e-6, rate_out=25e-6, read_ratio=0.1, rate_web=0.0):
    """`rows` with each cost filled in at the given rates, as Claude Code would record:
    cache writes at 1.25x input and cache reads at `read_ratio` of it."""
    out = []
    for row in rows:
        cost = ((row.get("input_tokens", 0) + 1.25 * row.get("cache_creation", 0)) * rate_in
                + row.get("cache_read", 0) * rate_in * read_ratio
                + row.get("output_tokens", 0) * rate_out + row.get("web_searches", 0) * rate_web)
        out.append({**row, "cost_usd": cost})
    return out


def test_a_models_rates_are_recovered_from_its_own_cost_records():
    price = fit_prices(usage(priced(RECORDS[:4])))["claude-opus-5"]
    assert round(price.input_rate * 1e6, 3) == 5.0
    assert round(price.cache_read_rate * 1e6, 3) == 0.5
    assert round(price.output_rate * 1e6, 3) == 25.0
    assert price.residual < 1e-9


def test_a_model_that_reads_the_cache_at_a_twentieth_of_its_input_rate_is_recovered_exactly():
    # claude-opus-5-5 at Claude Code 2.1.280's published $4 in, $20 out and $0.20 per Mtok
    # read: 0.05x where every earlier model charged 0.1x. With the ratio fixed at 0.1x this
    # model could never be priced, its residual crossing MAX_RESIDUAL on every fit.
    rows = priced(RECORDS[:4], rate_in=4e-6, rate_out=20e-6, read_ratio=0.05)
    price = fit_prices(usage(rows, model="claude-opus-5-5"))["claude-opus-5-5"]
    assert round(price.input_rate * 1e6, 3) == 4.0
    assert round(price.cache_read_rate * 1e6, 3) == 0.2
    assert round(price.output_rate * 1e6, 3) == 20.0
    assert price.residual < 1e-9


def test_a_model_billed_per_web_search_is_still_recovered_exactly():
    rows = priced([{**row, "web_searches": searches} for row, searches in zip(RECORDS, (3, 0, 11, 1, 0))],
                  rate_in=1e-6, rate_out=5e-6, rate_web=0.01)
    price = fit_prices(usage(rows, model="claude-haiku-4-5"))["claude-haiku-4-5"]
    assert round(price.input_rate * 1e6, 3) == 1.0 and round(price.cache_read_rate * 1e6, 3) == 0.1
    assert round(price.web_search_rate, 4) == 0.01
    # A model that searched fits four columns now, so it needs a fifth record where 0.12.1
    # priced it at four. The four here are independent: the row bar refuses them, not rank.
    assert fit_prices(usage(rows[:4], model="claude-haiku-4-5")) == {}


def test_a_model_with_no_web_searches_is_still_fitted():
    # An all-zero web column makes the design rank-deficient if it is kept, which would
    # refuse every model that never searched: that is most of them.
    price = fit_prices(usage(priced(RECORDS[:4])))["claude-opus-5"]
    assert price.web_search_rate == 0.0 and price.residual < 1e-9


def test_a_model_with_too_few_records_is_not_priced():
    assert fit_prices(usage(priced(RECORDS[:1]))) == {}


def test_a_model_is_priced_one_record_past_its_free_parameters_and_not_at_them():
    # Three records against the three columns of a model that never searched are exactly
    # determined: the fit is perfect by construction, so its 0.00% residual is evidence of
    # nothing. MIN_EXTRA_ROWS is the whole of what refuses it, and the fourth record is what
    # makes the residual an observation rather than an identity. The three are independent,
    # so the rank check passes them and this refusal is MIN_EXTRA_ROWS's alone. A model with
    # three records was priced by 0.12.1, whose fit had two columns.
    rows = priced(RECORDS[:4])
    assert fit_prices(usage(rows[:3])) == {}
    assert round(fit_prices(usage(rows))["claude-opus-5"].input_rate * 1e6, 3) == 5.0


def test_a_model_whose_costs_do_not_add_up_is_not_priced():
    rows = priced(RECORDS)
    rows[0]["cost_usd"] *= 3          # one record Claude Code priced by another rule
    assert fit_prices(usage(rows)) == {}


def test_a_fit_that_wants_a_negative_price_is_refused():
    # The real failure this guard exists for: on the owner's corpus a free fit returned
    # -$11.58 per Mtok of input for claude-opus-5. Costs generated at a negative input rate
    # stay positive, because output outweighs it on every record, so this refusal is the
    # negative-rate guard's and not the one on a total that is not positive.
    rows = priced(RECORDS[:4], rate_in=-1e-6, read_ratio=-0.5)
    assert all(row["cost_usd"] > 0 for row in rows)
    assert fit_prices(usage(rows)) == {}


def test_a_fit_that_wants_a_negative_cache_read_price_is_refused():
    # The guard covers the column this fit gained: nothing charges for reading the cache in
    # reverse. Input and output rates are ordinary here; only the read rate is negative.
    rows = priced(RECORDS[:4], read_ratio=-0.02)
    assert all(row["cost_usd"] > 0 for row in rows)
    assert fit_prices(usage(rows)) == {}


def test_a_model_whose_records_never_read_the_cache_is_not_priced():
    # Its read rate cannot be told from nothing, and charging its reads at zero would be a
    # wrong price rather than an absent one. 0.12.1 priced it, fixing reads at 0.1x input.
    rows = priced([{**row, "cache_read": 0} for row in RECORDS])
    assert fit_prices(usage(rows)) == {}


def test_a_record_with_no_cost_drops_from_the_fit_rather_than_the_model():
    rows = priced(RECORDS[:4])
    rows.append({"input_tokens": 999, "output_tokens": 9, "cache_read": 99, "cost_usd": None})
    price = fit_prices(usage(rows))["claude-opus-5"]
    assert price.rows == 4 and round(price.input_rate * 1e6, 3) == 5.0


def test_a_record_that_left_its_cache_read_count_out_leaves_the_model_unpriced_not_mispriced():
    # The parser reads a count Claude Code left out of a record as zero before it is stored
    # (logs._num). When the count was not really zero, the record's cost still includes what
    # it stood for, the fit cannot explain it, and the model is refused rather than priced on
    # reads it never saw. The rates the fit wants here are all positive, so this refusal is
    # the residual bound's, at 2.25%.
    rows = priced(RECORDS)
    rows[0]["cache_read"] = 0                  # 9,000 reads left out, all of them in its cost
    assert fit_prices(usage(rows)) == {}


def test_a_fit_reports_how_far_off_it_was():
    assert fit_prices(usage(priced(RECORDS)))["claude-opus-5"].residual <= MAX_RESIDUAL


def test_a_cache_write_is_billed_at_its_ratio_and_a_cache_read_at_the_models_own_rate():
    # 1000 input + 1.25 * 400 written = 1500 input-priced tokens. The read rate here is
    # 0.05x input, not 0.1x, so a charge that fell back to a shared ratio would bill the
    # 10,000 reads at $0.005 rather than $0.0025 and come to $0.015.
    assert input_billed_tokens(pd.Series([1000.0]), pd.Series([400.0])).tolist() == [1500.0]
    price = Price(input_rate=5e-6, cache_read_rate=0.25e-6, output_rate=25e-6, web_search_rate=0.0,
                  residual=0.0, rows=4)
    charged = price.charge(pd.Series([1000.0]), pd.Series([400.0]), pd.Series([10000.0]), pd.Series([100.0]))
    assert charged.tolist() == [pytest.approx(0.0125)]


def test_charging_a_record_at_its_own_fitted_price_reproduces_what_claude_code_recorded():
    # The fit and the charge are two halves of one formula; this is what makes them one.
    rows = priced(RECORDS, rate_in=4e-6, rate_out=20e-6, read_ratio=0.05)
    records = usage(rows, model="claude-opus-5-5")
    price = fit_prices(records)["claude-opus-5-5"]
    charged = price.charge(records["input_tokens"], records["cache_creation"], records["cache_read"],
                           records["output_tokens"])
    assert charged.tolist() == pytest.approx(records["cost_usd"].tolist())


def test_a_price_is_built_by_keyword_and_never_by_position():
    # Five of its fields are floats of a similar size: a positional call that shifted one
    # into another's place would price every response wrong and say nothing.
    with pytest.raises(TypeError):
        Price(5e-6, 0.5e-6, 25e-6, 0.0, 0.0, 4)


def test_an_empty_history_prices_nothing():
    assert fit_prices(pd.DataFrame()) == {}
