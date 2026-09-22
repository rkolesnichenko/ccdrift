"""Partitioning the history by where its tokens went."""

import json
from datetime import date

import pandas as pd
import pytest

from ccdrift.history import load_history
from ccdrift.logs import parse_source
from ccdrift.prices import Price
from ccdrift.spend import (DIMENSIONS, MATERIAL_SHARE, priced_total, run_spend, spend_json, spend_rows,
                           spend_turns, total_tokens)
from ccdrift.state import new_state, save_state
from ccdrift.texts import DIMENSION_NAMES
from tests.helpers import at, cost_state, line, text, write

TODAY = date(2026, 9, 10)


def corpus(tmp_path):
    """Four responses: two main thread, two in subagents, with attribution that overlaps
    across dimensions the way Claude Code's does. a2 carries a different model from the
    other three, so the model dimension genuinely splits and the subagent thread bucket
    holds two models, one priced and one not."""
    write(tmp_path / "proj-a" / "s1.jsonl", [
        line("m1", text(40), ts=at(0), entrypoint="cli", branch="main"),
        line("m2", text(40), ts=at(60), entrypoint="cli", branch="main",
             skill="superpowers:writing-plans", plugin="superpowers"),
    ])
    write(tmp_path / "proj-b" / "s2.jsonl", [
        line("a1", text(40), ts=at(120), entrypoint="cli", sidechain=True, agent_type="general-purpose",
             branch="topic"),
        line("a2", text(40), ts=at(180), entrypoint="cli", sidechain=True, agent_type="Explore",
             branch="topic", mcp_server="context7", model="claude-haiku-4-5"),
    ])
    return spend_turns(parse_source(tmp_path), TODAY)


# What the cost records in money_corpus say each model costs, per token in and out. Round
# numbers, $5.00/$25.00 and $1.00/$5.00 per Mtok, so every dollar figure below can be
# checked by hand. What the owner's corpus actually fitted is in docs/findings.md.
RATES = {"claude-opus-5": (5e-6, 25e-6), "claude-haiku-4-5": (1e-6, 5e-6)}


def cost_record(ts, start, counts):
    """One cost-state record with every model's costUSD worked out at RATES, the way
    Claude Code records what a session actually cost."""
    usage = {}
    for model, (rate_in, rate_out) in RATES.items():
        one = counts[model]
        billed = one["input"] + 1.25 * one.get("cache_creation", 0) + 0.1 * one.get("cache_read", 0)
        usage[model] = {**one, "costUSD": billed * rate_in + one["output"] * rate_out}
    return cost_state(ts, usage, start=start)


def money_corpus(tmp_path, unpriced_out=None):
    """Two responses whose dollars are known exactly, beside the cost records the price
    fit has to recover RATES from: four of them, one more than the two free parameters,
    with counts that keep the two columns independent. `unpriced_out` adds a subagent
    response on a model no cost record mentions, so the fit refuses it.

    claude-opus-5 on the main thread: 10 input + 1,000,000 output = $25.00005.
    claude-haiku-4-5 in a subagent: 10 input + 800,000 written + 2,000,000 output = $11.00001."""
    responses = [line("m1", text(40), ts=at(0), entrypoint="cli", out=1_000_000),
                 line("a1", text(40), ts=at(60), entrypoint="cli", sidechain=True,
                      model="claude-haiku-4-5", out=2_000_000, cache_creation=800_000)]
    if unpriced_out is not None:
        responses.append(line("a2", text(40), ts=at(120), entrypoint="cli", sidechain=True,
                              model="claude-fable-5-1", out=unpriced_out))
    counts = [{"claude-opus-5": {"input": 1_000, "output": 50, "cache_creation": 200, "cache_read": 9_000},
               "claude-haiku-4-5": {"input": 40, "output": 900}},
              {"claude-opus-5": {"input": 3_000, "output": 700},
               "claude-haiku-4-5": {"input": 5_000, "output": 60, "cache_read": 70_000}},
              {"claude-opus-5": {"input": 50, "output": 5_000, "cache_creation": 900},
               "claude-haiku-4-5": {"input": 20, "output": 8_000}},
              {"claude-opus-5": {"input": 7_000, "output": 20, "cache_read": 60_000},
               "claude-haiku-4-5": {"input": 900, "output": 30, "cache_creation": 400}}]
    write(tmp_path / "proj-a" / "s1.jsonl",
          responses + [cost_record(at(600 * i), i + 1, one) for i, one in enumerate(counts)])


def five_small_models(tmp_path):
    """One model carrying 98.5% of the window and five carrying 0.3% each: not one of the
    five is material on its own, 1.5% of the window between them."""
    records = [line("m0", text(40), ts=at(0), entrypoint="cli", out=36_107)]
    records += [line(f"m{i}", text(40), ts=at(60 * i), entrypoint="cli", model=f"claude-tiny-{i}")
                for i in range(1, 6)]
    write(tmp_path / "p" / "s1.jsonl", records)
    return spend_turns(parse_source(tmp_path / "p"), TODAY)


OPUS = {"claude-opus-5": Price(5e-6, 25e-6, 0.0, 0.0, 9)}


def test_every_dimension_has_a_heading_and_every_heading_a_dimension():
    # Two parallel structures: --by takes its choices from DIMENSION_NAMES and spend_lines
    # reads a heading out of it per DIMENSIONS, so a name missing from either is a
    # KeyError or a dimension no one can ask for.
    assert set(DIMENSIONS) == set(DIMENSION_NAMES)


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_every_dimension_accounts_for_all_the_tokens(tmp_path, dimension):
    turns = corpus(tmp_path)
    rows = spend_rows(turns, dimension, {})
    assert rows["tokens"].sum() == pytest.approx(total_tokens(turns))
    assert rows["share"].sum() == pytest.approx(1.0)


def test_a_response_carrying_a_plugin_and_a_skill_is_counted_once_in_each(tmp_path):
    turns = corpus(tmp_path)
    skills = spend_rows(turns, "skill", {})
    plugins = spend_rows(turns, "plugin", {})
    assert skills.loc[skills["bucket"] == "superpowers:writing-plans", "responses"].iloc[0] == 1
    assert plugins.loc[plugins["bucket"] == "superpowers", "responses"].iloc[0] == 1


def test_responses_a_dimension_does_not_name_get_their_own_bucket(tmp_path):
    rows = spend_rows(corpus(tmp_path), "skill", {})
    assert rows.loc[rows["bucket"] == "no skill", "responses"].iloc[0] == 3


def test_the_thread_dimension_splits_main_from_subagents(tmp_path):
    rows = spend_rows(corpus(tmp_path), "thread", {})
    assert dict(zip(rows["bucket"], rows["responses"])) == {"main thread": 2, "subagent": 2}


def test_the_project_dimension_reads_back_as_a_path_not_its_raw_encoding(tmp_path):
    rows = spend_rows(corpus(tmp_path), "project", {})
    assert set(rows["bucket"]) == {"/proj/a", "/proj/b"}
    assert "proj-a" not in rows["bucket"].tolist() and "proj-b" not in rows["bucket"].tolist()


def test_a_source_holding_no_project_folder_is_one_project_and_not_one_per_session(tmp_path):
    # --source pointed at one project's own folder leaves every transcript directly under
    # it, with nothing above it to read back as a path. Split per transcript, each bucket
    # would be a session id with its dashes turned into slashes, and ccdrift prints no
    # session ids anywhere: `peek` prints an id as its length alone.
    write(tmp_path / "p" / "0199c3d0-1111-2222-3333-444455556666.jsonl",
          [line("m1", text(40), ts=at(0), entrypoint="cli", sid="0199c3d0-1111-2222-3333-444455556666")])
    write(tmp_path / "p" / "0199c3d0-aaaa-bbbb-cccc-ddddeeeeffff.jsonl",
          [line("m2", text(40), ts=at(60), entrypoint="cli", sid="0199c3d0-aaaa-bbbb-cccc-ddddeeeeffff")])
    rows = spend_rows(spend_turns(parse_source(tmp_path / "p"), TODAY), "project", {})
    assert rows["bucket"].tolist() == ["the source folder"]
    assert "0199c3d0" not in "".join(rows["bucket"])


def test_buckets_come_out_largest_first_with_ties_broken_by_name(tmp_path):
    rows = spend_rows(corpus(tmp_path), "branch", {})
    tokens = rows["tokens"].tolist()
    assert tokens == sorted(tokens, reverse=True)
    pd.testing.assert_frame_equal(rows, spend_rows(corpus(tmp_path), "branch", {}))


def test_the_model_dimension_splits_two_models_into_two_buckets(tmp_path):
    rows = spend_rows(corpus(tmp_path), "model", {})
    assert dict(zip(rows["bucket"], rows["responses"])) == {"claude-opus-5": 3, "claude-haiku-4-5": 1}


def test_a_price_joins_its_model_by_the_whole_name_and_never_by_a_prefix_of_it(tmp_path):
    # Claude Code keys its cost records by a name message.model does not always repeat:
    # claude-opus-5[1m] is priced apart from claude-opus-5 and no response ever says so.
    # Joining on anything looser than the whole name would charge the 1m-context tier at
    # the plain rate and call it measured, which is the one thing this module will not do.
    write(tmp_path / "p" / "s1.jsonl", [
        line("m1", text(40), ts=at(0), entrypoint="cli"),
        line("m2", text(40), ts=at(60), entrypoint="cli", model="claude-opus-5[1m]"),
    ])
    rows = spend_rows(spend_turns(parse_source(tmp_path / "p"), TODAY), "model", OPUS)
    assert dict(zip(rows["bucket"], rows["dollars"].notna())) == {"claude-opus-5": True, "claude-opus-5[1m]": False}


def test_a_bucket_is_priced_only_when_every_model_in_it_is(tmp_path):
    turns = corpus(tmp_path)
    # The subagent bucket holds both claude-opus-5 (a1) and claude-haiku-4-5 (a2); pricing
    # only the former must leave the whole bucket unpriced, not just a1's own dollars.
    opus_only = {"claude-opus-5": Price(5e-6, 25e-6, 0.0, 0.0, 9)}
    rows = spend_rows(turns, "thread", opus_only)
    assert pd.notna(rows.loc[rows["bucket"] == "main thread", "dollars"].iloc[0])
    assert pd.isna(rows.loc[rows["bucket"] == "subagent", "dollars"].iloc[0])

    both_priced = {**opus_only, "claude-haiku-4-5": Price(1e-6, 5e-6, 0.0, 0.0, 9)}
    assert spend_rows(turns, "thread", both_priced)["dollars"].notna().all()
    assert spend_rows(turns, "thread", {})["dollars"].isna().all()


def test_the_total_is_withheld_when_a_material_model_is_unpriced_and_present_when_all_are(tmp_path):
    turns = corpus(tmp_path)
    by_model = spend_rows(turns, "model", OPUS)
    haiku = by_model.loc[by_model["bucket"] == "claude-haiku-4-5", "share"].iloc[0]
    assert haiku >= MATERIAL_SHARE                            # a quarter of the window, not a rounding error
    assert priced_total(turns, OPUS) is None
    both = {**OPUS, "claude-haiku-4-5": Price(1e-6, 5e-6, 0.0, 0.0, 9)}
    # 3 opus responses of 10 input and 100 output, one haiku response of the same.
    assert priced_total(turns, both) == pytest.approx(3 * (10 * 5e-6 + 100 * 25e-6) + (10 * 1e-6 + 100 * 5e-6))


def test_unpriced_models_withhold_the_total_once_they_add_up_though_none_is_material_alone(tmp_path):
    turns = five_small_models(tmp_path)
    rows = spend_rows(turns, "model", OPUS)
    small = rows[rows["bucket"] != "claude-opus-5"]["share"]
    assert len(small) == 5 and (small < MATERIAL_SHARE).all()  # each one passes the per-model test
    assert small.sum() >= MATERIAL_SHARE                       # and 1.5% of the window would be counted as $0
    assert priced_total(turns, OPUS) is None


def test_the_day_still_in_progress_is_left_out(tmp_path):
    write(tmp_path / "p" / "s1.jsonl", [line("m1", text(40), ts=at(0), entrypoint="cli")])
    assert spend_turns(parse_source(tmp_path / "p"), date(2026, 9, 1)).empty


def test_agent_sdk_sessions_are_left_out(tmp_path):
    write(tmp_path / "p" / "s1.jsonl", [line("m1", text(40), ts=at(0), entrypoint="sdk-py")])
    assert spend_turns(parse_source(tmp_path / "p"), TODAY).empty


def test_the_default_view_leads_with_thread_then_agent(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    corpus(tmp_path / "logs")
    assert run_spend(tmp_path / "logs", state, today=TODAY) == 0
    out = capsys.readouterr().out
    assert out.index("By thread") < out.index("By agent")


def test_a_named_dimension_shows_only_that_one(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    corpus(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, by="skill", today=TODAY)
    out = capsys.readouterr().out
    assert "By skill" in out and "By agent" not in out


def bucket_line(out, bucket):
    """The printed line for one bucket, found by its name at the start."""
    return next(row for row in out.splitlines() if row.strip().startswith(bucket + " "))


def test_the_dollars_printed_are_the_ones_claude_codes_own_cost_records_imply(tmp_path, capsys):
    # The whole money path in one run, with nothing handing the command a price:
    # cost-state records on disk, rows in the history store, a fitted rate per model,
    # each response's own tokens charged at it, a dollar figure printed. The store is
    # claimed first so `cost` reads its cost records back out of it rather than parsing
    # the transcripts again, which is the leg of the chain no other test covers.
    state = tmp_path / "state.json"
    save_state(state, new_state())
    money_corpus(tmp_path / "logs")
    load_history(tmp_path / "logs", state, claim=True)
    assert run_spend(tmp_path / "logs", state, today=TODAY) == 0
    out = capsys.readouterr().out
    assert "$36.00" in out.splitlines()[0]                     # 25.00005 + 11.00001
    assert "$25.00" in bucket_line(out, "main thread") and "$11.00" in bucket_line(out, "subagent")
    assert "$25.00" in bucket_line(out, "claude-opus-5") and "$11.00" in bucket_line(out, "claude-haiku-4-5")

    run_spend(tmp_path / "logs", state, as_json=True, today=TODAY)
    payload = json.loads(capsys.readouterr().out)
    assert payload["dollars"] == pytest.approx(25.00005 + 11.00001)
    dollars = {row["bucket"]: row["dollars"] for row in payload["dimensions"]["model"]}
    assert dollars == {"claude-opus-5": pytest.approx(25.00005), "claude-haiku-4-5": pytest.approx(11.00001)}


def test_a_bucket_with_no_dollars_names_the_model_that_left_it_without_any(tmp_path, capsys):
    # The failure this exists for, from the owner's corpus: one model of 0.0% of the
    # window is refused by the fit, the total still prints, and the whole subagent bucket
    # goes blank beside it. Unexplained, that reads as broken arithmetic.
    state = tmp_path / "state.json"
    save_state(state, new_state())
    money_corpus(tmp_path / "logs", unpriced_out=100)
    run_spend(tmp_path / "logs", state, today=TODAY)
    out = capsys.readouterr().out
    assert "$36.00" in out.splitlines()[0]
    assert out.splitlines()[2].startswith("No price for claude-fable-5-1, 0.0% of the window's tokens:")
    assert "no price: claude-fable-5-1" in bucket_line(out, "subagent")
    assert "$25.00" in bucket_line(out, "main thread")
    # The bucket that is the model says it once: repeating its own name explains nothing.
    assert bucket_line(out, "claude-fable-5-1").endswith("no price")


def test_a_withheld_total_says_which_model_withheld_it(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    money_corpus(tmp_path / "logs", unpriced_out=1_000_000)
    run_spend(tmp_path / "logs", state, today=TODAY)
    out = capsys.readouterr().out
    assert "$" not in out.splitlines()[0]
    assert out.splitlines()[2].startswith("No total: no price for claude-fable-5-1, 20.8% of the window's tokens.")


def test_a_history_with_no_cost_records_reports_tokens_and_no_dollars(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    corpus(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, today=TODAY)
    out = capsys.readouterr().out
    assert "tokens" in out and "$" not in out
    # And says nothing about prices: a new install has no cost record, which is normal
    # rather than a failure, so "no price" on every line would be noise, not a reason.
    assert "no price" not in out


def test_a_source_with_no_transcripts_says_so(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    (tmp_path / "empty").mkdir()
    assert run_spend(tmp_path / "empty", state, today=TODAY) == 2


def test_the_json_holds_every_dimension_but_the_ones_that_name_your_folders(tmp_path):
    # Asked for all eight dimensions, including the two private ones, so the absence
    # of "project" and "branch" below is a claim the filter has to earn rather than a
    # fact about DEFAULT_ORDER, which never carries either one regardless of filtering.
    turns = corpus(tmp_path)
    window = sorted(turns["day"].astype(str).unique())
    payload = json.loads(spend_json(turns, list(DIMENSIONS), {}, window))
    assert "thread" in payload["dimensions"] and "skill" in payload["dimensions"]
    assert "project" not in payload["dimensions"] and "branch" not in payload["dimensions"]
    assert set(payload["withheld"]) == {"project", "branch"}


def test_the_json_names_no_project_and_no_branch_even_when_asked_for_one(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    corpus(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, by="branch", as_json=True, today=TODAY)
    out = capsys.readouterr().out
    assert "topic" not in out and "proj-a" not in out
    assert json.loads(out)["withheld"] == ["branch"]


def test_the_json_names_no_project_even_when_asked_for_one(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    corpus(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, by="project", as_json=True, today=TODAY)
    out = capsys.readouterr().out
    # The project bucket renders as a real filesystem path (see
    # test_the_project_dimension_reads_back_as_a_path_not_its_raw_encoding), so the
    # withheld check must be against that rendered form, not the raw folder name.
    assert "/proj/a" not in out and "/proj/b" not in out
    assert json.loads(out)["withheld"] == ["project"]


def test_the_json_says_what_it_could_not_price(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    corpus(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, as_json=True, today=TODAY)
    payload = json.loads(capsys.readouterr().out)
    assert payload["dollars"] is None and payload["priced_models"] == []


def test_the_json_carries_each_prices_fit_quality_and_names_what_it_could_not_price(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    money_corpus(tmp_path / "logs", unpriced_out=1_000_000)
    run_spend(tmp_path / "logs", state, as_json=True, today=TODAY)
    payload = json.loads(capsys.readouterr().out)
    assert payload["dollars"] is None
    # Four cost records a model, reproducing their costs exactly: the evidence behind
    # every dollar figure in the same document as the figures.
    assert payload["priced_models"] == [{"model": "claude-haiku-4-5", "residual": pytest.approx(0, abs=1e-9),
                                         "rows": 4},
                                        {"model": "claude-opus-5", "residual": pytest.approx(0, abs=1e-9),
                                         "rows": 4}]
    assert payload["unpriced_models"] == [{"model": "claude-fable-5-1", "share": pytest.approx(0.2083, abs=1e-4)}]
    subagent = next(row for row in payload["dimensions"]["thread"] if row["bucket"] == "subagent")
    assert subagent["dollars"] is None and subagent["unpriced"] == ["claude-fable-5-1"]


def test_the_json_prices_only_the_models_the_window_actually_ran(tmp_path):
    turns = corpus(tmp_path)
    # claude-sonnet-5 was fitted from a cost record of some older session. Listing it as
    # priced beside a dollars of null would say a price was found for spend that is there,
    # when the model is simply absent from the window.
    prices = {**OPUS, "claude-sonnet-5": Price(2e-6, 10e-6, 0.0, 0.0, 12)}
    payload = json.loads(spend_json(turns, ["model"], prices, ["2026-09-01"]))
    assert [row["model"] for row in payload["priced_models"]] == ["claude-opus-5"]
