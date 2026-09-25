"""Partitioning the history by where its tokens went."""

import json
from datetime import date

import pandas as pd
import pytest

from ccdrift.history import load_history
from ccdrift.logs import parse_source
from ccdrift.prices import Price
from ccdrift.spend import (DEFAULT_ORDER, DIMENSIONS, MATERIAL_SHARE, PRIVATE_DIMENSIONS, TOKEN_COLUMNS,
                           branch_projects, priced_total, record_write_tiers, response_dollars, run_spend,
                           spend_json, spend_rows, spend_turns, total_tokens)
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


def branches_across_projects(tmp_path):
    """`main` reached from two project folders and `topic` from one: the shape that makes
    a branch row name more than one place at once. On the owner's corpus on 2026-09-22,
    3 branch names of 191 did this and carried 18.0% of the window between them."""
    write(tmp_path / "proj-a" / "s1.jsonl", [
        line("m1", text(40), ts=at(0), entrypoint="cli", branch="main"),
        line("m2", text(40), ts=at(60), entrypoint="cli", branch="topic"),
    ])
    write(tmp_path / "proj-b" / "s2.jsonl", [
        line("m3", text(40), ts=at(120), entrypoint="cli", branch="main"),
    ])
    return spend_turns(parse_source(tmp_path), TODAY)


def branch_and_agent_share_a_name(tmp_path):
    """A branch called `general-purpose`, reached from two project folders, and an agent
    of that same name in a third response on one of them. Nothing stops a branch being
    named after an agent, so the collision is a fair case and not a contrived one: it is
    the only fixture shape that can tell "the project count is looked up for the branch
    dimension" apart from "the project count is looked up by bucket name and happens not
    to collide", since `branch_projects`' keys are branch names and a lookup that ignores
    the dimension guard would hit any other dimension's bucket of the identical name."""
    write(tmp_path / "proj-a" / "s1.jsonl", [
        line("m1", text(40), ts=at(0), entrypoint="cli", branch="general-purpose"),
    ])
    write(tmp_path / "proj-b" / "s2.jsonl", [
        line("m2", text(40), ts=at(60), entrypoint="cli", branch="general-purpose"),
        line("a1", text(40), ts=at(120), entrypoint="cli", sidechain=True, agent_type="general-purpose",
             branch="topic"),
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
    fit has to recover RATES from: four of them, one more than the three free parameters,
    with counts that keep the three columns independent, and cache reads on two records
    per model, since a read rate one record sets is not priced. `unpriced_out` adds a subagent
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
               "claude-haiku-4-5": {"input": 20, "output": 8_000, "cache_read": 3_000}},
              {"claude-opus-5": {"input": 7_000, "output": 20, "cache_read": 60_000},
               "claude-haiku-4-5": {"input": 900, "output": 30, "cache_creation": 400}}]
    write(tmp_path / "proj-a" / "s1.jsonl",
          responses + [cost_record(at(600 * i), i + 1, one) for i, one in enumerate(counts)])


# Two models reading the cache at different shares of their input rate, as the models Claude
# Code shipped by 2.1.280 do: claude-opus-5 at $5/$25 per Mtok reading at 0.1x, and
# claude-opus-5-5 at $4/$20 reading at 0.05x, the rates Claude Code published for it.
CACHE_RATIOS = {"claude-opus-5": (5e-6, 25e-6, 0.1), "claude-opus-5-5": (4e-6, 20e-6, 0.05)}


def two_cache_ratios_corpus(tmp_path, keys=None, rates=None):
    """One main-thread response per model, each reading a million tokens from the cache,
    beside four cost records per model priced at CACHE_RATIOS. A fit sharing one read ratio
    between them could price at most one of the two. `keys` renames one of those models'
    cost-state key, as Claude Code keys a 1M-context tier `claude-opus-5-5[1m]`; `rates` adds
    cost records under further keys, taken as written, as (rate in, rate out, read ratio,
    how many records).

    claude-opus-5:   10 input + 1,000,000 read at $0.50/Mtok + 100,000 output = $3.00005.
    claude-opus-5-5: 10 input + 1,000,000 read at $0.20/Mtok + 100,000 output = $2.20004."""
    responses = [line("m1", text(40), ts=at(0), entrypoint="cli", out=100_000, cache_read=1_000_000),
                 line("m2", text(40), ts=at(60), entrypoint="cli", out=100_000, cache_read=1_000_000,
                      model="claude-opus-5-5")]
    counts = [{"input": 1_000, "output": 50, "cache_creation": 200, "cache_read": 9_000},
              {"input": 3_000, "output": 700, "cache_creation": 10, "cache_read": 100},
              {"input": 50, "output": 5_000, "cache_creation": 900, "cache_read": 40},
              {"input": 7_000, "output": 20, "cache_creation": 5, "cache_read": 60_000}]
    records = []
    priced = [((keys or {}).get(model, model), *ratios, len(counts)) for model, ratios in CACHE_RATIOS.items()]
    priced += [(key, *ratios) for key, ratios in (rates or {}).items()]
    for i, one in enumerate(counts):
        usage = {}
        for key, rate_in, rate_out, read_ratio, many in priced:
            if i >= many:
                continue
            cost = ((one["input"] + 1.25 * one["cache_creation"]) * rate_in
                    + one["cache_read"] * rate_in * read_ratio + one["output"] * rate_out)
            usage[key] = {**one, "costUSD": cost}
        records.append(cost_state(at(600 * i), usage, start=i + 1))
    write(tmp_path / "proj-a" / "s1.jsonl", responses + records)


def five_small_models(tmp_path):
    """One model carrying 98.5% of the window and five carrying 0.3% each: not one of the
    five is material on its own, 1.5% of the window between them."""
    records = [line("m0", text(40), ts=at(0), entrypoint="cli", out=36_107)]
    records += [line(f"m{i}", text(40), ts=at(60 * i), entrypoint="cli", model=f"claude-tiny-{i}")
                for i in range(1, 6)]
    write(tmp_path / "p" / "s1.jsonl", records)
    return spend_turns(parse_source(tmp_path / "p"), TODAY)


def price(rate_in, rate_out, read_ratio=0.1, rows=9):
    """A price as fit_prices builds one, reading the cache at `read_ratio` of its input
    rate: 0.1x, the share every model before claude-opus-5-5 charged."""
    return Price(input_rate=rate_in, cache_read_rate=rate_in * read_ratio, output_rate=rate_out,
                 web_search_rate=0.0, residual=0.0, rows=rows)


OPUS = {"claude-opus-5": price(5e-6, 25e-6)}
# Both the models money_corpus records costs for, at the rates RATES states, so a bucket
# holding a1 alongside an unpriced response has a dollar figure to be priced from.
OPUS_AND_HAIKU = {**OPUS, "claude-haiku-4-5": price(1e-6, 5e-6)}


def test_every_dimension_has_a_heading_and_every_heading_a_dimension():
    # Two parallel structures: --by takes its choices from DIMENSION_NAMES and spend_lines
    # reads a heading out of it per DIMENSIONS, so a name missing from either is a
    # KeyError or a dimension no one can ask for.
    assert set(DIMENSIONS) == set(DIMENSION_NAMES)


def test_the_terminals_default_dimensions_are_the_same_ones_the_json_defaults_to():
    # run_spend passes DEFAULT_ORDER to the terminal path and list(DIMENSIONS) to the JSON
    # one; they agree today only by coincidence of content and order, unpinned here.
    assert tuple(d for d in DIMENSIONS if d not in PRIVATE_DIMENSIONS) == DEFAULT_ORDER


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


def test_a_session_whose_subagents_have_a_folder_of_their_own_is_still_not_a_project(tmp_path):
    # Claude Code keeps a session's subagent transcripts in a folder named by its session id.
    sid = "0199c3d0-1111-2222-3333-444455556666"
    write(tmp_path / "p" / f"{sid}.jsonl", [line("m1", text(40), ts=at(0), entrypoint="cli", sid=sid)])
    write(tmp_path / "p" / sid / "subagents" / "agent-a1.jsonl",
          [line("m2", text(40), ts=at(60), entrypoint="cli", sid=sid, sidechain=True)])
    rows = spend_rows(spend_turns(parse_source(tmp_path / "p"), TODAY), "project", {})
    assert rows["bucket"].tolist() == ["the source folder"]


def test_a_transcript_beside_project_folders_counts_under_the_source_folder(tmp_path):
    sid = "0199c3d0-1111-2222-3333-444455556666"
    write(tmp_path / "logs" / f"{sid}.jsonl", [line("m1", text(40), ts=at(0), entrypoint="cli", sid=sid)])
    write(tmp_path / "logs" / "-proj-a" / "s2.jsonl", [line("m2", text(40), ts=at(60), entrypoint="cli", sid="s2")])
    rows = spend_rows(spend_turns(parse_source(tmp_path / "logs"), TODAY), "project", {})
    assert sorted(rows["bucket"]) == ["/proj/a", "the source folder"]


def test_a_repository_with_no_branch_checked_out_is_kept_out_of_the_branch_names(tmp_path, capsys):
    # "HEAD" is what git answers with nothing checked out, so it is not a branch name and
    # must not sort among them. It stays apart from "no branch", which means the field is
    # absent: a Claude Code version fact rather than a git one, and not the same thing.
    write(tmp_path / "logs" / "p" / "s1.jsonl", [
        line("m1", text(40), ts=at(0), entrypoint="cli", branch="HEAD"),
        line("m2", text(40), ts=at(60), entrypoint="cli", branch="main"),
        line("m3", text(40), ts=at(120), entrypoint="cli"),
    ])
    # A second project folder detached at the same time as the first, so the detached
    # bucket is the one place this suite ties the rename to branch_projects' own lookup
    # and to the count the printed row carries, rather than to the raw "HEAD" value.
    write(tmp_path / "logs" / "q" / "s2.jsonl", [
        line("m4", text(40), ts=at(180), entrypoint="cli", branch="HEAD"),
    ])
    turns = spend_turns(parse_source(tmp_path / "logs"), TODAY)
    rows = spend_rows(turns, "branch", {})
    assert set(rows["bucket"]) == {"detached HEAD", "main", "no branch"}
    assert branch_projects(turns)["detached HEAD"] == 2

    state = tmp_path / "state.json"
    save_state(state, new_state())
    run_spend(tmp_path / "logs", state, by="branch", today=TODAY)
    assert bucket_line(capsys.readouterr().out, "detached HEAD").endswith("2 projects")


def test_the_partition_still_holds_with_a_detached_bucket_in_it(tmp_path):
    write(tmp_path / "p" / "s1.jsonl", [
        line("m1", text(40), ts=at(0), entrypoint="cli", branch="HEAD"),
        line("m2", text(40), ts=at(60), entrypoint="cli", branch="main"),
    ])
    turns = spend_turns(parse_source(tmp_path / "p"), TODAY)
    rows = spend_rows(turns, "branch", {})
    assert rows["tokens"].sum() == pytest.approx(total_tokens(turns))
    assert rows["share"].sum() == pytest.approx(1.0)


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


def test_a_bucket_blanks_when_its_unpriced_models_are_material_to_that_bucket(tmp_path):
    turns = corpus(tmp_path)
    # The subagent bucket holds both claude-opus-5 (a1) and claude-haiku-4-5 (a2), half
    # its tokens each, so pricing only the former must leave the whole bucket unpriced.
    # Half is not a rounding error; the bucket below, where the unpriced share is, is.
    opus_only = {"claude-opus-5": price(5e-6, 25e-6)}
    rows = spend_rows(turns, "thread", opus_only)
    assert pd.notna(rows.loc[rows["bucket"] == "main thread", "dollars"].iloc[0])
    assert pd.isna(rows.loc[rows["bucket"] == "subagent", "dollars"].iloc[0])

    both_priced = {**opus_only, "claude-haiku-4-5": price(1e-6, 5e-6)}
    assert spend_rows(turns, "thread", both_priced)["dollars"].notna().all()
    assert spend_rows(turns, "thread", {})["dollars"].isna().all()


def test_a_bucket_is_priced_from_the_rest_when_its_unpriced_share_is_a_rounding_error(tmp_path):
    # The failure this rule exists for, from the owner's corpus on 2026-09-22: 11
    # responses of a model with no price, 0.009% of the window's tokens, blanked the
    # dollars on 60.5% of it, while the total one line above printed because the same
    # 0.009% cleared the same cutoff.
    money_corpus(tmp_path / "logs", unpriced_out=100)
    turns = spend_turns(parse_source(tmp_path / "logs"), TODAY)
    row = spend_rows(turns, "thread", OPUS_AND_HAIKU).set_index("bucket").loc["subagent"]
    assert row["unpriced"] == ("claude-fable-5-1",)          # still named, and still in the JSON
    assert row["dollars"] == pytest.approx(11.00001)         # a1's own dollars, a2's left out


def test_a_bucket_blanks_once_its_unpriced_share_reaches_the_cutoff(tmp_path):
    money_corpus(tmp_path / "logs", unpriced_out=35_000)
    turns = spend_turns(parse_source(tmp_path / "logs"), TODAY)
    rows = spend_rows(turns, "thread", OPUS_AND_HAIKU).set_index("bucket")
    assert pd.isna(rows.loc["subagent", "dollars"])
    assert pd.notna(rows.loc["main thread", "dollars"])
    # 1.2% of the bucket and 0.9% of the window: the one case that tells the two rules
    # apart, so the bucket goes blank while the window's own total still prints.
    assert priced_total(turns, OPUS_AND_HAIKU) is not None


def test_the_cutoff_is_measured_against_the_bucket_and_not_against_the_window(tmp_path):
    # A bucket that is entirely unpriced must blank however small it is. Measured against
    # the window instead, every small bucket would pass, including this one.
    write(tmp_path / "p" / "s1.jsonl", [
        line("m1", text(40), ts=at(0), entrypoint="cli", out=100_000),
        line("a1", text(40), ts=at(60), entrypoint="cli", sidechain=True, out=10, model="claude-fable-5-1"),
    ])
    turns = spend_turns(parse_source(tmp_path / "p"), TODAY)
    rows = spend_rows(turns, "thread", OPUS).set_index("bucket")
    assert rows.loc["subagent", "share"] < MATERIAL_SHARE     # 0.02% of the window
    assert pd.isna(rows.loc["subagent", "dollars"])           # and 100% of itself


def test_a_bucket_with_no_tokens_at_all_blanks_rather_than_dividing_by_them(tmp_path):
    # A response whose message carries no usage parses to zero tokens, so a bucket can
    # sum to zero and the share it would be tested on does not exist.
    turns = corpus(tmp_path).assign(**{column: 0 for column in TOKEN_COLUMNS})
    rows = spend_rows(turns, "thread", OPUS).set_index("bucket")
    assert pd.isna(rows.loc["subagent", "dollars"])          # holds an unpriced model
    assert rows.loc["main thread", "dollars"] == 0.0         # priced, and worth nothing


def test_the_model_dimension_is_all_or_nothing_whatever_the_cutoff_is(tmp_path):
    # A model bucket's key is the model, so it is 0% or 100% unpriced and never in
    # between. priced_total reads that dimension, so the new rule must not reach it.
    money_corpus(tmp_path / "logs", unpriced_out=100)
    turns = spend_turns(parse_source(tmp_path / "logs"), TODAY)
    rows = spend_rows(turns, "model", OPUS_AND_HAIKU).set_index("bucket")
    assert pd.isna(rows.loc["claude-fable-5-1", "dollars"])
    assert pd.notna(rows.loc["claude-opus-5", "dollars"])


def test_the_total_is_withheld_when_a_material_model_is_unpriced_and_present_when_all_are(tmp_path):
    turns = corpus(tmp_path)
    by_model = spend_rows(turns, "model", OPUS)
    haiku = by_model.loc[by_model["bucket"] == "claude-haiku-4-5", "share"].iloc[0]
    assert haiku >= MATERIAL_SHARE                            # a quarter of the window, not a rounding error
    assert priced_total(turns, OPUS) is None
    both = {**OPUS, "claude-haiku-4-5": price(1e-6, 5e-6)}
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


def write_tier_corpus(tmp_path):
    """One session of claude-opus-5-5 at its list price, $4 in, $20 out and reads at 0.05x,
    writing as Claude Code does: a million tokens for an hour on the main thread and a
    million for five minutes in a subagent, so half the session's writes are at each tier.
    Its four cost records, keyed by the 1M tier as Claude Code keys this model, carry no
    tiers of their own and are billed at that half.

    main thread: 10 input + 1,000,000 written at 2x + 100 output = $8.00204.
    subagent:    10 input + 1,000,000 written at 1.25x + 100 output = $5.00204."""
    responses = [line("m1", text(40), ts=at(0), entrypoint="cli", model="claude-opus-5-5",
                      cache_creation=1_000_000, cache_1h=1_000_000, cache_5m=0),
                 line("a1", text(40), ts=at(60), entrypoint="cli", sidechain=True, model="claude-opus-5-5",
                      cache_creation=1_000_000, cache_1h=0, cache_5m=1_000_000)]
    counts = [{"input": 1_000, "output": 50, "cache_creation": 200, "cache_read": 9_000},
              {"input": 3_000, "output": 700, "cache_creation": 10, "cache_read": 100},
              {"input": 50, "output": 5_000, "cache_creation": 900, "cache_read": 40},
              {"input": 7_000, "output": 20, "cache_creation": 5, "cache_read": 60_000}]
    records = []
    for i, one in enumerate(counts):
        written = 0.5 * 1.25 * one["cache_creation"] + 0.5 * 2 * one["cache_creation"]
        cost = (one["input"] + written) * 4e-6 + one["cache_read"] * 0.2e-6 + one["output"] * 20e-6
        records.append(cost_state(at(600 * i), {"claude-opus-5-5[1m]": {**one, "costUSD": cost}}, start=i + 1))
    write(tmp_path / "proj-a" / "s1.jsonl", responses + records)


def test_the_dollars_printed_charge_each_cache_write_at_its_own_tier(tmp_path, capsys):
    # Measured on 2026-09-25: claude-opus-5-5 writes its main-thread cache for an hour, at 2x
    # input, and its subagent cache for five minutes, at 1.25x. Every write billed at 1.25x
    # had the fit move the difference into output, $37.16 per Mtok against a list price of $20.
    state = tmp_path / "state.json"
    save_state(state, new_state())
    write_tier_corpus(tmp_path / "logs")
    load_history(tmp_path / "logs", state, claim=True)
    assert run_spend(tmp_path / "logs", state, today=TODAY) == 0
    out = capsys.readouterr().out
    assert "$13.00" in out.splitlines()[0]                     # 8.00204 + 5.00204
    assert "$8.00" in bucket_line(out, "main thread") and "$5.00" in bucket_line(out, "subagent")


def test_a_cost_records_one_hour_writes_come_from_its_own_sessions_responses_of_its_model():
    # A record logs only how much it wrote; its session's responses say at which tier. The
    # record's 1M-tier key is its model's plain name to a response, another model's writes in
    # the same session are not its own, and nor are the same model's in another session.
    usage = pd.DataFrame([{"session_id": "s1", "model": "claude-opus-5-5[1m]", "cache_creation": 800.0},
                          {"session_id": "s1", "model": "claude-haiku-4-5", "cache_creation": 400.0},
                          {"session_id": "s2", "model": "claude-opus-5-5", "cache_creation": 500.0}])
    responses = pd.DataFrame([
        {"session_id": "s1", "model": "claude-opus-5-5", "cache_creation": 400.0, "cache_1h": 300.0, "cache_5m": 100.0},
        {"session_id": "s1", "model": "claude-haiku-4-5", "cache_creation": 50.0, "cache_1h": 0.0, "cache_5m": 50.0},
        {"session_id": "s3", "model": "claude-opus-5-5", "cache_creation": 900.0, "cache_1h": 900.0, "cache_5m": 0.0}])
    assert record_write_tiers(usage, responses)["cache_1h"].tolist() == [600.0, 0.0, 0.0]


def test_a_cost_records_one_hour_share_counts_its_sessions_untiered_writes_at_five_minutes():
    # response_dollars charges a response that logged no tier at five minutes, so the record
    # its session wrote is fitted the same way: 300 of the session's 800 written tokens were
    # at one hour, not 300 of the 400 that logged a tier. Found in review on 2026-09-25; the
    # owner's store then held no response with untiered writes, so it changed no price.
    usage = pd.DataFrame([{"session_id": "s1", "model": "claude-opus-5-5", "cache_creation": 1000.0}])
    responses = pd.DataFrame([
        {"session_id": "s1", "model": "claude-opus-5-5", "cache_creation": 400.0, "cache_1h": 300.0, "cache_5m": 100.0},
        {"session_id": "s1", "model": "claude-opus-5-5", "cache_creation": 400.0, "cache_1h": 0.0, "cache_5m": 0.0}])
    assert record_write_tiers(usage, responses)["cache_1h"].tolist() == [375.0]


def test_a_response_is_charged_for_its_own_one_hour_writes_and_one_logging_no_tier_at_five_minutes(tmp_path):
    # A response that logs no tier, as a Claude Code too old to log them wrote, is charged at
    # five minutes, which is how every write was billed before.
    write(tmp_path / "p" / "s1.jsonl", [
        line("m1", text(40), ts=at(0), entrypoint="cli", model="claude-opus-5-5", cache_creation=1_000_000,
             cache_1h=1_000_000, cache_5m=0),
        line("m2", text(40), ts=at(60), entrypoint="cli", model="claude-opus-5-5", cache_creation=1_000_000)])
    turns = spend_turns(parse_source(tmp_path / "p"), TODAY)
    dollars = response_dollars(turns, {"claude-opus-5-5": price(4e-6, 20e-6, read_ratio=0.05)})
    assert dollars.tolist() == [pytest.approx(8.00204), pytest.approx(5.00204)]


def test_two_models_reading_the_cache_at_different_ratios_are_each_charged_at_their_own(tmp_path, capsys):
    # The defect 0.13.0 exists for. 0.12.1 fixed every model's cache reads at 0.1x its input
    # rate, so claude-opus-5-5's records never fit, it went unpriced, and at half of this
    # window's tokens it withheld the total. Each model's reads are now billed at its own.
    state = tmp_path / "state.json"
    save_state(state, new_state())
    two_cache_ratios_corpus(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, today=TODAY)
    out = capsys.readouterr().out
    assert "$5.20" in out.splitlines()[0]                      # 3.00005 + 2.20004
    assert "$3.00" in bucket_line(out, "claude-opus-5") and "$2.20" in bucket_line(out, "claude-opus-5-5")
    assert "no price" not in out


def test_a_model_whose_cost_records_name_only_its_1m_context_tier_is_charged_from_them(tmp_path, capsys):
    # claude-opus-5-5's first cost record, on 2026-09-24, was keyed claude-opus-5-5[1m] while
    # all 3,001 of its responses said claude-opus-5-5: joined exactly, it could never be
    # priced, and at 5.6% of the window it withheld the total for good.
    state = tmp_path / "state.json"
    save_state(state, new_state())
    two_cache_ratios_corpus(tmp_path / "logs", keys={"claude-opus-5-5": "claude-opus-5-5[1m]"})
    run_spend(tmp_path / "logs", state, today=TODAY)
    out = capsys.readouterr().out
    assert "$5.20" in out.splitlines()[0]                      # 3.00005 + 2.20004
    assert "$2.20" in bucket_line(out, "claude-opus-5-5") and "no price" not in out


def test_a_model_with_a_plain_key_is_charged_at_it_beside_its_1m_context_tier(tmp_path, capsys):
    # claude-opus-5 has both keys. A response doesn't say which tier it ran on, so it keeps
    # the plain price; the 1M tier's, here twice as dear, must not replace it.
    state = tmp_path / "state.json"
    save_state(state, new_state())
    two_cache_ratios_corpus(tmp_path / "logs", rates={"claude-opus-5[1m]": (10e-6, 50e-6, 0.1, 4)})
    run_spend(tmp_path / "logs", state, today=TODAY)
    assert "$3.00" in bucket_line(capsys.readouterr().out, "claude-opus-5")


def test_a_model_whose_plain_key_went_unpriced_does_not_borrow_its_1m_context_price(tmp_path, capsys):
    # With a plain key on record the 1M tier is a different price, not a second name for the
    # same one: too few plain records leave the model unpriced rather than charged as 1M.
    state = tmp_path / "state.json"
    save_state(state, new_state())
    two_cache_ratios_corpus(tmp_path / "logs", keys={"claude-opus-5-5": "claude-opus-5-5[1m]"},
                            rates={"claude-opus-5-5": (4e-6, 20e-6, 0.05, 1)})
    run_spend(tmp_path / "logs", state, today=TODAY)
    assert bucket_line(capsys.readouterr().out, "claude-opus-5-5").endswith("no price")


def test_a_rounding_error_model_no_longer_blanks_the_bucket_it_landed_in(tmp_path, capsys):
    # The failure this rule exists for, end to end: one model of 0.0% of the window is
    # refused by the fit, and the bucket it landed in keeps its dollars rather than going
    # blank beside a total that counted the same spend as immaterial.
    state = tmp_path / "state.json"
    save_state(state, new_state())
    money_corpus(tmp_path / "logs", unpriced_out=100)
    run_spend(tmp_path / "logs", state, today=TODAY)
    out = capsys.readouterr().out
    assert "$36.00" in out.splitlines()[0]
    assert out.splitlines()[2].startswith("No price for claude-fable-5-1, 0.0% of the window's tokens:")
    assert "$11.00" in bucket_line(out, "subagent")
    assert "$25.00" in bucket_line(out, "main thread")
    # The bucket that is the model says it once: repeating its own name explains nothing.
    assert bucket_line(out, "claude-fable-5-1").endswith("no price")


def test_a_bucket_a_model_with_no_price_weighs_on_still_names_it(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    money_corpus(tmp_path / "logs", unpriced_out=35_000)
    run_spend(tmp_path / "logs", state, today=TODAY)
    out = capsys.readouterr().out
    assert "$" in out.splitlines()[0]                          # the window's total still prints
    assert "no price: claude-fable-5-1" in bucket_line(out, "subagent")
    assert "$25.00" in bucket_line(out, "main thread")


def test_the_header_states_the_cutoff_that_decides_each_bucket_too(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    money_corpus(tmp_path / "logs", unpriced_out=100)
    run_spend(tmp_path / "logs", state, today=TODAY)
    said = capsys.readouterr().out.splitlines()[2]
    # One rule decides the total and every bucket, so it is stated once, with its number.
    assert said.endswith("out of any bucket where it stays under 1%. A bucket where it reaches 1% "
                         "shows no dollars at all.")


def test_a_withheld_total_says_which_model_withheld_it(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    money_corpus(tmp_path / "logs", unpriced_out=1_000_000)
    run_spend(tmp_path / "logs", state, today=TODAY)
    out = capsys.readouterr().out
    assert "$" not in out.splitlines()[0]
    assert out.splitlines()[2].startswith("No total: no price for claude-fable-5-1, 20.8% of the window's tokens.")
    assert out.splitlines()[2].endswith("A bucket where it reaches 1% shows no dollars either.")


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


def test_the_default_json_names_both_private_dimensions_withheld_instead_of_dropping_them(tmp_path, capsys):
    # run_spend's own default dimension list (DEFAULT_ORDER) never asks for project or
    # branch, so with no --by at all the withheld check above never even ran: shipped
    # 0.12.0 returned "withheld": [] here, silently dropping both instead of naming them.
    state = tmp_path / "state.json"
    save_state(state, new_state())
    corpus(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, as_json=True, today=TODAY)
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["withheld"]) == {"project", "branch"}
    assert "project" not in payload["dimensions"] and "branch" not in payload["dimensions"]


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
                                         "rows": 4, "cache_read_ratio": 0.1},
                                        {"model": "claude-opus-5", "residual": pytest.approx(0, abs=1e-9),
                                         "rows": 4, "cache_read_ratio": 0.1}]
    assert payload["unpriced_models"] == [{"model": "claude-fable-5-1", "share": pytest.approx(0.2083, abs=1e-4)}]
    subagent = next(row for row in payload["dimensions"]["thread"] if row["bucket"] == "subagent")
    assert subagent["dollars"] is None and subagent["unpriced"] == ["claude-fable-5-1"]


def test_the_json_says_what_share_of_its_input_rate_each_model_charges_for_a_cache_read(tmp_path, capsys):
    # The fact 0.13.0 exists for, where a reader can see it: models stopped sharing one
    # cache-read ratio, and the fit found each one's from Claude Code's own records.
    state = tmp_path / "state.json"
    save_state(state, new_state())
    two_cache_ratios_corpus(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, as_json=True, today=TODAY)
    payload = json.loads(capsys.readouterr().out)
    ratios = {row["model"]: row["cache_read_ratio"] for row in payload["priced_models"]}
    assert ratios == {"claude-opus-5": 0.1, "claude-opus-5-5": 0.05}
    assert payload["dollars"] == pytest.approx(3.00005 + 2.20004)
    # The ratio is evidence about a price, not a figure of spend, so the terminal view
    # that reports spend does not carry it. Neither ratio's digits occur anywhere else in
    # this output, so any rendering of either, "0.05x" as much as "ratio", shows up here.
    run_spend(tmp_path / "logs", state, today=TODAY)
    out = capsys.readouterr().out
    assert not [shown for shown in ("ratio", "0.05", "0.1") if shown in out]


def test_the_json_prices_only_the_models_the_window_actually_ran(tmp_path):
    turns = corpus(tmp_path)
    # claude-sonnet-5 was fitted from a cost record of some older session. Listing it as
    # priced beside a dollars of null would say a price was found for spend that is there,
    # when the model is simply absent from the window.
    prices = {**OPUS, "claude-sonnet-5": price(2e-6, 10e-6, rows=12)}
    payload = json.loads(spend_json(turns, ["model"], prices, ["2026-09-01"]))
    assert [row["model"] for row in payload["priced_models"]] == ["claude-opus-5"]


def test_the_json_names_no_cache_read_ratio_for_an_input_rate_that_all_but_vanished(tmp_path):
    # A fit can land an input rate just above zero by near-cancellation, and the ratio then
    # reads as reads costing 9e13 times input. Every model priced so far reads the cache at a
    # discount, so a ratio above 1 says the fit's input rate collapsed, not that reads cost more.
    turns = corpus(tmp_path)
    prices = {"claude-opus-5": Price(input_rate=5.4e-21, cache_read_rate=0.5e-6, output_rate=25e-6,
                                     web_search_rate=0.0, residual=0.0, rows=9)}
    payload = json.loads(spend_json(turns, ["model"], prices, ["2026-09-01"]))
    assert [(row["model"], row["cache_read_ratio"]) for row in payload["priced_models"]] == [("claude-opus-5", None)]


def test_the_json_names_no_cache_read_ratio_for_a_price_with_no_input_rate(tmp_path):
    # The ratio divides by the input rate. A fit returns exactly zero only by an exact
    # cancellation, but a price built with one can still reach the JSON, and there is no ratio
    # to report: null, rather than a ZeroDivisionError that takes the whole document down.
    turns = corpus(tmp_path)
    prices = {"claude-opus-5": Price(input_rate=0.0, cache_read_rate=0.5e-6, output_rate=25e-6,
                                     web_search_rate=0.0, residual=0.0, rows=9)}
    payload = json.loads(spend_json(turns, ["model"], prices, ["2026-09-01"]))
    assert [(row["model"], row["cache_read_ratio"]) for row in payload["priced_models"]] == [("claude-opus-5", None)]


def test_a_branch_bucket_counts_the_project_folders_it_drew_on(tmp_path):
    assert branch_projects(branches_across_projects(tmp_path)) == {"main": 2, "topic": 1}


def test_a_branch_reached_from_more_than_one_project_says_how_many(tmp_path, capsys):
    state = tmp_path / "state.json"
    save_state(state, new_state())
    branches_across_projects(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, by="branch", today=TODAY)
    out = capsys.readouterr().out
    assert bucket_line(out, "main").endswith("2 projects")
    assert not bucket_line(out, "topic").rstrip().endswith("projects")


def test_the_project_count_belongs_to_the_branch_dimension_even_when_another_buckets_name_matches_it(tmp_path,
                                                                                                     capsys):
    # `pooled` is keyed by branch name alone, so a lookup that forgot to restrict itself to
    # the branch dimension would hit any other dimension's bucket sharing that name, not by
    # rule but by luck of the fixture not colliding. This fixture makes them collide on
    # purpose: do not "simplify" it back to non-overlapping names, or the guard it pins
    # stops being pinned by anything.
    state = tmp_path / "state.json"
    save_state(state, new_state())
    branch_and_agent_share_a_name(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, by="branch", today=TODAY)
    assert bucket_line(capsys.readouterr().out, "general-purpose").endswith("2 projects")

    run_spend(tmp_path / "logs", state, by="agent", today=TODAY)
    assert not bucket_line(capsys.readouterr().out, "general-purpose").rstrip().endswith("projects")


def test_the_project_count_stays_out_of_the_json(tmp_path, capsys):
    # The branch dimension is withheld from --json entirely, so a count there would
    # describe folders the JSON exists not to name.
    state = tmp_path / "state.json"
    save_state(state, new_state())
    branches_across_projects(tmp_path / "logs")
    run_spend(tmp_path / "logs", state, as_json=True, today=TODAY)
    payload = json.loads(capsys.readouterr().out)
    assert "branch" in payload["withheld"]
    assert all("projects" not in row for rows in payload["dimensions"].values() for row in rows)
