"""`ccdrift report --html`: the day view as one self-contained page."""

from datetime import date

import pandas as pd

from ccdrift.detector import DetectorConfig
from ccdrift.page import blocks, chart, escape, points, render, section, table, top, z_strip

DAYS = ["2026-09-01", "2026-09-02", "2026-09-03"]


def rows_of(ratios, zs, shares=None, haiku_zs=None, flagged=None):
    return pd.DataFrame({
        "day": DAYS, "responses": [60] * len(DAYS), "cache_ratio": ratios, "cache_z": zs,
        "haiku_share": shares or [0.0] * len(DAYS), "haiku_z": haiku_zs or [0.0] * len(DAYS),
        "loop_turns": [0] * len(DAYS), "loop_misses": [0] * len(DAYS),
        "subagent_loop_turns": [0] * len(DAYS), "subagent_loop_misses": [0] * len(DAYS),
        "flagged": flagged or [""] * len(DAYS)})


def test_a_day_sits_evenly_across_the_chart_and_a_missing_one_leaves_a_gap():
    assert points([0.9, None, 0.5], 0.0, 1.0) == [(48.0, 28.0), None, (768.0, 92.0)]
    # One day sits in the middle rather than at the left edge.
    assert points([0.5], 0.0, 1.0) == [(408.0, 92.0)]


def test_a_chart_shades_an_incidents_days_marks_the_flagged_ones_and_draws_the_line():
    drawn = chart(DAYS, [0.9, 0.5, 0.9], title="Cache read ratio per day", low=0.0, high=1.0,
                  marked=["2026-09-02"], shaded=["2026-09-02"])
    assert drawn.startswith('<figure><figcaption>Cache read ratio per day</figcaption>'
                            '<svg viewBox="0 0 780 200" role="img" aria-label="Cache read ratio per day">')
    assert '<rect class="incident" x="405.0" y="12" width="6" height="160" />' in drawn
    assert '<polyline class="line" points="48.0,28.0 408.0,92.0 768.0,28.0" />' in drawn
    assert '<circle class="flagged" cx="408.0" cy="92.0" r="3.5" />' in drawn
    assert '<text class="tick" x="42" y="16" text-anchor="end">1.00</text>' in drawn
    assert drawn.endswith("</svg></figure>")


def test_a_chart_breaks_its_line_at_a_day_it_has_no_value_for():
    drawn = chart(DAYS, [0.9, None, 0.5], title="t", low=0.0, high=1.0)
    # Two lone days on either side of the gap: nothing to draw a line between.
    assert "<polyline" not in drawn
    assert '<circle class="point" cx="48.0" cy="28.0" r="2" />' in drawn
    assert '<circle class="point" cx="768.0" cy="92.0" r="2" />' in drawn


def test_the_z_strip_draws_a_bar_per_judged_day_and_the_cutoff_on_the_side_judged_from():
    below = z_strip(["2026-09-01", "2026-09-02"], [None, -4.0], -3.0, above=False)
    assert '<rect class="z" x="766.0" y="32.0" width="4" height="17.8" />' in below
    assert '<line class="cutoff" x1="48" y1="45.3" x2="768" y2="45.3" />' in below
    assert '<text class="tick" x="42" y="49.3" text-anchor="end">-3.0</text>' in below
    above = z_strip(["2026-09-01", "2026-09-02"], [0.0, 4.0], 3.5, above=True)
    assert '<line class="cutoff" x1="48" y1="18.7" x2="768" y2="18.7" />' in above
    assert '<text class="tick" x="42" y="22.7" text-anchor="end">+3.5</text>' in above


def test_a_zero_cutoff_with_no_larger_z_draws_an_empty_strip_instead_of_dividing_by_zero():
    # reach would be 0 here with nothing larger to divide by; it must not raise.
    strip = z_strip(["2026-09-01"], [0.0], 0.0, above=True)
    assert '<rect class="z" x="406.0" y="32.0" width="4" height="0.0" />' in strip
    assert '<line class="cutoff" x1="48" y1="32.0" x2="768" y2="32.0" />' in strip
    empty = z_strip(["2026-09-01"], [], 0.0, above=False)
    assert "<rect" not in empty


def test_everything_the_page_prints_is_escaped():
    assert escape('/Users/me/a&b<script>') == "/Users/me/a&amp;b&lt;script&gt;"
    assert '&lt;script&gt;' in section("Session starts", ["  /Users/me/<script>x</script> ~1k"])


def test_a_section_keeps_the_terminal_wording_and_a_one_line_part_stays_a_sentence():
    assert section("Incidents", ["  cache 2026-08-18..2026-09-03"]) == (
        "<h2>Incidents</h2><pre>cache 2026-08-18..2026-09-03</pre>")
    # The heading's trailing colon belongs to a terminal line, not to a heading.
    assert section("Settings over these days:", ["  claude-opus-5: effort xhigh 100%"]).startswith(
        "<h2>Settings over these days</h2>")
    assert section("Session starts by project over these days: /Users/me/app ~1k (3 sessions)", []) == (
        '<p class="note">Session starts by project over these days: /Users/me/app ~1k (3 sessions)</p>')


def test_the_terminal_tail_splits_into_its_own_parts():
    assert blocks(["", "Hooks over these days: 3 runs", "", "Settings:", "  opus: effort xhigh"]) == [
        ("Hooks over these days: 3 runs", []), ("Settings:", ["opus: effort xhigh"])]


def test_a_body_line_before_any_heading_starts_its_own_block_instead_of_vanishing():
    # No tail helper emits this shape today, but a future one that did shouldn't lose the
    # line: it opens a block of its own rather than being swallowed with nothing to catch it.
    assert blocks(["  stray line", "Heading:", "  body"]) == [
        ("stray line", []), ("Heading:", ["body"])]


def test_the_table_carries_the_terminal_columns_and_marks_a_flagged_day():
    drawn = table(rows_of([0.9, 0.44, 0.9], [None, -20.4, 0.0], flagged=["", "cache", ""]))
    assert "<th>day</th><th>responses</th><th>cache ratio</th>" in drawn
    assert '<tr class=""><td>2026-09-01</td><td>60</td><td>0.900</td><td>-</td>' in drawn
    assert '<tr class="flagged"><td>2026-09-02</td><td>60</td><td>0.440</td><td>-20.4</td>' in drawn


def test_the_page_holds_the_rule_both_charts_the_table_and_the_footer_and_no_script():
    page = render(rows_of([0.9, 0.44, 0.9], [None, -20.4, 0.0], flagged=["", "cache", ""]),
                  entries=[], reported={}, summary=[], extra=["", "Hooks over these days: 3 runs"],
                  cfg=DetectorConfig(), version="0.9.0", today=date(2026, 9, 18),
                  source="/Users/me/.claude/projects", incident_days=["2026-09-02"])
    assert page.startswith("<!DOCTYPE html>\n")
    assert "<script" not in page
    assert ("A metric is flagged once 3 of any 4 days in a row pass the cutoff: z ≤ −3.0 for the cache ratio, "
            "z ≥ +3.5 for the Haiku share.") in page
    assert "<figcaption>Cache read ratio per day</figcaption>" in page
    assert "<figcaption>Haiku share of main-thread responses per day</figcaption>" in page
    assert '<rect class="incident" x="405.0"' in page
    assert "<h2>Incidents</h2><pre>none yet</pre>" in page
    assert '<p class="note">Hooks over these days: 3 runs</p>' in page
    assert ("<footer>Written by ccdrift 0.9.0 on 2026-09-18 from the transcripts in "
            "/Users/me/.claude/projects. This page holds local paths, and nothing left this machine to "
            "make it.</footer>") in page


def test_a_page_with_no_days_still_renders_its_sections():
    empty = rows_of([0.9] * 3, [0.0] * 3).iloc[0:0]
    page = render(empty, entries=[], reported={}, summary=[], extra=[], cfg=DetectorConfig(),
                  version="0.9.0", today=date(2026, 9, 18), source="/logs")
    assert "The last 0 complete UTC days" in page
    assert "<figcaption>" not in page and "<table>" not in page
    assert "<h2>Incidents</h2>" in page


def test_a_share_chart_scales_to_its_own_range_so_a_small_rise_is_still_visible():
    assert top([0.0, 0.0], 0.2) == 0.2          # a metric that never leaves the floor keeps its scale
    assert top([0.12, 0.03], 0.2) == 0.2
    assert top([0.5, 0.62], 0.2) == 0.8         # room above the busiest day, rounded to a tenth
    assert top([], 0.2) == 0.2


def test_the_haiku_chart_labels_the_scale_it_drew():
    page = render(rows_of([0.9] * 3, [0.0] * 3, shares=[0.0, 0.5, 0.1]), entries=[], reported={}, summary=[],
                  extra=[], cfg=DetectorConfig(), version="0.9.0", today=date(2026, 9, 18), source="/logs")
    haiku = page.split("<figcaption>Haiku share")[1]
    assert '<text class="tick" x="42" y="16" text-anchor="end">0.70</text>' in haiku
