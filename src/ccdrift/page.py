"""`ccdrift report --html`: the day view as one self-contained page. Inline CSS and
hand-drawn SVG, no scripts and nothing fetched when it is opened, so it works offline and
survives being emailed. Everything here is a pure function from the numbers `report`
already computed to a string: the page claims nothing the terminal view doesn't."""

from __future__ import annotations

import html
import math
from datetime import date
from typing import Any, Optional, Sequence

import pandas as pd

from ccdrift.settings import settings_lines
from ccdrift.texts import INCIDENT_METRICS, incident_line, misses, number

WIDTH = 720          # the drawing area of a chart, in SVG user units
HEIGHT = 160         # the metric line's height
STRIP = 40           # the z strip below it
PAD_LEFT = 48        # room for the axis labels
PAD_RIGHT = 12
PAD_TOP = 12
PAD_BOTTOM = 28      # room for the day labels
LABEL_EVERY = 5      # days between the labels under a chart


def escape(value: Any) -> str:
    """Whatever the page prints, safe to put in it — project paths included."""
    return html.escape(str(value), quote=True)


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _x(index: int, count: int) -> float:
    """Where a day sits across the drawing area; one day sits in the middle."""
    if count <= 1:
        return round(PAD_LEFT + WIDTH / 2, 1)
    return round(PAD_LEFT + WIDTH * index / (count - 1), 1)


def points(values: Sequence[Any], low: float, high: float) -> list[Optional[tuple[float, float]]]:
    """Each value's place in the drawing area, left to right, evenly spaced. A missing day
    is None, so the line breaks at it rather than drawing through it."""
    span = high - low or 1.0
    placed: list[Optional[tuple[float, float]]] = []
    for i, value in enumerate(values):
        if _missing(value):
            placed.append(None)
            continue
        share = (float(value) - low) / span
        placed.append((_x(i, len(values)), round(PAD_TOP + HEIGHT * (1 - share), 1)))
    return placed


def _runs(placed: Sequence[Optional[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """The unbroken stretches of a line: a missing day starts a new one."""
    runs: list[list[tuple[float, float]]] = []
    for point in placed:
        if point is None:
            runs.append([])
        elif runs and runs[-1]:
            runs[-1].append(point)
        else:
            runs.append([point])
    return [run for run in runs if run]


def _path(run: Sequence[tuple[float, float]]) -> str:
    return " ".join(f"{x},{y}" for x, y in run)


def top(values: Sequence[Any], floor: float) -> float:
    """The top of a chart whose values sit well below 1: the largest day with room above
    it, never less than `floor`, so a share that never leaves the floor still reads as a
    share and a small rise is still visible. Rounded to a tenth, so the axis label is a
    number someone can hold in their head."""
    seen = [float(value) for value in values if not _missing(value)]
    return max(floor, round(max(seen, default=0.0) * 1.25 + 0.049, 1))


def chart(days: Sequence[str], values: Sequence[Any], *, title: str, low: float, high: float,
          marked: Sequence[str] = (), shaded: Sequence[str] = (), fmt: str = ".2f") -> str:
    """One metric over the days shown: the days of an incident shaded, the days that passed
    the cutoff marked, and the line itself. Values are placed between `low` and `high`, so
    two charts of the same metric are always read on the same scale."""
    placed = points(values, low, high)
    height = PAD_TOP + HEIGHT + PAD_BOTTOM
    parts = [f'<svg viewBox="0 0 {PAD_LEFT + WIDTH + PAD_RIGHT} {height}" role="img" '
             f'aria-label="{escape(title)}">']
    for i, day in enumerate(days):
        if day in set(shaded):
            parts.append(f'<rect class="incident" x="{_x(i, len(days)) - 3}" y="{PAD_TOP}" width="6" '
                         f'height="{HEIGHT}" />')
    parts.append(f'<line class="axis" x1="{PAD_LEFT}" y1="{PAD_TOP + HEIGHT}" x2="{PAD_LEFT + WIDTH}" '
                 f'y2="{PAD_TOP + HEIGHT}" />')
    for run in _runs(placed):
        if len(run) == 1:
            parts.append(f'<circle class="point" cx="{run[0][0]}" cy="{run[0][1]}" r="2" />')
        else:
            parts.append(f'<polyline class="line" points="{_path(run)}" />')
    for i, day in enumerate(days):
        if day in set(marked) and placed[i] is not None:
            parts.append(f'<circle class="flagged" cx="{placed[i][0]}" cy="{placed[i][1]}" r="3.5" />')
    parts.append(f'<text class="tick" x="{PAD_LEFT - 6}" y="{PAD_TOP + 4}" text-anchor="end">'
                 f'{escape(format(high, fmt))}</text>')
    parts.append(f'<text class="tick" x="{PAD_LEFT - 6}" y="{PAD_TOP + HEIGHT + 4}" text-anchor="end">'
                 f'{escape(format(low, fmt))}</text>')
    for i, day in enumerate(days):
        if i % LABEL_EVERY == 0 or i == len(days) - 1:
            parts.append(f'<text class="tick" x="{_x(i, len(days))}" y="{height - 8}" text-anchor="middle">'
                         f'{escape(day[5:])}</text>')
    parts.append("</svg>")
    return f'<figure><figcaption>{escape(title)}</figcaption>' + "".join(parts) + "</figure>"


def z_strip(days: Sequence[str], zs: Sequence[Any], cutoff: float, *, above: bool) -> str:
    """How far each day sat from its usual level, as bars from the middle, with the cutoff
    drawn across the side the metric is judged from. A day ccdrift couldn't judge has no
    bar."""
    reach = max([abs(cutoff) * 1.5] + [abs(float(z)) for z in zs if not _missing(z)])
    middle = PAD_TOP + STRIP / 2
    scale = (STRIP / 2) / reach
    parts = [f'<svg viewBox="0 0 {PAD_LEFT + WIDTH + PAD_RIGHT} {PAD_TOP * 2 + STRIP}" role="img" '
             f'aria-label="z per day">']
    parts.append(f'<line class="axis" x1="{PAD_LEFT}" y1="{middle}" x2="{PAD_LEFT + WIDTH}" y2="{middle}" />')
    for i, z in enumerate(zs):
        if _missing(z):
            continue
        height = round(abs(float(z)) * scale, 1)
        top = round(middle - height if float(z) > 0 else middle, 1)
        parts.append(f'<rect class="z" x="{_x(i, len(zs)) - 2}" y="{top}" width="4" height="{height}" />')
    line = round(middle - cutoff * scale if above else middle + abs(cutoff) * scale, 1)
    parts.append(f'<line class="cutoff" x1="{PAD_LEFT}" y1="{line}" x2="{PAD_LEFT + WIDTH}" y2="{line}" />')
    parts.append(f'<text class="tick" x="{PAD_LEFT - 6}" y="{line + 4}" text-anchor="end">'
                 f'{escape(format(cutoff if above else -abs(cutoff), "+.1f"))}</text>')
    parts.append("</svg>")
    return "".join(parts)


def section(title: str, lines: Sequence[str]) -> str:
    """A heading and the lines the terminal report already writes, escaped. The page keeps
    one wording for both views rather than a second to hold in step. A part the terminal
    writes as a single sentence — "Session starts by project over these days: …" — has no
    body to head, so it stays a sentence."""
    kept = [line.strip() for line in lines if line.strip()]
    heading = title.rstrip(":")
    if not kept:
        return f'<p class="note">{escape(heading)}</p>'
    return f"<h2>{escape(heading)}</h2><pre>" + escape("\n".join(kept)) + "</pre>"


def blocks(lines: Sequence[str]) -> list[tuple[str, list[str]]]:
    """The terminal report's tail split into (heading, body) by the blank lines it already
    puts between its parts, so the page and the terminal never drift apart in wording."""
    found: list[tuple[str, list[str]]] = []
    for line in lines:
        if not line.strip():
            continue
        if line.startswith(" "):
            if found:
                found[-1][1].append(line.strip())
        else:
            found.append((line, []))
    return found


TABLE_COLUMNS = [("day", "day"), ("responses", "responses"), ("cache_ratio", "cache ratio"), ("cache_z", "z"),
                 ("haiku_share", "haiku share"), ("haiku_z", "z"), ("loop", "loop misses"),
                 ("subagent", "subagent misses"), ("flagged", "flagged")]


def _cell(row: Any, key: str) -> str:
    """One cell, formatted as the terminal report formats it."""
    if key == "day":
        return escape(row.day)
    if key == "responses":
        return escape(f"{int(row.responses):,}")
    if key == "loop":
        return escape(misses(row.loop_misses, row.loop_turns))
    if key == "subagent":
        return escape(misses(row.subagent_loop_misses, row.subagent_loop_turns))
    if key == "flagged":
        return escape(row.flagged)
    spec = ".3f" if key in ("cache_ratio", "haiku_share") else "+.1f"
    return escape(number(getattr(row, key), spec))


def table(rows: pd.DataFrame) -> str:
    """The day table, the same columns the terminal view prints. A flagged day carries a
    class, so it reads as flagged without a second column of punctuation."""
    head = "".join(f"<th>{escape(label)}</th>" for _, label in TABLE_COLUMNS)
    body = []
    for row in rows.itertuples(index=False):
        cells = "".join(f"<td>{_cell(row, key)}</td>" for key, _ in TABLE_COLUMNS)
        body.append(f'<tr class="{"flagged" if row.flagged else ""}">{cells}</tr>')
    return f"<table><thead><tr>{head}</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


STYLE = """:root { color-scheme: light dark; --ink: #1a1a1a; --muted: #6b6b6b; --line: #d5d5d5;
  --flag: #b23a2e; --band: #f0d9d5; --paper: #ffffff; }
@media (prefers-color-scheme: dark) { :root { --ink: #e8e8e8; --muted: #9a9a9a; --line: #3a3a3a;
  --flag: #ff8a7a; --band: #4a2b28; --paper: #161616; } }
* { box-sizing: border-box; }
body { margin: 0 auto; padding: 2rem 1rem 4rem; max-width: 56rem; background: var(--paper); color: var(--ink);
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
h1 { font-size: 1.4rem; margin: 0 0 .25rem; }
h2 { font-size: 1.05rem; margin: 2rem 0 .5rem; }
p.rule { color: var(--muted); margin: 0 0 1.5rem; }
figure { margin: 1.5rem 0 .25rem; }
figcaption { color: var(--muted); margin-bottom: .25rem; }
svg { width: 100%; height: auto; display: block; }
.line { fill: none; stroke: var(--ink); stroke-width: 1.5; }
.point { fill: var(--ink); }
.flagged circle, circle.flagged { fill: var(--flag); }
.incident { fill: var(--band); }
.axis { stroke: var(--line); stroke-width: 1; }
.z { fill: var(--muted); }
.cutoff { stroke: var(--flag); stroke-width: 1; stroke-dasharray: 4 3; }
.tick { fill: var(--muted); font-size: 11px; }
table { border-collapse: collapse; width: 100%; margin-top: .5rem; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: .3rem .5rem; border-bottom: 1px solid var(--line); }
th:first-child, td:first-child { text-align: left; }
tr.flagged td { color: var(--flag); }
pre { margin: 0; white-space: pre-wrap; color: var(--ink); }
p.note { margin: 1.25rem 0 0; }
footer { margin-top: 3rem; color: var(--muted); font-size: .9rem; }"""


def render(rows: pd.DataFrame, entries: Sequence[tuple[dict, float]], reported: dict, summary: Sequence[dict],
           extra: Sequence[str], cfg: Any, *, version: str, today: date, source: Any,
           incident_days: Sequence[str] = ()) -> str:
    """The whole page: the rule in words, a chart per metric with its z below it, the day
    table, and the sections the terminal report prints under it."""
    days = [str(day) for day in rows["day"]] if not rows.empty else []
    cache_cutoff = cfg.metric_z_thresholds.get("cache_ratio", cfg.z_threshold)
    haiku_cutoff = cfg.metric_z_thresholds.get("haiku_fraction", cfg.z_threshold)
    flagged_cache = [str(row.day) for row in rows.itertuples(index=False)
                     if not _missing(row.cache_z) and float(row.cache_z) <= -cache_cutoff]
    flagged_haiku = [str(row.day) for row in rows.itertuples(index=False)
                     if not _missing(row.haiku_z) and float(row.haiku_z) >= haiku_cutoff]
    parts = [
        "<!DOCTYPE html>", '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>ccdrift report {escape(today.isoformat())}</title>", f"<style>{STYLE}</style>",
        "</head><body>",
        f"<h1>ccdrift report, {escape(today.isoformat())}</h1>",
        f'<p class="rule">The last {len(days)} complete UTC days with main-thread activity. A metric is '
        f"flagged once {cfg.deviant_bins} of any {cfg.flag_window} days in a row pass the cutoff: "
        f"z ≤ −{cache_cutoff:.1f} for the cache ratio, z ≥ +{haiku_cutoff:.1f} for the Haiku share. "
        "Shaded days belong to a recorded incident.</p>",
    ]
    if days:
        parts.append(chart(days, [row.cache_ratio for row in rows.itertuples(index=False)],
                           title="Cache read ratio per day", low=0.0, high=1.0,
                           marked=flagged_cache, shaded=incident_days, fmt=".2f"))
        parts.append(z_strip(days, [row.cache_z for row in rows.itertuples(index=False)],
                             -cache_cutoff, above=False))
        shares = [row.haiku_share for row in rows.itertuples(index=False)]
        parts.append(chart(days, shares, title="Haiku share of main-thread responses per day", low=0.0,
                           high=top(shares, 0.2), marked=flagged_haiku, shaded=incident_days, fmt=".2f"))
        parts.append(z_strip(days, [row.haiku_z for row in rows.itertuples(index=False)],
                             haiku_cutoff, above=True))
        parts.append(table(rows))
    incidents = [incident_line(incident, cost) for incident, cost in entries]
    parts.append(section("Incidents", incidents or ["none yet"]))
    legacy = [f"{label} from {day}" for metric, label in INCIDENT_METRICS.items() for day in reported.get(metric, [])]
    if legacy:
        parts.append(section("Flags reported before ccdrift followed incidents", legacy))
    for heading, body in blocks([*settings_lines(list(summary)), *extra]):
        parts.append(section(heading, body))
    parts.append(f"<footer>Written by ccdrift {escape(version)} on {escape(today.isoformat())} from the "
                 f"transcripts in {escape(source)}. This page holds local paths, and nothing left this "
                 "machine to make it.</footer>")
    parts.append("</body></html>")
    return "\n".join(part for part in parts if part) + "\n"
