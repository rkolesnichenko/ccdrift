"""Failed requests and responses cut short: the error banners Claude Code writes when a
request fails, the retries it logs, and the responses that stop at the token limit. Both
are far too rare for a usual rate, so neither is judged like the cache metric: a day
alerts when it stands well above the days before it."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Callable, Iterator, Optional, Sequence

import pandas as pd

from ccdrift.logs import outside_sdk
# BEFORE_DAYS, the window a day is judged against, lives in texts with before_text: the
# status line names it too, and nothing that only prints should have to import pandas.
from ccdrift.texts import BEFORE_DAYS, before_text, kinds_text, run_text

# Every kind parse_file records, and those a rule counts: a banner blaming the user's
# own Mac for going to sleep is no drift, so it is reported but never alerts.
KINDS = ("overloaded", "stream", "other", "retry", "slept")
COUNTED = ("overloaded", "stream", "other", "retry")
FAILURE_DAY_COLUMNS = ["day", "responses", "requests", "truncated", "refused", *KINDS]

REQUEST_FLOOR = 5      # counted failures for a day to alert
REQUEST_RATIO = 2      # times the busiest of the days before, which counts as at least 1
CUT_FLOOR = 5          # responses that stopped at the token limit or refused
CUT_SHARE = 0.005      # of that day's main-thread responses
CUT_RATIO = 3          # times the worst share of the days before, floored at CUT_USUAL
CUT_USUAL = 0.001      # the share a quiet day is taken to have, so one cut day can't mask the next
ACTIVE_RESPONSES = 50  # main-thread responses for a day to count as active
MIN_BEFORE_DAYS = 5    # judged days to compare with, among the BEFORE_DAYS before
RECENT_DAYS = 14
SPELL_DAYS = 3         # days after a reported episode that belong to the same spell


def judged_failures(failures: pd.DataFrame, today: date) -> pd.DataFrame:
    """Failed requests of complete UTC days, without Agent SDK sessions. Subagent
    failures stay in: the report counts them, and only the rules leave them out."""
    if failures.empty:
        return failures
    return failures[(failures["day"].astype(str) < today.isoformat()) & outside_sdk(failures)]


def failure_counts(failures: pd.DataFrame, turns: pd.DataFrame) -> pd.DataFrame:
    """Per day over the judged turns' days and the judged failures' days together:
    main-thread responses, failures of each kind, the counted ones together, and the
    responses that stopped at the token limit or refused. Failures count wherever they
    happened — a request a subagent made is one Claude Code made — while responses and
    cut-short responses stay main-thread, as every other daily verdict is. A day whose
    requests failed has few responses to show for them, and one whose requests all failed
    has none: such a day still gets a row, so a rule can judge it, though only active days
    are ever the baseline another day is judged against."""
    turn_days = turns["day"].astype(str) if not turns.empty else pd.Series(dtype="object")
    fail_days = failures["day"].astype(str) if not failures.empty else pd.Series(dtype="object")
    index = pd.Index(sorted(set(turn_days) | set(fail_days)), name="day")
    if index.empty:
        return pd.DataFrame(columns=FAILURE_DAY_COLUMNS)

    def per_day(values: pd.Series, days: pd.Series) -> pd.Series:
        """`values` summed per day of `days`, 0 on the days holding none of them."""
        if days.empty:
            return pd.Series(0, index=index, dtype=int)
        return values.groupby(days).sum().reindex(index, fill_value=0)

    stop = turns["stop_reason"].astype("string") if "stop_reason" in turns else pd.Series(pd.NA, index=turns.index)
    counts = pd.DataFrame({"responses": per_day(pd.Series(1, index=turns.index), turn_days),
                           "truncated": per_day(stop == "max_tokens", turn_days),
                           "refused": per_day(stop == "refusal", turn_days)}, index=index)
    for kind in KINDS:
        chosen = failures[failures["kind"].astype(str) == kind] if not failures.empty else failures
        counts[kind] = (per_day(pd.Series(1, index=chosen.index), chosen["day"].astype(str))
                        if not chosen.empty else 0)
    counts["requests"] = sum(counts[kind] for kind in COUNTED)
    counts = counts.rename_axis("day").reset_index()
    return counts[FAILURE_DAY_COLUMNS].astype({name: int for name in FAILURE_DAY_COLUMNS[1:]})


def _judged_days(counts: pd.DataFrame, state_key: str, state: dict[str, Any], today: date,
                 hit: Callable[[pd.Series, pd.DataFrame], bool], quiet_days: bool = False,
                 ongoing: Optional[Callable[[dict[str, Any], pd.Series, pd.DataFrame], bool]] = None,
                 ) -> Iterator[tuple[pd.Series, pd.DataFrame]]:
    """The days within RECENT_DAYS that `hit` accepts, each with the active days before it
    that `hit` judged it against, skipping those a reported episode already covers.
    `quiet_days` says whether a day too quiet to be active may be judged: the harder the
    API fails, the fewer responses that day holds, so the worst day of an outage can be
    too quiet to count. The days before are the active ones either way — a quiet day is
    judged, never a baseline. `ongoing(episode, row, before)` says whether a day merely
    carries on what `episode` already reported; such a day is skipped however long ago the
    episode was, so one lasting regression alerts once rather than every SPELL_DAYS.
    A generator on purpose: a rule records each episode as it takes it, so the next day's
    suppression test sees it, and one run over a fortnight — the first check after an
    upgrade, or after days with the machine off — alerts once per spell, just as a
    day-by-day sequence of runs would."""
    if counts.empty:
        return
    active = counts[counts["responses"] >= ACTIVE_RESPONSES].reset_index(drop=True)
    active_days = active["day"].astype(str)
    candidates = counts.reset_index(drop=True) if quiet_days else active
    days = candidates["day"].astype(str)
    since = (today - timedelta(days=RECENT_DAYS)).isoformat()
    for i in range(len(candidates)):
        day = str(days[i])
        if day < since:
            continue
        earliest = (date.fromisoformat(day) - timedelta(days=BEFORE_DAYS)).isoformat()
        before = active[(active_days < day) & (active_days >= earliest)]
        if len(before) < MIN_BEFORE_DAYS or not hit(candidates.loc[i], before):
            continue
        # One spell of failures alerts once: a day within SPELL_DAYS of a reported
        # episode belongs to it. A later burst is judged on its own, and the ratio
        # test keeps one that isn't worse from alerting again.
        spell = (date.fromisoformat(day) - timedelta(days=SPELL_DAYS)).isoformat()
        if any(episode["since"] >= spell for episode in state[state_key]):
            continue
        if ongoing is not None and any(ongoing(episode, candidates.loc[i], before)
                                       for episode in state[state_key]):
            continue
        yield candidates.loc[i], before


def failing_requests(counts: pd.DataFrame, state: dict[str, Any], today: date,
                     floor: int = REQUEST_FLOOR, ratio: int = REQUEST_RATIO) -> list[dict[str, Any]]:
    """Days with at least `floor` failed requests, at least `ratio` times the busiest of
    the active days before them; each is recorded in state["failed_requests"], once per
    spell. A day too quiet to be active is judged too: a day whose requests kept failing
    has little else to show. The lab tries other floors and ratios; the check keeps the
    defaults."""
    def hit(row: pd.Series, before: pd.DataFrame) -> bool:
        return row["requests"] >= floor and row["requests"] >= ratio * max(1, int(before["requests"].max()))

    new = []
    for row, before in _judged_days(counts, "failed_requests", state, today, hit, quiet_days=True):
        day = str(row["day"])
        episode = {"since": day, "days": [day], "requests": int(row["requests"]),
                   "kinds": {kind: int(row[kind]) for kind in COUNTED if int(row[kind])},
                   "before": int(before["requests"].max()) if len(before) else 0,
                   "reported_on": today.isoformat()}
        state["failed_requests"].append(episode)
        new.append(episode)
    return new


def _cut_shares(frame: pd.DataFrame) -> pd.Series:
    """The share of each day's responses that stopped at the token limit or refused."""
    return (frame["truncated"] + frame["refused"]) / frame["responses"]


def _worst_share(frame: pd.DataFrame) -> float:
    """The worst of those shares, 0.0 over no days at all."""
    return float(_cut_shares(frame).max()) if len(frame) else 0.0


def _usual_days(before: pd.DataFrame, share: float) -> pd.DataFrame:
    """The days before a candidate that stand for its usual level: `before` without the
    unbroken run of days at or above `share` that ends at its latest day. A day inside the
    same run of bad days is the regression, not the usual level — the idea
    incidents.exclusions applies to the cache metric, in the small. Without it a
    regression that began on a day too small for the floor would set a bar the days
    carrying it on could never clear, so it would never be reported at all."""
    keep = len(before)
    if keep:
        shares = _cut_shares(before)
        while keep and float(shares.iloc[keep - 1]) >= share:
            keep -= 1
    return before.iloc[:keep]


def cut_short(counts: pd.DataFrame, state: dict[str, Any], today: date,
              floor: int = CUT_FLOOR, share: float = CUT_SHARE) -> list[dict[str, Any]]:
    """Days where at least `floor` responses stopped at the token limit or refused, on at
    least `share` of the day's responses and CUT_RATIO times the worst share of the active
    days before them that stand for the usual level (see _usual_days; a clean day counting
    as CUT_USUAL); each is recorded in state["cut_short"], once per run. Only active days
    are judged: this rule is a share of a day's responses, which a day too quiet to be
    active can't support. The lab tries other floors and shares; the check keeps the
    defaults."""
    def hit(row: pd.Series, before: pd.DataFrame) -> bool:
        cut = int(row["truncated"] + row["refused"])
        today_share = cut / int(row["responses"])
        usual = max(CUT_USUAL, _worst_share(_usual_days(before, share)))
        return cut >= floor and today_share >= share and today_share >= CUT_RATIO * usual

    def ongoing(episode: dict[str, Any], row: pd.Series, before: pd.DataFrame) -> bool:
        """Whether this day only carries on the regression `episode` reported: the episode
        is still among the days judged, and every judged day from its day to this one
        stayed at or above `share`. One regression is one alert while it lasts. An episode
        whose day has left the comparison window says nothing about this day — otherwise a
        cut-short alert from months ago would silence a new regression for good, once the
        window happened to be all bad — so a run outliving BEFORE_DAYS is reported again,
        about every BEFORE_DAYS: a second word after a fortnight beats silence."""
        if before.empty or episode["since"] < str(before["day"].astype(str).iloc[0]):
            return False
        carried = before[before["day"].astype(str) >= episode["since"]]
        return bool(len(carried)) and bool((_cut_shares(carried) >= share).all())

    new = []
    for row, before in _judged_days(counts, "cut_short", state, today, hit, ongoing=ongoing):
        usual = _usual_days(before, share)
        episode = {"since": str(row["day"]), "days": [str(row["day"])],
                   "cut": int(row["truncated"] + row["refused"]), "truncated": int(row["truncated"]),
                   "refused": int(row["refused"]), "responses": int(row["responses"]),
                   "before_share": _worst_share(usual),
                   # The days this day's own run took out of the comparison, so the alert
                   # can't call a fortnight clean that wasn't.
                   "run_days": len(before) - len(usual),
                   "reported_on": today.isoformat()}
        state["cut_short"].append(episode)
        new.append(episode)
    return new


def _on(versions: Sequence[str]) -> str:
    return f", on Claude Code {', '.join(versions)}" if versions else ""


def requests_message(episode: dict[str, Any], versions: Sequence[str]) -> str:
    named = kinds_text(episode["kinds"])
    before = before_text(episode["before"])
    return (f"{episode['requests']} requests failed on {episode['since']}"
            f"{f' ({named})' if named else ''}, {before}{_on(versions)}. Claude Code retries these itself; a run "
            "of them points at the API or your connection, not your setup.")


def cut_short_message(episode: dict[str, Any], versions: Sequence[str]) -> str:
    what = "stopped at the token limit or refused" if episode["refused"] else "stopped at the token limit"
    share = episode["cut"] / episode["responses"] if episode["responses"] else 0.0
    before = before_text(f"{episode['before_share']:.2%}" if episode["before_share"] > 0 else None)
    return (f"{episode['cut']} of {episode['responses']:,} main-thread responses {what} on {episode['since']} "
            f"({share:.2%}), {before}{run_text(episode.get('run_days', 0))}{_on(versions)}. "
            "A Claude Code update may have changed the output limit.")


def failure_summary(counts: pd.DataFrame, days: Sequence[str]) -> Optional[dict[str, Any]]:
    """Failures over `days`: how many of each kind, and how many responses were cut
    short; None when there were none."""
    window = counts[counts["day"].astype(str).isin(list(days))] if not counts.empty else counts
    if window.empty:
        return None
    kinds = {kind: int(window[kind].sum()) for kind in KINDS if int(window[kind].sum())}
    cut = int(window["truncated"].sum() + window["refused"].sum())
    if not kinds and not cut:
        return None
    return {"kinds": kinds, "cut": cut}


def failure_lines(summary: Optional[dict[str, Any]]) -> list[str]:
    """The report's failures line, starting with a blank line; empty without failures."""
    if summary is None:
        return []
    parts = [kinds_text(summary["kinds"])] if summary["kinds"] else []
    if summary["cut"]:
        parts.append(f"{summary['cut']} response{'' if summary['cut'] == 1 else 's'} cut short")
    return ["", "Failures over these days: " + ", ".join(parts)]


def digest_part(counts: pd.DataFrame, days: Sequence[str]) -> str:
    """The weekly summary's failures part: "no failed requests", or what there was."""
    summary = failure_summary(counts, days)
    counted = sum(count for kind, count in (summary or {"kinds": {}})["kinds"].items() if kind in COUNTED)
    if summary is None or (not counted and not summary["cut"]):
        return "no failed requests"
    parts = [f"{counted} failed request{'' if counted == 1 else 's'}"] if counted else []
    if summary["cut"]:
        parts.append(f"{summary['cut']} response{'' if summary['cut'] == 1 else 's'} cut short")
    return ", ".join(parts)
