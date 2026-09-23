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
from ccdrift.texts import BEFORE_DAYS

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
    happened (a request a subagent made is one Claude Code made), while responses and
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
                 louder: Optional[Callable[[dict[str, Any], pd.Series], bool]] = None,
                 ) -> Iterator[tuple[pd.Series, pd.DataFrame, list[dict[str, Any]]]]:
    """The days within RECENT_DAYS that `hit` accepts, each with the active days before it
    that `hit` judged it against and the reported episodes whose word still covers it,
    skipping the days an episode already covers. `quiet_days` says whether a day too quiet
    to be active may be judged: the harder the API fails, the fewer responses that day
    holds, so the worst day of an outage can be too quiet to count. The days before are the
    active ones either way: a quiet day is judged, never a baseline.
    An episode covers a day within SPELL_DAYS of it, and a day that
    `ongoing(episode, row, before)` says merely carries on what the episode reported,
    however long ago that was, so one lasting regression alerts once rather than every
    SPELL_DAYS. `louder(episode, row)` says the day stands so far above what the episode
    reported that it is the regression deepening rather than that regression carrying on; a
    day louder than every episode covering it is yielded whatever those episodes would
    otherwise have said, because both kinds of cover exist to stop one level being reported
    twice and such a day is not that level.
    A generator on purpose: a rule records each episode as it takes it, so the next day's
    suppression test sees it, and one run over a fortnight (the first check after an
    upgrade, or after days with the machine off) alerts once per spell, just as a
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
        # One spell of failures alerts once: a day within SPELL_DAYS of a reported episode
        # belongs to it, as does one carrying its run on. A later burst is judged on its
        # own, and the ratio test keeps one that isn't worse from alerting again.
        spell = (date.fromisoformat(day) - timedelta(days=SPELL_DAYS)).isoformat()
        covering = [episode for episode in state[state_key]
                    if episode["since"] >= spell
                    or (ongoing is not None and ongoing(episode, candidates.loc[i], before))]
        if covering and not (louder is not None
                             and all(louder(episode, candidates.loc[i]) for episode in covering)):
            continue
        yield candidates.loc[i], before, covering


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
    for row, before, _ in _judged_days(counts, "failed_requests", state, today, hit, quiet_days=True):
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


def _day_share(row: pd.Series) -> float:
    """The share of one day's responses that stopped at the token limit or refused."""
    return int(row["truncated"] + row["refused"]) / int(row["responses"])


def _episode_share(episode: dict[str, Any]) -> float:
    """The share a reported episode said was cut short: the number the owner was given."""
    return episode["cut"] / episode["responses"] if episode["responses"] else 0.0


def _worst_share(frame: pd.DataFrame) -> float:
    """The worst of those shares, 0.0 over no days at all."""
    return float(_cut_shares(frame).max()) if len(frame) else 0.0


def _usual_days(before: pd.DataFrame, share: float) -> pd.DataFrame:
    """The days before a candidate that stand for its usual level: `before` without the
    unbroken run of days at or above `share` that ends at its latest day. A day inside the
    same run of bad days is the regression, not the usual level: the idea
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
    defaults. A day standing CUT_RATIO above the share a covering episode reported is the
    regression deepening rather than that episode carrying on, and is reported again,
    carrying "worse_than", the level and day it escalated from."""
    def hit(row: pd.Series, before: pd.DataFrame) -> bool:
        cut = int(row["truncated"] + row["refused"])
        today_share = _day_share(row)
        usual = max(CUT_USUAL, _worst_share(_usual_days(before, share)))
        return cut >= floor and today_share >= share and today_share >= CUT_RATIO * usual

    def ongoing(episode: dict[str, Any], row: pd.Series, before: pd.DataFrame) -> bool:
        """Whether this day only carries on the regression `episode` reported: the episode
        is still among the days judged, and every judged day from its day to this one
        stayed at or above `share`. One regression is one alert while it lasts. An episode
        whose day has left the comparison window says nothing about this day. Otherwise a
        cut-short alert from months ago would silence a new regression for good, once the
        window happened to be all bad, so a run outliving BEFORE_DAYS is reported again,
        about every BEFORE_DAYS: a second word after a fortnight beats silence."""
        if before.empty or episode["since"] < str(before["day"].astype(str).iloc[0]):
            return False
        carried = before[before["day"].astype(str) >= episode["since"]]
        return bool(len(carried)) and bool((_cut_shares(carried) >= share).all())

    def louder(episode: dict[str, Any], row: pd.Series) -> bool:
        """Whether this day stands CUT_RATIO above the share `episode` reported: the same
        factor a first alert needs over its baseline, so no second number decides this.
        A regression that deepens that far is not the reported one carrying on, and says so
        whatever would have held it: both the spell and the run exist to stop one level
        being reported twice, and this day is not that level. Each further word costs
        another tripling, so a run first reported at 0.6% can speak four more times before
        it is cutting every response short."""
        return _day_share(row) >= CUT_RATIO * _episode_share(episode)

    new = []
    for row, before, covering in _judged_days(counts, "cut_short", state, today, hit,
                                              ongoing=ongoing, louder=louder):
        usual = _usual_days(before, share)
        episode = {"since": str(row["day"]), "days": [str(row["day"])],
                   "cut": int(row["truncated"] + row["refused"]), "truncated": int(row["truncated"]),
                   "refused": int(row["refused"]), "responses": int(row["responses"]),
                   "before_share": _worst_share(usual),
                   # The days this day's own run took out of the comparison, so the alert
                   # can't call a fortnight clean that wasn't.
                   "run_days": len(before) - len(usual),
                   "reported_on": today.isoformat()}
        if covering:
            # The last word said about this regression, which this day stands three times
            # above. The alert names it rather than a baseline the run left behind days
            # ago, and the next escalation is judged against this episode in turn.
            last = max(covering, key=lambda reported: str(reported["since"]))
            episode["worse_than"] = {"share": _episode_share(last), "since": str(last["since"])}
        state["cut_short"].append(episode)
        new.append(episode)
    return new


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


def week_failures(counts: pd.DataFrame, days: Sequence[str]) -> tuple[int, int]:
    """The failed requests that count and the responses cut short on `days`, for the
    weekly summary (texts.week_failures_text words them)."""
    summary = failure_summary(counts, days)
    if summary is None:
        return 0, 0
    return sum(count for kind, count in summary["kinds"].items() if kind in COUNTED), summary["cut"]
