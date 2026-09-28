"""Settings Claude Code chooses: the cache tier and effort each model usually gets."""

from datetime import date

import pandas as pd

from ccdrift.settings import setting_changes, settings_summary
from ccdrift.state import new_state
from ccdrift.texts import change_line, change_message, settings_lines
from tests.helpers import nth_day


def responses(days, setting="cache_tier", model="claude-opus-5"):
    """Judged responses: each day maps a value of `setting` to how many responses had it."""
    rows = [{"day": nth_day(i), "model": model, "version": "2.1.226", setting: value}
            for i, counts in enumerate(days) for value, n in counts.items() for _ in range(n)]
    return pd.DataFrame(rows)


def test_a_two_day_move_to_the_5_minute_cache_is_reported_once():
    state = new_state()
    turns = responses([{"1h": 40}] * 14 + [{"5m": 40}] * 3)
    assert [(c["from"], c["to"], c["since"], c["days"]) for c in setting_changes(turns, state, date(2026, 9, 18))] \
        == [("1h", "5m", "2026-09-15", ["2026-09-15", "2026-09-16"])]
    assert setting_changes(turns, state, date(2026, 9, 18)) == []
    assert len(state["settings"]) == 1


def test_a_one_day_change_is_not_reported():
    turns = responses([{"1h": 40}] * 14 + [{"5m": 40}] + [{"1h": 40}] * 2)
    assert setting_changes(turns, new_state(), date(2026, 9, 18)) == []


def test_nothing_is_reported_without_a_usual_value():
    turns = responses([{"1h": 40}, {"5m": 40}] * 7 + [{"5m": 40}] * 3)
    assert setting_changes(turns, new_state(), date(2026, 9, 18)) == []


def test_quiet_days_dont_count_as_a_change():
    turns = responses([{"1h": 40}] * 14 + [{"5m": 10}] * 3)
    assert setting_changes(turns, new_state(), date(2026, 9, 18)) == []


def test_changes_from_weeks_ago_are_not_reported():
    turns = responses([{"1h": 40}] * 14 + [{"5m": 40}] * 3)
    assert setting_changes(turns, new_state(), date(2026, 10, 10)) == []


def test_an_effort_change_says_the_default_may_have_changed():
    change = setting_changes(responses([{"xhigh": 40}] * 14 + [{"high": 40}] * 2, setting="effort"),
                             new_state(), date(2026, 9, 17))[0]
    assert change_message(change, ["2.1.280 (since 09-15)"]) == (
        "Effort for claude-opus-5 changed from xhigh to high from 2026-09-15, on Claude Code 2.1.280 "
        "(since 09-15). If you didn't change it, Claude Code's default did.")


def test_a_cache_tier_change_names_both_caches():
    change = {"setting": "cache_tier", "model": "claude-opus-5", "from": "1h", "to": "5m", "since": "2026-09-20"}
    assert change_message(change, []) == \
        "Cache writes for claude-opus-5 moved from the 1-hour to the 5-minute cache from 2026-09-20."
    assert change_line(change) == "cache tier for claude-opus-5: 1h -> 5m from 2026-09-20"


def model_switch(setting="effort", before="xhigh", after="high", days_after=2):
    """14 days of claude-opus-5 at `before`, then `days_after` days of claude-opus-5-5 at `after`."""
    new = responses([{}] * 14 + [{after: 40}] * days_after, setting=setting, model="claude-opus-5-5")
    return pd.concat([responses([{before: 40}] * 14, setting=setting), new], ignore_index=True)


def test_a_setting_that_arrives_with_a_new_model_is_reported_for_the_main_thread():
    # On 2026-09-22 the main thread moved from claude-opus-5 at xhigh to claude-opus-5-5 at
    # high. Judged per model, one had no days after and the other none before, so nothing fired.
    changes = setting_changes(model_switch(), new_state(), date(2026, 9, 17))
    assert changes == [{"setting": "effort", "scope": "thread", "model": None, "from": "xhigh", "to": "high",
                        "since": "2026-09-15", "days": ["2026-09-15", "2026-09-16"],
                        "from_model": "claude-opus-5", "to_model": "claude-opus-5-5", "reported_on": "2026-09-17"}]
    assert change_message(changes[0], ["2.1.280 (since 09-15)"]) == (
        "Effort on the main thread changed from xhigh to high from 2026-09-15, on Claude Code 2.1.280 "
        "(since 09-15), as its model moved from claude-opus-5 to claude-opus-5-5. If you didn't change it, "
        "Claude Code's default did.")
    assert change_line(changes[0]) == "effort on the main thread: xhigh -> high from 2026-09-15"


def test_a_cache_tier_that_arrives_with_a_new_model_is_reported_for_the_main_thread_too():
    [change] = setting_changes(model_switch("cache_tier", "1h", "5m"), new_state(), date(2026, 9, 17))
    assert change_message(change, []) == ("Cache writes on the main thread moved from the 1-hour to the 5-minute "
                                          "cache from 2026-09-15, as its model moved from claude-opus-5 to "
                                          "claude-opus-5-5.")


def test_a_main_thread_change_on_one_model_names_no_model_move():
    change = {"setting": "effort", "scope": "thread", "model": None, "from": "xhigh", "to": "high",
              "since": "2026-09-15", "from_model": "claude-opus-5", "to_model": "claude-opus-5"}
    assert change_message(change, []) == ("Effort on the main thread changed from xhigh to high from 2026-09-15. "
                                          "If you didn't change it, Claude Code's default did.")


def test_a_change_on_one_model_is_reported_for_that_model_and_not_again_for_the_main_thread():
    state = new_state()
    changes = setting_changes(responses([{"xhigh": 40}] * 14 + [{"high": 40}] * 2, setting="effort"), state,
                              date(2026, 9, 17))
    assert [(c["model"], c.get("scope")) for c in changes] == [("claude-opus-5", None)]
    assert len(state["settings"]) == 1


def test_a_main_thread_change_already_reported_holds_back_the_same_change_on_one_model():
    # An earlier run reported the change for the main thread; the model's own judgement of the
    # same change, reached on a later run, isn't a second alert.
    state = new_state()
    state["settings"].append({"setting": "effort", "scope": "thread", "model": None, "from": "xhigh",
                              "to": "high", "since": "2026-09-15", "days": ["2026-09-15", "2026-09-16"],
                              "from_model": "claude-opus-5", "to_model": "claude-opus-5-5",
                              "reported_on": "2026-09-17"})
    turns = responses([{"xhigh": 40}] * 14 + [{"high": 40}] * 2, setting="effort")
    assert setting_changes(turns, state, date(2026, 9, 17)) == []


def test_a_main_thread_that_alternates_between_two_models_settings_raises_nothing():
    # Two guards keep it quiet, each enough alone: no usual value on a mixed baseline, and a
    # pair of days never both off it. Only removing both fails this test.
    turns = pd.concat([responses([{"xhigh": 40}, {}] * 8, setting="effort"),
                       responses([{}, {"high": 40}] * 8, setting="effort", model="claude-opus-5-5")],
                      ignore_index=True)
    assert setting_changes(turns, new_state(), date(2026, 9, 17)) == []


def test_the_settings_summary_lists_shares_and_day_to_day_changes():
    turns = pd.concat([responses([{"1h": 30}, {"5m": 30}]),
                       responses([{"xhigh": 30}] * 2, setting="effort")], axis=1)
    turns = turns.loc[:, ~turns.columns.duplicated()]
    summary = settings_summary(turns, [nth_day(0), nth_day(1)])
    assert summary == [{"model": "claude-opus-5", "responses": 60,
                        "shares": {"cache_tier": {"1h": 0.5, "5m": 0.5}, "effort": {"xhigh": 1.0},
                                   "speed": {"not logged": 1.0}, "service_tier": {"not logged": 1.0}},
                        "changes": [{"day": "2026-09-02", "setting": "cache_tier", "from": "1h", "to": "5m"}]}]
    assert settings_lines(summary) == [
        "",
        "Settings on the CLI main thread over these days (share of responses):",
        "  claude-opus-5: cache tier 1h 50%, 5m 50%; effort xhigh 100%; speed not logged 100%; "
        "service tier not logged 100%",
        "    2026-09-02: cache tier 1h -> 5m",
    ]


def test_a_change_names_the_new_value_even_when_the_old_one_is_still_the_most_common():
    # With three values on a day, the usual one can fall under half and still come first.
    turns = responses([{"high": 40}] * 10 + [{"high": 18, "medium": 12, "low": 10}] * 2, setting="effort")
    assert [(c["from"], c["to"]) for c in setting_changes(turns, new_state(), date(2026, 9, 13))] == \
        [("high", "medium")]
