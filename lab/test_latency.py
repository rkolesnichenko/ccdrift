"""The turn latency spike: daily medians, the flag rule, and planted slowdowns."""

import pandas as pd

from lab.latency import daily_latency, flag_days, main, sweep_latency
from tests.helpers import DAY, at, nth_day, turn_duration, write


def durations(days=30, per_day=20, seconds=(60, 62, 58, 61, 59)):
    """Main-thread CLI turns of 4 messages whose durations cycle through `seconds`,
    so every day has the same median."""
    return pd.DataFrame([{"day": nth_day(d), "duration_ms": seconds[(d + k) % len(seconds)] * 1000.0,
                          "message_count": 4, "is_sidechain": False, "entrypoint": "cli"}
                         for d in range(days) for k in range(per_day)])


def test_daily_latency_uses_main_thread_cli_turns_only():
    extra = pd.DataFrame([
        {"day": nth_day(0), "duration_ms": 999_000.0, "message_count": 1, "is_sidechain": True, "entrypoint": "cli"},
        {"day": nth_day(0), "duration_ms": 999_000.0, "message_count": 1, "is_sidechain": False,
         "entrypoint": "sdk-py"}])
    daily = daily_latency(pd.concat([durations(days=1, per_day=5), extra], ignore_index=True))
    assert daily[["day", "turns", "duration_s", "per_message_s"]].values.tolist() == [["2026-09-01", 5, 60.0, 15.0]]


def test_steady_latency_raises_no_flags():
    assert not any(flag_days(daily_latency(durations())["duration_s"].to_numpy(dtype=float)))


def test_a_doubled_latency_is_caught_from_every_start():
    res = sweep_latency(durations(), "per_message_s", factors=(2.0,), n_starts=4)
    assert len(res) == 4
    assert res["caught"].all()
    assert res.attrs["clean_flag_days"] == []


def test_latency_spike_prints_milliseconds_per_message(tmp_path, capsys):
    # messageCount counts the whole session so far, often thousands of messages, so
    # seconds per message printed as 0.0 or 0.1.
    for d in range(3):
        write(tmp_path / "logs" / f"s{d}.jsonl",
              [turn_duration(at(d * DAY + 60 * k), 60_000, 1_500, sid=f"s{d}", uuid=f"d{d}-{k}") for k in range(5)])
    assert main(["--source", str(tmp_path / "logs"), "--starts", "1"]) == 0
    out = capsys.readouterr().out.splitlines()
    i = out.index("=== median milliseconds per message (messageCount: the session's messages so far) ===")
    assert out[i + 1] == "daily median 40.0, range 40.0-40.0, spread (MAD/median) 0.00"


def test_latency_spike_runs_on_synthetic_logs(capsys):
    assert main(["--synthetic", "--starts", "2"]) == 0
    assert "smallest slowdown caught from every start" in capsys.readouterr().out
