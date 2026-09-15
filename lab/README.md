# Lab

The research harness behind [the findings](../docs/findings.md). It plants controlled
changes in real or synthetic Claude Code logs to measure how small a change the daily
detector catches, and how fast a turn-by-turn CUSUM catches one against its false
alarms. It isn't installed with the package; run it from a clone.

```sh
# Smoke-test on synthetic logs
uv run --group lab python lab/harness.py --synthetic --out ./out

# Daily metrics, flags and plots for your own logs
uv run --group lab python lab/harness.py --out ./out

# Smallest planted cache drop caught, from up to 10 starting days,
# leaving out a known incident
uv run --group lab python lab/harness.py --sweep cache --incident 2026-08-16..2026-09-04 --out ./out

# Turn-by-turn detection time against false alarms
uv run --group lab python lab/harness.py --stream cache --incident 2026-08-16..2026-09-04 --out ./out
```

`--help` lists every option, including `--main-thread-only`, `--since`, `--until` and
the detector settings.
