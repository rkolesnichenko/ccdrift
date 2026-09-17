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

# Logged thinking token counts against the estimate, leaving out a known incident
uv run --group lab python lab/harness.py --compare-thinking --incident 2026-08-16..2026-09-04

# Effort sweep with effort computed from logged thinking token counts
uv run --group lab python lab/harness.py --sweep effort --thinking logged --incident 2026-08-16..2026-09-04 --out ./out

# Turn latency: day-to-day spread and the smallest slowdown caught
uv run --group lab python -m lab.latency --incident 2026-08-16..2026-09-04

# G2: session-start size per version, and whether its alerts would be sound
uv run --group lab python -m lab.session_start

# G3: an early warning on cache misses — false alarms, time to catch, the real regression
uv run --group lab python -m lab.early_warning --incident 2026-08-16..2026-09-04

# G8 and G9: an early warning on tool-loop cache misses, on the main thread and in subagents
uv run --group lab python -m lab.loop_cache --incident 2026-08-16..2026-09-04

# G7: do built-in subagents keep one model?
uv run --group lab python -m lab.subagent_models
```

`--help` lists every option, including `--main-thread-only`, `--since`, `--until` and
the detector settings.
