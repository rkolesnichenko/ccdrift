---
name: threshold-change
description: The procedure for changing a shipped detection cutoff in ccdrift (cache z score, Haiku z score, CUSUM h or p1, the session-start step, the failure floor, share or ratio). Use before editing any constant that a lab gate backs, and when docs/findings.md needs to reflect a new measurement.
---

Every shipped cutoff is a measurement, not a preference. The rule is: re-run the sweep, record the number, keep the gate green. Do not change a constant on judgment alone.

## 1. Find the gate that backs it

docs/findings.md records one entry per numbered gate (G2, G3, G7-G13) and names the sweep behind it. Match the constant to its gate before touching anything. The sweeps live in lab/: `lab/harness.py`, `lab/early_warning.py`, `lab/loop_cache.py`, `lab/failures.py`, `lab/context.py`, `lab/session_start.py`.

## 2. Re-run the sweep

Run the matching sweep, not the whole harness, and keep the output aggregate. These read real transcripts:

```
uv run --group lab python -m lab.early_warning --incident 2026-08-16..2026-09-04
uv run --group lab python -m lab.loop_cache
uv run --group lab python -m lab.failures
uv run --group lab python -m lab.context
uv run --group lab python -m lab.session_start
uv run --group lab python lab/harness.py --synthetic --out ./out
```

Sweeps are seeded explicitly, so a re-run on the same corpus reproduces. If it does not, something broke determinism: a frame lost its `kind="stable"` sort, or a seed went missing. Fix that before reading the numbers.

## 3. Record the measurement

Update the gate's entry in docs/findings.md with the new number. Aggregates only, from the sweep you just ran. In the code, the constant carries a comment stating the measurement behind it; update that comment too, in the style of src/ccdrift/detector.py:20 or src/ccdrift/check.py:50.

## 4. Confirm the gate still passes

```
uv run --group dev --group lab pytest -q lab/
```

The gate tests in lab/test_*.py assert that the shipped setting still holds. A gate test that now fails means the change is not ready, not that the test needs relaxing.

## 5. Report

Give the old value, the new value, the sweep command you ran, its headline number, and the docs/findings.md line you edited.
