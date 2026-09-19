---
name: verify
description: Run the full ccdrift suite plus the privacy, determinism and status-line-weight invariant checks, and report only what was actually observed. Use before claiming work is complete, before tagging a release, and after any change to logs.py, history.py, state.py, report.py, draft.py or page.py.
---

Run each step and paste the real output. Do not summarize a step you did not run, and do not call anything passing without the line that says so.

## 1. Full suite, CI parity

```
uv run --group dev --group lab pytest -q
```

Expect 578 passing at present, roughly 35s. The `lab` group is required even if only `tests/` matters; lab collection imports matplotlib. If the count dropped, find out which test disappeared before doing anything else.

## 2. Status line stays import-light

`ccdrift status --short` runs on every status-line refresh, so the status path must import neither pandas nor numpy. pandas alone costs about 0.3s.

```
uv run --group dev --group lab pytest -q tests/test_status.py::test_status_short_loads_neither_pandas_nor_numpy
```

If a heavy import crept in, the fix is to defer it into the command branch in cli.py, not to relax the test.

## 3. Privacy invariants

These emit aggregates only: no paths, session ids or project names. `peek` reports text, ids and paths as lengths, and omits the transcript path because it names the project folder.

```
uv run --group dev --group lab pytest -q tests/test_report.py tests/test_draft.py tests/test_page.py tests/test_cli.py
```

## 4. Determinism

Every frame is sorted with `kind="stable"` before any positional or first/last logic, and lab sweeps seed explicitly. If the diff touched a sort, a groupby or a sweep, run the lab gates and confirm the numbers still match docs/findings.md:

```
uv run --group dev --group lab pytest -q lab/
```

## 5. Report

State the suite count and duration you saw, name each invariant check you ran, and list anything you skipped and why. If any step failed, stop and report the failure rather than continuing to the next step.
