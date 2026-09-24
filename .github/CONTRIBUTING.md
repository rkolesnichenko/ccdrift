# Contributing to ccdrift

Thanks for helping. ccdrift is maintained by one person, so a small PR with a clear reason
lands faster than a large one. For anything bigger than a fix, open an issue first.

## Privacy first

ccdrift reads Claude Code transcripts, and yours hold your code and prompts. Never put real
transcript content, paths, session ids or project names in an issue, a PR or a test fixture.
`ccdrift peek`, `ccdrift report --json` and `ccdrift cost --json` print output that is safe to share.

## Running the tests

```sh
uv run --group dev --group lab pytest
```

There is no install step. The `lab` group is needed even for `tests/` alone, because
collecting `lab/` imports matplotlib. A single test:
`uv run --group dev --group lab pytest tests/test_check.py::test_name`, or `-k <substring>`.

No test touches your real `~/.ccdrift`: an autouse fixture in tests/conftest.py points
`CCDRIFT_HOME` at a temporary directory. Commands you run by hand are not isolated, so set
`CCDRIFT_HOME` yourself when trying a change against real logs.

## Style

There is no formatter, linter or type checker, on purpose. Please don't run one: `black` or
`ruff format` at their defaults would rewrite the whole tree. Match the surrounding code by hand.

- Lines sit around 120 characters.
- Annotate everything, with `from __future__ import annotations` at the top of every
  substantive module.
- Wrap multi-name imports in parentheses, continuations aligned under the opening one.
- No em dashes anywhere, in prose, comments or docstrings. Use hyphens or en dashes.
- All user-visible wording lives in `src/ccdrift/texts.py`, except option help, which stays beside its
  option in cli.py; text other programs read (the crontab marker, unit files, the page's CSS, the
  AppleScript, Claude Code's banners) stays where it is. tests/test_command_texts.py enforces it.
- A constant that came from a measurement carries a comment with that measurement.
- Test names are full English sentences.
- Build transcript fixtures with `tests.helpers.line()`, never hand-rolled JSON. It writes
  JSONL the way Claude Code does. Tests pin behavior at frozen dates (`helpers.T0`).

## Rules that fail silently

- **Changing what `parse_file` returns** means bumping `PARSER_VERSION` in history.py and
  extending its numbered comment list. Otherwise unchanged transcripts keep stale rows.
- **A new log field variant** goes in the `CANDIDATES` dict in logs.py, never parsed ad hoc.
  It is a parser change, so bump `PARSER_VERSION` too.
- **A new history column** needs an additive `ALTER TABLE` branch in `_migrate` and an edit
  to `SCHEMA`.
- **`ccdrift status --short` stays import-light.** It runs on every status-line refresh and
  must not import pandas or numpy.
- **`report --json`, `cost --json` and `incident draft` emit aggregates only.** `peek` shows
  text, ids and paths only as their length, and a content block as its type and the size of
  the rest.
- **`report --html` stays self-contained**: no scripts, no network fetches, everything escaped.
- **Output is deterministic.** Sort every frame with `kind="stable"` before any positional or
  first/last logic, and seed lab sweeps explicitly.
- **A new alert kind** needs its own key in `new_state()` in state.py and a suppression rule.
- **Thresholds are measured, not chosen.** Changing a shipped cutoff means re-running the
  matching sweep in `lab/` and recording the new number in docs/findings.md. Only the sweep,
  run by hand on real logs, says whether the shipped setting still passes: the gate tests in
  `lab/test_*.py` exercise each gate's logic on synthetic inputs and stay green either way.
  See lab/README.md.

## Pull requests

CI runs the suite on Linux and macOS with Python 3.10 and 3.13, and all four jobs must pass.
PRs are squash-merged, so the PR title becomes the commit subject: write it the way the
history reads, as a sentence saying what the change does.

If you use Claude Code, the checked-in `.claude/settings.json` wires two hooks that warn
(never block) on an em dash, on a heavy import on the status line path, on a copy of the
sdist's paths that has fallen behind `only-include`, and on a logs.py change without a
`PARSER_VERSION` bump. They need `jq`.
