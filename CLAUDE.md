# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Tests: `uv run --group dev --group lab pytest` (578 tests, ~35s). `testpaths = ["tests", "lab"]`, so a bare `pytest` runs both suites.
- Single test: `uv run --group dev --group lab pytest tests/test_check.py::test_name`, or `-k <substring>`. No install step is needed; `pythonpath = [".", "src"]` is set in pyproject.toml.
- The `lab` group is required even when running only `tests/`: lab collection imports matplotlib.
- Build: `uv build`. There is no lint, format or type-check step, by design (see Style).

## Style

- Do not run a formatter. No ruff/black/mypy config exists and none is wanted. Lines sit around 120 chars with outliers to 142, so `ruff format` or `black` at defaults would rewrite the tree. Match the surrounding style by hand.
- Annotate everything; `from __future__ import annotations` at the top of every substantive module. The existing `# type: ignore[...]` codes are documentation, not suppressions for a checker that runs.
- Wrap multi-name imports in parens with continuations aligned under the opening paren, not 4-space indented.
- No em dashes anywhere: prose, comments or docstrings. Use hyphens or en dashes.
- A constant that came from a measurement carries a comment with that measurement (detector.py:20, logs.py:104, check.py:50). Never add a bare magic number.
- All user-visible wording and formatting lives in src/ccdrift/texts.py, so the check, report and status line stay consistent and the import stays light.
- Test names are full English sentences.
- Two hooks in `.claude/hooks/`, wired by the checked-in `.claude/settings.json`, warn and never block: write-guard.sh on an em dash, a heavy import on the status line path, or a copy of the sdist's paths that has fallen behind `only-include`, and parser-version-guard.sh on a `logs.py` change with no `PARSER_VERSION` bump. Both need `jq`.

## Invariants

Each of these fails silently when ignored.

- **Bump `PARSER_VERSION`** (history.py:34) whenever `parse_file` output changes, and extend its numbered comment list. Without the bump, `History.update` skips unchanged transcripts and the store keeps stale rows.
- **Log-format drift goes in exactly one place**: the `CANDIDATES` dict at logs.py:31. Add a new field variant there, never parse it ad hoc. That is a parser change, so bump `PARSER_VERSION` as well.
- **`SCHEMA_VERSION`** (history.py:26): a new column needs both an additive `ALTER TABLE` branch in `_migrate` and an edit to `SCHEMA`. The version check runs before any DDL on purpose; a store written by a newer ccdrift is refused, not migrated.
- **`STATE_VERSION` and `CONTEXT_RULE`** (state.py:21, state.py:23): changing the session-start judging rule requires bumping `CONTEXT_RULE` so existing states are re-judged once. A state from a newer ccdrift is never overwritten.
- **The status line stays import-light**: `ccdrift status --short` runs on every status-line refresh and must import neither pandas nor numpy (pandas alone costs ~0.3s). cli.py defers heavy imports into the command branch and imports only state and texts at module level. Guarded by tests/test_status.py:197.
- **Privacy**: `report --json` and `incident draft` emit aggregates only, no paths, session ids or project names. `report --html` is the exception and is not shareable: it names project folders exactly as the terminal report does, and the page's own last line says so (tests/test_page.py:145). `peek` prints text, ids and paths as their length only, and deliberately omits the transcript path because it names the project folder. State, history and log are forced to owner-only 0600 through `make_private`.
- **`report --html` is self-contained**: no scripts, no network fetches, everything escaped.
- **Deterministic output**: sort every frame with `kind="stable"` before any positional or first/last logic, and seed lab sweeps explicitly. Breaking this breaks reproducibility of the gates.
- **Alert-once**: a new alert kind needs its own key in `new_state()` (state.py:44) plus a suppression rule.
- **Thresholds are measured, not chosen.** Changing a shipped cutoff means re-running the matching sweep in lab/ and recording the new number in docs/findings.md. The gate tests in lab/test_*.py assert that the shipped setting still passes.

## Testing

- tests/conftest.py has one autouse fixture pointing `CCDRIFT_HOME` at a tmp dir, so no test touches the real `~/.ccdrift`.
- Build log fixtures through `tests.helpers.line()`, which writes JSONL exactly as Claude Code does: one content block per line, usage repeated per line, fields omitted when None to mimic older versions. Both suites import from `tests.helpers`. Do not hand-roll transcript JSON.
- Tests pin behavior at frozen dates (`helpers.T0`).
- Scheduler changes are exercised for real in CI on Linux (install, status, remove round trip), so schedule.py can fail CI even when the unit tests pass.
- CI allows only GitHub-owned actions plus `astral-sh/setup-uv` and `astral-sh/attest-action`, each pinned by full SHA. A new action needs both a SHA pin and an allowlist entry in the repository's Actions settings, or the run fails before its first step. Dependabot moves the pins weekly.

## Release

Tag-triggered Trusted Publishing over OIDC, no secrets.

First check that anything ships: `git diff --stat v<previous>..HEAD -- src tests lab docs/findings.md README.md LICENSE pyproject.toml`, those paths being `only-include` in pyproject.toml, so change one and change the other. An empty result means the distributions would differ from the last by the version string alone, and a PyPI version can never be reused. Work on `.claude/`, `.github/` and this file never reaches the package and is finished once it is on `main`.

1. Bump `__version__` in src/ccdrift/__init__.py to match the intended tag. The workflow installs wheel and sdist in isolation and fails unless each reports `ccdrift <tag>`.
2. Update docs/findings.md, and refresh docs/what-ccdrift-caught.html (hand-maintained, no generator in the repo; it fetches nothing over the network, so a font or library goes inline or not at all).
3. Land the bump through a PR, like every change to `main`, then tag the squashed commit on `main`: `git tag v0.11.0 && git push origin v0.11.0`.
4. `publish` waits for the maintainer's approval in the `pypi` environment and does not wait for `tests.yml`. Approve only once the tag's tests are green, and only when told to. Never `gh run watch` the release run: `publish` holds at `waiting` until the review, so the watch blocks on a person. Watch the tag's `tests.yml`, and read the release run with `gh run list`.
5. Write the GitHub Release notes by hand. There is no CHANGELOG file; the Changelog URL points at GitHub Releases.

## Layout

- src/ccdrift/ is the shipped package: flat, no subpackages.
- lab/ is the research harness, with its own numbered gate tests (G2, G3, G7-G13) matching the entries in docs/findings.md. It is not installed; run it from a clone, for example `uv run --group lab python lab/harness.py --synthetic --out ./out`. See lab/README.md for the full list.
- Optional env vars, none required: `CCDRIFT_HOME` (default ~/.ccdrift), `CLAUDE_CONFIG_DIR` (default ~/.claude/projects), `XDG_CONFIG_HOME` (systemd user unit dir).
