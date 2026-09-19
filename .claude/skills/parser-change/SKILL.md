---
name: parser-change
description: The checklist for changing how ccdrift reads Claude Code transcripts. Use whenever touching parse_file, the CANDIDATES dict, or any field extraction in src/ccdrift/logs.py, or when Claude Code has changed its log format and a new field variant needs absorbing.
---

Transcript parsing has three coupled pieces. Changing one without the others fails silently: the store keeps stale rows and no test complains.

## 1. Absorb drift in one place only

New or renamed fields go in the `CANDIDATES` dict at src/ccdrift/logs.py:31, as an extra dotted path on the existing key. That dict is the single place schema drift is absorbed; `field_get` resolves by first match. Never parse a variant ad hoc at the call site, and never add a second lookup path elsewhere in the module.

## 2. Bump PARSER_VERSION

If `parse_file`'s output changes at all, including a new candidate path, bump `PARSER_VERSION` at src/ccdrift/history.py:34 and add a numbered line to the comment list above it saying what changed.

Why it matters: `History.update` compares the stored parser version and skips transcripts that look unchanged. Without the bump, every transcript already on disk keeps its old rows and the change appears to do nothing.

## 3. Build fixtures through the helper

Write test transcripts with `tests.helpers.line()`, never hand-rolled JSON. It reproduces Claude Code's real shape: one content block per line, `message.id` and usage repeated per line, fields omitted when None so older versions are represented. `helpers.BANNER_TEXTS` and `helpers.api_error()` reproduce the API-error banners. Both suites import from `tests.helpers`.

Cover the old variant and the new one in the same test, so the fallback chain stays proven rather than assumed.

## 4. Verify

```
uv run --group dev --group lab pytest -q
```

Then check the field-gap path specifically: a field that has never been logged before must not read as a gap. If the change affects what counts as a gap or a drop, confirm the alert-once keys in `new_state()` (src/ccdrift/state.py:44) still cover it.

## 5. Report

Name the candidate paths added, the new `PARSER_VERSION`, and the numbered comment line you wrote.
