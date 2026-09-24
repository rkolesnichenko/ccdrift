## What and why



## Checklist

- [ ] `uv run --group dev --group lab pytest` passes locally.
- [ ] If `parse_file` output changed: `PARSER_VERSION` is bumped and its numbered list extended.
- [ ] If a shipped cutoff changed: the lab sweep was re-run and docs/findings.md records the new number.
- [ ] User-visible wording lives in `src/ccdrift/texts.py` (option help and formats other programs read excepted).
- [ ] No em dashes, no formatter run, test names are full sentences.
- [ ] Nothing here comes from a real transcript: no paths, session ids, project names or text.
