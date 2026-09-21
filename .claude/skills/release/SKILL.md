---
name: release
description: Cut a ccdrift release. Bumps __version__ to match the tag, verifies, tags and pushes, then drafts the GitHub Release notes. Takes the target version as an argument, for example /release 0.11.0.
disable-model-invocation: true
---

Target version: $ARGUMENTS

Publishing is tag-triggered Trusted Publishing over OIDC. The tag is the trigger and it cannot be taken back cleanly, so every check happens before the push.

## 1. Preflight

Confirm and report each:

```
git status --porcelain
git rev-parse --abbrev-ref HEAD
git remote -v
git tag --list 'v*' | tail -5
```

- Working tree clean, on `main`.
- `origin` must point at `https://github.com/rkolesnichenko/ccdrift.git`. If `git remote -v` is empty or names anything else, stop and ask before going further.
- The target tag must not already exist.

If no version was given in the argument, ask for it. Do not infer one.

## 2. Bump the version

Set `__version__` in src/ccdrift/__init__.py to the target, with no leading `v`. The workflow installs the wheel and the sdist in isolation and fails unless each reports `ccdrift <tag>` exactly, so a mismatch burns the tag.

## 3. Docs that ship with the version

- docs/findings.md reflects the shipped thresholds. If any cutoff changed in this release, its entry must carry the current measurement.
- docs/what-ccdrift-caught.html is hand-maintained with no generator. Refresh it, or state plainly that it was left alone and why.

## 4. Verify

```
uv run --group dev --group lab pytest -q
uv build
```

Run the full suite, not a subset, and paste the count. Do not proceed on a failure.

## 5. Commit, tag, push

```
git commit -am "Release <version>"
git tag v<version>
git push origin main
git push origin v<version>
```

Confirm the tag matches the pattern the workflow watches (`v*`, including rcN, aN, bN forms).

Both halves of Trusted Publishing are configured and neither needs touching for an ordinary release. Check them only when `publish` fails, since a mismatch there is what a failure looks like:

```
gh api repos/rkolesnichenko/ccdrift/environments/pypi --jq .name
gh api repos/rkolesnichenko/ccdrift/environments/pypi/deployment-branch-policies --jq '.branch_policies[].name'
```

The environment is `pypi`, restricted to `v*` tags. On the PyPI side the publisher claims owner rkolesnichenko, repository ccdrift, workflow release.yml, environment pypi, and every one of those must match or the token is refused. It started as a pending publisher, which is the only form available for a name with no project behind it; after the first publish it lives under the project's own publishing settings at pypi.org/manage/project/ccdrift/settings/publishing, not the account page it was created on.

## 6. Release notes

There is no CHANGELOG file; the Changelog URL in pyproject.toml points at GitHub Releases, so the notes are the changelog. Draft them from the commits since the previous tag:

```
git log --oneline v<previous>..v<version>
```

Write them in the repo's voice: no em dashes, no filler, one line per user-visible change, measurements named where a threshold moved. Show the draft and let me post it.

## 7. Report

Version bumped, suite count, tag pushed, workflow run URL if available, and whether the docs were refreshed.
