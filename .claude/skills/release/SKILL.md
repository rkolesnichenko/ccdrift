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
git diff --stat v<previous>..HEAD -- src tests lab docs/findings.md README.md LICENSE pyproject.toml
```

- Working tree clean, on `main`.
- `origin` must point at `https://github.com/rkolesnichenko/ccdrift.git`. If `git remote -v` is empty or names anything else, stop and ask before going further.
- The target tag must not already exist.
- Something must actually ship. That last command's paths are a copy of `only-include` in pyproject.toml, which is the whole of the sdist, so they are the only diff a release can carry; change one and change the other. An empty result means the distributions would differ from the last one by the version string alone: stop, say so, and let me decide, since a PyPI version can never be reused. Work on `.claude/`, `.github/` and CLAUDE.md never reaches the package and is finished once it is on `main`.

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

## 5. Commit, land through a PR, tag

`main` takes pull requests. The owner bypass lets a direct push through, so never use it.

```
git switch -c release-<version>
git commit -am "Release <version>"
git push -u origin release-<version>
gh pr create --fill
gh pr checks --watch
gh pr merge --squash --delete-branch
git switch main && git pull --ff-only
```

Squashing makes a new commit, so tag `main` after the pull, never the branch. Check `__version__` there first, then:

```
git tag v<version>
git push origin v<version>
```

Confirm the tag matches the pattern the workflow watches (`v*`, including rcN, aN, bN forms).

`publish` then waits for approval in the `pypi` environment, and it does not wait for `tests.yml`. Watch `tests.yml` on the tag, report its result, and leave the approval to me: approve only when I say so, and never while the tag's tests are failing or still running.

```
gh run list --branch v<version> --json workflowName,status,conclusion,databaseId
```

Never `gh run watch` the release run. Its `publish` job sits in `waiting` until the deployment is reviewed, so the watch blocks on a human and burns its whole timeout without reporting anything. `gh run watch` the tag's `tests.yml` instead, which finishes on its own, and read the release run's state with the single query above. After I approve, one more of those says whether `publish` succeeded.

Both halves of Trusted Publishing are configured and neither needs touching for an ordinary release. Check them only when `publish` fails, since a mismatch there is what a failure looks like:

```
gh api repos/rkolesnichenko/ccdrift/environments/pypi --jq .name
gh api repos/rkolesnichenko/ccdrift/environments/pypi/deployment-branch-policies --jq '.branch_policies[].name'
```

The environment is `pypi`, restricted to `v*` tags, with me as its required reviewer and no admin bypass. A tag ruleset lets only admins create, move or delete a `v*` tag. On the PyPI side the publisher claims owner rkolesnichenko, repository ccdrift, workflow release.yml, environment pypi, and every one of those must match or the token is refused. It started as a pending publisher, which is the only form available for a name with no project behind it; after the first publish it lives under the project's own publishing settings at pypi.org/manage/project/ccdrift/settings/publishing, not the account page it was created on.

## 6. Release notes

There is no CHANGELOG file; the Changelog URL in pyproject.toml points at GitHub Releases, so the notes are the changelog. Draft them from the commits since the previous tag:

```
git log --oneline v<previous>..v<version>
```

Write them in the repo's voice: no em dashes, no filler, one line per user-visible change, measurements named where a threshold moved. Show the draft and let me post it.

## 7. Report

Version bumped, suite count, tag pushed, workflow run URL if available, and whether the docs were refreshed.
