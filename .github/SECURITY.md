# Security policy

## Reporting a vulnerability

Please report it privately through
[GitHub's private vulnerability reporting](https://github.com/rkolesnichenko/ccdrift/security/advisories/new),
not in a public issue.

Include the version (`ccdrift --version`), your OS, and the steps that show the problem.
Please don't attach real Claude Code transcripts: they can hold your code, prompts and
secrets. A synthetic transcript line that reproduces the problem is enough.

Only the latest release on PyPI receives fixes.

## What is in scope

ccdrift reads Claude Code's local transcripts, which can hold source code, prompts and
secrets. Anything that lets that material leave the machine, or reach another local user,
is in scope. For example:

- `report --json` or `incident draft` emitting a path, session id or project name. Both
  are meant to hold aggregates only.
- `peek` printing text, ids or paths. It is meant to show them only as their length.
- The state, history or log file under `~/.ccdrift` being readable by another user. They
  are meant to be owner-only (0600).
- Control characters or terminal escapes from a transcript reaching your terminal or the
  status line.
- `report --html` running a script or fetching anything from the network.
- The release workflow publishing something other than what the tag points at.

## What is not

`report --html` names project folders, exactly as the terminal report does. The page says
it is not for sharing, in its own last line.
