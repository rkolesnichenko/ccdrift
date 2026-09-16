# ccdrift

A daily check for silent changes in Claude Code, read from your own session logs.

Claude Code keeps a transcript of every session on your machine. ccdrift reads them
once a day and tells you when something shifts that you'd otherwise only notice on
your bill or your usage limits:

- **Prompt caching gets worse.** Less of each new prompt is read from cache, so Claude
  Code resends the conversation more often. ccdrift caught a real regression this way
  (Claude Code 2.1.233–2.1.258, August 2026) and follows such a regression until it's
  fixed.
- **Haiku appears on the main thread**, where your chosen model normally answers.
- **A setting Claude Code picks changes:** the main thread moves between the 1-hour
  and 5-minute prompt cache, or its effort level changes.

Each alert names the Claude Code version that was running, and what a regression has
cost: tokens re-cached, or extra Haiku responses.

It can't tell you whether responses think less: in one person's logs, effort swings
more from day to day than a 70% cut in thinking moves it. See
[docs/findings.md](docs/findings.md).

Everything stays on your machine. ccdrift reads the transcripts and keeps a state
file, a log and its own history of responses in `~/.ccdrift`. The history holds token
counts, models, versions and settings, plus each transcript's path (which includes
your project folder names) and session id; no prompt or response text. ccdrift sends
nothing anywhere, unless you give it a command to run with `--exec`.

## Install

Needs Python 3.10 or newer.

```sh
uv tool install git+https://github.com/rkolesnichenko/ccdrift
# or
pipx install git+https://github.com/rkolesnichenko/ccdrift
```

Then schedule the daily check:

```sh
ccdrift schedule install                         # daily at 09:00, with notifications
ccdrift schedule install --at 18:30 --no-notify  # another time, log only
ccdrift schedule status                          # installed? how did the last run go?
```

On macOS this adds a launchd agent. On Linux it adds a systemd user timer, or a
crontab line where systemd user sessions aren't available. Installing again replaces
the job. On Windows, run `ccdrift check` from Task Scheduler instead.

Claude Code deletes transcripts after 30 days by default. From its first run on,
ccdrift keeps its own history of every response it has read, so later deletions don't
affect it; the history grows by about 65 MB a year for a heavy user. Its first run
can only see what's still on disk, and it judges each day against the 14 before it.
To give that first run more to go on, and to keep transcripts for your own digging,
set this in `~/.claude/settings.json`:

```json
{ "cleanupPeriodDays": 365 }
```

## When an alert arrives

| Alert | What it means | What to do |
|---|---|---|
| **ccdrift flag** | 3 of the last 4 days passed the cutoff for the cache ratio or main-thread Haiku share. ccdrift opens an incident and names the version and the cost so far. | `ccdrift report` lists the days; `ccdrift report --by version` compares versions. A false alarm? `ccdrift incident dismiss`. |
| **ccdrift: back to normal** | The metric has been back inside the cutoff, 3 days pooled, on 3 days in a row. | Nothing. `ccdrift incident list` keeps the record. |
| **ccdrift: change persists** | The metric hasn't recovered 30 days after the incident started. ccdrift now treats the new level as normal. | Check whether you changed something: hooks, MCP servers, model. |
| **ccdrift: setting changed** | The cache tier or effort level a model usually gets on the main thread changed, 2 days in a row. | If you didn't change it, Claude Code's default did. |
| **ccdrift can't compute the cache metric** | 3 busy days had no usable cache values. Claude Code's log format has most likely changed. | `ccdrift peek` shows the first response ccdrift finds and the fields it reads from it. Please open an issue, removing any prompt or response text from what you paste. |
| **ccdrift check failed** | The check itself stopped with an error. | `~/.ccdrift/check.log` has the details. |

Each alert is sent once, except a failing check, which alerts on every run until it
works again. Claude Code's documentation says the transcript format "is internal to
Claude Code and changes between versions, so scripts that parse these files directly
can break on any release", which is why the cache-metric alert exists.

## Incidents

While an incident is open, its days stay out of the baseline, so a regression that
lasts for weeks is still judged against the days before it began. Against a rolling
baseline, the August regression looked normal again within 8 days, while 5–10% of
prompt turns kept missing the cache.

```text
ccdrift incident list                               every incident, its cost and versions
ccdrift incident add cache 2026-08-16..2026-09-04   record one from before ccdrift ran
ccdrift incident close cache                        end the open one as of yesterday (UTC)
ccdrift incident dismiss cache 2026-09-14           a false alarm: its days rejoin the baseline
```

## Status line

`ccdrift status --short` prints one line when something needs attention, and nothing
otherwise: a failing check, no check for 3 days, or an open incident.

```console
$ ccdrift status --short
ccdrift: cache ratio down since 08-18
```

It reads only the state file and always exits 0, so it's cheap and safe to call from
the command your Claude Code status line runs. `ccdrift status` shows the last run,
open and recent incidents, and setting changes.

## Alerts elsewhere

`--exec` runs a command through the shell for each alert, with `CCDRIFT_ALERT` (`flag`,
`recovered`, `persistent`, `setting`, `blank_cache` or `failed`), `CCDRIFT_TITLE` and
`CCDRIFT_MESSAGE` set. For example, to send alerts to [ntfy](https://ntfy.sh):

```sh
ccdrift schedule install --exec 'curl -s -d "$CCDRIFT_MESSAGE" ntfy.sh/your-topic'
```

A command that fails or runs longer than 30 seconds is noted in the log and doesn't
stop the check.

## Commands

```text
ccdrift check [--notify] [--exec CMD] [--source DIR] [--state FILE]   what the schedule runs
ccdrift report [--days N] [--by day|version] [--json] [--source DIR] [--state FILE]
ccdrift status [--short] [--state FILE]
ccdrift incident list [--source DIR] [--state FILE]
ccdrift incident add {cache|haiku} START..END [--state FILE]
ccdrift incident close {cache|haiku} [--state FILE]
ccdrift incident dismiss {cache|haiku} START [--state FILE]
ccdrift peek [--source DIR]                                            the fields ccdrift reads
ccdrift schedule install [--at HH:MM] [--no-notify] [--exec CMD] [--source DIR]
ccdrift schedule remove
ccdrift schedule status
```

`report --json` holds aggregates only: no paths, session ids or project names.

Transcripts are read from `$CLAUDE_CONFIG_DIR/projects` when that variable is set,
otherwise from `~/.claude/projects`. The state file, history and log live in
`$CCDRIFT_HOME`, otherwise in `~/.ccdrift`. Schedulers don't see your shell's
variables, so a schedule installed while either one is set keeps its value. Install
the schedule again after moving or reinstalling ccdrift.

## How it decides

For each complete UTC day, ccdrift looks at main-thread responses from the Claude Code
CLI (Agent SDK sessions are your own scripts and are left out) and computes:

- the cache read ratio on turns that open with a new prompt, within an hour of the
  previous response and not right after a compaction;
- the share of responses from a Haiku model.

Each day is compared with the 14 days before it, leaving out the days of open and
recovered incidents, using their median and spread, with the spread floored at
sampling noise. A day is deviant past z = −3.0 for the cache ratio or z = +3.5 for
Haiku share, and a metric is flagged once 3 of any 4 days in a row are deviant. These
defaults were tuned on one person's logs; the research harness in
[lab/](lab/README.md) measures how small a change they catch on yours.

## Linux notes

- systemd user timers run only while you're logged in, unless lingering is on:
  `loginctl enable-linger $USER`.
- The timer appends to the log with `StandardOutput=append:`, which needs systemd 240 or newer.
- A timer catches up on a run missed while the machine was off; cron doesn't.
- Notifications use `notify-send`. They usually appear from a systemd timer but not
  from cron, so with cron, watch the log or use `--exec`.
- The systemd path is covered by tests but hasn't yet run on a real machine. Reports
  are welcome.

## Uninstall

```sh
ccdrift schedule remove
uv tool uninstall ccdrift        # or: pipx uninstall ccdrift
rm -rf ~/.ccdrift                # state, history and log
```

## Development

```sh
uv run --group dev --group lab pytest
```

## License

MIT
