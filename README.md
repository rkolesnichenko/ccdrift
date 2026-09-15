# ccdrift

A daily check for silent changes in Claude Code, read from your own session logs.

Claude Code keeps a transcript of every session on your machine. ccdrift reads them
once a day and tells you when something shifts that you'd otherwise only notice on
your bill or your usage limits:

- **Prompt caching gets worse.** Less of each new prompt is read from cache, so Claude
  Code resends the conversation more often. ccdrift caught a real regression this way
  (Claude Code 2.1.233–2.1.258, August 2026).
- **Haiku appears on the main thread**, where your chosen model normally answers.

It can't tell you whether responses think less: in one person's logs, effort swings
more from day to day than a 70% cut in thinking moves it. See
[docs/findings.md](docs/findings.md).

Everything stays on your machine. ccdrift reads the transcripts, keeps a small state
file and a log in `~/.ccdrift`, and sends nothing anywhere.

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

Claude Code deletes transcripts after 30 days by default, and ccdrift judges each day
against the 14 before it. To keep more history, set this in `~/.claude/settings.json`:

```json
{ "cleanupPeriodDays": 365 }
```

## When an alert arrives

| Alert | What it means | What to do |
|---|---|---|
| **ccdrift flag** | 3 of the last 4 days passed the cutoff for the cache ratio or main-thread Haiku share. | `ccdrift report` lists the days, their values and z-scores. |
| **ccdrift can't compute the cache metric** | 3 busy days had no usable cache values. Claude Code's log format has most likely changed. | `ccdrift peek` shows the first response ccdrift finds and the fields it reads from it. Please open an issue, removing any prompt or response text from what you paste. |
| **ccdrift check failed** | The check itself stopped with an error. | `~/.ccdrift/check.log` has the details. |

The flag and cache-metric alerts are sent once each; a failing check alerts on every
run until it works again. Claude Code's documentation says the transcript format "is
internal to Claude Code and changes between versions, so scripts that parse these
files directly can break on any release", which is why the second alert exists.

## Commands

```text
ccdrift check [--notify] [--source DIR] [--state FILE]    what the schedule runs
ccdrift report [--days N] [--source DIR] [--state FILE]   recent days, z-scores and flags
ccdrift peek [--source DIR]                               the fields ccdrift reads
ccdrift schedule install [--at HH:MM] [--no-notify] [--source DIR]
ccdrift schedule remove
ccdrift schedule status
```

Transcripts are read from `$CLAUDE_CONFIG_DIR/projects` when that variable is set,
otherwise from `~/.claude/projects`. The state file and log live in `$CCDRIFT_HOME`,
otherwise in `~/.ccdrift`. Schedulers don't see your shell's variables, so a schedule
installed while either one is set keeps its value. Install the schedule again after
moving or reinstalling ccdrift.

## How it decides

For each complete UTC day, ccdrift looks at main-thread responses and computes:

- the cache read ratio on turns that open with a new prompt, within an hour of the
  previous response and not right after a compaction;
- the share of responses from a Haiku model.

Each day is compared with the 14 days before it, using their median and spread, with
the spread floored at sampling noise. A day is deviant past z = −3.0 for the cache
ratio or z = +3.5 for Haiku share, and a metric is flagged once 3 of any 4 days in a
row are deviant. These defaults were tuned on one person's logs; the research harness
in [lab/](lab/README.md) measures how small a change they catch on yours.

## Linux notes

- systemd user timers run only while you're logged in, unless lingering is on:
  `loginctl enable-linger $USER`.
- The timer appends to the log with `StandardOutput=append:`, which needs systemd 240 or newer.
- A timer catches up on a run missed while the machine was off; cron doesn't.
- Notifications use `notify-send`. They usually appear from a systemd timer but not
  from cron, so with cron, watch the log.
- The systemd path is covered by tests but hasn't yet run on a real machine. Reports
  are welcome.

## Uninstall

```sh
ccdrift schedule remove
uv tool uninstall ccdrift        # or: pipx uninstall ccdrift
rm -rf ~/.ccdrift
```

## Development

```sh
uv run --group dev --group lab pytest
```

## License

MIT
