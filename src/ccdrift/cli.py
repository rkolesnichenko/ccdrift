"""The ccdrift command line."""

from __future__ import annotations

import argparse
import shlex
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from ccdrift import __version__
from ccdrift.state import ccdrift_home
from ccdrift.texts import DIMENSION_NAMES, METRIC_ARGS


def choose_backend():
    """The scheduler for this system (see ccdrift.schedule.choose_backend); a name
    here so tests can replace it."""
    from ccdrift.schedule import choose_backend as choose
    return choose()


def _add_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", help="folder of Claude Code transcripts "
                        "(default: $CLAUDE_CONFIG_DIR/projects, otherwise ~/.claude/projects)")


def _add_state(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state", help="the daily check's state file "
                        "(default: check-state.json in $CCDRIFT_HOME, otherwise in ~/.ccdrift)")


def _source(args: argparse.Namespace) -> Path:
    from ccdrift.logs import default_source
    return Path(args.source).expanduser() if args.source else default_source()


def _state(args: argparse.Namespace) -> Path:
    return Path(args.state).expanduser() if args.state else ccdrift_home() / "check-state.json"


def _days(text: str) -> int:
    try:
        days = int(text)
    except ValueError:
        days = 0
    if days < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number of days, 1 or more, not {text!r}")
    return days


def _day(text: str) -> str:
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a day like 2026-08-18, not {text!r}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ccdrift",
        description="A check for silent changes in Claude Code's prompt caching, Haiku use, settings, context "
                    "and hooks, read from your local session logs.")
    parser.add_argument("--version", action="version", version=f"ccdrift {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    check = commands.add_parser("check", help="follow incidents and changes, and alert on each "
                                "(what the schedule runs)")
    check.add_argument("--notify", action="store_true", help="also show a desktop notification for each alert")
    check.add_argument("--exec", metavar="CMD", help="also run CMD through the shell for each alert, with "
                       "CCDRIFT_ALERT, CCDRIFT_TITLE and CCDRIFT_MESSAGE set")
    check.add_argument("--no-digest", action="store_true", help="skip the weekly summary on Mondays")
    _add_source(check)
    _add_state(check)

    peek_command = commands.add_parser("peek", help="show the first response and the fields ccdrift reads from it")
    _add_source(peek_command)

    report = commands.add_parser("report", help="recent days or Claude Code versions, incidents and settings")
    report.add_argument("--days", type=_days, default=None,
                        help="how many recent days to cover (default: 21 by day, all by version)")
    report.add_argument("--by", choices=["day", "version"], default="day",
                        help="one row per day (default) or per Claude Code version")
    output = report.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="print the report as JSON")
    output.add_argument("--html", type=Path, metavar="FILE",
                        help="write the day view to FILE as a self-contained page, charts and all")
    _add_source(report)
    _add_state(report)
    # So a refusal `main` raises prints `usage: ccdrift report …`, as argparse's own do.
    report.set_defaults(_parser=report)

    cost = commands.add_parser("cost", help="where the tokens went, by thread, agent, skill, "
                               "plugin, MCP server, model, project or branch")
    cost.add_argument("--days", type=_days, default=None,
                      help="how many recent days to cover (default: 30)")
    cost.add_argument("--by", choices=list(DIMENSION_NAMES), default=None,
                      help="show one dimension instead of the usual set")
    cost.add_argument("--json", action="store_true", help="print the breakdown as JSON")
    _add_source(cost)
    _add_state(cost)

    status = commands.add_parser("status", help="how the last check went and what ccdrift is following")
    status.add_argument("--short", action="store_true",
                        help="one line when something needs attention, nothing otherwise (for a status line)")
    _add_state(status)

    incident = commands.add_parser("incident", help="list incidents, or add, close or dismiss one")
    incident_actions = incident.add_subparsers(dest="action", required=True, metavar="ACTION")
    list_action = incident_actions.add_parser("list", help="every incident, newest first, with what it cost")
    _add_source(list_action)
    _add_state(list_action)
    add_action = incident_actions.add_parser(
        "add", help="record a past incident, so its days stay out of the baseline")
    add_action.add_argument("metric", choices=list(METRIC_ARGS))
    add_action.add_argument("days", metavar="START..END", help="UTC dates, inclusive")
    _add_state(add_action)
    close_action = incident_actions.add_parser("close", help="end the open incident as of yesterday (UTC)")
    close_action.add_argument("metric", choices=list(METRIC_ARGS))
    _add_state(close_action)
    dismiss_action = incident_actions.add_parser(
        "dismiss", help="mark an incident as a false alarm, so its days rejoin the baseline")
    dismiss_action.add_argument("metric", choices=list(METRIC_ARGS))
    dismiss_action.add_argument("start", help="the incident's first day")
    _add_state(dismiss_action)
    draft_action = incident_actions.add_parser(
        "draft", help="print a GitHub issue draft about an incident with its evidence, aggregates only")
    draft_action.add_argument("metric", choices=list(METRIC_ARGS))
    draft_action.add_argument("start", nargs="?", default=None, type=_day,
                              help="the incident's first day (default: the latest incident not dismissed)")
    _add_source(draft_action)
    _add_state(draft_action)

    replay = commands.add_parser("replay", help="replay the check day by day over your history and show the "
                                 "incidents it would have followed, without recording anything")
    _add_source(replay)
    _add_state(replay)

    schedule = commands.add_parser("schedule", help="run the check every hour, or once a day")
    actions = schedule.add_subparsers(dest="action", required=True, metavar="ACTION")
    install_action = actions.add_parser("install", help="set up the job, replacing an existing one")
    install_action.add_argument("--at", default=None,
                                help="run once a day at this local time, 24-hour HH:MM (default: every hour)")
    install_action.add_argument("--no-notify", action="store_true",
                                help="write alerts to the log without desktop notifications")
    install_action.add_argument("--exec", metavar="CMD", help="also run CMD through the shell for each alert "
                                "(see `ccdrift check --help`)")
    install_action.add_argument("--no-digest", action="store_true", help="the job skips the weekly summary")
    _add_source(install_action)
    actions.add_parser("remove", help="remove the job")
    actions.add_parser("status", help="show whether the job is installed and how its last run went")
    return parser


def _schedule(args: argparse.Namespace) -> int:
    from ccdrift.schedule import ScheduleError, install as install_job, make_job
    try:
        job = make_job(args.at if args.action == "install" else None,
                       notify=not getattr(args, "no_notify", False),
                       source=getattr(args, "source", None),
                       exec_command=getattr(args, "exec", None),
                       no_digest=getattr(args, "no_digest", False))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    backend = choose_backend()
    if backend is None:
        print("ccdrift can't set up a scheduled job on this system. Run this command every hour, or once a day, "
              "with your system's scheduler:", file=sys.stderr)
        print("  " + " ".join(shlex.quote(arg) for arg in job.argv()), file=sys.stderr)
        return 2
    if args.action == "install":
        try:
            install_job(job, backend)
        except ScheduleError as exc:
            print(f"Nothing installed: {exc}", file=sys.stderr)
            return 1
        print(f"Installed a {backend.name} job: `ccdrift check` runs {job.when()}.")
        print(f"Log: {job.log}")
        print("A first run has started. Check `ccdrift schedule status` in a minute.")
        for note in backend.install_notes(job):
            print(note)
        return 0
    if args.action == "remove":
        try:
            removed = backend.remove()
        except ScheduleError as exc:
            print(f"Nothing removed: {exc}", file=sys.stderr)
            return 1
        print("Removed the ccdrift job." if removed else "No ccdrift job was installed.")
        return 0
    try:
        status = backend.status()
    except ScheduleError as exc:
        print(f"Couldn't read the schedule: {exc}", file=sys.stderr)
        return 1
    print("\n".join(status))
    return 0


def _incident(args: argparse.Namespace) -> int:
    from ccdrift.incidents import run_list
    from ccdrift.state import state_lock
    state_path = _state(args)
    if args.action == "list":
        return run_list(_source(args), state_path)
    if args.action == "draft":
        from ccdrift.draft import run_draft
        return run_draft(_source(args), state_path, METRIC_ARGS[args.metric], args.start)
    # A check running now would otherwise save the state it read before this change.
    try:
        with state_lock(state_path):
            return _change_incident(args, state_path)
    except OSError as exc:
        print(f"Can't change the state file {state_path}: {exc}", file=sys.stderr)
        return 1


def _change_incident(args: argparse.Namespace, state_path: Path) -> int:
    from ccdrift.incidents import add_incident, close_incident, dismiss_incident, incident_line, parse_days
    from ccdrift.state import load_state, save_state
    try:
        state = load_state(state_path)
    except (OSError, ValueError) as exc:
        print(f"Can't read the state file {state_path}: {exc}", file=sys.stderr)
        return 1
    today = datetime.now(timezone.utc).date()
    metric = METRIC_ARGS[args.metric]
    try:
        if args.action == "add":
            incident = add_incident(state["incidents"], metric, *parse_days(args.days), today)
        elif args.action == "close":
            incident = close_incident(state["incidents"], metric, today)
        else:
            incident = dismiss_incident(state["incidents"], metric, date.fromisoformat(args.start).isoformat(), today)
    except ValueError as exc:
        print(f"Nothing changed: {exc}", file=sys.stderr)
        return 2
    save_state(state_path, state)
    if args.action == "add":
        # Its cost comes from the history, which only `incident list` and the check read.
        print(f"Added {args.metric} {incident['start']}..{incident['end']}. "
              "`ccdrift incident list` shows what it cost.")
    else:
        print(incident_line(incident))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "check":
        from ccdrift.check import run_check
        return run_check(_source(args), _state(args), notify_user=args.notify, exec_command=args.exec,
                         digest=not args.no_digest)
    if args.command == "peek":
        from ccdrift.logs import peek
        return 0 if peek(_source(args)) else 2
    if args.command == "report":
        from ccdrift.report import run_report
        if args.html is not None and args.by == "version":
            args._parser.error("--html draws the day view; drop --by version")
        return run_report(_source(args), _state(args), days=args.days, by=args.by, as_json=args.json,
                          html_path=args.html)
    if args.command == "cost":
        from ccdrift.spend import run_spend
        return run_spend(_source(args), _state(args), days=args.days, by=args.by, as_json=args.json)
    if args.command == "status":
        from ccdrift.status import run_status
        return run_status(_state(args), short=args.short)
    if args.command == "incident":
        return _incident(args)
    if args.command == "replay":
        from ccdrift.replay import run_replay
        return run_replay(_source(args), _state(args))
    if args.command == "schedule":
        return _schedule(args)
    raise AssertionError(f"unhandled command: {args.command}")
