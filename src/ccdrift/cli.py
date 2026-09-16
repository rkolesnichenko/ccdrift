"""The ccdrift command line."""

from __future__ import annotations

import argparse
import shlex
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from ccdrift import __version__
from ccdrift.check import ccdrift_home, run_check
from ccdrift.incidents import (METRIC_ARGS, add_incident, close_incident, dismiss_incident, incident_line,
                               parse_days, run_list)
from ccdrift.logs import default_source, peek
from ccdrift.report import run_report
from ccdrift.schedule import ScheduleError, choose_backend, install as install_job, make_job
from ccdrift.state import load_state, save_state
from ccdrift.status import run_status


def _add_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", help="folder of Claude Code transcripts "
                        "(default: $CLAUDE_CONFIG_DIR/projects, otherwise ~/.claude/projects)")


def _add_state(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state", help="the daily check's state file "
                        "(default: check-state.json in $CCDRIFT_HOME, otherwise in ~/.ccdrift)")


def _source(args: argparse.Namespace) -> Path:
    return Path(args.source).expanduser() if args.source else default_source()


def _state(args: argparse.Namespace) -> Path:
    return Path(args.state).expanduser() if args.state else ccdrift_home() / "check-state.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ccdrift",
        description="A daily check for silent changes in Claude Code's prompt caching and Haiku use, "
                    "read from your local session logs.")
    parser.add_argument("--version", action="version", version=f"ccdrift {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    check = commands.add_parser("check", help="report new flags once (what the schedule runs)")
    check.add_argument("--notify", action="store_true", help="also show a desktop notification for each alert")
    check.add_argument("--exec", metavar="CMD", help="also run CMD through the shell for each alert, with "
                       "CCDRIFT_ALERT, CCDRIFT_TITLE and CCDRIFT_MESSAGE set")
    _add_source(check)
    _add_state(check)

    peek_command = commands.add_parser("peek", help="show the first response and the fields ccdrift reads from it")
    _add_source(peek_command)

    report = commands.add_parser("report", help="recent days or Claude Code versions, incidents and settings")
    report.add_argument("--days", type=int, default=None,
                        help="how many recent days to cover (default: 21 by day, all by version)")
    report.add_argument("--by", choices=["day", "version"], default="day",
                        help="one row per day (default) or per Claude Code version")
    report.add_argument("--json", action="store_true", help="print the report as JSON")
    _add_source(report)
    _add_state(report)

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

    schedule = commands.add_parser("schedule", help="run the check once a day")
    actions = schedule.add_subparsers(dest="action", required=True, metavar="ACTION")
    install_action = actions.add_parser("install", help="set up the daily job, replacing an existing one")
    install_action.add_argument("--at", default="09:00", help="local time to run, 24-hour HH:MM (default: 09:00)")
    install_action.add_argument("--no-notify", action="store_true",
                                help="write alerts to the log without desktop notifications")
    install_action.add_argument("--exec", metavar="CMD", help="also run CMD through the shell for each alert "
                                "(see `ccdrift check --help`)")
    _add_source(install_action)
    actions.add_parser("remove", help="remove the daily job")
    actions.add_parser("status", help="show whether the job is installed and how its last run went")
    return parser


def _schedule(args: argparse.Namespace) -> int:
    try:
        job = make_job(args.at if args.action == "install" else "09:00",
                       notify=not getattr(args, "no_notify", False),
                       source=getattr(args, "source", None),
                       exec_command=getattr(args, "exec", None))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    backend = choose_backend()
    if backend is None:
        print("ccdrift can't set up a daily job on this system. Run this command once a day "
              "with your system's scheduler:", file=sys.stderr)
        print("  " + " ".join(shlex.quote(arg) for arg in job.argv()), file=sys.stderr)
        return 2
    if args.action == "install":
        try:
            install_job(job, backend)
        except ScheduleError as exc:
            print(f"Nothing installed: {exc}", file=sys.stderr)
            return 1
        print(f"Installed a {backend.name} job: `ccdrift check` runs daily at {job.hour:02d}:{job.minute:02d}.")
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
        print("Removed the daily job." if removed else "No daily job was installed.")
        return 0
    try:
        status = backend.status()
    except ScheduleError as exc:
        print(f"Couldn't read the schedule: {exc}", file=sys.stderr)
        return 1
    print("\n".join(status))
    return 0


def _incident(args: argparse.Namespace) -> int:
    state_path = _state(args)
    if args.action == "list":
        return run_list(_source(args), state_path)
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
    print(incident_line(incident))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        return run_check(_source(args), _state(args), notify_user=args.notify, exec_command=args.exec)
    if args.command == "peek":
        return 0 if peek(_source(args)) else 2
    if args.command == "report":
        return run_report(_source(args), _state(args), days=args.days, by=args.by, as_json=args.json)
    if args.command == "status":
        return run_status(_state(args), short=args.short)
    if args.command == "incident":
        return _incident(args)
    if args.command == "schedule":
        return _schedule(args)
    raise AssertionError(f"unhandled command: {args.command}")
