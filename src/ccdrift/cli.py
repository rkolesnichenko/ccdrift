"""The ccdrift command line."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from ccdrift import __version__
from ccdrift.check import ccdrift_home, run_check
from ccdrift.logs import default_source, peek
from ccdrift.report import run_report


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
    _add_source(check)
    _add_state(check)

    peek_command = commands.add_parser("peek", help="show the first response and the fields ccdrift reads from it")
    _add_source(peek_command)

    report = commands.add_parser("report", help="recent daily metrics and flags: the details behind an alert")
    report.add_argument("--days", type=int, default=21, help="how many recent days to list (default: 21)")
    _add_source(report)
    _add_state(report)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "check":
        return run_check(_source(args), _state(args), notify_user=args.notify)
    if args.command == "peek":
        return 0 if peek(_source(args)) else 2
    if args.command == "report":
        return run_report(_source(args), _state(args), days=args.days)
    raise AssertionError(f"unhandled command: {args.command}")
