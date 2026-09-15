"""Run the daily check once a day: launchd on macOS, a systemd user timer or cron on
Linux. Every scheduler command goes through a `run` callable, so tests can stand in
for the real system."""

from __future__ import annotations

import os
import plistlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from ccdrift.check import ccdrift_home
from ccdrift.logs import default_source
from ccdrift.notify import notify as send_notification

LAUNCHD_LABEL = "io.github.rkolesnichenko.ccdrift"
SYSTEMD_UNIT = "ccdrift-check"

Run = Callable[..., subprocess.CompletedProcess]


class ScheduleError(RuntimeError):
    """A scheduler command failed; the message carries its output."""


def run_command(argv: list[str], input: Optional[str] = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, input=input, capture_output=True, text=True, check=False)
    except OSError as exc:
        return subprocess.CompletedProcess(argv, 127, stdout="", stderr=str(exc))


def _failure(argv: list[str], result: subprocess.CompletedProcess) -> str:
    output = (result.stderr or result.stdout or "").strip()
    return f"`{' '.join(argv)}` exited with {result.returncode}" + (f": {output}" if output else "")


def _checked(run: Run, argv: list[str], input: Optional[str] = None) -> subprocess.CompletedProcess:
    result = run(argv, input=input)
    if result.returncode != 0:
        raise ScheduleError(_failure(argv, result))
    return result


@dataclass(frozen=True)
class Job:
    """What the scheduler runs: `python -m ccdrift check`, carrying the folders set
    when the schedule was installed, since schedulers don't see shell variables."""

    python: str
    hour: int
    minute: int
    notify: bool
    log: Path
    source: Optional[Path] = None
    state: Optional[Path] = None

    def argv(self) -> list[str]:
        args = [self.python, "-m", "ccdrift", "check"]
        if self.notify:
            args.append("--notify")
        if self.source is not None:
            args += ["--source", str(self.source)]
        if self.state is not None:
            args += ["--state", str(self.state)]
        return args


def parse_at(text: str) -> tuple[int, int]:
    """'09:00' -> (9, 0). Raises ValueError for anything but a 24-hour HH:MM time."""
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if not match or int(match.group(1)) > 23 or int(match.group(2)) > 59:
        raise ValueError(f"--at takes a 24-hour time like 09:00, not {text!r}")
    return int(match.group(1)), int(match.group(2))


def make_job(at: str, notify: bool, source: Optional[str] = None,
             environ: Mapping[str, str] = os.environ, python: str = sys.executable) -> Job:
    hour, minute = parse_at(at)
    if source:
        source_path: Optional[Path] = Path(source).expanduser().absolute()
    elif environ.get("CLAUDE_CONFIG_DIR"):
        source_path = default_source(environ).absolute()
    else:
        source_path = None
    home = ccdrift_home(environ).absolute()
    state = home / "check-state.json" if environ.get("CCDRIFT_HOME") else None
    return Job(python=python, hour=hour, minute=minute, notify=notify, log=home / "check.log",
               source=source_path, state=state)


def last_log_line(log: Path) -> str:
    try:
        lines = [line for line in log.read_text(errors="replace").splitlines() if line.strip()]
    except OSError:
        lines = []
    return f"last log line: {lines[-1]}" if lines else f"log: nothing written yet ({log})"


def launchd_plist(job: Job) -> str:
    return plistlib.dumps({
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": job.argv(),
        "StartCalendarInterval": {"Hour": job.hour, "Minute": job.minute},
        "StandardOutPath": str(job.log),
        "StandardErrorPath": str(job.log),
    }, sort_keys=False).decode()


class Launchd:
    """A per-user launchd agent in ~/Library/LaunchAgents."""

    name = "launchd"

    def __init__(self, run: Run = run_command, home: Optional[Path] = None, uid: Optional[int] = None):
        self.run = run
        self.plist = (home or Path.home()) / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
        self.domain = f"gui/{os.getuid() if uid is None else uid}"
        self.service = f"{self.domain}/{LAUNCHD_LABEL}"

    def install_notes(self, job: Job) -> list[str]:
        return []

    def install(self, job: Job) -> None:
        """Write and load the agent, then start a first run. On failure, unload and
        delete it and raise ScheduleError."""
        self.run(["launchctl", "bootout", self.service])  # replaces a loaded agent; fails harmlessly otherwise
        self.plist.parent.mkdir(parents=True, exist_ok=True)
        self.plist.write_text(launchd_plist(job))
        try:
            _checked(self.run, ["launchctl", "bootstrap", self.domain, str(self.plist)])
            _checked(self.run, ["launchctl", "kickstart", self.service])
        except ScheduleError:
            self.run(["launchctl", "bootout", self.service])
            self.plist.unlink(missing_ok=True)
            raise

    def remove(self) -> bool:
        if not self.plist.exists():
            return False
        self.run(["launchctl", "bootout", self.service])
        self.plist.unlink()
        return True

    def status(self) -> list[str]:
        if not self.plist.exists():
            return ["not installed"]
        job = plistlib.loads(self.plist.read_bytes())
        when = job["StartCalendarInterval"]
        lines = [f"installed: launchd agent {LAUNCHD_LABEL}, daily at {when['Hour']:02d}:{when['Minute']:02d}"]
        printed = self.run(["launchctl", "print", self.service])
        if printed.returncode != 0:
            lines.append("not loaded; run `ccdrift schedule install` again")
        for key in ("runs", "last exit code"):
            found = re.search(rf"^\s*{key} = (.+)$", printed.stdout or "", re.MULTILINE)
            if found:
                lines.append(f"{key}: {found.group(1).strip()}")
        lines.append(last_log_line(Path(job["StandardOutPath"])))
        return lines


def _systemd_quote(arg: str) -> str:
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$")
    return f'"{escaped}"'


def systemd_units(job: Job) -> dict[str, str]:
    """The service and timer files for a systemd user timer that runs `job` daily."""
    command = " ".join(_systemd_quote(arg) for arg in job.argv())
    log = str(job.log).replace("%", "%%")
    service = (
        "[Unit]\n"
        "Description=ccdrift daily check\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"ExecStart={command}\n"
        f"StandardOutput=append:{log}\n"
        f"StandardError=append:{log}\n"
    )
    timer = (
        "[Unit]\n"
        "Description=Run the ccdrift daily check\n"
        "\n"
        "[Timer]\n"
        f"OnCalendar=*-*-* {job.hour:02d}:{job.minute:02d}:00\n"
        "Persistent=true\n"
        "\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )
    return {f"{SYSTEMD_UNIT}.service": service, f"{SYSTEMD_UNIT}.timer": timer}


class Systemd:
    """A systemd user service and timer in ~/.config/systemd/user."""

    name = "systemd"

    def __init__(self, run: Run = run_command, config_home: Optional[Path] = None,
                 environ: Mapping[str, str] = os.environ):
        self.run = run
        base = config_home or Path(environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
        self.unit_dir = base / "systemd" / "user"
        self.timer = self.unit_dir / f"{SYSTEMD_UNIT}.timer"
        self.service = self.unit_dir / f"{SYSTEMD_UNIT}.service"

    def install_notes(self, job: Job) -> list[str]:
        return ["systemd user timers run only while you're logged in, unless lingering is on "
                "(loginctl enable-linger)."]

    def install(self, job: Job) -> None:
        """Write both units, enable the timer and start a first run. On failure,
        disable and delete them and raise ScheduleError."""
        self.unit_dir.mkdir(parents=True, exist_ok=True)
        for name, text in systemd_units(job).items():
            (self.unit_dir / name).write_text(text)
        try:
            _checked(self.run, ["systemctl", "--user", "daemon-reload"])
            _checked(self.run, ["systemctl", "--user", "enable", "--now", self.timer.name])
            _checked(self.run, ["systemctl", "--user", "start", "--no-block", self.service.name])
        except ScheduleError:
            self._delete()
            raise

    def _delete(self) -> None:
        self.run(["systemctl", "--user", "disable", "--now", self.timer.name])
        self.timer.unlink(missing_ok=True)
        self.service.unlink(missing_ok=True)
        self.run(["systemctl", "--user", "daemon-reload"])

    def remove(self) -> bool:
        if not self.timer.exists():
            return False
        self._delete()
        return True

    def status(self) -> list[str]:
        if not self.timer.exists():
            return ["not installed"]
        when = re.search(r"^OnCalendar=\*-\*-\* (\d\d:\d\d):00$", self.timer.read_text(), re.MULTILINE)
        lines = [f"installed: systemd timer {self.timer.name}, "
                 f"daily at {when.group(1) if when else 'an unreadable time'}"]
        enabled = self.run(["systemctl", "--user", "is-enabled", self.timer.name])
        lines.append(f"enabled: {(enabled.stdout or '').strip() or 'unknown'}")
        shown = self.run(["systemctl", "--user", "show", self.service.name,
                          "-p", "ExecMainStartTimestamp", "-p", "ExecMainStatus"])
        props = dict(line.split("=", 1) for line in (shown.stdout or "").splitlines() if "=" in line)
        lines.append(f"last run: {props.get('ExecMainStartTimestamp') or 'never'}")
        lines.append(f"last exit code: {props.get('ExecMainStatus', 'unknown')}")
        log = re.search(r"^StandardOutput=append:(.+)$", self.service.read_text(), re.MULTILINE)
        if log:
            lines.append(last_log_line(Path(log.group(1).replace("%%", "%"))))
        return lines


def install(job: Job, backend, send: Callable[[str, str], None] = send_notification) -> None:
    """Create the log folder, install the job (the backend also starts a first run),
    and send a test notification unless the job runs without notifications."""
    job.log.parent.mkdir(parents=True, exist_ok=True)
    backend.install(job)
    if job.notify:
        send("ccdrift", f"The daily check will run at {job.hour:02d}:{job.minute:02d}. "
                        "Alerts will look like this.")


def choose_backend(platform: str = sys.platform, run: Run = run_command,
                   which: Callable[[str], Optional[str]] = shutil.which):
    """The scheduler ccdrift can set up on this system, or None."""
    if platform == "darwin":
        return Launchd(run=run)
    return None
