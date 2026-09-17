"""Run the check every hour or once a day: launchd on macOS, a systemd user timer or cron on
Linux. Every scheduler command goes through a `run` callable, so tests can stand in
for the real system."""

from __future__ import annotations

import os
import plistlib
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from ccdrift.logs import CONTROL_CHARS, default_source
from ccdrift.notify import notify as send_notification
from ccdrift.state import ccdrift_home, make_private

LAUNCHD_LABEL = "io.github.rkolesnichenko.ccdrift"
SYSTEMD_UNIT = "ccdrift-check"
CRON_MARKER = "# ccdrift check"
BOOTSTRAP_ATTEMPTS = 5

Run = Callable[..., subprocess.CompletedProcess]


class ScheduleError(RuntimeError):
    """A job can't be scheduled, or a scheduler command failed; the message says why,
    with the command's output."""


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
    hour: Optional[int]
    minute: Optional[int]  # both None: every hour
    notify: bool
    log: Path
    source: Optional[Path] = None
    state: Optional[Path] = None
    exec_command: Optional[str] = None
    no_digest: bool = False

    def argv(self) -> list[str]:
        args = [self.python, "-m", "ccdrift", "check"]
        if self.notify:
            args.append("--notify")
        if self.no_digest:
            args.append("--no-digest")
        if self.source is not None:
            args += ["--source", str(self.source)]
        if self.state is not None:
            args += ["--state", str(self.state)]
        if self.exec_command:
            args += ["--exec", self.exec_command]
        return args

    @property
    def hourly(self) -> bool:
        return self.hour is None

    def when(self) -> str:
        return "every hour" if self.hourly else f"daily at {self.hour:02d}:{self.minute:02d}"


def parse_at(text: str) -> tuple[int, int]:
    """'09:00' -> (9, 0). Raises ValueError for anything but a 24-hour HH:MM time."""
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if not match or int(match.group(1)) > 23 or int(match.group(2)) > 59:
        raise ValueError(f"--at takes a 24-hour time like 09:00, not {text!r}")
    return int(match.group(1)), int(match.group(2))


def make_job(at: Optional[str], notify: bool, source: Optional[str] = None, exec_command: Optional[str] = None,
             no_digest: bool = False, environ: Mapping[str, str] = os.environ,
             python: str = sys.executable) -> Job:
    hour, minute = parse_at(at) if at is not None else (None, None)
    if source:
        source_path: Optional[Path] = Path(source).expanduser().absolute()
    elif environ.get("CLAUDE_CONFIG_DIR"):
        source_path = default_source(environ).absolute()
    else:
        source_path = None
    home = ccdrift_home(environ).absolute()
    state = home / "check-state.json" if environ.get("CCDRIFT_HOME") else None
    return Job(python=python, hour=hour, minute=minute, notify=notify, log=home / "check.log",
               source=source_path, state=state, exec_command=exec_command, no_digest=no_digest)


def last_log_line(log: Path) -> str:
    try:
        lines = [line for line in log.read_text(errors="replace").splitlines() if line.strip()]
    except OSError:
        lines = []
    return f"last log line: {lines[-1]}" if lines else f"log: nothing written yet ({log})"


def launchd_plist(job: Job) -> str:
    when = ({"StartInterval": 3600} if job.hourly
            else {"StartCalendarInterval": {"Hour": job.hour, "Minute": job.minute}})
    return plistlib.dumps({
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": job.argv(),
        **when,
        "StandardOutPath": str(job.log),
        "StandardErrorPath": str(job.log),
    }, sort_keys=False).decode()


class Launchd:
    """A per-user launchd agent in ~/Library/LaunchAgents."""

    name = "launchd"

    def __init__(self, run: Run = run_command, home: Optional[Path] = None, uid: Optional[int] = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.run = run
        self.sleep = sleep
        self.plist = (home or Path.home()) / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
        self.domain = f"gui/{os.getuid() if uid is None else uid}"
        self.service = f"{self.domain}/{LAUNCHD_LABEL}"

    def install_notes(self, job: Job) -> list[str]:
        return []

    def install(self, job: Job) -> None:
        """Write and load the agent, then start a first run. On failure, unload it, put
        back the agent it replaced or delete it when there was none, and raise
        ScheduleError."""
        previous = self.plist.read_bytes() if self.plist.exists() else None
        self.run(["launchctl", "bootout", self.service])  # replaces a loaded agent; fails harmlessly otherwise
        self.plist.parent.mkdir(parents=True, exist_ok=True)
        self.plist.write_text(launchd_plist(job))
        try:
            self._bootstrap()
            _checked(self.run, ["launchctl", "kickstart", self.service])
        except ScheduleError as exc:
            self.run(["launchctl", "bootout", self.service])
            if previous is None:
                self.plist.unlink(missing_ok=True)
                raise
            self.plist.write_bytes(previous)
            try:
                self._bootstrap()
            except ScheduleError:
                raise ScheduleError(f"{exc}. The job installed before is back in place but didn't load; "
                                    "`ccdrift schedule status` shows it.") from exc
            raise ScheduleError(f"{exc}. The job installed before is back in place.") from exc

    def _bootstrap(self) -> None:
        """Load the agent. bootout can return while the agent it unloads, or a check
        that agent runs, is still going, and bootstrap fails until it is gone, so a
        failure is tried again for a few seconds."""
        argv = ["launchctl", "bootstrap", self.domain, str(self.plist)]
        for attempt in range(BOOTSTRAP_ATTEMPTS):
            if attempt:
                self.sleep(1)
            result = self.run(argv)
            if result.returncode == 0:
                return
        raise ScheduleError(_failure(argv, result))

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
        when = job.get("StartCalendarInterval")
        schedule = f"daily at {when['Hour']:02d}:{when['Minute']:02d}" if when else "every hour"
        lines = [f"installed: launchd agent {LAUNCHD_LABEL}, {schedule}"]
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
    """The service and timer files for a systemd user timer that runs `job` every hour or daily."""
    command = " ".join(_systemd_quote(arg) for arg in job.argv())
    log = str(job.log).replace("%", "%%")
    service = (
        "[Unit]\n"
        "Description=ccdrift check\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        f"ExecStart={command}\n"
        f"StandardOutput=append:{log}\n"
        f"StandardError=append:{log}\n"
    )
    timer = (
        "[Unit]\n"
        "Description=Run the ccdrift check\n"
        "\n"
        "[Timer]\n"
        f"OnCalendar={'hourly' if job.hourly else f'*-*-* {job.hour:02d}:{job.minute:02d}:00'}\n"
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
        """Write both units, enable the timer and start a first run. On failure, put
        back the units they replaced, or disable and delete them when there were none,
        and raise ScheduleError."""
        previous = {path: path.read_text() for path in (self.timer, self.service) if path.exists()}
        self.unit_dir.mkdir(parents=True, exist_ok=True)
        for name, text in systemd_units(job).items():
            (self.unit_dir / name).write_text(text)
        try:
            _checked(self.run, ["systemctl", "--user", "daemon-reload"])
            _checked(self.run, ["systemctl", "--user", "enable", "--now", self.timer.name])
            _checked(self.run, ["systemctl", "--user", "start", "--no-block", self.service.name])
        except ScheduleError as exc:
            if self.timer not in previous:
                self._delete()
                raise
            for path in (self.timer, self.service):
                if path in previous:
                    path.write_text(previous[path])
                else:
                    path.unlink(missing_ok=True)
            self.run(["systemctl", "--user", "daemon-reload"])
            if self.run(["systemctl", "--user", "enable", "--now", self.timer.name]).returncode != 0:
                raise ScheduleError(f"{exc}. The job installed before is back in place but couldn't be enabled; "
                                    "`ccdrift schedule status` shows it.") from exc
            raise ScheduleError(f"{exc}. The job installed before is back in place.") from exc

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
        text = self.timer.read_text()
        daily = re.search(r"^OnCalendar=\*-\*-\* (\d\d:\d\d):00$", text, re.MULTILINE)
        if re.search(r"^OnCalendar=hourly$", text, re.MULTILINE):
            schedule = "every hour"
        else:
            schedule = f"daily at {daily.group(1) if daily else 'an unreadable time'}"
        lines = [f"installed: systemd timer {self.timer.name}, {schedule}"]
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


def cron_line(job: Job) -> str:
    """A crontab line that runs `job` every hour or daily, appending its output to the log."""
    command = " ".join(shlex.quote(arg) for arg in job.argv()).replace("%", "\\%")
    log = shlex.quote(str(job.log)).replace("%", "\\%")
    timing = "0 * * * *" if job.hourly else f"{job.minute} {job.hour} * * *"
    return f"{timing} {command} >> {log} 2>&1 {CRON_MARKER}"


def _spawn(argv: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as out:
        subprocess.Popen(argv, stdout=out, stderr=out, stdin=subprocess.DEVNULL, start_new_session=True)


class Cron:
    """A marked line in the user's crontab, for Linux without a systemd user session."""

    name = "cron"

    def __init__(self, run: Run = run_command, spawn: Callable[[list[str], Path], None] = _spawn):
        self.run = run
        self.spawn = spawn

    def install_notes(self, job: Job) -> list[str]:
        notes = ["Cron doesn't catch up on runs missed while the machine was off."]
        if job.notify:
            notes.append("Jobs started by cron usually can't show notifications, so alerts will mostly "
                         "reach only the log.")
        return notes

    @staticmethod
    def _ours(line: str) -> bool:
        return line.rstrip().endswith(CRON_MARKER)

    def _lines(self) -> list[str]:
        argv = ["crontab", "-l"]
        listed = self.run(argv)
        if listed.returncode == 0:
            return (listed.stdout or "").splitlines()
        if "no crontab" in (listed.stderr or "").lower():
            return []
        raise ScheduleError(_failure(argv, listed))

    def _write(self, lines: list[str]) -> None:
        _checked(self.run, ["crontab", "-"], input="".join(line + "\n" for line in lines))

    def install(self, job: Job) -> None:
        """Replace ccdrift's crontab line, keeping every other line. Cron can't start a
        run on demand, so the job's command also runs once now, in the background. When
        it can't, the crontab is put back as it was."""
        lines = self._lines()
        self._write([line for line in lines if not self._ours(line)] + [cron_line(job)])
        try:
            self.spawn(job.argv(), job.log)
        except OSError as exc:
            self._write(lines)
            raise ScheduleError(f"couldn't start a first run: {exc}") from exc

    def remove(self) -> bool:
        lines = self._lines()
        kept = [line for line in lines if not self._ours(line)]
        if len(kept) == len(lines):
            return False
        self._write(kept)
        return True

    def status(self) -> list[str]:
        ours = [line for line in self._lines() if self._ours(line)]
        if not ours:
            return ["not installed"]
        minute, hour = ours[0].split()[:2]
        schedule = "every hour" if hour == "*" else f"daily at {int(hour):02d}:{int(minute):02d}"
        lines = [f"installed: crontab line, {schedule}",
                 "cron keeps no run history; the log shows each run"]
        # The line ends `>> LOG 2>&1 # ccdrift check`; an --exec command can hold `>> ` too.
        try:
            words = shlex.split(ours[0].replace("\\%", "%"))
        except ValueError:
            words = []
        if len(words) >= 6 and words[-6] == ">>":
            lines.append(last_log_line(Path(words[-5])))
        return lines


def install(job: Job, backend, send: Callable[[str, str], None] = send_notification) -> None:
    """Create the log only its owner can read, install the job (the backend also
    starts a first run), and send a test notification unless the job runs without
    notifications. Raises ScheduleError, before changing anything, for a job holding a
    control character: in a systemd unit or a crontab, what follows a line break reads
    as a line of its own."""
    for value in (*job.argv(), str(job.log)):
        if CONTROL_CHARS.search(value):
            raise ScheduleError(f"a line break or other control character can't go into a scheduled job: {value!r}")
    job.log.parent.mkdir(parents=True, exist_ok=True)
    # launchd, systemd and cron append to a log that exists and keep its permissions.
    make_private(job.log)
    backend.install(job)
    if job.notify:
        send("ccdrift", f"The check will run {job.when()}. Alerts will look like this.")


def choose_backend(platform: str = sys.platform, run: Run = run_command,
                   which: Callable[[str], Optional[str]] = shutil.which):
    """The scheduler ccdrift can set up on this system, or None: launchd on macOS; on
    Linux a systemd user timer when a user session answers, otherwise cron."""
    if platform == "darwin":
        return Launchd(run=run)
    if platform.startswith("linux"):
        if which("systemctl") and run(["systemctl", "--user", "show-environment"]).returncode == 0:
            return Systemd(run=run)
        if which("crontab"):
            return Cron(run=run)
    return None
