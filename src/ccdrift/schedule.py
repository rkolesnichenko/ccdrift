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
from ccdrift.state import LOG_FILE, ccdrift_home, make_private
from ccdrift.texts import SCHEDULE_LINES

LAUNCHD_LABEL = "io.github.rkolesnichenko.ccdrift"
SYSTEMD_UNIT = "ccdrift-check"
BOOTSTRAP_ATTEMPTS = 5

# The schedulers' own formats, not ccdrift's wording: the marker that finds ccdrift's
# crontab line again, what `crontab -l` prints without a crontab, the fields of `launchctl
# print` shown as they are, and the systemd unit files.
CRON_MARKER = "# ccdrift check"
CRONTAB_MISSING = "no crontab"
LAUNCHD_FIELDS = ("runs", "last exit code")
SERVICE_UNIT = ("[Unit]\n"
                "Description=ccdrift check\n"
                "\n"
                "[Service]\n"
                "Type=oneshot\n"
                "ExecStart={command}\n"
                "StandardOutput=append:{log}\n"
                "StandardError=append:{log}\n")
TIMER_UNIT = ("[Unit]\n"
              "Description=Run the ccdrift check\n"
              "\n"
              "[Timer]\n"
              "OnCalendar={calendar}\n"
              "Persistent=true\n"
              "\n"
              "[Install]\n"
              "WantedBy=timers.target\n")

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
    return (SCHEDULE_LINES["exited"].format(command=" ".join(argv), code=result.returncode)
            + (SCHEDULE_LINES["output"].format(output=output) if output else ""))


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
        if self.hourly:
            return SCHEDULE_LINES["hourly"]
        return SCHEDULE_LINES["daily"].format(time=SCHEDULE_LINES["clock"].format(hour=self.hour, minute=self.minute))


def parse_at(text: str) -> tuple[int, int]:
    """'09:00' -> (9, 0). Raises ValueError for anything but a 24-hour HH:MM time."""
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if not match or int(match.group(1)) > 23 or int(match.group(2)) > 59:
        raise ValueError(SCHEDULE_LINES["bad_at"].format(text=text))
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
    return Job(python=python, hour=hour, minute=minute, notify=notify, log=home / LOG_FILE,
               source=source_path, state=state, exec_command=exec_command, no_digest=no_digest)


def last_log_line(log: Path) -> str:
    try:
        lines = [line for line in log.read_text(errors="replace").splitlines() if line.strip()]
    except OSError:
        lines = []
    return SCHEDULE_LINES["last_log"].format(line=lines[-1]) if lines else SCHEDULE_LINES["no_log"].format(log=log)


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
                raise ScheduleError(SCHEDULE_LINES["restored_unloaded"].format(error=exc)) from exc
            raise ScheduleError(SCHEDULE_LINES["restored"].format(error=exc)) from exc

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
            return [SCHEDULE_LINES["not_installed"]]
        # The plist is ccdrift's own, but a half-written or hand-edited one must read as a
        # status rather than as a traceback: `status` is what someone runs to find out why
        # the job is misbehaving.
        try:
            job = plistlib.loads(self.plist.read_bytes())
        except (OSError, ValueError, plistlib.InvalidFileException) as exc:
            raise ScheduleError(SCHEDULE_LINES["bad_plist"].format(plist=self.plist, error=exc)) from exc
        when = job.get("StartCalendarInterval") or {}
        clock = SCHEDULE_LINES["clock"].format(hour=when.get("Hour", 0), minute=when.get("Minute", 0))
        schedule = SCHEDULE_LINES["daily"].format(time=clock) if when else SCHEDULE_LINES["hourly"]
        lines = [SCHEDULE_LINES["launchd_installed"].format(label=LAUNCHD_LABEL, schedule=schedule)]
        printed = self.run(["launchctl", "print", self.service])
        if printed.returncode != 0:
            lines.append(SCHEDULE_LINES["not_loaded"])
        for key in LAUNCHD_FIELDS:
            found = re.search(rf"^\s*{key} = (.+)$", printed.stdout or "", re.MULTILINE)
            if found:
                lines.append(SCHEDULE_LINES["field"].format(key=key, value=found.group(1).strip()))
        log = job.get("StandardOutPath")
        lines.append(last_log_line(Path(log)) if log else SCHEDULE_LINES["no_log_file"])
        return lines


def _systemd_quote(arg: str) -> str:
    escaped = arg.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$")
    return f'"{escaped}"'


def systemd_units(job: Job) -> dict[str, str]:
    """The service and timer files for a systemd user timer that runs `job` every hour or daily."""
    command = " ".join(_systemd_quote(arg) for arg in job.argv())
    log = str(job.log).replace("%", "%%")
    calendar = "hourly" if job.hourly else f"*-*-* {job.hour:02d}:{job.minute:02d}:00"
    return {f"{SYSTEMD_UNIT}.service": SERVICE_UNIT.format(command=command, log=log),
            f"{SYSTEMD_UNIT}.timer": TIMER_UNIT.format(calendar=calendar)}


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
        return [SCHEDULE_LINES["systemd_note"]]

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
                raise ScheduleError(SCHEDULE_LINES["restored_disabled"].format(error=exc)) from exc
            raise ScheduleError(SCHEDULE_LINES["restored"].format(error=exc)) from exc

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
            return [SCHEDULE_LINES["not_installed"]]
        text = self.timer.read_text()
        daily = re.search(r"^OnCalendar=\*-\*-\* (\d\d:\d\d):00$", text, re.MULTILINE)
        if re.search(r"^OnCalendar=hourly$", text, re.MULTILINE):
            schedule = SCHEDULE_LINES["hourly"]
        else:
            clock = daily.group(1) if daily else SCHEDULE_LINES["unreadable_time"]
            schedule = SCHEDULE_LINES["daily"].format(time=clock)
        lines = [SCHEDULE_LINES["systemd_installed"].format(timer=self.timer.name, schedule=schedule)]
        enabled = self.run(["systemctl", "--user", "is-enabled", self.timer.name])
        state = (enabled.stdout or "").strip() or SCHEDULE_LINES["unknown"]
        lines.append(SCHEDULE_LINES["enabled"].format(state=state))
        shown = self.run(["systemctl", "--user", "show", self.service.name,
                          "-p", "ExecMainStartTimestamp", "-p", "ExecMainStatus"])
        props = dict(line.split("=", 1) for line in (shown.stdout or "").splitlines() if "=" in line)
        started = props.get("ExecMainStartTimestamp") or SCHEDULE_LINES["never"]
        lines.append(SCHEDULE_LINES["last_run"].format(at=started))
        lines.append(SCHEDULE_LINES["last_exit"].format(code=props.get("ExecMainStatus", SCHEDULE_LINES["unknown"])))
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


def _cron_schedule(line: str) -> str:
    """How a crontab line reads as a schedule. ccdrift writes `0 * * * *` or `M H * * *`,
    but the crontab is the user's, and a line they edited into a range or a step is still
    ccdrift's job: it is described as unreadable rather than crashing the status."""
    fields = line.split()
    if len(fields) < 2:
        return SCHEDULE_LINES["unreadable_schedule"]
    minute, hour = fields[0], fields[1]
    if hour == "*":
        return SCHEDULE_LINES["hourly"] if minute == "0" else SCHEDULE_LINES["hourly_at"].format(minute=minute)
    try:
        return SCHEDULE_LINES["daily"].format(time=SCHEDULE_LINES["clock"].format(hour=int(hour), minute=int(minute)))
    except ValueError:
        return SCHEDULE_LINES["cron_schedule"].format(fields=" ".join(fields[:5]))


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
        notes = [SCHEDULE_LINES["cron_catch_up"]]
        if job.notify:
            notes.append(SCHEDULE_LINES["cron_notify"])
        return notes

    @staticmethod
    def _ours(line: str) -> bool:
        return line.rstrip().endswith(CRON_MARKER)

    def _lines(self) -> list[str]:
        argv = ["crontab", "-l"]
        listed = self.run(argv)
        if listed.returncode == 0:
            return (listed.stdout or "").splitlines()
        if CRONTAB_MISSING in (listed.stderr or "").lower():
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
            raise ScheduleError(SCHEDULE_LINES["first_run_failed"].format(error=exc)) from exc

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
            return [SCHEDULE_LINES["not_installed"]]
        lines = [SCHEDULE_LINES["cron_installed"].format(schedule=_cron_schedule(ours[0])),
                 SCHEDULE_LINES["cron_history"]]
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
            raise ScheduleError(SCHEDULE_LINES["control_char"].format(value=value))
    job.log.parent.mkdir(parents=True, exist_ok=True)
    # launchd, systemd and cron append to a log that exists and keep its permissions.
    make_private(job.log)
    backend.install(job)
    if job.notify:
        send(SCHEDULE_LINES["test_title"], SCHEDULE_LINES["test_message"].format(when=job.when()))


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
