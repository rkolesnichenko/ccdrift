"""The daily schedule: the job, the scheduler files, and the commands that install,
remove and report on it. Scheduler commands run through fakes; nothing touches the
real system."""

import plistlib
import subprocess
from pathlib import Path

import pytest

from ccdrift.cli import main
from ccdrift.schedule import (Launchd, ScheduleError, Systemd, choose_backend, install, launchd_plist,
                              make_job, parse_at, systemd_units)


class FakeRun:
    """Stands in for running scheduler commands: records each command and answers
    with the canned (exit code, stdout) for the first matching prefix, else success."""

    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def __call__(self, argv, input=None):
        self.calls.append(argv)
        for prefix, (code, stdout) in self.results.items():
            if tuple(argv[:len(prefix)]) == prefix:
                return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr="failed" if code else "")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def launchd(tmp_path, results=None):
    run = FakeRun(results)
    return run, Launchd(run=run, home=tmp_path, uid=501)


def job_for(tmp_path):
    return make_job("09:00", notify=True, python="/venv/bin/python",
                    environ={"CCDRIFT_HOME": str(tmp_path / "data")})


@pytest.mark.parametrize("text, expected", [("09:00", (9, 0)), ("7:05", (7, 5)), ("23:59", (23, 59))])
def test_at_reads_24_hour_times(text, expected):
    assert parse_at(text) == expected


@pytest.mark.parametrize("text", ["9am", "24:00", "12:60", "12"])
def test_at_rejects_anything_else(text):
    with pytest.raises(ValueError):
        parse_at(text)


def test_job_runs_the_check_with_notifications_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    job = make_job("09:00", notify=True, environ={}, python="/venv/bin/python")
    assert job.argv() == ["/venv/bin/python", "-m", "ccdrift", "check", "--notify"]
    assert job.log == tmp_path / ".ccdrift" / "check.log"


def test_job_keeps_the_folders_set_at_install_time():
    # launchd and systemd don't see shell variables, so the job carries them.
    job = make_job("09:00", notify=False, python="/venv/bin/python",
                   environ={"CLAUDE_CONFIG_DIR": "/cfg", "CCDRIFT_HOME": "/data"})
    assert job.argv() == ["/venv/bin/python", "-m", "ccdrift", "check",
                          "--source", "/cfg/projects", "--state", "/data/check-state.json"]
    assert job.log == Path("/data/check.log")


def test_job_prefers_a_source_given_to_install():
    job = make_job("09:00", notify=False, source="/logs", python="/venv/bin/python",
                   environ={"CLAUDE_CONFIG_DIR": "/cfg"})
    assert job.argv()[-2:] == ["--source", "/logs"]


def test_launchd_plist_runs_the_job_daily_and_logs_its_output():
    job = make_job("07:30", notify=True, python="/venv/bin/python", environ={"CCDRIFT_HOME": "/data"})
    assert plistlib.loads(launchd_plist(job).encode()) == {
        "Label": "io.github.rkolesnichenko.ccdrift",
        "ProgramArguments": ["/venv/bin/python", "-m", "ccdrift", "check", "--notify",
                             "--state", "/data/check-state.json"],
        "StartCalendarInterval": {"Hour": 7, "Minute": 30},
        "StandardOutPath": "/data/check.log",
        "StandardErrorPath": "/data/check.log",
    }


def test_launchd_install_writes_the_agent_and_starts_a_first_run(tmp_path):
    run, backend = launchd(tmp_path)
    backend.install(job_for(tmp_path))
    plist = tmp_path / "Library" / "LaunchAgents" / "io.github.rkolesnichenko.ccdrift.plist"
    assert plist.exists()
    assert run.calls == [
        ["launchctl", "bootout", "gui/501/io.github.rkolesnichenko.ccdrift"],
        ["launchctl", "bootstrap", "gui/501", str(plist)],
        ["launchctl", "kickstart", "gui/501/io.github.rkolesnichenko.ccdrift"],
    ]


def test_launchd_install_leaves_nothing_behind_when_loading_fails(tmp_path):
    run, backend = launchd(tmp_path, {("launchctl", "bootstrap"): (5, "")})
    with pytest.raises(ScheduleError):
        backend.install(job_for(tmp_path))
    assert not (tmp_path / "Library" / "LaunchAgents" / "io.github.rkolesnichenko.ccdrift.plist").exists()
    assert run.calls[-1] == ["launchctl", "bootout", "gui/501/io.github.rkolesnichenko.ccdrift"]


def test_launchd_remove_unloads_the_agent_and_deletes_it(tmp_path):
    run, backend = launchd(tmp_path)
    backend.install(job_for(tmp_path))
    run.calls.clear()
    assert backend.remove() is True
    assert run.calls == [["launchctl", "bootout", "gui/501/io.github.rkolesnichenko.ccdrift"]]
    assert not backend.plist.exists()


def test_launchd_remove_says_so_when_nothing_is_installed(tmp_path):
    run, backend = launchd(tmp_path)
    assert backend.remove() is False
    assert run.calls == []


def test_launchd_status_shows_runs_exit_code_and_the_last_log_line(tmp_path):
    printed = "gui/501/io.github.rkolesnichenko.ccdrift = {\n\tstate = not running\n\truns = 2\n\tlast exit code = 0\n}\n"
    run, backend = launchd(tmp_path, {("launchctl", "print"): (0, printed)})
    job = job_for(tmp_path)
    backend.install(job)
    job.log.parent.mkdir(parents=True)
    job.log.write_text("[check 2026-09-16 09:00] no new flags\n")
    assert backend.status() == [
        "installed: launchd agent io.github.rkolesnichenko.ccdrift, daily at 09:00",
        "runs: 2",
        "last exit code: 0",
        "last log line: [check 2026-09-16 09:00] no new flags",
    ]


def test_launchd_status_says_when_nothing_is_installed(tmp_path):
    run, backend = launchd(tmp_path)
    assert backend.status() == ["not installed"]


def test_install_creates_the_log_folder_and_sends_a_test_notification(tmp_path):
    run, backend = launchd(tmp_path)
    sent = []
    job = job_for(tmp_path)
    install(job, backend, send=lambda title, message: sent.append(title))
    assert job.log.parent.is_dir()
    assert sent == ["ccdrift"]


def test_install_without_notifications_sends_no_test(tmp_path):
    run, backend = launchd(tmp_path)
    sent = []
    job = make_job("09:00", notify=False, python="/venv/bin/python",
                   environ={"CCDRIFT_HOME": str(tmp_path / "data")})
    install(job, backend, send=lambda title, message: sent.append(title))
    assert sent == []


def test_macos_gets_launchd():
    assert isinstance(choose_backend(platform="darwin", run=FakeRun()), Launchd)


def test_windows_gets_no_scheduler():
    assert choose_backend(platform="win32", run=FakeRun()) is None


def test_schedule_install_rejects_a_bad_time():
    assert main(["schedule", "install", "--at", "25:00"]) == 2


def test_schedule_prints_the_command_where_it_cant_set_up_a_job(monkeypatch, capsys):
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CCDRIFT_HOME", raising=False)
    monkeypatch.setattr("ccdrift.cli.choose_backend", lambda: None)
    assert main(["schedule", "install"]) == 2
    assert "-m ccdrift check --notify" in capsys.readouterr().err


def test_schedule_install_exits_1_when_the_scheduler_refuses(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CCDRIFT_HOME", str(tmp_path / "data"))
    run, backend = launchd(tmp_path, {("launchctl", "bootstrap"): (5, "")})
    monkeypatch.setattr("ccdrift.cli.choose_backend", lambda: backend)
    assert main(["schedule", "install", "--no-notify"]) == 1
    assert "Nothing installed" in capsys.readouterr().err


def test_schedule_install_reports_the_job_and_its_log(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CCDRIFT_HOME", str(tmp_path / "data"))
    run, backend = launchd(tmp_path)
    monkeypatch.setattr("ccdrift.cli.choose_backend", lambda: backend)
    assert main(["schedule", "install", "--at", "18:30", "--no-notify"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Installed a launchd job: `ccdrift check` runs daily at 18:30.",
        f"Log: {tmp_path / 'data' / 'check.log'}",
        "A first run has started. Check `ccdrift schedule status` in a minute.",
    ]


def systemd(tmp_path, results=None):
    run = FakeRun(results)
    return run, Systemd(run=run, config_home=tmp_path / "config")


def test_systemd_units_run_the_job_daily_with_quoted_arguments():
    job = make_job("06:05", notify=True, python="/opt/my env/bin/python",
                   environ={"CCDRIFT_HOME": "/home/u/.ccdrift"})
    units = systemd_units(job)
    assert units["ccdrift-check.service"] == (
        "[Unit]\n"
        "Description=ccdrift daily check\n"
        "\n"
        "[Service]\n"
        "Type=oneshot\n"
        'ExecStart="/opt/my env/bin/python" "-m" "ccdrift" "check" "--notify" "--state" '
        '"/home/u/.ccdrift/check-state.json"\n'
        "StandardOutput=append:/home/u/.ccdrift/check.log\n"
        "StandardError=append:/home/u/.ccdrift/check.log\n"
    )
    assert units["ccdrift-check.timer"] == (
        "[Unit]\n"
        "Description=Run the ccdrift daily check\n"
        "\n"
        "[Timer]\n"
        "OnCalendar=*-*-* 06:05:00\n"
        "Persistent=true\n"
        "\n"
        "[Install]\n"
        "WantedBy=timers.target\n"
    )


def test_systemd_install_writes_both_units_and_starts_a_first_run(tmp_path):
    run, backend = systemd(tmp_path)
    backend.install(job_for(tmp_path))
    unit_dir = tmp_path / "config" / "systemd" / "user"
    assert sorted(p.name for p in unit_dir.iterdir()) == ["ccdrift-check.service", "ccdrift-check.timer"]
    assert run.calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", "--now", "ccdrift-check.timer"],
        ["systemctl", "--user", "start", "--no-block", "ccdrift-check.service"],
    ]


def test_systemd_install_leaves_nothing_behind_when_enabling_fails(tmp_path):
    run, backend = systemd(tmp_path, {("systemctl", "--user", "enable"): (1, "")})
    with pytest.raises(ScheduleError):
        backend.install(job_for(tmp_path))
    assert list((tmp_path / "config" / "systemd" / "user").iterdir()) == []


def test_systemd_remove_disables_the_timer_and_deletes_both_units(tmp_path):
    run, backend = systemd(tmp_path)
    backend.install(job_for(tmp_path))
    run.calls.clear()
    assert backend.remove() is True
    assert run.calls == [
        ["systemctl", "--user", "disable", "--now", "ccdrift-check.timer"],
        ["systemctl", "--user", "daemon-reload"],
    ]
    assert list((tmp_path / "config" / "systemd" / "user").iterdir()) == []


def test_systemd_status_shows_the_last_run_and_log_line(tmp_path):
    shown = "ExecMainStartTimestamp=Wed 2026-09-16 09:00:01 EEST\nExecMainStatus=0\n"
    run, backend = systemd(tmp_path, {("systemctl", "--user", "is-enabled"): (0, "enabled\n"),
                                      ("systemctl", "--user", "show"): (0, shown)})
    job = job_for(tmp_path)
    backend.install(job)
    job.log.parent.mkdir(parents=True)
    job.log.write_text("[check 2026-09-16 09:00] no new flags\n")
    assert backend.status() == [
        "installed: systemd timer ccdrift-check.timer, daily at 09:00",
        "enabled: enabled",
        "last run: Wed 2026-09-16 09:00:01 EEST",
        "last exit code: 0",
        "last log line: [check 2026-09-16 09:00] no new flags",
    ]
