"""The daily schedule: the job, the scheduler files, and the commands that install,
remove and report on it. Scheduler commands run through fakes; nothing touches the
real system."""

import plistlib
import subprocess
from pathlib import Path

import pytest

from ccdrift.cli import main
from ccdrift.schedule import (Cron, Launchd, ScheduleError, Systemd, choose_backend, cron_line, install,
                              launchd_plist, make_job, parse_at, systemd_units)


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


def launchd(tmp_path, results=None, sleep=lambda seconds: None):
    run = FakeRun(results)
    return run, Launchd(run=run, home=tmp_path, uid=501, sleep=sleep)


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


def test_launchd_install_waits_for_the_agent_it_replaces_to_unload(tmp_path):
    # bootout can return while the old agent, or a check it runs, is still going; bootstrap then fails.
    run, backend = launchd(tmp_path, sleep=lambda seconds: run.results.clear())
    run.results[("launchctl", "bootstrap")] = (5, "")
    backend.install(job_for(tmp_path))
    assert [argv[1] for argv in run.calls] == ["bootout", "bootstrap", "bootstrap", "kickstart"]


def test_a_failed_launchd_reinstall_puts_back_the_agent_it_replaced(tmp_path):
    run, backend = launchd(tmp_path)
    backend.install(make_job("09:00", notify=True, python="/venv/bin/python", environ={}))
    before = backend.plist.read_bytes()
    run.results[("launchctl", "kickstart")] = (1, "")
    run.calls.clear()
    with pytest.raises(ScheduleError, match="The job installed before is back in place"):
        backend.install(job_for(tmp_path))
    assert backend.plist.read_bytes() == before
    assert run.calls[-1] == ["launchctl", "bootstrap", "gui/501", str(backend.plist)]


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


def test_schedule_status_exits_1_when_the_scheduler_cant_be_read(monkeypatch, capsys):
    backend = Cron(run=FakeRun({("crontab", "-l"): (1, "")}))
    monkeypatch.setattr("ccdrift.cli.choose_backend", lambda: backend)
    assert main(["schedule", "status"]) == 1
    err = capsys.readouterr().err
    assert "Couldn't read the schedule" in err
    assert "crontab -l" in err


def test_schedule_remove_exits_1_when_the_scheduler_cant_be_read(monkeypatch, capsys):
    backend = Cron(run=FakeRun({("crontab", "-l"): (1, "")}))
    monkeypatch.setattr("ccdrift.cli.choose_backend", lambda: backend)
    assert main(["schedule", "remove"]) == 1
    err = capsys.readouterr().err
    assert "Nothing removed" in err
    assert "crontab -l" in err


def test_schedule_remove_through_cli_reports_whether_a_job_was_installed(tmp_path, monkeypatch, capsys):
    run, backend = launchd(tmp_path)
    monkeypatch.setattr("ccdrift.cli.choose_backend", lambda: backend)
    assert main(["schedule", "remove"]) == 0
    assert capsys.readouterr().out.splitlines() == ["No ccdrift job was installed."]
    backend.install(job_for(tmp_path))
    assert main(["schedule", "remove"]) == 0
    assert capsys.readouterr().out.splitlines() == ["Removed the ccdrift job."]


def test_schedule_status_through_cli_says_when_nothing_is_installed(tmp_path, monkeypatch, capsys):
    run, backend = launchd(tmp_path)
    monkeypatch.setattr("ccdrift.cli.choose_backend", lambda: backend)
    assert main(["schedule", "status"]) == 0
    assert capsys.readouterr().out.splitlines() == ["not installed"]


def systemd(tmp_path, results=None):
    run = FakeRun(results)
    return run, Systemd(run=run, config_home=tmp_path / "config")


def test_systemd_units_run_the_job_daily_with_quoted_arguments():
    job = make_job("06:05", notify=True, python="/opt/my env/bin/python",
                   environ={"CCDRIFT_HOME": "/home/u/.ccdrift"})
    units = systemd_units(job)
    assert units["ccdrift-check.service"] == (
        "[Unit]\n"
        "Description=ccdrift check\n"
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
        "Description=Run the ccdrift check\n"
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


def test_a_failed_systemd_reinstall_puts_back_the_units_it_replaced(tmp_path):
    run, backend = systemd(tmp_path)
    backend.install(make_job("09:00", notify=True, python="/venv/bin/python", environ={}))
    before = {path.name: path.read_text() for path in (backend.timer, backend.service)}
    run.results[("systemctl", "--user", "start")] = (1, "")
    run.calls.clear()
    with pytest.raises(ScheduleError, match="The job installed before is back in place"):
        backend.install(make_job(None, notify=False, python="/venv/bin/python", environ={}))
    assert {path.name: path.read_text() for path in (backend.timer, backend.service)} == before
    assert run.calls[-2:] == [["systemctl", "--user", "daemon-reload"],
                              ["systemctl", "--user", "enable", "--now", "ccdrift-check.timer"]]


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


class FakeCrontab:
    """Stands in for `crontab -l` and `crontab -`, keeping the table in memory."""

    def __init__(self, table=None):
        self.table = table  # None: the user has no crontab yet

    def __call__(self, argv, input=None):
        if argv == ["crontab", "-l"]:
            if self.table is None:
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="crontab: no crontab for u\n")
            return subprocess.CompletedProcess(argv, 0, stdout=self.table, stderr="")
        if argv == ["crontab", "-"]:
            self.table = input
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected command {argv}")


def test_cron_line_quotes_paths_and_ends_with_the_marker():
    job = make_job("06:05", notify=True, python="/opt/my env/bin/python",
                   environ={"CCDRIFT_HOME": "/home/u/.ccdrift"})
    assert cron_line(job) == (
        "5 6 * * * '/opt/my env/bin/python' -m ccdrift check --notify --state /home/u/.ccdrift/check-state.json"
        " >> /home/u/.ccdrift/check.log 2>&1 # ccdrift check"
    )


def test_cron_install_keeps_other_lines_and_replaces_its_own(tmp_path):
    table = FakeCrontab("0 1 * * * backup.sh\n0 9 * * * old-command >> old.log 2>&1 # ccdrift check\n")
    started = []
    job = job_for(tmp_path)
    Cron(run=table, spawn=lambda argv, log: started.append(argv)).install(job)
    assert table.table == "0 1 * * * backup.sh\n" + cron_line(job) + "\n"
    assert started == [job.argv()]


def test_cron_install_starts_a_table_when_there_is_none(tmp_path):
    table = FakeCrontab(None)
    job = job_for(tmp_path)
    Cron(run=table, spawn=lambda argv, log: None).install(job)
    assert table.table == cron_line(job) + "\n"


def test_a_failed_cron_reinstall_keeps_the_line_it_would_replace(tmp_path):
    table = FakeCrontab("0 1 * * * backup.sh\n0 9 * * * old-command >> old.log 2>&1 # ccdrift check\n")
    before = table.table

    def spawn(argv, log):
        raise OSError("no such python")

    with pytest.raises(ScheduleError, match="couldn't start a first run"):
        Cron(run=table, spawn=spawn).install(job_for(tmp_path))
    assert table.table == before


def test_cron_remove_deletes_only_its_line():
    table = FakeCrontab("0 1 * * * backup.sh\n0 9 * * * x >> y 2>&1 # ccdrift check\n")
    assert Cron(run=table).remove() is True
    assert table.table == "0 1 * * * backup.sh\n"


def test_cron_remove_says_so_when_nothing_is_installed():
    table = FakeCrontab("0 1 * * * backup.sh\n")
    assert Cron(run=table).remove() is False
    assert table.table == "0 1 * * * backup.sh\n"


def test_cron_status_shows_the_time_and_the_last_log_line(tmp_path):
    job = job_for(tmp_path)
    table = FakeCrontab(cron_line(job) + "\n")
    job.log.parent.mkdir(parents=True)
    job.log.write_text("[check 2026-09-16 09:00] no new flags\n")
    assert Cron(run=table).status() == [
        "installed: crontab line, daily at 09:00",
        "cron keeps no run history; the log shows each run",
        "last log line: [check 2026-09-16 09:00] no new flags",
    ]


def test_cron_status_finds_the_log_when_the_exec_command_appends_to_a_file(tmp_path):
    job = make_job("09:00", notify=False, exec_command='echo "$CCDRIFT_MESSAGE" >> ~/alerts.txt',
                   python="/venv/bin/python", environ={"CCDRIFT_HOME": str(tmp_path / "data")})
    job.log.parent.mkdir(parents=True)
    job.log.write_text("[check 2026-09-16 09:00] no alerts\n")
    assert Cron(run=FakeCrontab(cron_line(job) + "\n")).status()[-1] == \
        "last log line: [check 2026-09-16 09:00] no alerts"


def test_linux_with_a_systemd_user_session_gets_systemd():
    backend = choose_backend(platform="linux", run=FakeRun(), which=lambda name: f"/usr/bin/{name}")
    assert isinstance(backend, Systemd)


def test_linux_without_a_systemd_user_session_gets_cron():
    run = FakeRun({("systemctl", "--user", "show-environment"): (1, "")})
    backend = choose_backend(platform="linux", run=run, which=lambda name: f"/usr/bin/{name}")
    assert isinstance(backend, Cron)


def test_linux_without_systemd_or_cron_gets_no_scheduler():
    # Passes before this task's change too; it guards against a fallback that
    # picks a scheduler the system doesn't have.
    assert choose_backend(platform="linux", run=FakeRun(), which=lambda name: None) is None


NTFY = 'curl -d "$CCDRIFT_MESSAGE" ntfy.sh/topic 100%'


def test_job_carries_the_exec_command_given_to_install():
    job = make_job("09:00", notify=False, exec_command=NTFY, python="/venv/bin/python", environ={})
    assert job.argv()[-2:] == ["--exec", NTFY]


def test_exec_command_survives_launchd_systemd_and_cron_quoting():
    job = make_job("06:05", notify=False, exec_command=NTFY, python="/venv/bin/python",
                   environ={"CCDRIFT_HOME": "/data"})
    assert plistlib.loads(launchd_plist(job).encode())["ProgramArguments"][-2:] == ["--exec", NTFY]
    assert systemd_units(job)["ccdrift-check.service"].splitlines()[5].endswith(
        '"--exec" "curl -d \\"$$CCDRIFT_MESSAGE\\" ntfy.sh/topic 100%%"')
    assert "--exec 'curl -d \"$CCDRIFT_MESSAGE\" ntfy.sh/topic 100\\%'" in cron_line(job)


def test_schedule_install_passes_exec_to_the_job(tmp_path, monkeypatch):
    monkeypatch.setenv("CCDRIFT_HOME", str(tmp_path / "data"))
    run, backend = launchd(tmp_path)
    monkeypatch.setattr("ccdrift.cli.choose_backend", lambda: backend)
    assert main(["schedule", "install", "--no-notify", "--exec", NTFY]) == 0
    assert plistlib.loads(backend.plist.read_bytes())["ProgramArguments"][-2:] == ["--exec", NTFY]


def test_a_job_without_a_time_runs_every_hour_on_each_scheduler():
    job = make_job(None, notify=True, python="/venv/bin/python", environ={"CCDRIFT_HOME": "/data"})
    assert job.when() == "every hour"
    plist = plistlib.loads(launchd_plist(job).encode())
    assert (plist["StartInterval"], "StartCalendarInterval" in plist) == (3600, False)
    assert "OnCalendar=hourly\nPersistent=true\n" in systemd_units(job)["ccdrift-check.timer"]
    assert cron_line(job).startswith("0 * * * * /venv/bin/python -m ccdrift check --notify ")


def test_each_scheduler_reports_an_hourly_job_as_every_hour(tmp_path):
    job = make_job(None, notify=False, python="/venv/bin/python", environ={"CCDRIFT_HOME": str(tmp_path / "data")})
    _, agent = launchd(tmp_path)
    agent.install(job)
    assert agent.status()[0] == "installed: launchd agent io.github.rkolesnichenko.ccdrift, every hour"
    _, timer = systemd(tmp_path)
    timer.install(job)
    assert timer.status()[0] == "installed: systemd timer ccdrift-check.timer, every hour"
    assert Cron(run=FakeCrontab(cron_line(job) + "\n")).status()[0] == "installed: crontab line, every hour"


def test_schedule_install_without_a_time_sets_up_an_hourly_job(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CCDRIFT_HOME", str(tmp_path / "data"))
    run, backend = launchd(tmp_path)
    monkeypatch.setattr("ccdrift.cli.choose_backend", lambda: backend)
    assert main(["schedule", "install", "--no-notify"]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "Installed a launchd job: `ccdrift check` runs every hour."
    assert plistlib.loads(backend.plist.read_bytes())["StartInterval"] == 3600


def test_job_carries_no_digest_given_to_install():
    job = make_job("09:00", notify=True, no_digest=True, python="/venv/bin/python", environ={})
    assert job.argv() == ["/venv/bin/python", "-m", "ccdrift", "check", "--notify", "--no-digest"]
