"""Desktop notifications for alerts."""

import subprocess

from ccdrift.notify import notify, run_exec


def recorder():
    calls = []
    return calls, lambda argv, **kwargs: calls.append(argv)


def test_notify_uses_osascript_on_macos():
    calls, run = recorder()
    notify("ccdrift flag", 'Haiku share "on the main thread"', platform="darwin", run=run,
           which=lambda name: f"/usr/bin/{name}")
    assert calls == [["osascript", "-e",
                      'display notification "Haiku share \\"on the main thread\\"" with title "ccdrift flag"']]


def test_notify_uses_notify_send_on_linux():
    calls, run = recorder()
    notify("ccdrift flag", "Cache read ratio on new prompts flagged from 2026-09-20", platform="linux",
           run=run, which=lambda name: f"/usr/bin/{name}")
    assert calls == [["notify-send", "ccdrift flag", "Cache read ratio on new prompts flagged from 2026-09-20"]]


def test_notify_does_nothing_without_a_notifier():
    calls, run = recorder()
    notify("ccdrift flag", "message", platform="linux", run=run, which=lambda name: None)
    assert calls == []


def test_notify_ignores_a_notifier_that_fails_to_start():
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        raise OSError("not executable")

    notify("ccdrift flag", "message", platform="linux", run=run, which=lambda name: "/usr/bin/notify-send")
    assert calls == [["notify-send", "ccdrift flag", "message"]]


def test_notify_ignores_a_notifier_that_times_out():
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        raise subprocess.TimeoutExpired(argv, 10)

    notify("ccdrift flag", "message", platform="linux", run=run, which=lambda name: "/usr/bin/notify-send")
    assert [argv for argv, kwargs in calls] == [["notify-send", "ccdrift flag", "message"]]
    assert calls[0][1]["timeout"] == 10


def test_exec_runs_the_command_through_the_shell_with_the_alert_in_its_environment(tmp_path):
    out = tmp_path / "alert.txt"
    command = f'printf "%s|%s|%s" "$CCDRIFT_ALERT" "$CCDRIFT_TITLE" "$CCDRIFT_MESSAGE" > "{out}"'
    assert run_exec(command, "flag", "ccdrift flag", "Haiku share 'up' from 2026-09-15") is None
    assert out.read_text() == "flag|ccdrift flag|Haiku share 'up' from 2026-09-15"


def test_exec_says_why_a_command_failed():
    assert run_exec("echo no route >&2; exit 3", "flag", "ccdrift flag", "message") == "exit 3: no route"


def test_exec_says_why_a_command_failed_when_its_output_isnt_utf8():
    assert run_exec(r"printf '\377\376' >&2; exit 1", "flag", "ccdrift flag", "message") == "exit 1: ��"


def test_exec_gives_up_on_a_command_after_30_seconds():
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    assert run_exec("sleep 100", "flag", "ccdrift flag", "message", run=run) == "timed out after 30 s"
    assert calls == [30]
