"""Desktop notifications for alerts."""

import subprocess

from ccdrift.notify import notify


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
