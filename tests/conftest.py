"""Keeps every test's ccdrift files out of the real home folder, and its transcripts out of Claude Code's."""

import os
import time

import pytest


@pytest.fixture(autouse=True)
def isolated_ccdrift_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CCDRIFT_HOME", str(tmp_path / "ccdrift-home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))


@pytest.fixture
def machine_zone():
    """Sets the machine's own time zone for one test, from a POSIX TZ string, and puts the
    old one back after it. Skips where Python can't change it (Windows has no time.tzset)."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is POSIX only")
    old = os.environ.get("TZ")

    def set_zone(tz):
        os.environ["TZ"] = tz
        time.tzset()

    yield set_zone
    if old is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = old
    time.tzset()
