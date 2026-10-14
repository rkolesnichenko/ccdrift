"""Keeps every test's ccdrift files out of the real home folder, and its transcripts out of Claude Code's."""

import pytest


@pytest.fixture(autouse=True)
def isolated_ccdrift_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CCDRIFT_HOME", str(tmp_path / "ccdrift-home"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config"))
