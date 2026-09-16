"""Keeps every test's ccdrift files out of the real home folder."""

import pytest


@pytest.fixture(autouse=True)
def isolated_ccdrift_home(tmp_path, monkeypatch):
    monkeypatch.setenv("CCDRIFT_HOME", str(tmp_path / "ccdrift-home"))
