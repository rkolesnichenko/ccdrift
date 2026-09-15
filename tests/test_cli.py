"""The ccdrift command line: check, peek, --version, and where it looks by default."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from ccdrift import __version__
from ccdrift.check import ccdrift_home
from ccdrift.cli import main
from ccdrift.logs import default_source
from tests.helpers import at, busy_days, line, prompt, text, write


def one_response(folder):
    write(folder / "s1.jsonl", [prompt(at(0)), line("m1", text(40), ts=at(0))])


def test_source_defaults_to_projects_in_the_claude_config_dir(tmp_path):
    assert default_source({"CLAUDE_CONFIG_DIR": str(tmp_path / "cfg")}) == tmp_path / "cfg" / "projects"


def test_source_defaults_to_claude_projects_in_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_source({}) == tmp_path / ".claude" / "projects"


def test_ccdrift_home_follows_its_variable_then_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert ccdrift_home({"CCDRIFT_HOME": str(tmp_path / "data")}) == tmp_path / "data"
    assert ccdrift_home({}) == tmp_path / ".ccdrift"


def test_check_writes_nothing_in_the_working_folder(tmp_path, monkeypatch):
    # launchd starts jobs in /, where nothing can be written.
    one_response(tmp_path / "logs")
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    assert main(["check", "--source", str(tmp_path / "logs"), "--state", str(tmp_path / "state.json")]) == 0
    assert list(workdir.iterdir()) == []


def test_check_keeps_its_state_in_ccdrift_home(tmp_path, monkeypatch):
    busy_days(tmp_path / "logs", days=3, per_day=60)  # no cache usage: an alert that saves state
    monkeypatch.setenv("CCDRIFT_HOME", str(tmp_path / "home"))
    assert main(["check", "--source", str(tmp_path / "logs")]) == 0
    assert (tmp_path / "home" / "check-state.json").exists()


def test_check_fails_and_names_the_folder_without_transcripts(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert main(["check", "--source", str(tmp_path / "empty"), "--state", str(tmp_path / "state.json")]) == 1
    assert str(tmp_path / "empty") in capsys.readouterr().out


def test_peek_shows_the_fields_read_from_the_first_response(tmp_path, capsys):
    one_response(tmp_path / "logs")
    assert main(["peek", "--source", str(tmp_path / "logs")]) == 0
    assert "  model" + " " * 12 + "-> 'claude-opus-5'" in capsys.readouterr().out.splitlines()


def test_peek_exits_2_and_names_the_folder_without_transcripts(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert main(["peek", "--source", str(tmp_path / "empty")]) == 2
    assert str(tmp_path / "empty") in capsys.readouterr().err


def test_version_prints_the_package_version(capsys):
    with pytest.raises(SystemExit) as exited:
        main(["--version"])
    assert exited.value.code == 0
    assert capsys.readouterr().out == f"ccdrift {__version__}\n"


def test_python_m_ccdrift_runs_the_command_line():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "ccdrift", "--version"], capture_output=True, text=True,
                            cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")})
    assert result.stdout == f"ccdrift {__version__}\n"
