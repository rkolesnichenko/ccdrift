"""The ccdrift command line: check, peek, --version, and where it looks by default."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ccdrift import __version__
from ccdrift.check import ccdrift_home
from ccdrift.cli import main
from ccdrift.logs import default_source
from tests.helpers import at, busy_days, decode_limit, deep_line, line, prompt, text, write


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


def test_peek_shows_text_ids_and_folders_only_as_their_length(tmp_path, capsys):
    # Alerts send users to `ccdrift peek` and on to an issue, where its output gets pasted.
    record = line("m1", {"type": "text", "text": "the plan for acme"}, ts=at(0), sid="0f6c-session",
                  version="2.1.260")
    record.update(cwd="/home/someone/acme-deal", gitBranch="acme/merger")
    write(tmp_path / "logs" / "-home-someone-acme-deal" / "s1.jsonl", [prompt(at(0)), record])
    assert main(["peek", "--source", str(tmp_path / "logs")]) == 0
    out = capsys.readouterr().out
    assert "acme" not in out and "0f6c-session" not in out
    assert '"text": "<17 chars>"' in out
    assert "  version" + " " * 10 + "-> '2.1.260'" in out.splitlines()


def test_peek_shows_a_content_block_by_its_type_and_the_size_of_the_rest(tmp_path, capsys):
    # A tool call's input is whatever the conversation put there, keys included, even
    # under a name peek shows as logged elsewhere, such as an MCP tool's "type" or "model".
    block = {"type": "tool_use", "id": "toolu_1", "name": "mcp__crm__lookup",
             "input": {"type": "acme payroll", "model": "/Users/someone/acme/plan.txt", "acme_ref": 7}}
    write(tmp_path / "logs" / "s1.jsonl", [prompt(at(0)), line("m1", block, ts=at(0))])
    assert main(["peek", "--source", str(tmp_path / "logs")]) == 0
    out = capsys.readouterr().out
    assert "acme" not in out
    assert '"type": "tool_use"' in out and '"input": "<3 keys>"' in out


def test_peek_does_not_fail_on_a_line_nested_too_deep_to_decode_or_show(tmp_path, capsys):
    # One line past the decoder's limit, and one it decodes but that is deeper than
    # the Python recursion limit a walk over it would need on 3.13.
    folder = tmp_path / "logs"
    folder.mkdir()
    (folder / "s1.jsonl").write_text(deep_line("m1", decode_limit() + 10, ts=at(0))
                                     + deep_line("m2", sys.getrecursionlimit(), ts=at(1))
                                     + json.dumps(line("m3", text(40), ts=at(2))) + "\n")
    assert main(["peek", "--source", str(folder)]) == 0
    assert "# first assistant line, text shown as its length" in capsys.readouterr().out


def test_peek_exits_2_and_names_the_folder_without_transcripts(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert main(["peek", "--source", str(tmp_path / "empty")]) == 2
    assert str(tmp_path / "empty") in capsys.readouterr().err


def test_help_says_what_the_check_does(capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "    check     follow incidents and changes, and alert on each " \
           "(what the schedule runs)" in capsys.readouterr().out.splitlines()


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


def test_check_and_schedule_install_take_no_digest():
    from ccdrift.cli import build_parser
    assert build_parser().parse_args(["check", "--no-digest"]).no_digest is True
    assert build_parser().parse_args(["schedule", "install", "--no-digest"]).no_digest is True


def test_cost_rejects_a_dimension_it_does_not_know(capsys):
    with pytest.raises(SystemExit):
        main(["cost", "--by", "nonsense"])
    assert "invalid choice" in capsys.readouterr().err
