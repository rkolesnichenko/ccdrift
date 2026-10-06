"""The Claude Code plugin: its manifests, and the hook script that finds ccdrift and hands it the payload."""

import ast
import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

from ccdrift import __version__

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugin"
SCRIPT = PLUGIN / "hooks" / "session-start.sh"
PAYLOAD = json.dumps({"session_id": "SESSION-ID-1", "hook_event_name": "SessionStart", "source": "startup"})


def manifest(path):
    return json.loads((ROOT / path).read_text())


def test_the_marketplace_lists_the_plugin_from_its_folder_under_the_plugins_own_name():
    market = manifest(".claude-plugin/marketplace.json")
    [entry] = market["plugins"]
    assert (market["name"], market["owner"]["name"]) == ("ccdrift", "Roman Kolesnichenko")
    assert entry["source"] == "./plugin" and (ROOT / entry["source"] / ".claude-plugin" / "plugin.json").is_file()
    assert entry["name"] == manifest("plugin/.claude-plugin/plugin.json")["name"] == "ccdrift"


def test_the_plugins_version_is_the_packages_so_each_release_reaches_plugin_users():
    # Claude Code keeps users on a plugin's set version until it changes.
    assert manifest("plugin/.claude-plugin/plugin.json")["version"] == __version__


def test_the_only_hook_is_a_synchronous_session_start_running_the_script():
    hooks = manifest("plugin/hooks/hooks.json")["hooks"]
    assert list(hooks) == ["SessionStart"]
    [group] = hooks["SessionStart"]
    [hook] = group["hooks"]
    # A matcher would limit which session starts begin a check; async would hide the line.
    assert "matcher" not in group and "async" not in hook
    assert (hook["type"], hook["timeout"]) == ("command", 10)
    path = Path(hook["command"].replace('"${CLAUDE_PLUGIN_ROOT}"', str(PLUGIN)))
    assert path == SCRIPT and os.access(path, os.X_OK)


def test_the_plugin_ships_through_its_marketplace_not_the_package():
    line = re.search(r"^only-include = (\[.*\])$", (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    assert not {"plugin", ".claude-plugin"} & {path.split("/")[0] for path in ast.literal_eval(line)}


def fake(folder, output="", code=0):
    """A ccdrift that notes its arguments and standard input beside itself, prints `output`,
    complains on stderr and exits with `code`."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "ccdrift"
    path.write_text(f"#!/bin/sh\n"
                    f"echo \"$@\" > \"{folder}/args\"\n"
                    f"cat > \"{folder}/stdin\"\n"
                    f"printf '%s' '{output}'\n"
                    f"echo 'Traceback: noise' >&2\n"
                    f"exit {code}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return folder


def hook(tmp_path, *path_dirs):
    env = {"HOME": str(tmp_path / "home"), "PATH": os.pathsep.join([*map(str, path_dirs), "/usr/bin", "/bin"])}
    return subprocess.run(["/bin/sh", str(SCRIPT)], input=PAYLOAD, capture_output=True, text=True, env=env)


def test_the_hook_says_how_to_install_ccdrift_when_there_is_none(tmp_path):
    result = hook(tmp_path)
    assert (result.returncode, result.stderr) == (0, "")
    assert json.loads(result.stdout) == {"systemMessage": "ccdrift: not installed. Run: uv tool install ccdrift"}


@pytest.mark.parametrize("path", ["/usr/bin:/bin:", ":/usr/bin:/bin", "/usr/bin:.:/bin", "/usr/bin:project:/bin"],
                         ids=["empty-last", "empty-first", "dot", "relative"])
def test_the_hook_never_runs_a_ccdrift_in_the_projects_folder_through_an_empty_or_relative_path_entry(tmp_path, path):
    # Claude Code runs the hook in the project's folder, so such an entry would run the project's own ccdrift.
    project = fake(tmp_path / "project", '{"systemMessage": "the project ran"}')
    fake(project / "project", '{"systemMessage": "the project ran"}')
    result = subprocess.run(["/bin/sh", str(SCRIPT)], input=PAYLOAD, capture_output=True, text=True, cwd=project,
                            env={"HOME": str(tmp_path / "home"), "PATH": path})
    assert json.loads(result.stdout) == {"systemMessage": "ccdrift: not installed. Run: uv tool install ccdrift"}
    assert not (project / "args").exists() and not (project / "project" / "args").exists()


def test_the_hook_hands_ccdrift_the_payload_and_passes_on_what_it_prints(tmp_path):
    shown = '{"systemMessage": "ccdrift: no check yet"}'
    bin_dir = fake(tmp_path / "bin", shown)
    result = hook(tmp_path, bin_dir)
    assert (result.returncode, result.stdout, result.stderr) == (0, shown, "")
    assert (bin_dir / "args").read_text() == "hook session-start\n"
    assert (bin_dir / "stdin").read_text() == PAYLOAD


def test_the_hook_prefers_the_ccdrift_in_local_bin_to_one_earlier_on_path(tmp_path):
    local = fake(tmp_path / "home" / ".local" / "bin", '{"systemMessage": "local"}')
    other = fake(tmp_path / "venv", '{"systemMessage": "venv"}')
    assert json.loads(hook(tmp_path, other).stdout) == {"systemMessage": "local"}
    assert not (other / "args").exists() and (local / "args").exists()


def test_the_hook_says_to_upgrade_a_ccdrift_too_old_to_know_the_hook(tmp_path):
    result = hook(tmp_path, fake(tmp_path / "bin", code=2))
    assert (result.returncode, result.stderr) == (0, "")
    assert json.loads(result.stdout) == {
        "systemMessage": "ccdrift: the plugin needs ccdrift 0.26.0 or later. Run: uv tool upgrade ccdrift"}


@pytest.mark.parametrize("code", [1, 127])
def test_the_hook_says_ccdrift_failed_when_it_dies_before_its_own_handler_can(tmp_path, code):
    result = hook(tmp_path, fake(tmp_path / "bin", code=code))
    assert (result.returncode, result.stderr) == (0, "")
    assert json.loads(result.stdout) == {
        "systemMessage": "ccdrift: the session-start hook failed. Run: ccdrift hook session-start"}
