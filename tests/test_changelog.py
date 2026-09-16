"""Claude Code's release notes, read from the copy it keeps on disk."""

import pandas as pd

from ccdrift.changelog import changelog_path, load_changelog, new_versions, note_lines, release_notes
from tests.helpers import nth_day

CHANGELOG = """# Changelog

## 2.1.267

- Fixed switching models with /model re-sending every tool definition (a prompt-cache miss)
- Added a theme picker
Some prose that isn't a bullet

## 2.1.261

- Improved startup time
- Hooks now receive the session's effort level
"""


def test_the_changelog_is_read_per_version_bullets_only(tmp_path):
    (tmp_path / "changelog.md").write_text(CHANGELOG)
    assert load_changelog(tmp_path / "changelog.md") == {
        "2.1.267": ["Fixed switching models with /model re-sending every tool definition (a prompt-cache miss)",
                    "Added a theme picker"],
        "2.1.261": ["Improved startup time", "Hooks now receive the session's effort level"]}


def test_a_missing_changelog_reads_as_empty(tmp_path):
    assert load_changelog(tmp_path / "nope.md") == {}


def test_the_changelog_sits_in_the_config_folder_next_to_the_transcripts(tmp_path):
    assert changelog_path(tmp_path / "cfg" / "projects") == tmp_path / "cfg" / "cache" / "changelog.md"


def test_release_notes_keep_lines_on_the_topic_in_version_order(tmp_path):
    (tmp_path / "changelog.md").write_text(CHANGELOG)
    notes = load_changelog(tmp_path / "changelog.md")
    assert release_notes(notes, ["2.1.261", "2.1.267"], "cache") == [
        ("2.1.267", "Fixed switching models with /model re-sending every tool definition (a prompt-cache miss)")]
    assert release_notes(notes, ["2.1.261", "2.1.267"], ("hooks", "effort")) == [
        ("2.1.261", "Hooks now receive the session's effort level")]
    assert release_notes(notes, ["2.1.261", "2.1.267"], ("hooks", "cache"), limit=1) == [
        ("2.1.261", "Hooks now receive the session's effort level")]


def test_long_release_notes_are_cut_at_160_characters():
    notes = release_notes({"2.1.1": ["cache " + "x" * 300]}, ["2.1.1"], "cache")
    assert len(notes[0][1]) == 160 and notes[0][1].endswith("…")
    assert note_lines(notes)[0].startswith("release notes 2.1.1: cache x")


def test_new_versions_are_those_first_seen_in_the_window():
    turns = pd.DataFrame({"day": [nth_day(0), nth_day(3), nth_day(5), nth_day(9)],
                          "version": ["2.1.99", "2.1.233", "2.1.99", "2.1.240"]})
    assert new_versions(turns, nth_day(2), nth_day(9)) == ["2.1.233", "2.1.240"]
