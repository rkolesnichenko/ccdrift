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


def test_a_release_note_cant_carry_a_terminal_escape_into_the_log(tmp_path):
    """Claude Code writes this file, not ccdrift, and its lines are quoted into the check's
    log and printed by `ccdrift schedule status`, so they are cleaned like every other text
    ccdrift reads."""
    (tmp_path / "changelog.md").write_text("## 2.1.267\n\n- \x1b[5;31mFixed a cache miss\x1b[0m\n")
    assert load_changelog(tmp_path / "changelog.md") == {"2.1.267": ["[5;31mFixed a cache miss[0m"]}


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


def test_the_changelog_is_found_from_one_projects_folder_too(tmp_path):
    # --source pointed at one project's own folder sits a level deeper in the config folder.
    (tmp_path / "cfg" / "cache").mkdir(parents=True)
    (tmp_path / "cfg" / "cache" / "changelog.md").write_text(CHANGELOG)
    assert changelog_path(tmp_path / "cfg" / "projects" / "-Users-me-app") == tmp_path / "cfg" / "cache" / "changelog.md"


def test_the_changelog_is_not_looked_for_above_the_config_folder(tmp_path):
    # Above ~/.claude is the home folder, where cache/changelog.md is some other program's.
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "changelog.md").write_text(CHANGELOG)
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


def test_release_notes_quote_at_most_2_lines_a_version():
    notes = {"2.1.267": ["Fixed a prompt-cache miss after /rewind", "Fixed a prompt-cache miss on resume",
                         "Fixed a prompt-cache miss after a login"],
             "2.1.265": ["Fixed a prompt-cache miss in print mode"]}
    assert release_notes(notes, ["2.1.267", "2.1.265"], "cache") == [
        ("2.1.267", "Fixed a prompt-cache miss after /rewind"), ("2.1.267", "Fixed a prompt-cache miss on resume"),
        ("2.1.265", "Fixed a prompt-cache miss in print mode")]


def test_release_notes_skip_lines_that_only_mention_a_model_a_tool_an_agent_or_context():
    # Those words matched about half of each version's notes in Claude Code's real changelog.
    notes = {"2.1.267": ["Fixed the /model picker showing disabled models",
                         "Fixed background agents losing their tool results",
                         "Fixed a tip about 5x more context on Opus",
                         "Fixed MCP tools rewriting the tool list mid-session",
                         "Changed the default model for Enterprise seats to Opus 5",
                         "Changed CLAUDE_CODE_SUBAGENT_MODEL to set the default subagent model"]}
    assert release_notes(notes, ["2.1.267"], "context") == [
        ("2.1.267", "Fixed MCP tools rewriting the tool list mid-session")]
    assert release_notes(notes, ["2.1.267"], "haiku") == [
        ("2.1.267", "Changed the default model for Enterprise seats to Opus 5")]
    assert release_notes(notes, ["2.1.267"], "subagents") == [
        ("2.1.267", "Changed CLAUDE_CODE_SUBAGENT_MODEL to set the default subagent model")]


def test_release_notes_quote_the_lines_matching_most_of_a_topic_first():
    # From 2.1.267, whose sessions started with far less context: the likeliest
    # causes name both deferred tools or tool definitions and the system prompt.
    notes = {"2.1.267": ["Added `--system-prompt-snapshot off` to render the system prompt fresh on every request",
                         "Fixed a tool that disappears mid-conversation rewriting the tool list",
                         "Fixed MCP tools being added to the tool list mid-session; models now receive them as "
                         "deferred definitions",
                         "Improved prompt-cache stability: sessions record the system prompt and tool definitions once"],
             "2.1.266": ["Fixed the plugin cache not refreshing after an update",
                         "Fixed a prompt-cache miss after /rewind"]}
    assert release_notes(notes, ["2.1.267", "2.1.266"], "context") == [
        ("2.1.267", "Fixed MCP tools being added to the tool list mid-session; models now receive them as "
                    "deferred definitions"),
        ("2.1.267", "Improved prompt-cache stability: sessions record the system prompt and tool definitions once")]
    assert release_notes(notes, ["2.1.266"], "cache") == [
        ("2.1.266", "Fixed a prompt-cache miss after /rewind"),
        ("2.1.266", "Fixed the plugin cache not refreshing after an update")]


def test_release_notes_skip_lines_that_only_mention_thinking_or_the_transcript_view():
    # "thinking" and "transcript" matched mostly display fixes in Claude Code's real changelog.
    notes = {"2.1.251": ["Fixed the thinking toggle having no effect for the rest of a session",
                         "Fixed custom theme overrides for the effort badge colors being ignored",
                         "Changed /effort to save your default effort level per model",
                         "Fixed content jumping when scrolling up through long transcript history",
                         "Fixed history search breaking when ~/.claude/history.jsonl has a malformed entry",
                         "Changed session transcripts to record the reasoning effort level on each assistant message"]}
    assert release_notes(notes, ["2.1.251"], "effort") == [
        ("2.1.251", "Changed /effort to save your default effort level per model"),
        ("2.1.251", "Changed session transcripts to record the reasoning effort level on each assistant message")]
    assert release_notes(notes, ["2.1.251"], "fields") == [
        ("2.1.251", "Changed session transcripts to record the reasoning effort level on each assistant message")]


def test_release_notes_quote_a_changed_default_effort_or_logged_usage_with_the_broader_words():
    # Real 2.1.154 and 2.1.152 notes an effort or field alert should quote.
    notes = {"2.1.154": ["Improved /effort so changing effort mid-session keeps the prompt cache",
                         "Opus 4.8 is here! Now defaults to high effort"],
             "2.1.152": ["Improved memory usage in long sessions",
                         "Fixed `cache_creation_input_tokens` reporting as 0 in transcript and result usage"]}
    assert release_notes(notes, ["2.1.154"], "effort") == [("2.1.154", "Opus 4.8 is here! Now defaults to high effort")]
    assert release_notes(notes, ["2.1.152"], "fields") == [
        ("2.1.152", "Fixed `cache_creation_input_tokens` reporting as 0 in transcript and result usage")]


def test_long_release_notes_are_cut_at_160_characters():
    notes = release_notes({"2.1.1": ["cache " + "x" * 300]}, ["2.1.1"], "cache")
    assert len(notes[0][1]) == 160 and notes[0][1].endswith("…")
    assert note_lines(notes)[0].startswith("release notes 2.1.1: cache x")


def test_new_versions_are_those_first_seen_in_the_window():
    turns = pd.DataFrame({"day": [nth_day(0), nth_day(3), nth_day(5), nth_day(9)],
                          "version": ["2.1.99", "2.1.233", "2.1.99", "2.1.240"]})
    assert new_versions(turns, nth_day(2), nth_day(9)) == ["2.1.233", "2.1.240"]
