"""Quota samples: what ccdrift keeps from Claude Code's status line JSON, and how often."""

import io
import json
import stat
from datetime import datetime, timedelta, timezone

import pytest

from ccdrift.quota import (MAX_PAYLOAD, MIN_INTERVAL, TAIL_BYTES, last_sample, quota_path, read_payload,
                           record_sample, sample_from, samples_summary)

NOW = datetime(2026, 9, 28, 13, 55, 0, tzinfo=timezone.utc)
FIVE = {"used_percentage": 23.5, "resets_at": 1790600400}
SEVEN = {"used_percentage": 41.2, "resets_at": 1791000000}
# Every key of the documented payload that must never reach the file, each holding a marker.
PRIVATE = {"cwd": "/Users/me/secret-project", "session_id": "SESSION-ID-MARKER", "session_name": "NAME-MARKER",
           "prompt_id": "PROMPT-ID-MARKER", "transcript_path": "/Users/me/.claude/projects/secret/t.jsonl",
           "workspace": {"current_dir": "/Users/me/secret-project", "project_dir": "/Users/me/secret-project",
                         "added_dirs": ["/Users/me/other-secret"], "git_worktree": "WORKTREE-MARKER",
                         "repo": {"host": "github.com", "owner": "OWNER-MARKER", "name": "REPO-MARKER"}},
           "output_style": {"name": "STYLE-MARKER"}, "agent": {"name": "AGENT-MARKER"},
           "pr": {"number": 4242, "url": "https://github.com/OWNER-MARKER/REPO-MARKER/pull/4242",
                  "review_state": "pending"},
           "worktree": {"name": "WT-MARKER", "path": "/Users/me/secret-project/.claude/worktrees/wt",
                        "branch": "BRANCH-MARKER", "original_cwd": "/Users/me/secret-project",
                        "original_branch": "main"},
           "cost": {"total_cost_usd": 0.01234}, "context_window": {"used_percentage": 8},
           "prompt_cache": {"hit_ratio": 0.91}, "effort": {"level": "high"}}


def payload(five=FIVE, seven=SEVEN, **extra):
    rate_limits = {key: value for key, value in (("five_hour", five), ("seven_day", seven)) if value is not None}
    return {"model": {"id": "claude-opus-5-5", "display_name": "Opus"}, "version": "2.1.283",
            "rate_limits": rate_limits, **PRIVATE, **extra}


def lines(path):
    return path.read_text().splitlines() if path.exists() else []


def test_a_sample_keeps_the_time_version_model_and_windows_and_nothing_else():
    assert sample_from(payload(), NOW) == {"at": "2026-09-28T13:55:00+00:00", "version": "2.1.283",
                                           "model": "claude-opus-5-5", "five_hour": FIVE, "seven_day": SEVEN}


def test_no_id_path_or_name_from_the_status_line_reaches_the_file(tmp_path):
    path = tmp_path / "quota.jsonl"
    assert record_sample(json.dumps(payload()), path, NOW)
    text = path.read_text()
    for marker in ("secret", "MARKER", "4242", "0.01234", "0.91", "\"high\""):
        assert marker not in text
    assert set(json.loads(text)) == {"at", "version", "model", "five_hour", "seven_day"}


def test_a_spend_limit_is_kept_even_above_100_percent():
    spend = {"used_percentage": 112.8, "resets_at": 1792000000}
    sample = sample_from(payload(five=None, seven=None, rate_limits={"spend_limit": spend}), NOW)
    assert sample["spend_limit"] == spend and "five_hour" not in sample


@pytest.mark.parametrize("data", [
    {"version": "2.1.283"},                                   # no rate_limits: an API key, or before a response
    {"rate_limits": None},
    {"rate_limits": {}},
    {"rate_limits": {"five_hour": {"used_percentage": True, "resets_at": 1}}},    # a bool isn't a number
    {"rate_limits": {"five_hour": {"used_percentage": "23", "resets_at": 1}}},
    {"rate_limits": {"five_hour": {"used_percentage": float("nan"), "resets_at": 1}}},
    {"rate_limits": {"five_hour": {"used_percentage": 23.5}}},  # no reset time
    {"rate_limits": {"five_hour": [23.5, 1]}},
    ["not", "an", "object"],
    "text",
])
def test_a_payload_without_numeric_windows_gives_no_sample(data):
    assert sample_from(data, NOW) is None


def test_a_window_that_isnt_numeric_is_left_out_and_the_others_kept():
    sample = sample_from(payload(seven={"used_percentage": None, "resets_at": 1791000000}), NOW)
    assert sample["five_hour"] == FIVE and "seven_day" not in sample


def test_a_missing_version_or_model_is_kept_as_null():
    data = payload()
    del data["version"], data["model"]
    sample = sample_from(data, NOW)
    assert (sample["version"], sample["model"]) == (None, None)


def test_samples_within_a_minute_or_unchanged_write_one_line(tmp_path):
    path = tmp_path / "quota.jsonl"
    assert record_sample(json.dumps(payload()), path, NOW)
    assert not record_sample(json.dumps(payload()), path, NOW + timedelta(seconds=MIN_INTERVAL))
    changed = payload(five={"used_percentage": 24.0, "resets_at": 1790600400})
    assert not record_sample(json.dumps(changed), path, NOW + timedelta(seconds=MIN_INTERVAL - 1))
    assert len(lines(path)) == 1
    assert record_sample(json.dumps(changed), path, NOW + timedelta(seconds=MIN_INTERVAL))
    assert [json.loads(line)["five_hour"]["used_percentage"] for line in lines(path)] == [23.5, 24.0]


def test_a_new_version_or_model_alone_writes_no_sample(tmp_path):
    path = tmp_path / "quota.jsonl"
    record_sample(json.dumps(payload()), path, NOW)
    other = {**payload(), "version": "2.1.284", "model": {"id": "claude-fable-5-1"}}
    assert not record_sample(json.dumps(other), path, NOW + timedelta(hours=1))
    assert len(lines(path)) == 1


def test_a_window_that_resets_is_a_change_even_at_the_same_share(tmp_path):
    path = tmp_path / "quota.jsonl"
    record_sample(json.dumps(payload()), path, NOW)
    rolled = payload(five={"used_percentage": 23.5, "resets_at": 1790618400})
    assert record_sample(json.dumps(rolled), path, NOW + timedelta(minutes=2))


def test_a_window_that_appears_or_goes_is_a_change(tmp_path):
    path = tmp_path / "quota.jsonl"
    record_sample(json.dumps(payload()), path, NOW)
    assert record_sample(json.dumps(payload(seven=None)), path, NOW + timedelta(minutes=2))


@pytest.mark.parametrize("text", ["", "{", "null", "[]", "x" * (MAX_PAYLOAD + 1), json.dumps(payload()) + "}"])
def test_a_payload_that_doesnt_parse_writes_nothing_and_raises_nothing(tmp_path, text):
    path = tmp_path / "quota.jsonl"
    assert not record_sample(text, path, NOW)
    assert not path.exists()


def test_a_payload_over_the_cap_is_not_parsed_even_when_it_is_valid(tmp_path):
    big = json.dumps(payload(padding="p" * MAX_PAYLOAD))
    assert not record_sample(big, tmp_path / "quota.jsonl", NOW)


def test_a_last_line_that_doesnt_parse_counts_as_no_last_sample(tmp_path):
    path = tmp_path / "quota.jsonl"
    path.write_text('{"at": "2026-09-28T13:55:00+00:00", "five_ho\n')
    assert last_sample(path) is None
    assert record_sample(json.dumps(payload()), path, NOW)
    assert json.loads(lines(path)[-1])["five_hour"] == FIVE


@pytest.mark.parametrize("at", ["yesterday", "2026-09-27T13:55:00"])
def test_a_last_line_whose_time_doesnt_parse_counts_as_no_last_sample(tmp_path, at):
    # Found in the 0.19.0 review: a time that isn't an ISO stamp with a zone raised in the
    # interval check, which record_sample swallowed on every refresh, so sampling stopped
    # for good and nothing said so.
    path = tmp_path / "quota.jsonl"
    path.write_text(json.dumps({"at": at, "five_hour": {"used_percentage": 1.0, "resets_at": 1}}) + "\n")
    assert last_sample(path) is None
    assert record_sample(json.dumps(payload()), path, NOW)
    assert len(lines(path)) == 2


def test_the_last_sample_is_read_from_the_end_of_a_long_file(tmp_path):
    path = tmp_path / "quota.jsonl"
    older = json.dumps(sample_from(payload(five={"used_percentage": 1.0, "resets_at": 1}), NOW - timedelta(days=9)))
    path.write_text((older + "\n") * (TAIL_BYTES // len(older) + 5))
    record_sample(json.dumps(payload()), path, NOW)
    assert last_sample(path)["five_hour"] == FIVE


def test_an_unwritable_file_raises_nothing(tmp_path):
    folder = tmp_path / "locked"
    folder.mkdir()
    folder.chmod(0o500)
    try:
        assert not record_sample(json.dumps(payload()), folder / "quota.jsonl", NOW)
    finally:
        folder.chmod(0o700)


def test_the_file_is_created_private_and_its_folder_when_missing(tmp_path):
    path = tmp_path / "home" / "quota.jsonl"
    assert record_sample(json.dumps(payload()), path, NOW)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_the_file_sits_beside_the_state_file(tmp_path):
    assert quota_path(tmp_path / "state.json") == tmp_path / "quota.jsonl"


class Terminal(io.TextIOWrapper):
    def isatty(self):
        return True


def test_stdin_is_read_up_to_the_cap_and_never_from_a_terminal():
    text = json.dumps(payload())
    assert read_payload(io.TextIOWrapper(io.BytesIO(text.encode()))) == text
    assert read_payload(io.TextIOWrapper(io.BytesIO(b"x" * (MAX_PAYLOAD + 1)))) is None
    assert read_payload(Terminal(io.BytesIO(text.encode()))) is None
    assert read_payload(io.TextIOWrapper(io.BytesIO(b"\xff\xfe{"))) is None


def test_the_summary_counts_the_samples_and_gives_the_first_and_last_time(tmp_path):
    path = tmp_path / "quota.jsonl"
    assert samples_summary(path) is None
    record_sample(json.dumps(payload()), path, NOW)
    record_sample(json.dumps(payload(seven=None)), path, NOW + timedelta(hours=2))
    with path.open("a") as out:
        out.write("not json\n")
    assert samples_summary(path) == {"count": 2, "first": "2026-09-28T13:55:00+00:00",
                                     "last": "2026-09-28T15:55:00+00:00"}
