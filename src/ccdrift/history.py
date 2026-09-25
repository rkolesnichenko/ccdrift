"""ccdrift's own copy of the responses in Claude Code's transcripts.

Claude Code deletes transcripts after 30 days by default, so ccdrift keeps every
response it has read (token counts, model, version and settings; no text) in
SQLite next to its state file, and reads only transcripts that are new or changed
since. A year of one heavy user's responses takes about 65 MB."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ccdrift.logs import (ATTRIBUTION_FIELDS, COMPONENT_SETS, COMPONENT_SIZES, MAX_TIME, MIN_TIME, SDK_ENTRYPOINT_PREFIX,
                          SETTING_FIELDS, TOKEN_FIELDS, USAGE_COUNTS, ParsedFile, Tables, census_frame, compaction_frame,
                          components_frame, coverage_frame, coverage_rows, duration_frame, failure_frame, frame, hook_frame,
                          holds_no_response, jsonl_files, parse_all, parse_file, usage_frame)
from ccdrift.state import make_private
from ccdrift.texts import HISTORY_LINES

HISTORY_FILE = "history.sqlite"
SCHEMA_VERSION = 8
# Bump whenever parse_file's output changes, so every transcript still on disk is
# read again. Rows of transcripts Claude Code already deleted keep their values.
# 3: counts, times and ids out of range or of the wrong type read as missing.
# 4: each transcript's first main-thread response is marked, and text SQLite can't
#    store is cleaned.
# 5: control characters are dropped from text.
# 6: failed requests are kept, and each response's stop reason.
# 7: each response's cache-miss reason, and the census of the keys its record carries.
# 8: where each response's work came from, the branch it ran on, and the per-model
#    usage and cost of each cost-state record.
# 9: an error status past MAX_COUNT reads as missing, and a line nested past the JSON
#    decoder's limit counts as bad JSON.
# 10: what each session started with: the names and sizes of its skills, deferred tools,
#     agent types, MCP instructions, CLAUDE.md files, system prompt, tool definitions and
#     first message.
# 11: hook coverage: for each tool call, whether a PreToolUse and a PostToolUse hook ran on it.
PARSER_VERSION = 11

TEXT_COLUMNS = ("model", "stop_reason", "miss_reason") + ATTRIBUTION_FIELDS + SETTING_FIELDS
FLAG_COLUMNS = ("is_sidechain", "new_prompt", "after_compaction", "opens_transcript")
COUNT_COLUMNS = TOKEN_FIELDS + ("thinking_logged", "signature_chars", "visible_chars", "n_mcp_calls")
RESPONSE_COLUMNS = TEXT_COLUMNS + FLAG_COLUMNS + COUNT_COLUMNS
DURATION_COLUMNS = ("version", "entrypoint", "is_sidechain", "duration_ms", "message_count")
HOOK_COLUMNS = ("version", "entrypoint", "is_sidechain", "hook_count", "error_count", "duration_ms", "prevented")
COMPACTION_COLUMNS = ("version", "entrypoint", "is_sidechain", "trigger", "pre_tokens")
FAILURE_COLUMNS = ("version", "entrypoint", "is_sidechain", "kind", "status")
USAGE_COLUMNS = ("model",) + USAGE_COUNTS + ("cost_usd",)

# Integer keys, microsecond timestamps and file ids keep a year of responses near
# 65 MB; text keys, text timestamps and a path per row made it four times larger.
# files.last_ts is the latest time among a transcript's responses, so the check finds
# recent transcripts without reading every response.
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, size INTEGER, mtime_ns INTEGER, session_id TEXT,
    last_ts INTEGER);
CREATE TABLE IF NOT EXISTS responses (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    model TEXT, version TEXT, entrypoint TEXT, effort TEXT, speed TEXT, service_tier TEXT, agent_type TEXT,
    stop_reason TEXT, miss_reason TEXT,
    attribution_skill TEXT, attribution_plugin TEXT, attribution_mcp TEXT, git_branch TEXT,
    is_sidechain INTEGER, new_prompt INTEGER, after_compaction INTEGER,
    input_tokens INTEGER, output_tokens INTEGER, cache_creation INTEGER, cache_read INTEGER,
    cache_1h INTEGER, cache_5m INTEGER, thinking_logged INTEGER,
    signature_chars INTEGER, visible_chars INTEGER, n_mcp_calls INTEGER,
    opens_transcript INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS responses_file ON responses (file_id);
CREATE TABLE IF NOT EXISTS durations (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    version TEXT, entrypoint TEXT, is_sidechain INTEGER, duration_ms INTEGER, message_count INTEGER);
CREATE INDEX IF NOT EXISTS durations_file ON durations (file_id);
CREATE TABLE IF NOT EXISTS hook_runs (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    version TEXT, entrypoint TEXT, is_sidechain INTEGER, hook_count INTEGER, error_count INTEGER,
    duration_ms INTEGER, prevented INTEGER);
CREATE INDEX IF NOT EXISTS hook_runs_file ON hook_runs (file_id);
CREATE TABLE IF NOT EXISTS compactions (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    version TEXT, entrypoint TEXT, is_sidechain INTEGER, trigger TEXT, pre_tokens INTEGER);
CREATE INDEX IF NOT EXISTS compactions_file ON compactions (file_id);
CREATE TABLE IF NOT EXISTS failures (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    version TEXT, entrypoint TEXT, is_sidechain INTEGER, kind TEXT, status INTEGER);
CREATE INDEX IF NOT EXISTS failures_file ON failures (file_id);
CREATE TABLE IF NOT EXISTS model_usage (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER, model TEXT,
    input_tokens INTEGER, output_tokens INTEGER, cache_creation INTEGER, cache_read INTEGER,
    thinking_tokens INTEGER, web_searches INTEGER, cost_usd REAL);
CREATE INDEX IF NOT EXISTS model_usage_file ON model_usage (file_id);
CREATE TABLE IF NOT EXISTS field_census (
    file_id INTEGER NOT NULL, day TEXT NOT NULL, version TEXT NOT NULL, path TEXT NOT NULL,
    responses INTEGER NOT NULL, PRIMARY KEY (file_id, day, version, path));
CREATE TABLE IF NOT EXISTS field_days (
    file_id INTEGER NOT NULL, day TEXT NOT NULL, version TEXT NOT NULL,
    responses INTEGER NOT NULL, PRIMARY KEY (file_id, day, version));
CREATE TABLE IF NOT EXISTS components (
    file_id INTEGER PRIMARY KEY, skills TEXT, deferred TEXT, agents TEXT, mcp TEXT, tools TEXT,
    skills_chars INTEGER, deferred_chars INTEGER, agents_chars INTEGER, mcp_chars INTEGER, claude_md_files INTEGER,
    claude_md_chars INTEGER, system_chars INTEGER, tools_chars INTEGER, message_chars INTEGER);
CREATE TABLE IF NOT EXISTS hook_coverage (
    file_id INTEGER NOT NULL, day TEXT NOT NULL, version TEXT, entrypoint TEXT, is_sidechain INTEGER NOT NULL,
    event TEXT NOT NULL, tool TEXT NOT NULL, calls INTEGER NOT NULL, hooked INTEGER NOT NULL);
CREATE INDEX IF NOT EXISTS hook_coverage_file ON hook_coverage (file_id);
"""


class HistoryError(RuntimeError):
    """The history store can't be used; the message names the file."""


def history_path(state_path: Path) -> Path:
    return state_path.parent / HISTORY_FILE


def _unusable(path: Path, exc: Exception) -> HistoryError:
    # SQLite's words for a store another connection held past the wait: nothing is wrong
    # with it, and moving it aside would drop the rows of transcripts already deleted.
    if isinstance(exc, sqlite3.OperationalError) and str(exc).endswith("is locked"):
        return HistoryError(HISTORY_LINES["busy"].format(path=path, error=exc))
    return HistoryError(HISTORY_LINES["unusable"].format(path=path, error=exc))


def row_key(text: str) -> int:
    """A response's text key as a signed 64-bit integer, so SQLite keeps it as the row id.
    A lone surrogate in it (JSON "\\ud800") is hashed as it is."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "big",
                          signed=True)


def _micros(ts: Any) -> Optional[int]:
    return None if ts is None else round(ts.timestamp() * 1_000_000)


def _count(value: Any) -> Optional[int]:
    return None if value is None else round(value)


UPSERT = ("INSERT INTO {table} ({names}) VALUES ({marks}) "
          "ON CONFLICT (key) DO UPDATE SET {updates} "
          "WHERE (SELECT path FROM files WHERE id = excluded.file_id) "
          "< (SELECT path FROM files WHERE id = {table}.file_id)")


def _upsert(table: str, columns: tuple[str, ...]) -> str:
    """Insert a row; a key already stored stays with the transcript whose path sorts
    first, the rule parse_source follows."""
    names = ("key", "file_id", "ts") + columns
    updates = ", ".join(f"{name} = excluded.{name}" for name in names[1:])
    return UPSERT.format(table=table, names=", ".join(names), marks=", ".join("?" * len(names)), updates=updates)


# The microsecond times a row can hold. ccdrift 0.3.0 stored times pandas can't, and
# kept them after their transcript was deleted, so they read as missing.
TS_RANGE = (_micros(MIN_TIME), _micros(MAX_TIME))


# Responses on the main thread outside Agent SDK sessions with a usable time, as the
# check judges them (GLOB, like str.startswith, tells case apart); takes TS_RANGE.
MAIN_CLI = ("ts BETWEEN ? AND ? AND is_sidechain = 0 "
            f"AND (entrypoint IS NULL OR entrypoint NOT GLOB '{SDK_ENTRYPOINT_PREFIX}*')")
MICROS_PER_DAY = 86_400_000_000


def _day(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1_000_000, timezone.utc).date().isoformat()


def _day_start(day: str) -> int:
    """The first microsecond of a UTC day."""
    return _micros(datetime.fromisoformat(day).replace(tzinfo=timezone.utc))


def _decode(rows: pd.DataFrame, flags: tuple[str, ...]) -> pd.DataFrame:
    ts = rows.pop("ts").astype(float)
    rows["timestamp"] = pd.to_datetime(ts.where(ts.between(*TS_RANGE)), unit="us", utc=True)
    for col in flags:
        rows[col] = rows[col].astype(bool)
    return rows


class History:
    """The store in `path`, created when missing."""

    def __init__(self, path: Path):
        self._days: Optional[list[tuple[int, Optional[str], int, int]]] = None
        # What the last update couldn't read, as Tables.skipped and Tables.no_responses.
        self.skipped: list[tuple[str, str]] = []
        self.no_responses: list[tuple[str, int]] = []
        if sqlite3.sqlite_version_info < (3, 24, 0):
            raise HistoryError(HISTORY_LINES["old_sqlite"].format(version=sqlite3.sqlite_version))
        if path.is_dir():
            raise HistoryError(HISTORY_LINES["folder"].format(path=path))
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # A store an older ccdrift made may be readable by others; SQLite gives its journal the
            # store's permissions.
            make_private(path)
            self.db = sqlite3.connect(path, timeout=30)
        except (OSError, sqlite3.Error) as exc:
            # Nothing is wrong with the store itself, so moving it aside wouldn't help.
            raise HistoryError(HISTORY_LINES["cant_open"].format(path=path, error=exc)) from exc
        try:
            has_meta = self.db.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'").fetchone()
            self.meta = dict(self.db.execute("SELECT key, value FROM meta")) if has_meta else {}
        except sqlite3.Error as exc:
            self.db.close()
            raise _unusable(path, exc) from exc
        # Check the schema version before running any DDL: a newer store's tables may
        # not match this version's SCHEMA, and applying it could fail with a confusing
        # error, or silently add tables the newer ccdrift doesn't expect.
        stored = self.meta.get("schema_version", str(SCHEMA_VERSION))
        try:
            schema_version = int(stored)
        except (TypeError, ValueError) as exc:
            self.db.close()
            raise _unusable(path, ValueError(HISTORY_LINES["bad_schema"].format(stored=stored))) from exc
        if schema_version > SCHEMA_VERSION:
            self.db.close()
            raise HistoryError(HISTORY_LINES["newer"].format(path=path))
        try:
            if has_meta and schema_version < SCHEMA_VERSION:
                self._migrate(schema_version)
            self.db.executescript(SCHEMA)
            self.meta = dict(self.db.execute("SELECT key, value FROM meta"))
            if self.meta.get("schema_version") != str(SCHEMA_VERSION):
                self._set_meta("schema_version", str(SCHEMA_VERSION))
        except sqlite3.Error as exc:
            self.db.close()
            raise _unusable(path, exc) from exc

    def _migrate(self, from_version: int) -> None:
        """Bring a store from an older ccdrift up to date; SCHEMA then adds missing tables."""
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(responses)")}
        if from_version < 2 and columns and "agent_type" not in columns:
            with self.db:
                self.db.execute("ALTER TABLE responses ADD COLUMN agent_type TEXT")
        if from_version < 3 and columns and "opens_transcript" not in columns:
            with self.db:
                self.db.execute("ALTER TABLE responses ADD COLUMN opens_transcript INTEGER NOT NULL DEFAULT 0")
                # Transcripts Claude Code has deleted can't be read again: their earliest
                # main-thread response stays their session start, as before.
                self.db.execute(
                    "UPDATE responses SET opens_transcript = 1 WHERE key IN (SELECT r.key FROM responses r JOIN "
                    "(SELECT file_id, MIN(ts) AS ts FROM responses WHERE is_sidechain = 0 AND ts BETWEEN ? AND ? "
                    "GROUP BY file_id) first ON r.file_id = first.file_id AND r.ts = first.ts "
                    "WHERE r.is_sidechain = 0)", TS_RANGE)
        if from_version < 4 and columns and "stop_reason" not in columns:
            with self.db:
                self.db.execute("ALTER TABLE responses ADD COLUMN stop_reason TEXT")
        if from_version < 5 and columns and "miss_reason" not in columns:
            with self.db:
                self.db.execute("ALTER TABLE responses ADD COLUMN miss_reason TEXT")
        for column in ("attribution_skill", "attribution_plugin", "attribution_mcp", "git_branch"):
            if from_version < 6 and columns and column not in columns:
                with self.db:
                    self.db.execute(f"ALTER TABLE responses ADD COLUMN {column} TEXT")
        file_columns = {row[1] for row in self.db.execute("PRAGMA table_info(files)")}
        if from_version < 3 and file_columns and "last_ts" not in file_columns:
            with self.db:
                self.db.execute("ALTER TABLE files ADD COLUMN last_ts INTEGER")
                if columns:
                    self.db.execute(
                        "UPDATE files SET last_ts = (SELECT MAX(ts) FROM responses WHERE file_id = files.id)")

    def __enter__(self) -> "History":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        self.db.close()

    def _set_meta(self, key: str, value: str) -> None:
        with self.db:
            self.db.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                            "ON CONFLICT (key) DO UPDATE SET value = excluded.value", (key, value))
        self.meta[key] = value

    def built_from(self) -> Optional[str]:
        """The transcript folder the store holds, once it has read a transcript."""
        return self.meta.get("source")

    def update(self, source: Path, claim: bool = True) -> int:
        """Read the transcripts under `source` that are new or changed since the last
        update, or all of them after a parser change, and return how many were read.
        A known transcript that can't be read keeps its rows and is tried again next
        time. So does one the parser fails on, or whose rows the store can't take, with
        a line on stderr, which the check's log keeps: one bad transcript no longer fails
        every check until Claude Code deletes it. When two or more fail and none reads,
        the first failure is raised: that is the parser, not a transcript, and the check
        has to say so. One alone is skipped even when it is the only one to read, since
        an hourly check often reads just the transcript that changed. With `claim`, a
        store that holds no transcript folder yet is tied to `source` once it has read one."""
        known = {path: (size, mtime) for path, size, mtime in
                 self.db.execute("SELECT path, size, mtime_ns FROM files")}
        reread = self.meta.get("parser_version") != str(PARSER_VERSION)
        read = 0
        failed: list[Exception] = []
        self.skipped, self.no_responses = [], []
        for fp, rel in jsonl_files(source):
            try:
                stat = fp.stat()
                if not reread and known.get(rel) == (stat.st_size, stat.st_mtime_ns):
                    continue
                parsed = parse_file(fp, rel)
            except OSError:
                self._retry(rel, known)
                continue
            except Exception as exc:  # a parser bug no test has found yet
                self._skip(rel, known, exc, failed)
                continue
            try:
                self._replace(rel, stat.st_size, stat.st_mtime_ns, parsed)
            except (OverflowError, ValueError, TypeError) as exc:  # a value SQLite can't store
                self._skip(rel, known, exc, failed)
                continue
            if holds_no_response(parsed):
                self.no_responses.append((rel, parsed.lines))
            read += 1
        if len(failed) > 1 and not read:
            raise failed[0]
        if claim and read and self.built_from() is None:
            self._set_meta("source", str(source.expanduser().resolve()))
        self._set_meta("parser_version", str(PARSER_VERSION))
        self._days = None
        return read

    def _retry(self, rel: str, known: dict[str, tuple[int, int]]) -> None:
        """Keep a known transcript's rows and have the next update read it again."""
        if rel in known:
            with self.db:
                self.db.execute("UPDATE files SET size = NULL WHERE path = ?", (rel,))

    def _skip(self, rel: str, known: dict[str, tuple[int, int]], exc: Exception, failed: list[Exception]) -> None:
        """Leave out a transcript that failed, saying so, and try it again next time."""
        print(HISTORY_LINES["skipped"].format(path=rel, error=f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        failed.append(exc)
        self.skipped.append((rel, type(exc).__name__))
        self._retry(rel, known)

    def _replace(self, rel: str, size: int, mtime_ns: int, parsed: ParsedFile) -> None:
        """Replace one transcript's rows in a single transaction."""
        with self.db:
            self.db.execute(
                "INSERT INTO files (path, size, mtime_ns, session_id) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (path) DO UPDATE SET size = excluded.size, mtime_ns = excluded.mtime_ns, "
                "session_id = excluded.session_id",
                (rel, size, mtime_ns, parsed.session_id or Path(rel).stem))
            (file_id,) = self.db.execute("SELECT id FROM files WHERE path = ?", (rel,)).fetchone()
            self.db.execute("DELETE FROM responses WHERE file_id = ?", (file_id,))
            self.db.execute("DELETE FROM durations WHERE file_id = ?", (file_id,))
            self.db.execute("DELETE FROM hook_runs WHERE file_id = ?", (file_id,))
            self.db.execute("DELETE FROM compactions WHERE file_id = ?", (file_id,))
            self.db.execute("DELETE FROM failures WHERE file_id = ?", (file_id,))
            self.db.execute("DELETE FROM model_usage WHERE file_id = ?", (file_id,))
            self.db.execute("DELETE FROM field_census WHERE file_id = ?", (file_id,))
            self.db.execute("DELETE FROM field_days WHERE file_id = ?", (file_id,))
            self.db.execute("DELETE FROM components WHERE file_id = ?", (file_id,))
            self.db.execute("DELETE FROM hook_coverage WHERE file_id = ?", (file_id,))
            self.db.executemany(_upsert("responses", RESPONSE_COLUMNS), [
                (row_key(row["key"]), file_id, _micros(row["timestamp"]),
                 *(row[c] for c in TEXT_COLUMNS), *(int(row[c]) for c in FLAG_COLUMNS),
                 *(_count(row[c]) for c in COUNT_COLUMNS))
                for row in parsed.responses.values()])
            self.db.executemany(_upsert("durations", DURATION_COLUMNS), [
                (row_key(row["key"]), file_id, _micros(row["timestamp"]), row["version"], row["entrypoint"],
                 int(row["is_sidechain"]), _count(row["duration_ms"]), row["message_count"])
                for row in parsed.durations.values()])
            self.db.executemany(_upsert("hook_runs", HOOK_COLUMNS), [
                (row_key(row["key"]), file_id, _micros(row["timestamp"]), row["version"], row["entrypoint"],
                 int(row["is_sidechain"]), row["hook_count"], row["error_count"], _count(row["duration_ms"]),
                 int(row["prevented"]))
                for row in parsed.hook_runs.values()])
            self.db.executemany(_upsert("compactions", COMPACTION_COLUMNS), [
                (row_key(row["key"]), file_id, _micros(row["timestamp"]), row["version"], row["entrypoint"],
                 int(row["is_sidechain"]), row["trigger"], _count(row["pre_tokens"]))
                for row in parsed.compactions.values()])
            self.db.executemany(_upsert("failures", FAILURE_COLUMNS), [
                (row_key(row["key"]), file_id, _micros(row["timestamp"]), row["version"], row["entrypoint"],
                 int(row["is_sidechain"]), row["kind"], _count(row["status"]))
                for row in parsed.failures.values()])
            self.db.executemany(
                "INSERT INTO field_census (file_id, day, version, path, responses) VALUES (?, ?, ?, ?, ?)",
                [(file_id, day, version, path, count)
                 for (day, version, path), count in parsed.field_census.items()])
            self.db.executemany(
                "INSERT INTO field_days (file_id, day, version, responses) VALUES (?, ?, ?, ?)",
                [(file_id, day, version, count) for (day, version), count in parsed.field_days.items()])
            self.db.executemany(_upsert("model_usage", USAGE_COLUMNS), [
                (row_key(row["key"]), file_id, _micros(row["timestamp"]), row["model"],
                 *(_count(row[c]) for c in USAGE_COUNTS), row["cost_usd"])
                for row in parsed.model_usage.values()])
            row = parsed.components
            self.db.execute(
                f"INSERT INTO components (file_id, {', '.join(COMPONENT_SETS + COMPONENT_SIZES)}) "
                f"VALUES ({', '.join('?' * (1 + len(COMPONENT_SETS) + len(COMPONENT_SIZES)))})",
                (file_id, *(None if row[c] is None else json.dumps(row[c]) for c in COMPONENT_SETS),
                 *(_count(row[c]) for c in COMPONENT_SIZES)))
            self.db.executemany(
                "INSERT INTO hook_coverage (file_id, day, version, entrypoint, is_sidechain, event, tool, calls, hooked) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(file_id, row["day"], row["version"], row["entrypoint"], int(row["is_sidechain"]), row["event"],
                  row["tool"], row["calls"], row["hooked"]) for row in coverage_rows(parsed, rel)])
            self.db.execute("UPDATE files SET last_ts = (SELECT MAX(ts) FROM responses WHERE file_id = ?) WHERE id = ?",
                            (file_id, file_id))

    def responses(self, since: Optional[str] = None) -> pd.DataFrame:
        """Every stored response, as the table parse_source returns. With `since` (a UTC
        day), only the responses of transcripts with one on that day or later, whole, so
        idle gaps and session starts read as they do in the full table; `version_first_day`
        then holds the day each version first ran over the whole store."""
        query = ("SELECT f.path AS source_file, f.session_id, r.ts, " + ", ".join(f"r.{c}" for c in RESPONSE_COLUMNS)
                 + " FROM responses r JOIN files f ON f.id = r.file_id")
        params: tuple = ()
        if since is not None:
            query += " WHERE r.file_id IN (SELECT id FROM files WHERE last_ts >= ?)"
            params = (_day_start(since),)
        df = frame(_decode(pd.read_sql_query(query + " ORDER BY f.path, r.ts", self.db, params=params),
                           FLAG_COLUMNS))
        if since is not None and not df.empty:
            df["version_first_day"] = df["version"].map(self.version_first_days())
        return df

    def _main_cli_days(self) -> list[tuple[int, Optional[str], int, int]]:
        """Per UTC day (days since 1970) and Claude Code version: the responses on the
        main thread outside Agent SDK sessions, and the earliest one's time. One pass
        over the whole store, kept until the next update."""
        if self._days is None:
            self._days = self.db.execute(
                f"SELECT ts / {MICROS_PER_DAY}, version, COUNT(*), MIN(ts) FROM responses WHERE {MAIN_CLI} "
                "GROUP BY 1, 2", TS_RANGE).fetchall()
        return self._days

    def version_first_days(self) -> dict[str, str]:
        """The UTC day each Claude Code version first answered on the main thread outside
        Agent SDK sessions, over the whole store."""
        first: dict[str, int] = {}
        for _, version, _, ts in self._main_cli_days():
            if version is not None:
                first[version] = min(ts, first.get(version, ts))
        return {version: _day(ts) for version, ts in first.items()}

    def active_day_start(self, active_days: int, responses: int) -> Optional[str]:
        """The UTC day `active_days` active days back, counting days with at least
        `responses` responses on the main thread outside Agent SDK sessions; None when
        the store holds fewer such days."""
        counts: Counter[int] = Counter()
        for day, _, count, _ in self._main_cli_days():
            counts[day] += count
        active = sorted((day for day, count in counts.items() if count >= responses), reverse=True)
        return _day(active[active_days - 1] * MICROS_PER_DAY) if len(active) >= active_days else None

    def _records(self, table: str, columns: tuple[str, ...], since: Optional[str]) -> pd.DataFrame:
        """A record table's stored rows, from `since` (a UTC day) when given."""
        query = (f"SELECT f.path AS source_file, f.session_id, t.ts, {', '.join(f't.{c}' for c in columns)} "
                 f"FROM {table} t JOIN files f ON f.id = t.file_id")
        params: tuple = ()
        if since is not None:
            query += " WHERE t.ts >= ?"
            params = (_day_start(since),)
        query += " ORDER BY f.path, t.ts"
        return pd.read_sql_query(query, self.db, params=params)

    def durations(self, since: Optional[str] = None) -> pd.DataFrame:
        """Every stored turn duration, or those from `since`, as the table parse_durations returns."""
        return duration_frame(_decode(self._records("durations", DURATION_COLUMNS, since), ("is_sidechain",)))

    def hook_runs(self, since: Optional[str] = None) -> pd.DataFrame:
        """Every stored stop-hook run, or those from `since`, as parse_all's `hook_runs` table."""
        return hook_frame(_decode(self._records("hook_runs", HOOK_COLUMNS, since), ("is_sidechain", "prevented")))

    def compactions(self, since: Optional[str] = None) -> pd.DataFrame:
        """Every stored compaction, or those from `since`, as parse_all's `compactions` table."""
        return compaction_frame(_decode(self._records("compactions", COMPACTION_COLUMNS, since), ("is_sidechain",)))

    def failures(self, since: Optional[str] = None) -> pd.DataFrame:
        """Every stored failed request, or those from `since`, as parse_all's `failures` table."""
        return failure_frame(_decode(self._records("failures", FAILURE_COLUMNS, since), ("is_sidechain",)))

    def model_usage(self, since: Optional[str] = None) -> pd.DataFrame:
        """Every stored per-model cost record, or those from `since`, as parse_all's
        `model_usage` table."""
        return usage_frame(_decode(self._records("model_usage", USAGE_COLUMNS, since), ()))

    def hook_coverage(self, since: Optional[str] = None) -> pd.DataFrame:
        """Every stored hook coverage row, or those of days from `since`, as parse_all's
        `hook_coverage` table."""
        query = ("SELECT f.path AS source_file, f.session_id, h.day, h.version, h.entrypoint, h.is_sidechain, h.event, "
                 "h.tool, h.calls, h.hooked FROM hook_coverage h JOIN files f ON f.id = h.file_id")
        params: tuple = ()
        if since is not None:
            query += " WHERE h.day >= ?"
            params = (since,)
        rows = pd.read_sql_query(query, self.db, params=params)
        # pandas 3 reads NULL as NaN in a text column that holds values too; pandas 2 as None.
        for col in ("version", "entrypoint"):
            rows[col] = [value if isinstance(value, str) else None for value in rows[col]]
        return coverage_frame(rows.to_dict("records"))

    def field_census(self, since: Optional[str] = None) -> pd.DataFrame:
        """The census of the keys response records carry, summed over transcripts, or
        that of days from `since`, as parse_all's `field_census` table."""
        where, params = ("", ()) if since is None else (" WHERE day >= ?", (since,))
        census = {(day, version, path): count for day, version, path, count in self.db.execute(
            "SELECT day, version, path, SUM(responses) FROM field_census" + where + " GROUP BY 1, 2, 3", params)}
        days = {(day, version): count for day, version, count in self.db.execute(
            "SELECT day, version, SUM(responses) FROM field_days" + where + " GROUP BY 1, 2", params)}
        return census_frame(census, days)

    def components(self, since: Optional[str] = None) -> pd.DataFrame:
        """What each stored transcript's session started with, as parse_all's `components`
        table. With `since` (a UTC day), only the transcripts History.responses keeps."""
        query = (f"SELECT f.path AS source_file, f.session_id, {', '.join(f'c.{n}' for n in COMPONENT_SETS + COMPONENT_SIZES)} "
                 "FROM components c JOIN files f ON f.id = c.file_id")
        params: tuple = ()
        if since is not None:
            query += " WHERE c.file_id IN (SELECT id FROM files WHERE last_ts >= ?)"
            params = (_day_start(since),)
        rows = pd.read_sql_query(query + " ORDER BY f.path", self.db, params=params)
        # pandas 3 reads NULL as NaN in a text column that holds values too; pandas 2 as None.
        for col in COMPONENT_SETS:
            rows[col] = [json.loads(value) if isinstance(value, str) else None for value in rows[col]]
        return components_frame(rows.to_dict("records"))


def load_history(source: Path, state_path: Path, claim: bool, since: Optional[str] = None,
                 active_days: int = 0, active_responses: int = 0) -> Tables:
    """Everything ccdrift knows of for `source`: the history store next to the state
    file, first brought up to date. Only the daily check claims a store (`claim`):
    other commands read the transcripts directly while no check has tied the store to
    a folder, and so does anything run on a folder other than the store's. With
    `since`, the store's responses are only those of transcripts active since that UTC
    day or since `active_days` days with `active_responses` responses back, whichever
    is earlier (see History.responses), and all of them while the store holds fewer
    such days; its other records are only those since the same day."""
    path = history_path(state_path)
    if not claim and not path.exists():
        return parse_all(source)
    try:
        with History(path) as history:
            built_from = history.built_from()
            if built_from is not None and built_from != str(source.expanduser().resolve()):
                print(HISTORY_LINES["other_source"].format(path=path, built_from=built_from), file=sys.stderr)
                return parse_all(source)
            if built_from is None and not claim:
                return parse_all(source)
            history.update(source, claim=claim)
            if since is not None and active_days:
                active_start = history.active_day_start(active_days, active_responses)
                since = None if active_start is None else min(since, active_start)
            return Tables(history.responses(since), history.durations(since), history.hook_runs(since),
                          history.compactions(since), history.failures(since), history.field_census(since),
                          history.model_usage(since), history.components(since), history.hook_coverage(since),
                          history.skipped, history.no_responses)
    except sqlite3.Error as exc:
        raise _unusable(path, exc) from exc
    except pd.errors.DatabaseError as exc:
        # pandas wraps SQLite's error in one that repeats the whole query.
        raise _unusable(path, exc.__cause__ or exc) from exc


def load_turns(source: Path, state_path: Path) -> pd.DataFrame:
    """The responses of load_history, claiming the store, over the whole history. The check
    claims it too but reads from history_start on (load_history with `since`)."""
    return load_history(source, state_path, claim=True).responses
