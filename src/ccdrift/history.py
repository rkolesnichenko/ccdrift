"""ccdrift's own copy of the responses in Claude Code's transcripts.

Claude Code deletes transcripts after 30 days by default, so ccdrift keeps every
response it has read (token counts, model, version and settings; no text) in
SQLite next to its state file, and reads only transcripts that are new or changed
since. A year of one heavy user's responses takes about 65 MB."""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ccdrift.logs import (SETTING_FIELDS, TOKEN_FIELDS, ParsedFile, duration_frame, frame, jsonl_files,
                          parse_file, parse_source)

HISTORY_FILE = "history.sqlite"
SCHEMA_VERSION = 1
# Bump whenever parse_file's output changes, so every transcript still on disk is
# read again. Rows of transcripts Claude Code already deleted keep their values.
PARSER_VERSION = 1

TEXT_COLUMNS = ("model",) + SETTING_FIELDS
FLAG_COLUMNS = ("is_sidechain", "new_prompt", "after_compaction")
COUNT_COLUMNS = TOKEN_FIELDS + ("thinking_logged", "signature_chars", "visible_chars", "n_mcp_calls")
RESPONSE_COLUMNS = TEXT_COLUMNS + FLAG_COLUMNS + COUNT_COLUMNS
DURATION_COLUMNS = ("version", "entrypoint", "is_sidechain", "duration_ms", "message_count")

# Integer keys, microsecond timestamps and file ids keep a year of responses near
# 65 MB; text keys, text timestamps and a path per row made it four times larger.
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, size INTEGER, mtime_ns INTEGER, session_id TEXT);
CREATE TABLE IF NOT EXISTS responses (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    model TEXT, version TEXT, entrypoint TEXT, effort TEXT, speed TEXT, service_tier TEXT,
    is_sidechain INTEGER, new_prompt INTEGER, after_compaction INTEGER,
    input_tokens INTEGER, output_tokens INTEGER, cache_creation INTEGER, cache_read INTEGER,
    cache_1h INTEGER, cache_5m INTEGER, thinking_logged INTEGER,
    signature_chars INTEGER, visible_chars INTEGER, n_mcp_calls INTEGER);
CREATE INDEX IF NOT EXISTS responses_file ON responses (file_id);
CREATE TABLE IF NOT EXISTS durations (
    key INTEGER PRIMARY KEY, file_id INTEGER NOT NULL, ts INTEGER,
    version TEXT, entrypoint TEXT, is_sidechain INTEGER, duration_ms INTEGER, message_count INTEGER);
CREATE INDEX IF NOT EXISTS durations_file ON durations (file_id);
"""


class HistoryError(RuntimeError):
    """The history store can't be used; the message names the file."""


def history_path(state_path: Path) -> Path:
    return state_path.parent / HISTORY_FILE


def _unusable(path: Path, exc: Exception) -> HistoryError:
    return HistoryError(f"Can't use the history store {path}: {exc}. Move it aside to rebuild it "
                        "from the transcripts still on disk.")


def row_key(text: str) -> int:
    """A response's text key as a signed 64-bit integer, so SQLite keeps it as the row id."""
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big", signed=True)


def _micros(ts: Any) -> Optional[int]:
    return None if ts is None else round(ts.timestamp() * 1_000_000)


def _count(value: Any) -> Optional[int]:
    return None if value is None else round(value)


def _upsert(table: str, columns: tuple[str, ...]) -> str:
    """Insert a row; a key already stored stays with the transcript whose path sorts
    first, the rule parse_source follows."""
    names = ("key", "file_id", "ts") + columns
    updates = ", ".join(f"{name} = excluded.{name}" for name in names[1:])
    return (f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' * len(names))}) "
            f"ON CONFLICT (key) DO UPDATE SET {updates} "
            f"WHERE (SELECT path FROM files WHERE id = excluded.file_id) "
            f"< (SELECT path FROM files WHERE id = {table}.file_id)")


def _decode(rows: pd.DataFrame, flags: tuple[str, ...]) -> pd.DataFrame:
    rows["timestamp"] = pd.to_datetime(rows.pop("ts").astype(float), unit="us", utc=True)
    for col in flags:
        rows[col] = rows[col].astype(bool)
    return rows


class History:
    """The store in `path`, created when missing."""

    def __init__(self, path: Path):
        self.path = path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(path, timeout=30)
        except (OSError, sqlite3.Error) as exc:
            # Nothing is wrong with the store itself, so moving it aside wouldn't help.
            raise HistoryError(f"Can't open the history store {path}: {exc}") from exc
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
            raise _unusable(path, ValueError(f"its schema version {stored!r} isn't a number")) from exc
        if schema_version > SCHEMA_VERSION:
            self.db.close()
            raise HistoryError(f"The history store {path} was written by a newer ccdrift. Upgrade ccdrift, "
                               "or move the store aside to rebuild it from the transcripts still on disk.")
        try:
            self.db.executescript(SCHEMA)
            self.meta = dict(self.db.execute("SELECT key, value FROM meta"))
            if "schema_version" not in self.meta:
                self._set_meta("schema_version", str(SCHEMA_VERSION))
        except sqlite3.Error as exc:
            self.db.close()
            raise _unusable(path, exc) from exc

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

    def update(self, source: Path) -> int:
        """Read the transcripts under `source` that are new or changed since the last
        update, or all of them after a parser change, and return how many were read.
        A transcript that can't be read now is tried again next time."""
        known = {path: (size, mtime) for path, size, mtime in
                 self.db.execute("SELECT path, size, mtime_ns FROM files")}
        reread = self.meta.get("parser_version") != str(PARSER_VERSION)
        read = skipped = 0
        for fp, rel in jsonl_files(source):
            try:
                stat = fp.stat()
                if not reread and known.get(rel) == (stat.st_size, stat.st_mtime_ns):
                    continue
                parsed = parse_file(fp, rel)
            except OSError:
                skipped += 1
                continue
            self._replace(rel, stat.st_size, stat.st_mtime_ns, parsed)
            read += 1
        if read and self.built_from() is None:
            self._set_meta("source", str(source.expanduser().resolve()))
        if not skipped:
            self._set_meta("parser_version", str(PARSER_VERSION))
        return read

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
            self.db.executemany(_upsert("responses", RESPONSE_COLUMNS), [
                (row_key(row["key"]), file_id, _micros(row["timestamp"]),
                 *(row[c] for c in TEXT_COLUMNS), *(int(row[c]) for c in FLAG_COLUMNS),
                 *(_count(row[c]) for c in COUNT_COLUMNS))
                for row in parsed.responses.values()])
            self.db.executemany(_upsert("durations", DURATION_COLUMNS), [
                (row_key(row["key"]), file_id, _micros(row["timestamp"]), row["version"], row["entrypoint"],
                 int(row["is_sidechain"]), _count(row["duration_ms"]), row["message_count"])
                for row in parsed.durations.values()])

    def responses(self) -> pd.DataFrame:
        """Every stored response, as the table parse_source returns."""
        rows = pd.read_sql_query(
            "SELECT f.path AS source_file, f.session_id, r.ts, "
            + ", ".join(f"r.{c}" for c in RESPONSE_COLUMNS)
            + " FROM responses r JOIN files f ON f.id = r.file_id ORDER BY f.path, r.ts", self.db)
        return frame(_decode(rows, FLAG_COLUMNS))

    def durations(self) -> pd.DataFrame:
        """Every stored turn duration, as the table parse_durations returns."""
        rows = pd.read_sql_query(
            "SELECT f.path AS source_file, f.session_id, d.ts, "
            + ", ".join(f"d.{c}" for c in DURATION_COLUMNS)
            + " FROM durations d JOIN files f ON f.id = d.file_id ORDER BY f.path, d.ts", self.db)
        return duration_frame(_decode(rows, ("is_sidechain",)))


def load_turns(source: Path, state_path: Path) -> pd.DataFrame:
    """Every response ccdrift knows of for `source`: the history store next to the
    state file, first brought up to date. A store built from another folder is left
    alone and `source` is parsed directly, with a note on stderr."""
    path = history_path(state_path)
    try:
        with History(path) as history:
            built_from = history.built_from()
            if built_from is not None and built_from != str(source.expanduser().resolve()):
                print(f"Not using ccdrift's history in {path}: it was built from {built_from}.", file=sys.stderr)
                return parse_source(source)
            history.update(source)
            return history.responses()
    except sqlite3.Error as exc:
        raise _unusable(path, exc) from exc
    except pd.errors.DatabaseError as exc:
        # pandas wraps SQLite's error in one that repeats the whole query.
        raise _unusable(path, exc.__cause__ or exc) from exc
