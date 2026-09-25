"""SCHEMA_VERSION moves with SCHEMA, and a store from every earlier version migrates to the
tables and columns a new store gets. tests/fixtures/schema-v<N>.sql holds what ccdrift
created at version N, dumped from the commit that shipped it."""

import sqlite3
from pathlib import Path

import pytest

from ccdrift.history import SCHEMA_VERSION, History

FIXTURES = Path(__file__).parent / "fixtures"


def dump(path):
    """The statements that make the store at `path` and its meta rows, as its fixture holds them."""
    db = sqlite3.connect(path)
    try:
        out = [row[0] + ";" for row in db.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY rowid")]
        out += ["INSERT INTO meta (key, value) VALUES ('%s', '%s');" % pair
                for pair in db.execute("SELECT key, value FROM meta ORDER BY key")]
    finally:
        db.close()
    return "\n".join(out) + "\n"


def shape(path):
    """Each table's columns by name, with type, NOT NULL, default and key, and the indexes."""
    db = sqlite3.connect(path)
    try:
        tables = [row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")]
        columns = {table: sorted(tuple(row[1:]) for row in db.execute(f"PRAGMA table_info({table})"))
                   for table in tables}
        indexes = sorted(row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'index'"))
    finally:
        db.close()
    return columns, indexes


def fresh(tmp_path):
    path = tmp_path / "fresh.sqlite"
    History(path).close()
    return path


def test_a_new_store_is_what_the_current_version_s_fixture_holds(tmp_path):
    # SCHEMA changed without SCHEMA_VERSION: stores already on disk keep the old tables,
    # and nothing adds what is new to them. Bump the version, add the _migrate branch,
    # and dump the new store to a fixture of its own, keeping the old one.
    assert dump(fresh(tmp_path)) == (FIXTURES / f"schema-v{SCHEMA_VERSION}.sql").read_text()


def test_every_earlier_version_has_a_fixture():
    assert sorted(int(p.stem.split("-v")[1]) for p in FIXTURES.glob("schema-v*.sql")) == list(
        range(1, SCHEMA_VERSION + 1))


@pytest.mark.parametrize("version", range(1, SCHEMA_VERSION))
def test_a_store_from_an_earlier_version_migrates_to_what_a_new_store_holds(tmp_path, version):
    old = tmp_path / "old.sqlite"
    db = sqlite3.connect(old)
    db.executescript((FIXTURES / f"schema-v{version}.sql").read_text())
    db.close()
    History(old).close()
    assert shape(old) == shape(fresh(tmp_path))
