"""
manifest.py — the relational source of truth for customer timelines and
document versions. Chroma is only the searchable index; this decides which
meetings exist, which revision of each is current, and in what order an
account's conversations happened.

Portfolio scale: SQLite file next to the Chroma index (data/chroma/manifest.sqlite),
so the index and its manifest always travel together. Production: the same
schema in PostgreSQL.

Tables
  accounts          tenant_id, account_id, name, aliases (json)
  meetings          one row per meeting: account, occurred_at, conversation_sequence,
                    current_version, status (active | deleted)
  meeting_versions  every revision ever indexed: content_hash, fingerprint, source_updated_at,
                    status (pending | committed), is_current, supersedes_version, chunk_count
  index_runs        one row per indexing run: index_version (YYYY-MM-DD.N), status, counts
  collection_meta   per Chroma collection: the embedding model its vectors were made with

Two-phase versions: a revision is recorded as `pending`, its vectors are written (inactive),
the vectors are switched on, and only then is the revision `committed` and made current.
A crash at any point leaves either the old revision serving or a pending row that the next
run finishes — never a manifest that claims vectors that don't exist.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    tenant_id   TEXT NOT NULL,
    account_id  TEXT NOT NULL,
    name        TEXT NOT NULL,
    aliases     TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (tenant_id, account_id)
);
CREATE TABLE IF NOT EXISTS meetings (
    tenant_id             TEXT NOT NULL,
    meeting_id            TEXT NOT NULL,
    account_id            TEXT NOT NULL DEFAULT '',
    title                 TEXT NOT NULL,
    access_level          TEXT NOT NULL,
    occurred_at           TEXT NOT NULL,
    occurred_ts           INTEGER NOT NULL,
    date_int              INTEGER NOT NULL,
    conversation_sequence INTEGER NOT NULL DEFAULT 0,
    current_version       INTEGER NOT NULL DEFAULT 0,   -- 0 until a revision is committed
    status                TEXT NOT NULL DEFAULT 'active',
    first_seen_at         TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    PRIMARY KEY (tenant_id, meeting_id)
);
CREATE INDEX IF NOT EXISTS ix_meetings_account ON meetings (tenant_id, account_id, occurred_ts);
CREATE TABLE IF NOT EXISTS meeting_versions (
    tenant_id           TEXT NOT NULL,
    meeting_id          TEXT NOT NULL,
    version             INTEGER NOT NULL,
    content_hash        TEXT NOT NULL,
    fingerprint         TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'committed',
    source_updated_at   TEXT NOT NULL,
    indexed_at          TEXT NOT NULL,
    is_current          INTEGER NOT NULL,
    supersedes_version  INTEGER,
    chunk_count         INTEGER NOT NULL,
    index_version       TEXT NOT NULL,
    chunking_version    TEXT NOT NULL,
    PRIMARY KEY (tenant_id, meeting_id, version)
);
CREATE TABLE IF NOT EXISTS index_runs (
    index_version     TEXT PRIMARY KEY,
    tenant_id         TEXT NOT NULL,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    status            TEXT NOT NULL DEFAULT 'running',
    error             TEXT,
    embedding_model   TEXT NOT NULL,
    chunking_version  TEXT NOT NULL,
    inserted          INTEGER DEFAULT 0,
    updated           INTEGER DEFAULT 0,
    reembedded        INTEGER DEFAULT 0,
    repaired          INTEGER DEFAULT 0,
    unchanged         INTEGER DEFAULT 0,
    deleted           INTEGER DEFAULT 0,
    total_chunks      INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS collection_meta (
    collection   TEXT PRIMARY KEY,
    embed_model  TEXT NOT NULL,
    created_at   TEXT NOT NULL
);
"""

# Columns added after the first release — added in place to existing manifest files.
_MIGRATIONS = {
    "meeting_versions": {"fingerprint": "TEXT NOT NULL DEFAULT ''", "status": "TEXT NOT NULL DEFAULT 'committed'"},
    "index_runs": {"status": "TEXT NOT NULL DEFAULT 'ok'", "error": "TEXT",
                   "reembedded": "INTEGER DEFAULT 0", "repaired": "INTEGER DEFAULT 0"},
}


def default_path() -> Path:
    return Path(os.getenv("MANIFEST_PATH", Path(os.getenv("CHROMA_DIR", "./data/chroma")) / "manifest.sqlite"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Manifest:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else default_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        for table, cols in _MIGRATIONS.items():
            have = {r["name"] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for col, ddl in cols.items():
                if col not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")
        self.conn.commit()

    @classmethod
    def open_if_exists(cls, path: str | Path | None = None) -> "Manifest | None":
        p = Path(path) if path else default_path()
        return cls(p) if p.exists() else None

    @contextmanager
    def tx(self):
        try:
            yield self.conn
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def reset(self, tenant_id: str) -> None:
        """Forget one tenant's accounts, meetings and versions. Other tenants are untouched;
        index_runs are kept as audit history."""
        with self.tx() as c:
            for t in ("accounts", "meetings", "meeting_versions"):
                c.execute(f"DELETE FROM {t} WHERE tenant_id = ?", (tenant_id,))

    def reset_all(self) -> None:
        """Forget every tenant (used together with deleting the whole collection)."""
        with self.tx() as c:
            for t in ("accounts", "meetings", "meeting_versions", "collection_meta"):
                c.execute(f"DELETE FROM {t}")

    def has_tenant(self, tenant_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM meetings WHERE tenant_id=? LIMIT 1", (tenant_id,)).fetchone() is not None

    # ── collection-level settings ─────────────────────────────────────────────
    def collection_embed_model(self, collection: str) -> str | None:
        r = self.conn.execute("SELECT embed_model FROM collection_meta WHERE collection=?", (collection,)).fetchone()
        return r["embed_model"] if r else None

    def set_collection_embed_model(self, collection: str, embed_model: str) -> None:
        with self.tx() as c:
            c.execute("""INSERT INTO collection_meta (collection, embed_model, created_at) VALUES (?,?,?)
                         ON CONFLICT(collection) DO UPDATE SET embed_model=excluded.embed_model""",
                      (collection, embed_model, _now()))

    # ── accounts ──────────────────────────────────────────────────────────────
    def upsert_account(self, tenant_id: str, account_id: str, name: str, aliases: list[str]) -> None:
        with self.tx() as c:
            c.execute("""INSERT INTO accounts (tenant_id, account_id, name, aliases) VALUES (?,?,?,?)
                         ON CONFLICT(tenant_id, account_id) DO UPDATE SET name=excluded.name, aliases=excluded.aliases""",
                      (tenant_id, account_id, name, json.dumps(sorted(set(aliases)))))

    def accounts(self, tenant_id: str) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM accounts WHERE tenant_id = ? ORDER BY account_id", (tenant_id,))
        return [{"account_id": r["account_id"], "name": r["name"], "aliases": json.loads(r["aliases"])} for r in rows]

    # ── meetings / versions ───────────────────────────────────────────────────
    def get_meeting(self, tenant_id: str, meeting_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM meetings WHERE tenant_id=? AND meeting_id=?", (tenant_id, meeting_id)).fetchone()
        return dict(r) if r else None

    def current_version_row(self, tenant_id: str, meeting_id: str) -> dict | None:
        """The committed revision currently being served (pending rows never count)."""
        r = self.conn.execute("""SELECT * FROM meeting_versions WHERE tenant_id=? AND meeting_id=?
                                 AND is_current=1 AND status='committed'""", (tenant_id, meeting_id)).fetchone()
        return dict(r) if r else None

    def versions(self, tenant_id: str, meeting_id: str) -> list[dict]:
        rows = self.conn.execute("""SELECT * FROM meeting_versions WHERE tenant_id=? AND meeting_id=?
                                    ORDER BY version""", (tenant_id, meeting_id))
        return [dict(r) for r in rows]

    def upsert_meeting(self, tenant_id: str, meeting_id: str, *, account_id: str, title: str, access_level: str,
                       occurred_at: str, occurred_ts: int, date_int: int) -> None:
        """Record what the source says about a meeting. current_version is only set by commit_version()."""
        now = _now()
        with self.tx() as c:
            c.execute("""INSERT INTO meetings (tenant_id, meeting_id, account_id, title, access_level, occurred_at,
                                               occurred_ts, date_int, current_version, status, first_seen_at, updated_at)
                         VALUES (?,?,?,?,?,?,?,?,0, 'active', ?, ?)
                         ON CONFLICT(tenant_id, meeting_id) DO UPDATE SET
                           account_id=excluded.account_id, title=excluded.title, access_level=excluded.access_level,
                           occurred_at=excluded.occurred_at, occurred_ts=excluded.occurred_ts,
                           date_int=excluded.date_int, status='active', updated_at=excluded.updated_at""",
                      (tenant_id, meeting_id, account_id, title, access_level, occurred_at, occurred_ts, date_int,
                       now, now))

    def pending_version(self, tenant_id: str, meeting_id: str, content_hash: str) -> int | None:
        """A pending revision of exactly this content left by an interrupted run (reuse its number)."""
        r = self.conn.execute("""SELECT version FROM meeting_versions WHERE tenant_id=? AND meeting_id=?
                                 AND status='pending' AND content_hash=? ORDER BY version DESC LIMIT 1""",
                              (tenant_id, meeting_id, content_hash)).fetchone()
        return r["version"] if r else None

    def next_version(self, tenant_id: str, meeting_id: str) -> int:
        r = self.conn.execute("SELECT MAX(version) FROM meeting_versions WHERE tenant_id=? AND meeting_id=?",
                              (tenant_id, meeting_id)).fetchone()
        return (r[0] or 0) + 1

    def begin_version(self, tenant_id: str, meeting_id: str, *, version: int, content_hash: str, fingerprint: str,
                      source_updated_at: str, chunk_count: int, index_version: str, chunking_version: str) -> None:
        """Phase 1: record the revision as pending (not current) BEFORE its vectors are written."""
        with self.tx() as c:
            c.execute("""INSERT INTO meeting_versions (tenant_id, meeting_id, version, content_hash, fingerprint, status,
                                                       source_updated_at, indexed_at, is_current, supersedes_version,
                                                       chunk_count, index_version, chunking_version)
                         VALUES (?,?,?,?,?, 'pending', ?, ?, 0, NULL, ?, ?, ?)
                         ON CONFLICT(tenant_id, meeting_id, version) DO UPDATE SET
                           content_hash=excluded.content_hash, fingerprint=excluded.fingerprint, status='pending',
                           source_updated_at=excluded.source_updated_at, indexed_at=excluded.indexed_at,
                           chunk_count=excluded.chunk_count, index_version=excluded.index_version,
                           chunking_version=excluded.chunking_version""",
                      (tenant_id, meeting_id, version, content_hash, fingerprint, source_updated_at, _now(),
                       chunk_count, index_version, chunking_version))

    def commit_version(self, tenant_id: str, meeting_id: str, version: int) -> None:
        """Phase 2: AFTER the vectors are written and switched on, make this revision current."""
        with self.tx() as c:
            prev = c.execute("""SELECT version FROM meeting_versions WHERE tenant_id=? AND meeting_id=?
                                AND is_current=1 AND status='committed' AND version != ?""",
                             (tenant_id, meeting_id, version)).fetchone()
            c.execute("UPDATE meeting_versions SET is_current=0 WHERE tenant_id=? AND meeting_id=?", (tenant_id, meeting_id))
            c.execute("""UPDATE meeting_versions SET status='committed', is_current=1,
                           supersedes_version=COALESCE(supersedes_version, ?)
                         WHERE tenant_id=? AND meeting_id=? AND version=?""",
                      (prev["version"] if prev else None, tenant_id, meeting_id, version))
            c.execute("UPDATE meetings SET current_version=?, updated_at=? WHERE tenant_id=? AND meeting_id=?",
                      (version, _now(), tenant_id, meeting_id))

    def update_fingerprint(self, tenant_id: str, meeting_id: str, version: int, fingerprint: str,
                           chunk_count: int, index_version: str, chunking_version: str) -> None:
        """After a successful re-embed of the SAME revision (settings changed, content didn't)."""
        with self.tx() as c:
            c.execute("""UPDATE meeting_versions SET fingerprint=?, chunk_count=?, index_version=?, chunking_version=?,
                           indexed_at=? WHERE tenant_id=? AND meeting_id=? AND version=?""",
                      (fingerprint, chunk_count, index_version, chunking_version, _now(), tenant_id, meeting_id, version))

    def add_version(self, tenant_id: str, meeting_id: str, *, version: int, content_hash: str,
                    source_updated_at: str, chunk_count: int, index_version: str, chunking_version: str,
                    fingerprint: str = "") -> None:
        """begin + commit in one call (for callers that have already written the vectors)."""
        self.begin_version(tenant_id, meeting_id, version=version, content_hash=content_hash, fingerprint=fingerprint,
                           source_updated_at=source_updated_at, chunk_count=chunk_count,
                           index_version=index_version, chunking_version=chunking_version)
        self.commit_version(tenant_id, meeting_id, version)

    def tombstone(self, tenant_id: str, meeting_id: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE meetings SET status='deleted', updated_at=? WHERE tenant_id=? AND meeting_id=?",
                      (_now(), tenant_id, meeting_id))
            c.execute("UPDATE meeting_versions SET is_current=0 WHERE tenant_id=? AND meeting_id=?", (tenant_id, meeting_id))

    def active_meeting_ids(self, tenant_id: str) -> set[str]:
        rows = self.conn.execute("SELECT meeting_id FROM meetings WHERE tenant_id=? AND status='active'", (tenant_id,))
        return {r["meeting_id"] for r in rows}

    def recompute_sequences(self, tenant_id: str) -> dict[str, int]:
        """Number each account's active meetings 1..n by time. Returns {meeting_id: seq} for rows that changed."""
        changed: dict[str, int] = {}
        rows = self.conn.execute("""SELECT meeting_id, account_id, conversation_sequence FROM meetings
                                    WHERE tenant_id=? AND status='active' AND account_id != ''
                                    AND current_version > 0
                                    ORDER BY account_id, occurred_ts, meeting_id""", (tenant_id,)).fetchall()
        seq_by_acct: dict[str, int] = {}
        with self.tx() as c:
            for r in rows:
                seq = seq_by_acct[r["account_id"]] = seq_by_acct.get(r["account_id"], 0) + 1
                if r["conversation_sequence"] != seq:
                    c.execute("UPDATE meetings SET conversation_sequence=? WHERE tenant_id=? AND meeting_id=?",
                              (seq, tenant_id, r["meeting_id"]))
                    changed[r["meeting_id"]] = seq
        return changed

    def timeline(self, tenant_id: str, account_id: str, *, allowed_levels: list[str] | None = None,
                 date_from: int | None = None, date_to: int | None = None) -> list[dict]:
        """An account's active meetings, oldest → newest, restricted to what the caller may see."""
        # current_version > 0: only meetings whose vectors were committed
        sql = ["SELECT * FROM meetings WHERE tenant_id=? AND account_id=? AND status='active' AND current_version > 0"]
        args: list = [tenant_id, account_id]
        if allowed_levels is not None:
            sql.append(f"AND access_level IN ({','.join('?' * len(allowed_levels))})")
            args += list(allowed_levels)
        if date_from is not None:
            sql.append("AND date_int >= ?")
            args.append(date_from)
        if date_to is not None:
            sql.append("AND date_int <= ?")
            args.append(date_to)
        sql.append("ORDER BY occurred_ts, meeting_id")
        return [dict(r) for r in self.conn.execute(" ".join(sql), args)]

    # ── index runs ────────────────────────────────────────────────────────────
    def start_run(self, tenant_id: str, embedding_model: str, chunking_version: str, today: str | None = None) -> str:
        """index_version = YYYY-MM-DD.N — N is one more than the highest N used that day (never reused)."""
        day = today or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        rows = self.conn.execute("SELECT index_version FROM index_runs WHERE index_version LIKE ?", (f"{day}.%",))
        n = max((int(r[0].rsplit(".", 1)[1]) for r in rows if r[0].rsplit(".", 1)[1].isdigit()), default=0)
        version = f"{day}.{n + 1}"
        with self.tx() as c:
            c.execute("""INSERT INTO index_runs (index_version, tenant_id, started_at, embedding_model, chunking_version,
                                                 status) VALUES (?,?,?,?,?, 'running')""",
                      (version, tenant_id, _now(), embedding_model, chunking_version))
        return version

    def finish_run(self, index_version: str, **counts: int) -> None:
        cols = ", ".join(f"{k}=?" for k in counts)
        with self.tx() as c:
            c.execute(f"UPDATE index_runs SET finished_at=?, status='ok', {cols} WHERE index_version=?",
                      (_now(), *counts.values(), index_version))

    def fail_run(self, index_version: str, error: str) -> None:
        with self.tx() as c:
            c.execute("UPDATE index_runs SET finished_at=?, status='failed', error=? WHERE index_version=?",
                      (_now(), error[:2000], index_version))

    def last_run(self, tenant_id: str | None = None) -> dict | None:
        sql, args = "SELECT * FROM index_runs", []
        if tenant_id:
            sql, args = sql + " WHERE tenant_id=?", [tenant_id]
        r = self.conn.execute(sql + " ORDER BY started_at DESC LIMIT 1", args).fetchone()
        return dict(r) if r else None
