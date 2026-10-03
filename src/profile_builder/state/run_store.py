"""Durable domain state for a run (SQLite). Separate from the LangGraph checkpointer.

All writes are idempotent upserts keyed by natural keys so a replayed tool call after a
crash is harmless. Only parameterized SQL is used.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from profile_builder.config import RUN_DB

SCHEMA_VERSION = 1

_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS run (
  run_id TEXT PRIMARY KEY,
  start_url TEXT NOT NULL,
  product_focus TEXT,
  stage TEXT NOT NULL DEFAULT 'init',
  status TEXT NOT NULL DEFAULT 'running',
  settings_json TEXT NOT NULL,
  counters_json TEXT NOT NULL DEFAULT '{}',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  finished_at REAL,
  output_path TEXT
);
CREATE TABLE IF NOT EXISTS pages (
  url TEXT PRIMARY KEY,
  final_url TEXT,
  status TEXT NOT NULL,            -- fetched | skipped | failed
  http_status INTEGER,
  title TEXT,
  content_hash TEXT,
  char_count INTEGER DEFAULT 0,
  cache_file TEXT,
  error_code TEXT,
  error TEXT,
  selected INTEGER NOT NULL DEFAULT 1,
  fetched_at REAL
);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  url TEXT NOT NULL,
  ordinal INTEGER NOT NULL,
  heading TEXT,
  text TEXT NOT NULL,
  text_hash TEXT NOT NULL,
  embedding BLOB,
  UNIQUE(url, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(text_hash);
CREATE TABLE IF NOT EXISTS paragraphs (
  url TEXT NOT NULL,
  text_hash TEXT NOT NULL,
  PRIMARY KEY (url, text_hash)
);
CREATE TABLE IF NOT EXISTS evidence (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  field_path TEXT NOT NULL,
  kind TEXT NOT NULL,              -- website | interview
  source_url TEXT,
  excerpt TEXT,
  question_id TEXT,
  answer TEXT,
  superseded INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_path ON evidence(field_path);
CREATE TABLE IF NOT EXISTS questions (
  qid TEXT PRIMARY KEY,
  ordinal INTEGER NOT NULL,
  kind TEXT NOT NULL,
  question TEXT NOT NULL,
  why_unclear TEXT,
  field_paths_json TEXT NOT NULL,
  status TEXT NOT NULL,            -- pending | answered | skipped | unknown
  answer TEXT,
  asked_at REAL NOT NULL,
  answered_at REAL
);
CREATE TABLE IF NOT EXISTS warnings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  code TEXT NOT NULL,
  message TEXT NOT NULL,
  details_json TEXT,
  created_at REAL NOT NULL,
  UNIQUE(code, message)
);
CREATE TABLE IF NOT EXISTS conflicts (
  field_path TEXT PRIMARY KEY,
  claims_json TEXT NOT NULL,
  summary TEXT,
  status TEXT NOT NULL DEFAULT 'open',   -- open | resolved
  resolution TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS drafts (
  version INTEGER PRIMARY KEY AUTOINCREMENT,
  profile_json TEXT NOT NULL,
  source TEXT NOT NULL,            -- draft | update | finalize
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,              -- model | embedding
  model TEXT NOT NULL,
  input_tokens INTEGER NOT NULL DEFAULT 0,
  output_tokens INTEGER NOT NULL DEFAULT 0,
  cost_usd REAL NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
"""


@dataclass
class RunRecord:
    run_id: str
    start_url: str
    product_focus: str | None
    stage: str
    status: str
    settings: dict[str, Any]
    counters: dict[str, Any]
    created_at: float
    updated_at: float
    finished_at: float | None
    output_path: str | None


@dataclass
class PageRecord:
    url: str
    final_url: str | None
    status: str
    http_status: int | None
    title: str | None
    content_hash: str | None
    char_count: int
    cache_file: str | None
    error_code: str | None
    error: str | None
    selected: bool
    fetched_at: float | None


@dataclass
class ChunkRecord:
    id: int
    url: str
    ordinal: int
    heading: str
    text: str
    text_hash: str
    embedding: np.ndarray | None


class RunStore:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / RUN_DB
        run_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_DDL)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        try:
            self.path.chmod(0o600)
        except OSError:  # pragma: no cover
            pass

    def close(self) -> None:
        self._conn.close()

    # ---- run -----------------------------------------------------------------------
    def create_run(
        self, run_id: str, start_url: str, product_focus: str | None, settings: dict[str, Any]
    ) -> RunRecord:
        now = time.time()
        self._conn.execute(
            "INSERT OR IGNORE INTO run(run_id, start_url, product_focus, settings_json, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?)",
            (run_id, start_url, product_focus, json.dumps(settings), now, now),
        )
        return self.get_run()

    def get_run(self) -> RunRecord:
        row = self._conn.execute("SELECT * FROM run LIMIT 1").fetchone()
        if row is None:
            raise LookupError("run record not found")
        return RunRecord(
            run_id=row["run_id"],
            start_url=row["start_url"],
            product_focus=row["product_focus"],
            stage=row["stage"],
            status=row["status"],
            settings=json.loads(row["settings_json"]),
            counters=json.loads(row["counters_json"] or "{}"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            finished_at=row["finished_at"],
            output_path=row["output_path"],
        )

    def has_run(self) -> bool:
        return self._conn.execute("SELECT 1 FROM run LIMIT 1").fetchone() is not None

    def set_stage(self, stage: str) -> None:
        self._conn.execute("UPDATE run SET stage=?, updated_at=?", (stage, time.time()))

    def set_status(self, status: str, output_path: str | None = None) -> None:
        finished = time.time() if status in {"complete", "partial", "failed"} else None
        self._conn.execute(
            "UPDATE run SET status=?, updated_at=?, finished_at=COALESCE(?, finished_at),"
            " output_path=COALESCE(?, output_path)",
            (status, time.time(), finished, output_path),
        )

    def set_product_focus(self, product: str) -> None:
        self._conn.execute("UPDATE run SET product_focus=?, updated_at=?", (product, time.time()))

    def update_settings(self, settings: dict[str, Any]) -> None:
        self._conn.execute(
            "UPDATE run SET settings_json=?, updated_at=?", (json.dumps(settings), time.time())
        )

    def get_counter(self, key: str, default: int = 0) -> int:
        return int(self.get_run().counters.get(key, default))

    def set_counter(self, key: str, value: int) -> None:
        counters = self.get_run().counters
        counters[key] = value
        self._conn.execute(
            "UPDATE run SET counters_json=?, updated_at=?", (json.dumps(counters), time.time())
        )

    def increment_counter(self, key: str, by: int = 1) -> int:
        value = self.get_counter(key) + by
        self.set_counter(key, value)
        return value

    # ---- pages ---------------------------------------------------------------------
    def upsert_page(self, page: PageRecord) -> None:
        self._conn.execute(
            """INSERT INTO pages(url, final_url, status, http_status, title, content_hash, char_count,
                                 cache_file, error_code, error, selected, fetched_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(url) DO UPDATE SET final_url=excluded.final_url, status=excluded.status,
                 http_status=excluded.http_status, title=excluded.title,
                 content_hash=excluded.content_hash, char_count=excluded.char_count,
                 cache_file=excluded.cache_file, error_code=excluded.error_code,
                 error=excluded.error, selected=excluded.selected, fetched_at=excluded.fetched_at""",
            (
                page.url,
                page.final_url,
                page.status,
                page.http_status,
                page.title,
                page.content_hash,
                page.char_count,
                page.cache_file,
                page.error_code,
                page.error,
                int(page.selected),
                page.fetched_at,
            ),
        )

    def get_page(self, url: str) -> PageRecord | None:
        row = self._conn.execute("SELECT * FROM pages WHERE url=?", (url,)).fetchone()
        return self._page_from_row(row) if row else None

    def list_pages(self, status: str | None = None) -> list[PageRecord]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM pages WHERE status=? ORDER BY fetched_at", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM pages ORDER BY fetched_at").fetchall()
        return [self._page_from_row(r) for r in rows]

    def fetched_urls(self) -> set[str]:
        rows = self._conn.execute("SELECT url FROM pages WHERE status='fetched'").fetchall()
        return {r["url"] for r in rows}

    def unique_pages_scraped(self) -> int:
        """Distinct URLs for which a live scrape returned *something*: successful pages,
        duplicates/empties, off-site redirects and permanent failures. Transient failures,
        robots skips and guard rejections did not consume a scrape."""
        row = self._conn.execute(
            """SELECT COUNT(*) AS n FROM pages WHERE status='fetched'
               OR error_code IN ('DUPLICATE_CONTENT','EMPTY_CONTENT','REDIRECT_REJECTED','REDIRECT_OFFSITE')
               OR (status='failed' AND error_code NOT IN ('RATE_LIMITED','TIMEOUT','SERVER_ERROR','NETWORK','TRANSIENT','HTTP_429','HTTP_408')
                   AND error_code NOT LIKE 'HTTP_5%')"""
        ).fetchone()
        return int(row["n"])

    def shift_list_paths(self, base: str, removed_index: int) -> None:
        """After deleting list item `removed_index` of `base`, drop evidence/conflict rows that
        pointed at it and renumber rows pointing past it, so paths keep meaning."""
        for table in ("evidence", "conflicts"):
            rows = self._conn.execute(
                f"SELECT rowid AS rid, field_path FROM {table} WHERE field_path LIKE ?",
                (base + "[%",),
            ).fetchall()
            for r in rows:
                rest = r["field_path"][len(base) + 1 :]
                idx_str, _, tail = rest.partition("]")
                if not idx_str.isdigit():
                    continue
                idx = int(idx_str)
                if idx == removed_index:
                    if table == "evidence":  # conflicts stay as a record of what was omitted
                        self._conn.execute(
                            f"DELETE FROM {table} WHERE rowid=?",
                            (r["rid"],),
                        )
                elif idx > removed_index:
                    self._conn.execute(
                        f"UPDATE {table} SET field_path=? WHERE rowid=?",
                        (f"{base}[{idx - 1}]{tail}", r["rid"]),
                    )

    def set_conflict_status(self, field_path: str, status: str, resolution: str) -> None:
        self._conn.execute(
            "UPDATE conflicts SET status=?, resolution=? WHERE field_path=?",
            (status, resolution, field_path),
        )

    def interview_bases(self) -> set[str]:
        """Base field paths that currently carry a (non-superseded) user answer."""
        rows = self._conn.execute(
            "SELECT DISTINCT field_path FROM evidence WHERE kind='interview' AND superseded=0"
        ).fetchall()
        return {r["field_path"].split("[")[0] for r in rows}

    def content_hash_exists(self, content_hash: str, other_than: str) -> str | None:
        row = self._conn.execute(
            "SELECT url FROM pages WHERE content_hash=? AND url<>? AND status='fetched' LIMIT 1",
            (content_hash, other_than),
        ).fetchone()
        return row["url"] if row else None

    @staticmethod
    def _page_from_row(row: sqlite3.Row) -> PageRecord:
        return PageRecord(
            url=row["url"],
            final_url=row["final_url"],
            status=row["status"],
            http_status=row["http_status"],
            title=row["title"],
            content_hash=row["content_hash"],
            char_count=row["char_count"] or 0,
            cache_file=row["cache_file"],
            error_code=row["error_code"],
            error=row["error"],
            selected=bool(row["selected"]),
            fetched_at=row["fetched_at"],
        )

    # ---- chunks --------------------------------------------------------------------
    def replace_chunks(
        self, url: str, chunks: list[tuple[int, str, str, str, np.ndarray | None]]
    ) -> None:
        """chunks: (ordinal, heading, text, text_hash, embedding)."""
        self._conn.execute("DELETE FROM chunks WHERE url=?", (url,))
        self._conn.executemany(
            "INSERT INTO chunks(url, ordinal, heading, text, text_hash, embedding) VALUES (?,?,?,?,?,?)",
            [
                (url, o, h, t, th, (e.astype(np.float32).tobytes() if e is not None else None))
                for (o, h, t, th, e) in chunks
            ],
        )

    def chunk_hashes(self, exclude_url: str | None = None) -> set[str]:
        rows = self._conn.execute("SELECT text_hash FROM chunks WHERE url<>?", (exclude_url or "",))
        return {r["text_hash"] for r in rows}

    def replace_paragraphs(self, url: str, hashes: set[str]) -> None:
        self._conn.execute("DELETE FROM paragraphs WHERE url=?", (url,))
        self._conn.executemany(
            "INSERT OR IGNORE INTO paragraphs(url, text_hash) VALUES (?,?)",
            [(url, h) for h in hashes],
        )

    def paragraph_hashes(self, exclude_url: str | None = None) -> set[str]:
        rows = self._conn.execute(
            "SELECT text_hash FROM paragraphs WHERE url<>?", (exclude_url or "",)
        )
        return {r["text_hash"] for r in rows}

    def list_chunks(self, url: str | None = None) -> list[ChunkRecord]:
        if url:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE url=? ORDER BY ordinal", (url,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM chunks ORDER BY url, ordinal").fetchall()
        out = []
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else None
            out.append(
                ChunkRecord(
                    id=r["id"],
                    url=r["url"],
                    ordinal=r["ordinal"],
                    heading=r["heading"] or "",
                    text=r["text"],
                    text_hash=r["text_hash"],
                    embedding=emb,
                )
            )
        return out

    # ---- evidence ------------------------------------------------------------------
    def replace_website_evidence(
        self, rows: list[dict[str, Any]], *, only_fields: set[str] | None = None
    ) -> None:
        """Replace non-superseded website evidence. With `only_fields`, only evidence whose
        base field (e.g. ``customer.buyers``) is in the set is replaced; the rest is kept."""
        if only_fields is None:
            self._conn.execute("DELETE FROM evidence WHERE kind='website' AND superseded=0")
        else:
            for base in only_fields:
                self._conn.execute(
                    "DELETE FROM evidence WHERE kind='website' AND superseded=0 AND"
                    " (field_path=? OR field_path LIKE ? OR field_path LIKE ?)",
                    (base, base + "[%", base + ".%"),
                )
        now = time.time()
        self._conn.executemany(
            "INSERT INTO evidence(field_path, kind, source_url, excerpt, created_at) VALUES (?,?,?,?,?)",
            [
                (r["field_path"], "website", r.get("source_url"), r.get("excerpt"), now)
                for r in rows
            ],
        )

    def add_evidence(
        self,
        field_path: str,
        kind: str,
        *,
        source_url: str | None = None,
        excerpt: str | None = None,
        question_id: str | None = None,
        answer: str | None = None,
    ) -> None:
        exists = self._conn.execute(
            "SELECT 1 FROM evidence WHERE field_path=? AND kind=? AND IFNULL(source_url,'')=IFNULL(?,'')"
            " AND IFNULL(excerpt,'')=IFNULL(?,'') AND IFNULL(question_id,'')=IFNULL(?,'') AND superseded=0",
            (field_path, kind, source_url, excerpt, question_id),
        ).fetchone()
        if exists:
            return
        self._conn.execute(
            "INSERT INTO evidence(field_path, kind, source_url, excerpt, question_id, answer, created_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (field_path, kind, source_url, excerpt, question_id, answer, time.time()),
        )

    def supersede_evidence(self, field_path: str, kind: str = "website") -> int:
        cur = self._conn.execute(
            "UPDATE evidence SET superseded=1 WHERE field_path=? AND kind=? AND superseded=0",
            (field_path, kind),
        )
        return cur.rowcount

    def supersede_evidence_tree(self, field_path: str) -> dict[str, int]:
        """Mark evidence for `field_path` and its descendants as stale (value changed).
        Returns counts per kind so callers can warn about replaced website claims."""
        rows = self._conn.execute(
            "SELECT kind, COUNT(*) AS n FROM evidence WHERE superseded=0 AND"
            " (field_path=? OR field_path LIKE ? OR field_path LIKE ?) GROUP BY kind",
            (field_path, field_path + "[%", field_path + ".%"),
        ).fetchall()
        counts = {r["kind"]: int(r["n"]) for r in rows}
        self._conn.execute(
            "UPDATE evidence SET superseded=1 WHERE superseded=0 AND"
            " (field_path=? OR field_path LIKE ? OR field_path LIKE ?)",
            (field_path, field_path + "[%", field_path + ".%"),
        )
        return counts

    def list_evidence(self, include_superseded: bool = True) -> list[dict[str, Any]]:
        sql = "SELECT * FROM evidence" + ("" if include_superseded else " WHERE superseded=0")
        rows = self._conn.execute(sql + " ORDER BY field_path, id").fetchall()
        return [dict(r) for r in rows]

    # ---- questions -----------------------------------------------------------------
    def get_question(self, qid: str) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM questions WHERE qid=?", (qid,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["field_paths"] = json.loads(d.pop("field_paths_json") or "[]")
        return d

    def upsert_question(
        self, qid: str, kind: str, question: str, why_unclear: str, field_paths: list[str]
    ) -> dict[str, Any]:
        existing = self.get_question(qid)
        if existing:
            return existing
        ordinal = self._conn.execute("SELECT COUNT(*) AS n FROM questions").fetchone()["n"] + 1
        self._conn.execute(
            "INSERT INTO questions(qid, ordinal, kind, question, why_unclear, field_paths_json, status, asked_at)"
            " VALUES (?,?,?,?,?,?,'pending',?)",
            (qid, ordinal, kind, question, why_unclear, json.dumps(field_paths), time.time()),
        )
        return self.get_question(qid)  # type: ignore[return-value]

    def answer_question(self, qid: str, status: str, answer: str | None) -> None:
        self._conn.execute(
            "UPDATE questions SET status=?, answer=?, answered_at=? WHERE qid=?",
            (status, answer, time.time(), qid),
        )

    def list_questions(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM questions ORDER BY ordinal").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["field_paths"] = json.loads(d.pop("field_paths_json") or "[]")
            out.append(d)
        return out

    def questions_asked(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) AS n FROM questions WHERE status<>'pending'"
        ).fetchone()["n"]

    def pending_question(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM questions WHERE status='pending' ORDER BY ordinal LIMIT 1"
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["field_paths"] = json.loads(d.pop("field_paths_json") or "[]")
        return d

    # ---- warnings / conflicts ------------------------------------------------------
    def add_warning(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO warnings(code, message, details_json, created_at) VALUES (?,?,?,?)",
            (code, message, json.dumps(details) if details else None, time.time()),
        )

    def list_warnings(self) -> list[dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM warnings ORDER BY id").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["details"] = json.loads(d.pop("details_json") or "null")
            out.append(d)
        return out

    def has_warning(self, code: str) -> bool:
        return (
            self._conn.execute("SELECT 1 FROM warnings WHERE code=? LIMIT 1", (code,)).fetchone()
            is not None
        )

    def upsert_conflict(self, field_path: str, claims: list[dict[str, Any]], summary: str) -> None:
        self._conn.execute(
            """INSERT INTO conflicts(field_path, claims_json, summary, created_at) VALUES (?,?,?,?)
               ON CONFLICT(field_path) DO UPDATE SET claims_json=excluded.claims_json,
                 summary=excluded.summary""",
            (field_path, json.dumps(claims), summary, time.time()),
        )

    def resolve_conflict(self, field_path: str, resolution: str) -> None:
        self._conn.execute(
            "UPDATE conflicts SET status='resolved', resolution=? WHERE field_path=?",
            (resolution, field_path),
        )

    def list_conflicts(self, status: str | None = None) -> list[dict[str, Any]]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM conflicts WHERE status=? ORDER BY created_at", (status,)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM conflicts ORDER BY created_at").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["claims"] = json.loads(d.pop("claims_json") or "[]")
            out.append(d)
        return out

    # ---- drafts --------------------------------------------------------------------
    def save_draft(self, profile: dict[str, Any], source: str) -> int:
        cur = self._conn.execute(
            "INSERT INTO drafts(profile_json, source, created_at) VALUES (?,?,?)",
            (json.dumps(profile, ensure_ascii=False), source, time.time()),
        )
        return int(cur.lastrowid)

    def latest_draft(self) -> tuple[int, dict[str, Any]] | None:
        row = self._conn.execute(
            "SELECT version, profile_json FROM drafts ORDER BY version DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return int(row["version"]), json.loads(row["profile_json"])

    def draft_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) AS n FROM drafts").fetchone()["n"]

    # ---- usage ---------------------------------------------------------------------
    def add_usage(
        self, kind: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float
    ) -> None:
        self._conn.execute(
            "INSERT INTO usage(kind, model, input_tokens, output_tokens, cost_usd, created_at) VALUES (?,?,?,?,?,?)",
            (kind, model, input_tokens, output_tokens, cost_usd, time.time()),
        )

    def usage_totals(self) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT COUNT(*) AS calls, COALESCE(SUM(input_tokens),0) AS tin,"
            " COALESCE(SUM(output_tokens),0) AS tout, COALESCE(SUM(cost_usd),0) AS cost FROM usage"
        ).fetchone()
        model_calls = self._conn.execute(
            "SELECT COUNT(*) AS n FROM usage WHERE kind='model'"
        ).fetchone()["n"]
        return {
            "calls": row["calls"],
            "model_calls": model_calls,
            "input_tokens": row["tin"],
            "output_tokens": row["tout"],
            "cost_usd": float(row["cost"]),
        }

    def total_cost(self) -> float:
        return float(self.usage_totals()["cost_usd"])
