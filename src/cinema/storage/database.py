"""SQLite job store: jobs, assets (with dependency graph), validation decisions, worker runs, events.

Every write is its own transaction; JSON manifests are mirrored atomically by storage.jobs.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..schemas import Decision
from ..state_machine import (AssetState, JobState, TransitionError, can_transition_asset,
                             can_transition_job)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY, status TEXT NOT NULL, genre TEXT, duration INTEGER, preset TEXT,
  seed INTEGER, created_at TEXT, updated_at TEXT, error TEXT, params TEXT, resume_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS assets (
  job_id TEXT NOT NULL, asset_id TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL,
  attempts INTEGER NOT NULL DEFAULT 0, path TEXT, sha256 TEXT, depends_on TEXT NOT NULL DEFAULT '[]',
  meta TEXT NOT NULL DEFAULT '{}', feedback TEXT, updated_at TEXT,
  PRIMARY KEY (job_id, asset_id)
);
CREATE TABLE IF NOT EXISTS decisions (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, asset_id TEXT, stage TEXT, criterion TEXT,
  evaluator TEXT, passed INTEGER, mandatory INTEGER, score REAL, evidence TEXT, action TEXT,
  attempt INTEGER, created_at TEXT
);
CREATE TABLE IF NOT EXISTS worker_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, worker TEXT, model TEXT, command TEXT,
  started_at TEXT, ended_at TEXT, seconds REAL, exit_code INTEGER, peak_rss_gib REAL,
  peak_footprint_gib REAL, avail_before_gib REAL, avail_after_gib REAL, pressure_before TEXT,
  pressure_after TEXT, exited_clean INTEGER, note TEXT
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, level TEXT, message TEXT, data TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_decisions_job ON decisions(job_id, asset_id);
CREATE INDEX IF NOT EXISTS idx_events_job ON events(job_id);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DB:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(50):  # concurrent first-open of the same DB can race on the WAL switch
            c = sqlite3.connect(self.path, timeout=30)
            try:
                c.execute("PRAGMA busy_timeout=30000")
                if c.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
                    c.execute("PRAGMA journal_mode=WAL")
                c.executescript(SCHEMA)
                break
            except sqlite3.OperationalError:
                if attempt == 49:
                    raise
                time.sleep(0.1)
            finally:
                c.close()

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA busy_timeout=30000")
        try:
            c.execute("BEGIN IMMEDIATE")
            yield c
            c.execute("COMMIT")
        except BaseException:
            if c.in_transaction:
                c.execute("ROLLBACK")
            raise
        finally:
            c.close()

    # ------------------------------------------------------------ jobs
    def create_job(self, job_id: str, genre: str, duration: int, preset: str, seed: int, params: dict) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO jobs (id,status,genre,duration,preset,seed,created_at,updated_at,params)"
                      " VALUES (?,?,?,?,?,?,?,?,?)",
                      (job_id, JobState.PENDING, genre, duration, preset, seed, now(), now(), json.dumps(params)))

    def job(self, job_id: str) -> dict[str, Any] | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["params"] = json.loads(d["params"] or "{}")
        return d

    def jobs(self) -> list[dict[str, Any]]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM jobs ORDER BY created_at DESC")]

    def set_job_status(self, job_id: str, status: JobState, error: str | None = None) -> None:
        with self.conn() as c:
            cur = JobState(c.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()[0])
            if not can_transition_job(cur, status):
                raise TransitionError(f"job {job_id}: {cur} -> {status} not allowed")
            c.execute("UPDATE jobs SET status=?, updated_at=?, error=COALESCE(?, error) WHERE id=?",
                      (status, now(), error, job_id))

    def bump_resume(self, job_id: str) -> int:
        with self.conn() as c:
            c.execute("UPDATE jobs SET resume_count=resume_count+1 WHERE id=?", (job_id,))
            return c.execute("SELECT resume_count FROM jobs WHERE id=?", (job_id,)).fetchone()[0]

    # ------------------------------------------------------------ assets
    def ensure_asset(self, job_id: str, asset_id: str, kind: str, depends_on: list[str] | None = None,
                     meta: dict | None = None) -> dict:
        with self.conn() as c:
            row = c.execute("SELECT * FROM assets WHERE job_id=? AND asset_id=?", (job_id, asset_id)).fetchone()
            if row is None:
                c.execute("INSERT INTO assets (job_id,asset_id,kind,status,depends_on,meta,updated_at)"
                          " VALUES (?,?,?,?,?,?,?)",
                          (job_id, asset_id, kind, AssetState.PENDING, json.dumps(depends_on or []),
                           json.dumps(meta or {}), now()))
            elif depends_on is not None and json.loads(row["depends_on"]) != depends_on:
                c.execute("UPDATE assets SET depends_on=? WHERE job_id=? AND asset_id=?",
                          (json.dumps(depends_on), job_id, asset_id))
        return self.asset(job_id, asset_id)

    def asset(self, job_id: str, asset_id: str) -> dict:
        with self.conn() as c:
            row = c.execute("SELECT * FROM assets WHERE job_id=? AND asset_id=?", (job_id, asset_id)).fetchone()
        if row is None:
            raise KeyError(asset_id)
        return _asset_row(row)

    def assets(self, job_id: str, kind: str | None = None) -> list[dict]:
        q = "SELECT * FROM assets WHERE job_id=?" + (" AND kind=?" if kind else "") + " ORDER BY asset_id"
        with self.conn() as c:
            return [_asset_row(r) for r in c.execute(q, (job_id, kind) if kind else (job_id,))]

    def set_asset(self, job_id: str, asset_id: str, status: AssetState | None = None, *,
                  path: str | None = None, sha256: str | None = None, meta: dict | None = None,
                  feedback: str | None = None, attempts_inc: bool = False) -> None:
        with self.conn() as c:
            row = c.execute("SELECT status, meta FROM assets WHERE job_id=? AND asset_id=?",
                            (job_id, asset_id)).fetchone()
            if row is None:
                raise KeyError(asset_id)
            if status is not None and not can_transition_asset(AssetState(row["status"]), status):
                raise TransitionError(f"asset {asset_id}: {row['status']} -> {status} not allowed")
            merged = json.loads(row["meta"]) | (meta or {})
            c.execute("UPDATE assets SET status=COALESCE(?,status), path=COALESCE(?,path),"
                      " sha256=COALESCE(?,sha256), meta=?, feedback=COALESCE(?,feedback),"
                      " attempts=attempts+?, updated_at=? WHERE job_id=? AND asset_id=?",
                      (status, path, sha256, json.dumps(merged), feedback, int(attempts_inc), now(),
                       job_id, asset_id))

    def invalidate(self, job_id: str, asset_id: str, reason: str, reset_attempts: bool = False) -> list[str]:
        """Mark asset and every transitive dependent PENDING. Returns invalidated ids (ordered)."""
        all_assets = {a["asset_id"]: a for a in self.assets(job_id)}
        dependents: dict[str, list[str]] = {}
        for a in all_assets.values():
            for dep in a["depends_on"]:
                dependents.setdefault(dep, []).append(a["asset_id"])
        order, stack, seen = [], [asset_id], set()
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in all_assets:
                continue
            seen.add(cur)
            order.append(cur)
            stack.extend(dependents.get(cur, []))
        with self.conn() as c:
            for aid in order:
                c.execute("UPDATE assets SET status=?, feedback=?, updated_at=?"
                          + (", attempts=0" if reset_attempts else "")
                          + " WHERE job_id=? AND asset_id=?",
                          (AssetState.PENDING, f"invalidated: {reason}" if aid != asset_id else reason,
                           now(), job_id, aid))
        return order

    # ------------------------------------------------------------ decisions / runs / events
    def log_decision(self, job_id: str, d: Decision) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO decisions (job_id,asset_id,stage,criterion,evaluator,passed,mandatory,score,"
                      "evidence,action,attempt,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      (job_id, d.asset_id, d.stage, d.criterion, d.evaluator, int(d.passed), int(d.mandatory),
                       d.score, d.evidence[:4000], d.action[:2000], d.attempt, now()))

    def decisions(self, job_id: str, asset_id: str | None = None) -> list[dict]:
        q = "SELECT * FROM decisions WHERE job_id=?" + (" AND asset_id=?" if asset_id else "") + " ORDER BY id"
        with self.conn() as c:
            return [dict(r) for r in c.execute(q, (job_id, asset_id) if asset_id else (job_id,))]

    def log_worker(self, job_id: str | None, **kw: Any) -> None:
        cols = ",".join(kw)
        with self.conn() as c:
            c.execute(f"INSERT INTO worker_runs (job_id,{cols}) VALUES (?{',?' * len(kw)})",
                      (job_id, *kw.values()))

    def worker_runs(self, job_id: str | None = None) -> list[dict]:
        with self.conn() as c:
            if job_id:
                return [dict(r) for r in c.execute("SELECT * FROM worker_runs WHERE job_id=? ORDER BY id", (job_id,))]
            return [dict(r) for r in c.execute("SELECT * FROM worker_runs ORDER BY id")]

    def event(self, job_id: str | None, message: str, level: str = "info", **data: Any) -> None:
        with self.conn() as c:
            c.execute("INSERT INTO events (job_id,level,message,data,created_at) VALUES (?,?,?,?,?)",
                      (job_id, level, message, json.dumps(data, default=str), now()))

    def events(self, job_id: str, limit: int = 50) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT * FROM events WHERE job_id=? ORDER BY id DESC LIMIT ?", (job_id, limit))
            return [dict(r) for r in rows][::-1]


def _asset_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["depends_on"] = json.loads(d["depends_on"])
    d["meta"] = json.loads(d["meta"])
    return d


def wait_for(predicate, timeout: float, interval: float = 1.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()
