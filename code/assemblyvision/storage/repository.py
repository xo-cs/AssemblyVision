"""Repository (LLD 3.2): the SQLite run store of a data area, behind RunStore.

One database per data area, inside it: data/qisens/av.db and data/public/av_dev.db
(CLAUDE.md), so QiSens rows never leave data/qisens. Tables: runs, steps, metrics,
evaluations, notices and revisions. The run folders stay the record the pipeline writes
(manifest.json, steps.csv, metrics.csv, eval.json, export/revisions.json); the database
indexes them: index_run() replaces a run's rows in one transaction, sync() indexes every
run whose files changed since (by their modification times) and drops rows of runs that
are gone. migrate() is sync() over all runs of the area (tools/migrate_runs.py).
"""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from .files import RUN_ID_RE, source_root, utc_now

SCHEMA_VERSION = 1
DB_NAME = {"qisens": "av.db", "public": "av_dev.db"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, source TEXT NOT NULL, video TEXT, video_path TEXT, status TEXT,
    created_utc TEXT, finished_utc TEXT, variant TEXT, engine TEXT, video_s REAL, has_steps INTEGER,
    files_mtime REAL, indexed_utc TEXT);
CREATE TABLE IF NOT EXISTS steps (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE, seg INTEGER NOT NULL, label TEXT,
    start_s REAL, end_s REAL, source TEXT, score REAL, PRIMARY KEY (run_id, seg));
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE, row INTEGER NOT NULL, segment_id TEXT,
    step INTEGER, label TEXT, start_s REAL, end_s REAL, metric TEXT, value REAL, unit TEXT, coverage REAL,
    flags TEXT, PRIMARY KEY (run_id, row));
CREATE TABLE IF NOT EXISTS evaluations (
    run_id TEXT PRIMARY KEY REFERENCES runs(run_id) ON DELETE CASCADE, gt_file TEXT, gt_sha256 TEXT,
    frame_accuracy REAL, edit REAL, f1_10 REAL, f1_25 REAL, f1_50 REAL, boundary_mae_s REAL,
    created_utc TEXT, record TEXT);
CREATE TABLE IF NOT EXISTS notices (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE, idx INTEGER NOT NULL, stage TEXT,
    code TEXT, effect TEXT, text TEXT, detail TEXT, PRIMARY KEY (run_id, idx));
CREATE TABLE IF NOT EXISTS revisions (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE, revision INTEGER NOT NULL,
    based_on INTEGER, by_user TEXT, note TEXT, created_utc TEXT, PRIMARY KEY (run_id, revision));
CREATE INDEX IF NOT EXISTS metrics_step ON metrics(run_id, step);
"""
FILES = ("manifest.json", "steps.csv", "metrics.csv", "eval.json", "export/revisions.json")


def _f(x):
    return None if x in (None, "") else float(x)


class Repository:
    def __init__(self, source: str, paths_cfg: dict | None, repo_root: Path | None = None, area: Path | None = None):
        """area: the data area folder itself (default: from paths_cfg)."""
        from .files import REPO_ROOT
        self.source = source
        self.root = area if area is not None else source_root(source, paths_cfg, repo_root or REPO_ROOT)
        self.db_path = self.root / DB_NAME[source]
        self.runs_dir = self.root / "runs"
        self._con: sqlite3.Connection | None = None

    def con(self) -> sqlite3.Connection:
        if self._con is None:
            self.root.mkdir(parents=True, exist_ok=True)
            c = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            c.executescript(SCHEMA)
            c.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            self._con = c
        return self._con

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    # indexing ----------------------------------------------------------------------

    @staticmethod
    def files_mtime(folder: Path) -> float:
        return max((p.stat().st_mtime for p in (folder / f for f in FILES) if p.is_file()), default=0.0)

    def index_run(self, folder: Path) -> None:
        """Replaces the rows of the run in folder by its files (one transaction)."""
        m = json.loads((folder / "manifest.json").read_text())
        rid = folder.name
        ing = m.get("ingest", {})
        run = (rid, self.source, Path(m.get("video", "")).name, m.get("video", ""), m.get("status", "?"),
               m.get("created_utc", ""), m.get("finished_utc"), m.get("variant"),
               m.get("config", {}).get("seg_a", {}).get("engine") or m.get("engine"),
               (ing.get("cfr") or {}).get("duration_s") or (ing.get("input") or {}).get("duration_s"),
               int((folder / "steps.csv").is_file()), self.files_mtime(folder), utc_now())
        steps, metrics = [], []
        if (folder / "steps.csv").is_file():
            with open(folder / "steps.csv", newline="", encoding="utf-8") as f:
                for i, r in enumerate(csv.DictReader(f)):
                    steps.append((rid, i + 1, r["label"], _f(r["start_s"]), _f(r["end_s"]), r.get("source", ""),
                                  _f(r.get("score"))))
        if (folder / "metrics.csv").is_file():
            with open(folder / "metrics.csv", newline="", encoding="utf-8") as f:
                for i, r in enumerate(csv.DictReader(f)):
                    metrics.append((rid, i, r.get("segment_id"), int(r["step"]) if r.get("step") else None, r.get("label"),
                                    _f(r.get("start_s")), _f(r.get("end_s")), r.get("metric"), _f(r.get("value")),
                                    r.get("unit"), _f(r.get("coverage")), r.get("flags", "")))
        ev = None
        if (folder / "eval.json").is_file():
            try:
                e = json.loads((folder / "eval.json").read_text())
                em = e.get("metrics", {})
                ev = (rid, e.get("gt_file"), e.get("gt_sha256"), em.get("frame_accuracy"), em.get("edit"), em.get("f1_10"),
                      em.get("f1_25"), em.get("f1_50"), em.get("boundary_mae_s"), e.get("created_utc"), json.dumps(e))
            except ValueError:
                ev = None
        notices = [(rid, i, n.get("stage"), n.get("code"), n.get("effect"), n.get("text"), n.get("detail"))
                   for i, n in enumerate(m.get("notices", []))]
        revs = []
        rp = folder / "export" / "revisions.json"
        if rp.is_file():
            try:
                revs = [(rid, r["revision"], r.get("based_on"), r.get("by"), r.get("note"), r.get("created_utc"))
                        for r in json.loads(rp.read_text())]
            except (ValueError, KeyError):
                revs = []
        c = self.con()
        with c:
            c.execute("DELETE FROM runs WHERE run_id = ?", (rid,))          # cascades to every other table
            c.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", run)
            c.executemany("INSERT INTO steps VALUES (?,?,?,?,?,?,?)", steps)
            c.executemany("INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", metrics)
            if ev is not None:
                c.execute("INSERT INTO evaluations VALUES (?,?,?,?,?,?,?,?,?,?,?)", ev)
            c.executemany("INSERT INTO notices VALUES (?,?,?,?,?,?,?)", notices)
            c.executemany("INSERT INTO revisions VALUES (?,?,?,?,?,?)", revs)

    def sync(self, run_id: str | None = None) -> dict:
        """Indexes the runs (or one run) whose files changed; drops rows of runs that are gone."""
        c = self.con()
        known = {r["run_id"]: r["files_mtime"] for r in c.execute("SELECT run_id, files_mtime FROM runs")}
        folders = []
        if run_id is not None:
            folders = [self.runs_dir / run_id]
        elif self.runs_dir.is_dir():
            folders = [d for d in self.runs_dir.iterdir() if RUN_ID_RE.fullmatch(d.name)]
        done, gone, failed = 0, 0, []
        present = set()
        for d in folders:
            if not (d / "manifest.json").is_file():
                continue
            present.add(d.name)
            if known.get(d.name) == self.files_mtime(d):
                continue
            try:
                self.index_run(d)
                done += 1
            except (OSError, ValueError, KeyError, sqlite3.Error) as e:
                failed.append(f"{d.name}: {type(e).__name__}")
        if run_id is None:
            with c:
                for rid in set(known) - present:
                    c.execute("DELETE FROM runs WHERE run_id = ?", (rid,))
                    gone += 1
        elif run_id not in present and run_id in known:
            with c:
                c.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))
                gone += 1
        return {"indexed": done, "removed": gone, "failed": failed, "total": len(present)}

    migrate = sync

    # reads (the names RunStore offers) ----------------------------------------------

    def list_runs(self) -> list[dict]:
        rows = self.con().execute("SELECT * FROM runs ORDER BY run_id DESC").fetchall()
        return [{"run_id": r["run_id"], "source": r["source"], "video": r["video"], "status": r["status"],
                 "created_utc": r["created_utc"], "finished_utc": r["finished_utc"], "variant": r["variant"],
                 "video_s": r["video_s"], "video_path": r["video_path"], "has_steps": bool(r["has_steps"])}
                for r in rows]

    def get_steps(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self.con().execute(
            "SELECT label, start_s, end_s, source, score FROM steps WHERE run_id = ? ORDER BY seg", (run_id,))]

    def get_metrics(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self.con().execute(
            "SELECT segment_id, step, label, start_s, end_s, metric, value, unit, coverage, flags FROM metrics "
            "WHERE run_id = ? ORDER BY row", (run_id,))]

    def get_eval(self, run_id: str) -> dict | None:
        r = self.con().execute("SELECT record FROM evaluations WHERE run_id = ?", (run_id,)).fetchone()
        return json.loads(r["record"]) if r else None

    def get_notices(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self.con().execute(
            "SELECT stage, code, effect, text, detail FROM notices WHERE run_id = ? ORDER BY idx", (run_id,))]

    def get_revisions(self, run_id: str) -> list[dict]:
        return [dict(r) for r in self.con().execute(
            "SELECT revision, based_on, by_user, note, created_utc FROM revisions WHERE run_id = ? ORDER BY revision",
            (run_id,))]
