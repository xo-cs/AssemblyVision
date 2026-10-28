"""Read access to stored runs for the interface, until Repository exists.

LLD deviation: LLD 3.2 has the interface read everything through Repository
(SQLite plus the File Store). There is no database yet, so RunStore reads the
run folders (manifest.json, steps.csv, metrics.csv, eval.json, reference
frames, overlay) and offers the read methods Repository will have, with the
same names, so the interface does not change when Repository arrives.

RunStore only ever looks inside the data areas named in `sources`. A source
that is not listed is never opened, listed or resolved (CLAUDE.md rule 1): a
run ID is accepted only if it has the run ID form and its folder exists under
data/<source>/runs/ of an allowed source, else AV-APP-003.
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import AvError
from .files import REPO_ROOT, RUN_ID_RE, SOURCES, RunPaths, source_root, utc_now, write_atomic

VIDEO_DIRS = ("videos", "input")          # data/<source>/videos (stored), data/<source>/input (uploads)
VIDEO_EXT = (".mp4", ".avi")
FINISHED = ("done", "partial", "failed")
# share of a typical run's time per stage, for the progress bar
STAGE_WEIGHTS = {"ingest": 2, "shots": 6, "reference": 5, "pose_body": 60, "tracking": 3, "segmentation": 1,
                 "metrics": 1, "overlay": 20, "eval": 1}


@dataclass
class RunStatus:
    run_id: str
    source: str
    status: str                       # queued, running, done, partial, failed
    stage: str | None = None          # the running stage
    progress: float = 0.0             # 0 to 1
    stages: list = field(default_factory=list)   # [(name, status, seconds, progress)]
    summary: str = ""
    error: str = ""
    waiting: str = ""
    active: bool = False              # a worker process is alive


def worker_alive(pid: int | None, run_id: str) -> bool:
    """True if process pid exists and its command line names run_id (so a
    reused PID of another program is not taken for the worker)."""
    if not pid:
        return False
    try:
        cmd = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (OSError, ValueError):
        return False
    return run_id.encode() in cmd


def _safe_name(name: str) -> str:
    stem, ext = os.path.splitext(Path(name).name)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "video"
    return stem + ext.lower()


class RunStore:
    def __init__(self, sources: list[str], cfg, repo_root: Path = REPO_ROOT):
        bad = [s for s in sources if s not in SOURCES]
        if bad or not sources:
            raise ValueError(f"sources must be a non empty subset of {SOURCES}, got {sources}")
        self.sources = [s for s in SOURCES if s in sources]
        self.cfg = cfg
        self.repo_root = repo_root
        self.roots = {s: source_root(s, cfg["paths"], repo_root) for s in self.sources}

    # runs ------------------------------------------------------------------------

    def _locate(self, run_id: str) -> tuple[str, RunPaths]:
        if not isinstance(run_id, str) or not RUN_ID_RE.fullmatch(run_id):
            raise AvError("AV-APP-003", run_id=str(run_id))
        for s, root in self.roots.items():
            folder = root / "runs" / run_id
            if (folder / "manifest.json").is_file():
                return s, RunPaths(run_id, folder)
        raise AvError("AV-APP-003", run_id=run_id)

    def paths(self, run_id: str) -> RunPaths:
        return self._locate(run_id)[1]

    def list_runs(self) -> list[dict]:
        """Every run of the allowed sources, newest first."""
        out = []
        for s, root in self.roots.items():
            runs = root / "runs"
            if not runs.is_dir():
                continue
            for d in runs.iterdir():
                if not RUN_ID_RE.fullmatch(d.name) or not (d / "manifest.json").is_file():
                    continue
                try:
                    m = json.loads((d / "manifest.json").read_text())
                except (OSError, ValueError):
                    continue
                ing = m.get("ingest", {})
                out.append({"run_id": d.name, "source": s, "video": Path(m.get("video", "")).name,
                            "status": m.get("status", "?"), "created_utc": m.get("created_utc", ""),
                            "finished_utc": m.get("finished_utc"), "variant": m.get("variant"),
                            "video_s": (ing.get("cfr") or {}).get("duration_s")
                            or (ing.get("input") or {}).get("duration_s"), "video_path": m.get("video", ""),
                            "has_steps": (d / "steps.csv").is_file()})
        return sorted(out, key=lambda r: r["run_id"], reverse=True)

    def get_run(self, run_id: str) -> dict:
        """The run manifest."""
        p = self.paths(run_id)
        try:
            return json.loads(p.manifest.read_text())
        except (OSError, ValueError) as e:
            raise AvError("AV-STO-003", file="manifest.json", detail=str(e)) from e

    def get_shots(self, run_id: str) -> list[dict]:
        return self.get_run(run_id).get("shots", {}).get("items", [])

    def get_segments(self, run_id: str, engine: str = "A") -> list[dict]:
        """Steps of steps.csv (engine A), with their duration; [] if not stored."""
        p = self.paths(run_id).steps_csv
        if not p.is_file():
            return []
        from ..evaluation.steps_io import read_steps_csv
        out, k = [], 0
        for i, s in enumerate(read_steps_csv(p)):
            if s.label != "other":
                k += 1
            # seg: index of the step in the run counting other too, as segment_id and
            # the step column of metrics.csv; step: number among the real steps (steps.csv)
            out.append({"seg": i + 1, "step": k if s.label != "other" else 0, "label": s.label,
                        "start_s": s.start_s, "end_s": s.end_s, "duration_s": round(s.end_s - s.start_s, 3),
                        "source": s.source, "score": s.score})
        return out

    def get_metrics(self, run_id: str, engine: str = "A") -> list[dict]:
        """Rows of metrics.csv; value and coverage as float or None."""
        p = self.paths(run_id).metrics_csv
        if not p.is_file():
            return []
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for k in ("value", "coverage", "start_s", "end_s"):
                r[k] = float(r[k]) if r.get(k) not in (None, "") else None
            r["step"] = int(r["step"])
        return rows

    def get_baseline(self, run_id: str) -> list[dict]:
        return self.get_run(run_id).get("metrics", {}).get("baseline", [])

    def get_eval(self, run_id: str) -> dict | None:
        p = self.paths(run_id).eval_json
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            return None

    def overlay_path(self, run_id: str) -> Path | None:
        p = self.paths(run_id).tracks_overlay
        return p if p.is_file() else None

    def reference_items(self, run_id: str) -> list[dict]:
        """Per shot with a reference frame: PNG path, numbered boxes, the subject
        track, how it was chosen and its box at the reference frame (or None)."""
        m = self.get_run(run_id)
        p = self.paths(run_id)
        tracks = {t["track_id"]: t for t in m.get("tracking", {}).get("tracks", [])}
        per_shot = {x["shot_id"]: x for x in m.get("tracking", {}).get("per_shot", [])}
        out = []
        for shot in m.get("shots", {}).get("items", []):
            sid = shot["shot_id"]
            png, js = p.ref_png(sid), p.ref_json(sid)
            if not png.is_file():
                continue
            ref = json.loads(js.read_text()) if js.is_file() else {}
            ps = per_shot.get(sid, {})
            subject = ps.get("subject")
            box = box_frame = None
            if subject and subject in tracks and ref.get("frame_idx") is not None:
                box, box_frame = self._box_at(p.root / tracks[subject]["kts_path"], int(ref["frame_idx"]))
            out.append({"shot_id": sid, "shot_type": shot.get("shot_type"), "png": png,
                        "boxes": ref.get("boxes", []), "frame_idx": ref.get("frame_idx"),
                        "subject": subject, "subject_source": ps.get("subject_source"), "subject_box": box,
                        "box_frame": box_frame})
        return out

    @staticmethod
    def _box_at(npz: Path, frame: int) -> tuple[list[float] | None, int | None]:
        """(box, frame) of a KTS track at one frame, or at its first present frame
        when it is not seen there; (None, None) if never present."""
        try:
            with np.load(npz) as z:
                fi, present, bb = z["frame_idx"], z["present"].astype(bool), z["bbox_xyxy"]
        except (OSError, KeyError, ValueError):
            return None, None
        hit = np.flatnonzero((fi == frame) & present)
        if not len(hit):
            hit = np.flatnonzero(present)
        return ([float(v) for v in bb[hit[0]]], int(fi[hit[0]])) if len(hit) else (None, None)

    # status ----------------------------------------------------------------------

    def status(self, run_id: str) -> RunStatus:
        """RunStatus from the manifest only. A queued or running run whose worker
        is gone is stored as failed with AV-RUN-004."""
        source, p = self._locate(run_id)
        m = self.get_run(run_id)
        st = m.get("status", "queued")
        pid = self._worker_pid(p, m)
        alive = worker_alive(pid, run_id)
        if st not in FINISHED and pid and not alive:
            m = self._mark_lost(p, m)
            st = m["status"]
        stages = m.get("stages", {})
        rows, done_w, running = [], 0.0, None
        for name, w in STAGE_WEIGHTS.items():
            rec = stages.get(name)
            if rec is None:
                continue
            s = rec.get("status", "pending")
            frac = 1.0 if s in ("done", "skipped", "failed") else float(rec.get("progress", 0.0)) \
                if s == "running" else 0.0
            if s == "running":
                running = name
            done_w += w * frac
            rows.append((name, s, rec.get("seconds"), frac))
        total_w = sum(STAGE_WEIGHTS.values())     # all stages, so a stage that appears later cannot lower it
        progress = 1.0 if st in FINISHED else min(done_w / total_w, 0.99)
        err = m.get("error") or next((r.get("error") for r in stages.values() if r.get("status") == "failed"), None)
        summary = m.get("summary", "")
        if m.get("reruns") and m["reruns"][-1].get("summary"):
            summary = (summary + "\n" if summary else "") + m["reruns"][-1]["summary"]
        return RunStatus(run_id, source, st, running, round(progress, 4), rows, summary,
                         f"{err['code']}: {err['text']}" if err else "", m.get("waiting", ""),
                         alive and st not in FINISHED)

    @staticmethod
    def _worker_pid(p: RunPaths, m: dict) -> int | None:
        """PID of the worker: log/worker.pid written by RunManager, else the PID
        run_pose stored itself (runs started from the command line)."""
        try:
            return int(p.worker_pid.read_text().strip())
        except (OSError, ValueError):
            return (m.get("worker") or {}).get("pid")

    def _mark_lost(self, p: RunPaths, m: dict) -> dict:
        err = AvError("AV-RUN-004")
        m["status"] = "failed"
        m.setdefault("error", err.as_record())
        m["finished_utc"] = utc_now()
        for rec in m.get("stages", {}).values():
            if rec.get("status") == "running":
                rec["status"] = "failed"
                rec["error"] = err.as_record()
        m["updated_utc"] = utc_now()
        write_atomic(p.manifest, json.dumps(m, indent=2, ensure_ascii=False).encode("utf-8"), p)
        return m

    def active_run(self) -> str | None:
        """The newest run with a live worker, if any."""
        for r in self.list_runs():
            if r["status"] not in FINISHED:
                try:
                    if self.status(r["run_id"]).active:
                        return r["run_id"]
                except AvError:
                    continue
        return None

    # videos ----------------------------------------------------------------------

    def list_videos(self) -> list[tuple[str, str]]:
        """(label 'source/folder/name', absolute path) of every mp4 and avi."""
        out = []
        for s, root in self.roots.items():
            for d in VIDEO_DIRS:
                folder = root / d
                if not folder.is_dir():
                    continue
                for f in sorted(folder.iterdir()):
                    if f.is_file() and f.suffix.lower() in VIDEO_EXT:
                        out.append((f"{s}/{d}/{f.name}", str(f)))
        return out

    def save_upload(self, tmp_file: str | Path, name: str, source: str) -> Path:
        """Move an uploaded file to data/<source>/input/ (AV-APP-001 if not mp4 or
        avi; a name that exists gets _1, _2, ...)."""
        if source not in self.roots:
            raise AvError("AV-RUN-001", name=name, detail=f"source {source} is not open in this interface")
        safe = _safe_name(name)
        if Path(safe).suffix not in VIDEO_EXT:
            raise AvError("AV-APP-001")
        size_mb = Path(tmp_file).stat().st_size / 2 ** 20
        if size_mb > self.cfg["ui"]["max_upload_mb"]:
            raise AvError("AV-APP-001", detail=f"{size_mb:.0f} MB is more than ui.max_upload_mb")
        folder = self.roots[source] / "input"
        folder.mkdir(parents=True, exist_ok=True)
        stem, ext = os.path.splitext(safe)
        dest, k = folder / safe, 0
        while dest.exists():
            k += 1
            dest = folder / f"{stem}_{k}{ext}"
        shutil.move(str(tmp_file), dest)
        return dest
