"""File Store part of Storage (LLD 3.2), until Repository exists.

Every path of a run is built by RunPaths, and every result file is written
with write_atomic, which refuses paths outside the run folder (AV-STO-005).
check_video_source enforces the data layout of CLAUDE.md: a QiSens video and
everything derived from it stays under data/qisens/, a public one under
data/public/.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Callable

from ..errors import AvError

REPO_ROOT = Path(__file__).resolve().parents[3]            # ~/AssemblyVision
SOURCES = ("qisens", "public")
# LLD source_type values ('open' is the LLD name of public data)
SOURCE_TYPE = {"qisens": "qisens", "public": "open"}


def source_root(source: str, paths_cfg: dict, repo_root: Path = REPO_ROOT) -> Path:
    if source not in SOURCES:
        raise ValueError(f"unknown source {source}")
    return (repo_root / paths_cfg[f"{source}_root"]).resolve()


def check_video_source(video: Path, source: str, paths_cfg: dict,
                       repo_root: Path = REPO_ROOT) -> Path:
    """Return the resolved video path, or refuse with AV-RUN-001.

    Symlinks are resolved first, so a link under data/public/ that points into
    data/qisens/ is refused for source public (and the other way round).
    """
    real = Path(os.path.realpath(video))
    own = source_root(source, paths_cfg, repo_root)
    if not real.is_relative_to(own):
        raise AvError("AV-RUN-001", name=Path(video).name,
                      detail=f"source {source} needs a video under {own}")
    for other in SOURCES:
        if other != source and real.is_relative_to(source_root(other, paths_cfg, repo_root)):
            raise AvError("AV-RUN-001", name=Path(video).name,
                          detail=f"the video lies in the {other} data area")
    return real


def data_area(path: Path, paths_cfg: dict, repo_root: Path = REPO_ROOT) -> str | None:
    """'qisens' or 'public' if the resolved path lies in that data area, else None."""
    real = Path(os.path.realpath(path))
    return next((s for s in SOURCES if real.is_relative_to(source_root(s, paths_cfg, repo_root))), None)


def run_source(run_dir: Path, paths_cfg: dict, repo_root: Path = REPO_ROOT) -> tuple[str, Path]:
    """(source, resolved folder) of an existing run folder data/<source>/runs/<run_id>,
    or AV-STO-005."""
    root = Path(os.path.realpath(run_dir))
    for s in SOURCES:
        if root.parent == source_root(s, paths_cfg, repo_root) / "runs" and (root / "manifest.json").is_file():
            return s, root
    raise AvError("AV-STO-005", path=str(run_dir), detail="not a run folder under data/<source>/runs/")


_issued: set[str] = set()


def new_run_id() -> str:
    """'run_' + UTC YYYYMMDD_HHMMSS + '_' + 4 hex digits (24 characters, LLD 3.1).

    Unique within the process (TC-RUN-02); across processes RunPaths.create
    refuses an existing folder, so the caller draws again.
    """
    while True:
        run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)
        if run_id not in _issued:
            _issued.add(run_id)
            return run_id


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    root: Path   # data/<source>/runs/<run_id>

    @classmethod
    def for_source(cls, source: str, run_id: str, paths_cfg: dict,
                   repo_root: Path = REPO_ROOT) -> "RunPaths":
        return cls(run_id, source_root(source, paths_cfg, repo_root) / "runs" / run_id)

    video_dir = property(lambda self: self.root / "video")
    cfr = property(lambda self: self.root / "video" / "cfr.mp4")
    detections_dir = property(lambda self: self.root / "detections")
    reference_dir = property(lambda self: self.root / "reference")
    keypoints_dir = property(lambda self: self.root / "keypoints")
    overlay_dir = property(lambda self: self.root / "overlay")
    pose_overlay = property(lambda self: self.root / "overlay" / "pose_overlay.mp4")
    tracks_overlay = property(lambda self: self.root / "overlay" / "tracks_overlay.mp4")
    log_dir = property(lambda self: self.root / "log")
    run_log = property(lambda self: self.root / "log" / "run.log")
    failed_frames = property(lambda self: self.root / "failed_frames.csv")
    tmp_dir = property(lambda self: self.root / "tmp")
    manifest = property(lambda self: self.root / "manifest.json")
    run_config = property(lambda self: self.root / "run_config.toml")
    steps_csv = property(lambda self: self.root / "steps.csv")
    eval_json = property(lambda self: self.root / "eval.json")

    def det(self, shot_id: str) -> Path:
        return self.detections_dir / f"{shot_id}.npz"

    def assign(self, shot_id: str) -> Path:
        return self.detections_dir / f"{shot_id}_assign.npz"

    def ref_png(self, shot_id: str) -> Path:
        return self.reference_dir / f"{shot_id}.png"

    def ref_json(self, shot_id: str) -> Path:
        return self.reference_dir / f"{shot_id}.json"

    def kts(self, track_id: str) -> Path:
        return self.keypoints_dir / f"{track_id}.npz"

    def stage_log(self, stage: str) -> Path:
        return self.log_dir / f"{stage}.log"

    def create(self) -> None:
        """Make the run folder; FileExistsError if this run ID is already taken."""
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.root.mkdir()
        for d in (self.video_dir, self.detections_dir, self.reference_dir, self.keypoints_dir,
                  self.overlay_dir, self.log_dir, self.tmp_dir):
            d.mkdir(parents=True, exist_ok=True)

    def check_inside(self, path: Path) -> Path:
        """AV-STO-005 if path is not inside the run folder."""
        p = Path(path)
        if ".." in p.parts or not p.resolve().is_relative_to(self.root.resolve()):
            raise AvError("AV-STO-005", path=str(path))
        return p


def write_atomic(path: Path, data: bytes | Callable[[BinaryIO], None], paths: RunPaths) -> Path:
    """Write to <path>.tmp, fsync, rename (LLD 3.2). data is bytes or a writer(file)."""
    path = paths.check_inside(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            if callable(data):
                data(f)
            else:
                f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise AvError("AV-STO-004", detail=str(e), what=path.name) from e
    return path


@dataclass
class RunContext:
    """What a stage needs to store results without the Repository: paths,
    the manifest and the run logger. Repository takes this role later."""
    run_id: str
    source: str
    paths: RunPaths
    log: logging.Logger
    manifest: dict = field(default_factory=dict)

    def notice(self, err: AvError, stage: str) -> None:
        rec = {"stage": stage, **err.as_record(), "effect": "Notice"}
        self.manifest.setdefault("notices", []).append(rec)
        self.log.warning("%s %s%s", err.code, err.text, f" ({err.detail})" if err.detail else "")

    def write_manifest(self) -> Path:
        self.manifest["updated_utc"] = utc_now()
        text = json.dumps(self.manifest, indent=2, ensure_ascii=False, default=str)
        return write_atomic(self.paths.manifest, text.encode("utf-8"), self.paths)

    def rel(self, path: Path) -> str:
        """Path relative to the run folder, with / separators."""
        return Path(path).relative_to(self.paths.root).as_posix()
