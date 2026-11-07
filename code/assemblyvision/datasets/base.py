"""One loader interface for the public datasets used to pretrain Engine B (Week 10).

A Sample holds, per frame, COCO 17 keypoints (from our pose model, or the
dataset's own skeleton mapped to COCO 17), the action label as an (action,
repetition) pair and the boundary target of the taxonomy free boundary scorer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import Config
from ..errors import AvError
from ..storage.files import REPO_ROOT, data_area

KEYPOINT_SOURCES = ("ours", "dataset")


@dataclass
class VideoRef:
    video_id: str
    subject: str
    location: str
    fps: float
    n_frames: int | None = None


@dataclass
class Sample:
    video: VideoRef
    source: str                 # "ours" (YOLO26 x KTS) or "dataset" (own skeleton mapped to COCO 17)
    keypoints: np.ndarray       # float32 [T,17,2] image pixels, NaN = missing
    conf: np.ndarray            # float32 [T,17]
    present: np.ndarray         # bool [T]
    labels: np.ndarray          # int32 [T,2] (action, repetition); action -1 = unlabelled
    boundary: np.ndarray        # bool [T]
    meta: dict = field(default_factory=dict)

    @property
    def fps(self) -> float:
        return self.video.fps


def change_frames(labels: np.ndarray, ignore_dont_care: bool = False, dont_care: int = 0) -> np.ndarray:
    """Frames t where the (action, repetition) pair differs from frame t - 1.

    ignore_dont_care merges dont_care frames into the action before them (the
    first action after them at the start), so only changes between real actions
    count (boundaries.ignore_dont_care, an ablation)."""
    lab = np.asarray(labels).reshape(len(labels), -1).copy()
    if ignore_dont_care and len(lab):
        dc = lab[:, 0] == dont_care
        if dc.all():
            return np.zeros(0, np.int64)
        real = np.flatnonzero(~dc)
        prev = np.maximum.accumulate(np.where(~dc, np.arange(len(lab)), -1))
        src = np.where(prev >= 0, prev, real[0])
        lab = lab[src]
    diff = np.any(lab[1:] != lab[:-1], axis=1)
    return np.flatnonzero(diff) + 1


def boundary_targets(labels: np.ndarray, tol_frames: int, ignore_dont_care: bool = False,
                     dont_care: int = 0) -> np.ndarray:
    """bool [T]: True within tol_frames of a label change (the change at frame t is
    between t - 1 and t, so frames t - 1 - tol .. t + tol are positive)."""
    T = len(labels)
    out = np.zeros(T, bool)
    for c in change_frames(labels, ignore_dont_care, dont_care):
        out[max(0, c - 1 - tol_frames):min(T, c + tol_frames + 1)] = True
    return out


def tol_frames(cfg: Config, fps: float) -> int:
    return int(round(cfg["boundaries"]["tol_s"] * fps))


class DatasetLoader:
    """Base class: videos() lists the videos, load() returns one Sample."""

    name = "base"

    def __init__(self, cfg: Config | None = None, root: Path | None = None, repo_root: Path = REPO_ROOT):
        self.cfg = cfg or Config.load()
        self.repo_root = repo_root
        self.root = Path(root) if root else repo_root / self.cfg["datasets"]["root"] / self.name
        if data_area(self.root, self.cfg["paths"], repo_root) != "public":
            raise AvError("AV-STO-005", path=str(self.root), detail="datasets live under data/public only")

    def videos(self) -> list[VideoRef]:
        raise NotImplementedError

    def load(self, video_id: str, keypoints: str = "ours") -> Sample:
        raise NotImplementedError

    def _check_source(self, keypoints: str) -> None:
        if keypoints not in KEYPOINT_SOURCES:
            raise ValueError(f"keypoints must be one of {KEYPOINT_SOURCES}")

    def boundaries(self, labels: np.ndarray, fps: float) -> np.ndarray:
        b = self.cfg["boundaries"]
        return boundary_targets(labels, tol_frames(self.cfg, fps), b["ignore_dont_care"], b["dont_care_label"])
