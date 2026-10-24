"""Shared data types (LLD 3.0.3)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_s: float


@dataclass(frozen=True)
class SplitSpec:
    duration_s: float
    train_end_s: float   # 0.60 of the duration
    val_end_s: float     # 0.70 of the duration


@dataclass
class Shot:
    shot_id: str
    run_id: str
    start_frame: int
    end_frame: int       # exclusive
    start_s: float
    end_s: float
    shot_type: str       # 'fixed' or 'handheld'
    scale_m_per_px: float | None = None   # None = no measured floor scale


@dataclass
class KeypointSeries:
    """Content of one KTS v1 file (LLD 3.0.4)."""
    run_id: str
    shot_id: str
    track_id: str
    fps: float
    width: int
    height: int
    frame_idx: np.ndarray   # int32   [T]
    t_sec: np.ndarray       # float32 [T]
    present: np.ndarray     # bool    [T]
    bbox_xyxy: np.ndarray   # float32 [T,4]
    body_xy: np.ndarray     # float32 [T,17,2] NaN = missing (COCO 17 order)
    body_conf: np.ndarray   # float32 [T,17]
    hand_xy: np.ndarray     # float32 [T,42,2] NaN until the pose_hands stage ran
    hand_conf: np.ndarray   # float32 [T,42]


@dataclass
class Step:
    label: str
    start_s: float
    end_s: float
    source: str
    score: float | None = None


@dataclass
class MetricRecord:
    segment_id: str
    metric_name: str
    value: float | None
    unit: str
    coverage: float
    flags: str = ""
