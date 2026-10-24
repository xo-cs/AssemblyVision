"""KTS v1 keypoint files (LLD 3.0.4). Stub: filled in with the tracking stage."""
from __future__ import annotations

from pathlib import Path

from .types import KeypointSeries

KTS_VERSION = 1
BODY_LAYOUT = "coco17"
HAND_LAYOUT = "rtmw21x2"   # left 0 to 20, right 21 to 41
N_BODY = 17
N_HAND = 42


def write_kts(series: KeypointSeries, npz_path: Path, meta: dict) -> Path:
    """Write keypoints/<track_id>.npz and its .json sidecar atomically."""
    raise NotImplementedError("KTS writing comes with the tracking stage (LLD 3.5)")


def read_kts(npz_path: Path) -> KeypointSeries:
    """Read and check a KTS v1 file pair (AV-STO-003 on a bad file)."""
    raise NotImplementedError("KTS reading comes with the tracking stage (LLD 3.5)")
