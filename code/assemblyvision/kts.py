"""KTS v1 keypoint files (LLD 3.0.4): keypoints/<track_id>.npz plus a .json sidecar.

Row i is frame frame_start + i. Missing frames hold NaN with present = False;
nothing is interpolated. The sidecar holds the fields of LLD 3.0.4 plus
is_subject and subject_source ('auto', 'click' or null).
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from .errors import AvError
from .storage.files import RunPaths, utc_now, write_atomic
from .types import KeypointSeries

KTS_VERSION = 1
BODY_LAYOUT = "coco17"
HAND_LAYOUT = "rtmw21x2"   # left 0 to 20, right 21 to 41
N_BODY = 17
N_HAND = 42

_ARRAYS = {  # name: (dtype, shape after T)
    "frame_idx": (np.int32, ()),
    "t_sec": (np.float32, ()),
    "present": (np.bool_, ()),
    "bbox_xyxy": (np.float32, (4,)),
    "body_xy": (np.float32, (N_BODY, 2)),
    "body_conf": (np.float32, (N_BODY,)),
    "hand_xy": (np.float32, (N_HAND, 2)),
    "hand_conf": (np.float32, (N_HAND,)),
}


def empty_series(run_id: str, shot_id: str, track_id: str, fps: float, width: int, height: int,
                 frame_start: int, frame_count: int) -> KeypointSeries:
    """A series with T = frame_count rows, all missing."""
    T = frame_count
    idx = np.arange(frame_start, frame_start + T, dtype=np.int32)

    def nan(*shape):
        return np.full((T, *shape), np.nan, np.float32)
    return KeypointSeries(run_id, shot_id, track_id, fps, width, height,
                          frame_idx=idx, t_sec=(idx / fps).astype(np.float32),
                          present=np.zeros(T, bool), bbox_xyxy=nan(4),
                          body_xy=nan(N_BODY, 2), body_conf=nan(N_BODY),
                          hand_xy=nan(N_HAND, 2), hand_conf=nan(N_HAND))


def write_kts(series: KeypointSeries, paths: RunPaths, meta: dict) -> Path:
    """Write keypoints/<track_id>.npz and .json atomically. meta adds sidecar
    fields (pose_model, pose_variant, tracker, seed, is_subject, ...)."""
    T = series.frame_idx.shape[0]
    arrays = {}
    for name, (dtype, shape) in _ARRAYS.items():
        a = np.asarray(getattr(series, name))
        if a.shape != (T, *shape):
            raise AvError("AV-STO-004", what=f"Track {series.track_id}", detail=f"{name} has shape {a.shape}")
        arrays[name] = a.astype(dtype)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    npz = paths.kts(series.track_id)
    sidecar = {
        "kts_version": KTS_VERSION, "run_id": series.run_id, "shot_id": series.shot_id,
        "track_id": series.track_id, "fps": series.fps, "width": series.width, "height": series.height,
        "frame_start": int(series.frame_idx[0]) if T else 0, "frame_count": T,
        "body_layout": BODY_LAYOUT, "hand_layout": HAND_LAYOUT,
        "pose_model": None, "pose_variant": None, "hand_model": None, "tracker": None,
        "hands_done": False, "seed": None, "is_subject": False, "subject_source": None,
        "created_utc": utc_now(),
    }
    sidecar.update(meta)
    write_atomic(npz, buf.getvalue(), paths)
    write_atomic(npz.with_suffix(".json"), json.dumps(sidecar, indent=2).encode("utf-8"), paths)
    return npz


def read_sidecar(npz_path: Path) -> dict:
    try:
        return json.loads(Path(npz_path).with_suffix(".json").read_text())
    except (OSError, ValueError) as e:
        raise AvError("AV-STO-003", file=Path(npz_path).with_suffix(".json").name, detail=str(e)) from e


def read_kts(npz_path: Path) -> tuple[KeypointSeries, dict]:
    """Read and check a KTS v1 file pair (AV-STO-003). Returns (series, sidecar)."""
    name = Path(npz_path).name
    meta = read_sidecar(npz_path)
    if meta.get("kts_version") != KTS_VERSION:
        raise AvError("AV-STO-003", file=name, detail=f"kts_version {meta.get('kts_version')}")
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
    except (OSError, ValueError) as e:
        raise AvError("AV-STO-003", file=name, detail=str(e)) from e
    T = meta["frame_count"]
    for key, (_, shape) in _ARRAYS.items():
        if key not in arrays or arrays[key].shape != (T, *shape):
            raise AvError("AV-STO-003", file=name, detail=f"{key} shape is not {(T, *shape)}")
    series = KeypointSeries(meta["run_id"], meta["shot_id"], meta["track_id"], meta["fps"],
                            meta["width"], meta["height"], **arrays)
    return series, meta
