"""Engine B training data (Week 10): per frame features and boundary targets.

Source: HA4M's own skeletons (Azure Kinect 2D colour projections mapped to COCO
17); the RGB route was dropped (the server gives about 0.2 MB/s). The domain gap
experiment measures how far YOLO26 x is from these joints on the same frames.

Features per frame (FEATURES, version FEATURE_VERSION):
  body_xy [17,2]  joints in the body frame of the reach heatmap (analysis/reach):
                  mid hip origin, hip to shoulder axis up, torso lengths; NaN -> 0
  conf    [17]    keypoint confidence (0 = missing); YOLO like levels (engine_b
                  conf_visible, conf_predicted), also without augmentation
  hip     [2]     mid hip in image heights (where the body is in the picture)
  scale   [1]     log(torso length / the video's median torso length)
  valid   [1]     1 if the torso was measured (the frame has a body frame)
Augmentation (train only, [engine_b]) works on the image coordinates before
the normalization, so it looks like YOLO26 output: Gaussian joint jitter (sigma
jitter_torso torso lengths, scaled per joint by the COCO sigmas), joint and
frame dropout, and Kinect confidence levels mapped to YOLO like values plus noise.
Hip calibration (engine_b.hip_calibration, train and test data alike): every Kinect
joint is first moved by the mean YOLO26 x minus Kinect offset of that joint over both
labs of the domain gap experiment (body frame, torso lengths), one global offset per
joint (QiSens is neither lab, so no per lab offsets).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..analysis.reach import TORSO, body_axes, to_body
from ..config import Config
from ..storage.files import utc_now
from .base import Sample

FEATURE_VERSION = 1
FEATURES = [f"{n}_{a}" for n in range(17) for a in ("x", "y")] + [f"conf_{n}" for n in range(17)] + \
    ["hip_x", "hip_y", "scale", "valid"]
N_FEATURES = len(FEATURES)          # 55
COCO_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
KINECT_VISIBLE, KINECT_PREDICTED = 0.6, 0.3     # thresholds between the Kinect levels 0.33 and 0.67


def torso_length_px(xy: np.ndarray) -> np.ndarray:
    return body_axes(xy)[3]


def features(xy: np.ndarray, conf: np.ndarray, image_h: float, median_torso: float | None = None) -> np.ndarray:
    """[T,17,2] image joints and [T,17] confidences -> float32 [T, N_FEATURES]."""
    xy = np.asarray(xy, np.float64)
    conf = np.nan_to_num(np.asarray(conf, np.float64))
    T = len(xy)
    measured = np.isfinite(xy).all(axis=2) & (conf > 0)
    hip, ex, ey, L = body_axes(np.where(measured[..., None], xy, np.nan))
    valid = measured[:, list(TORSO)].all(axis=1) & np.isfinite(L) & (L > 0)
    body = np.zeros((T, 17, 2))
    for j in range(17):
        ok = valid & measured[:, j]
        if ok.any():
            body[ok, j] = to_body(xy[ok, j], hip[ok], ex[ok], ey[ok], L[ok])
    med = median_torso if median_torso else (float(np.median(L[valid])) if valid.any() else 1.0)
    out = np.zeros((T, N_FEATURES), np.float32)
    out[:, :34] = body.reshape(T, 34)
    out[:, 34:51] = np.where(measured, conf, 0.0)
    out[valid, 51:53] = hip[valid] / image_h
    with np.errstate(divide="ignore", invalid="ignore"):
        out[valid, 53] = np.log(L[valid] / med)
    out[:, 54] = valid
    return out


@dataclass
class AugmentConfig:
    jitter_torso: float = 0.03
    joint_dropout: float = 0.05
    frame_dropout: float = 0.02
    conf_visible: float = 0.9
    conf_predicted: float = 0.4
    conf_noise: float = 0.1

    @classmethod
    def from_cfg(cls, cfg: Config) -> "AugmentConfig":
        e = cfg["engine_b"]
        return cls(**{k: e[k] for k in cls.__dataclass_fields__})


def yolo_like_conf(kinect_conf: np.ndarray, a: AugmentConfig) -> np.ndarray:
    """Kinect levels (0, 0.33 predicted, 0.67 visible) -> YOLO like confidences, no noise."""
    c = np.asarray(kinect_conf, np.float64)
    return np.where(c >= KINECT_VISIBLE, a.conf_visible, np.where(c >= KINECT_PREDICTED, a.conf_predicted, 0.0))


def augment(xy: np.ndarray, kinect_conf: np.ndarray, a: AugmentConfig,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Image joints [T,17,2] and Kinect confidences [T,17] -> augmented joints and YOLO like
    confidences: jitter, joint dropout, frame dropout, confidence noise."""
    xy = np.array(xy, np.float64)
    T = len(xy)
    conf = yolo_like_conf(kinect_conf, a)
    L = torso_length_px(np.where((conf > 0)[..., None], xy, np.nan))
    Lf = np.where(np.isfinite(L) & (L > 0), L, np.nanmedian(L) if np.isfinite(L).any() else 0.0)
    scale = COCO_SIGMAS / COCO_SIGMAS.mean()
    xy += rng.normal(0.0, 1.0, (T, 17, 2)) * (a.jitter_torso * Lf[:, None, None] * scale[None, :, None])
    conf = np.clip(conf + rng.normal(0.0, a.conf_noise, conf.shape) * (conf > 0), 0.0, 1.0)
    drop = rng.random((T, 17)) < a.joint_dropout
    drop |= (rng.random(T) < a.frame_dropout)[:, None]
    xy[drop] = np.nan
    conf[drop] = 0.0
    return xy, conf


def calibration_offsets(domain_gap_results: dict) -> np.ndarray:
    """[17,2] mean YOLO26 x minus Kinect residual per joint (body frame, torso lengths), the
    plain mean over the videos (labs) of a domain gap experiment; NaN -> 0."""
    per = [[j.get("offset_torso") or [np.nan, np.nan] for j in v["joints"]]
           for v in domain_gap_results["videos"].values()]
    arr = np.array(per, np.float64)
    n = np.isfinite(arr).sum(axis=0)
    return np.where(n > 0, np.nansum(arr, axis=0) / np.maximum(n, 1), 0.0)


def load_calibration(repo_root: Path, exp_id: str) -> np.ndarray:
    path = Path(repo_root) / "experiments" / exp_id / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"engine_b.hip_calibration needs {path}")
    return calibration_offsets(json.loads(path.read_text()))


def calibrate(xy: np.ndarray, kinect_conf: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Kinect image joints [T,17,2] moved by the per joint offsets [17,2] (body frame of
    the Kinect pose, torso lengths), so they sit where YOLO26 x puts them on average.
    Frames without a torso stay as they are."""
    xy = np.array(xy, np.float64)
    m = (np.asarray(kinect_conf) > 0)[..., None] & np.isfinite(xy)
    _, ex, ey, L = body_axes(np.where(m, xy, np.nan))
    ok = np.isfinite(L) & (L > 0) & np.isfinite(ex).all(1) & np.isfinite(ey).all(1)
    d = (offsets[None, :, 0:1] * ex[:, None, :] + offsets[None, :, 1:2] * ey[:, None, :]) * L[:, None, None]
    xy[ok] += d[ok]
    return xy


def cache_dir(dataset_root: Path) -> Path:
    return dataset_root / "engine_b" / "kinect_coco17"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def build_video(sample: Sample, skeleton_sha: str, out: Path, image_h: float, a: AugmentConfig) -> dict:
    """One video's cache file: image joints and Kinect confidences (for augmentation at
    training time), clean features, labels, boundary targets and the label mask."""
    xy, conf = sample.keypoints, sample.conf
    L = torso_length_px(np.where((conf > 0)[..., None], xy, np.nan))
    med = float(np.nanmedian(L)) if np.isfinite(L).any() else 1.0
    feats = features(xy, yolo_like_conf(conf, a), image_h, med)
    labelled = sample.labels[:, 0] >= 0
    rec = {"video": sample.video.video_id, "subject": sample.video.subject, "location": sample.video.location,
           "fps": sample.fps, "frames": int(len(xy)), "frames_valid": int(feats[:, 54].sum()),
           "frames_labelled": int(labelled.sum()), "boundary_frames": int(sample.boundary.sum()),
           "median_torso_px": round(med, 2), "skeleton_sha256": skeleton_sha, "feature_version": FEATURE_VERSION}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, xy=xy.astype(np.float32), kinect_conf=conf.astype(np.float32), features=feats,
                        labels=sample.labels.astype(np.int32), boundary=sample.boundary, labelled=labelled,
                        median_torso_px=np.float32(med), meta=np.str_(json.dumps(rec)))
    os.replace(tmp, out)
    return rec


def build(loader, cfg: Config, image_h: float, log=print) -> dict:
    """Build (or extend) the cache for every video with fetched annotations. A video is
    rebuilt only if its skeleton file, the feature version or the boundary settings
    changed. Writes index.json; returns it."""
    root = cache_dir(loader.root)
    index_path = root / "index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {"videos": {}}
    settings = {"feature_version": FEATURE_VERSION, "boundaries": dict(cfg["boundaries"]),
                "conf_levels": [cfg["engine_b"]["conf_visible"], cfg["engine_b"]["conf_predicted"]],
                "source": "ha4m kinect 2D colour joints -> coco17"}
    if index.get("settings") != settings:
        index = {"videos": {}}
    built = 0
    for v in loader.videos():
        sk = loader.root / "skeleton" / f"{v.video_id}.npz"
        if not sk.is_file():
            continue
        sha = _sha(sk)
        old = index["videos"].get(v.video_id)
        if old and old.get("skeleton_sha256") == sha and (root / f"{v.video_id}.npz").is_file():
            continue
        index["videos"][v.video_id] = build_video(loader.load(v.video_id, keypoints="dataset"), sha,
                                                  root / f"{v.video_id}.npz", image_h, AugmentConfig.from_cfg(cfg))
        built += 1
    index.update({"settings": settings, "features": FEATURES, "updated_utc": utc_now(),
                  "n_videos": len(index["videos"]),
                  "n_frames": sum(r["frames"] for r in index["videos"].values())})
    root.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_name("index.json.tmp")
    tmp.write_text(json.dumps(index, indent=1) + "\n")
    os.replace(tmp, index_path)
    log(f"engine_b data: {built} videos built, {index['n_videos']} in the cache, {index['n_frames']} frames")
    return index


def load_video(cache: Path, video_id: str, cfg: Config | None = None, train: bool = False,
               rng: np.random.Generator | None = None, image_h: float = 1536.0) -> dict:
    """Features, targets and mask of one cached video; train=True augments (engine_b)."""
    with np.load(cache / f"{video_id}.npz") as z:
        d = {k: z[k] for k in z.files}
    feats = d["features"]
    if train and cfg is not None and cfg["engine_b"]["augment"]:
        rng = rng or np.random.default_rng(cfg["engine_b"]["aug_seed"])
        xy, conf = augment(d["xy"], d["kinect_conf"], AugmentConfig.from_cfg(cfg), rng)
        feats = features(xy, conf, image_h, float(d["median_torso_px"]))
    return {"features": feats, "boundary": d["boundary"], "labels": d["labels"], "mask": d["labelled"],
            "meta": json.loads(str(d["meta"]))}
