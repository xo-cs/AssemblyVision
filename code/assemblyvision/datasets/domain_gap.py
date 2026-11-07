"""Domain gap check (Week 10): YOLO26 x against HA4M's Kinect 2D joints on the same frames.

Engine B trains on the Kinect joints mapped to COCO 17, but at use time it gets
YOLO26 x keypoints. A short window of Color frames per lab is fetched; YOLO runs on
them, and per frame the YOLO person matched to the Kinect operator is compared
with the Kinect joints:
  found        a YOLO detection overlaps the Kinect person box (IoU >= 0.3)
  PCK@a        |YOLO - Kinect| <= a x longer side of the Kinect person box
  error        pixel distance, mean and median per joint
  residual     YOLO - Kinect in the body frame (torso lengths): mean (systematic
               offset, e.g. the hips) and standard deviation (calibrates the
               engine_b jitter)
over the joints the Kinect measured (confidence > 0), and separately over the
joints the Kinect calls visible (0.67).
"""
from __future__ import annotations

import numpy as np

from ..analysis.reach import body_axes
from .mapping import COCO17
from .engine_b import KINECT_VISIBLE

FOUND_IOU = 0.3


def kinect_box(xy32: np.ndarray, conf32: np.ndarray, margin: float = 0.1) -> np.ndarray | None:
    """Box around every measured Kinect joint (32 joints), widened by margin per side."""
    ok = np.isfinite(xy32).all(axis=1) & (conf32 > 0)
    if ok.sum() < 3:
        return None
    p = xy32[ok]
    x1, y1 = p.min(axis=0)
    x2, y2 = p.max(axis=0)
    w, h = x2 - x1, y2 - y1
    return np.array([x1 - margin * w, y1 - margin * h, x2 + margin * w, y2 + margin * h])


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def match(kbox: np.ndarray, boxes: np.ndarray) -> int | None:
    """Index of the YOLO detection with the highest IoU with the Kinect box (>= FOUND_IOU)."""
    if kbox is None or len(boxes) == 0:
        return None
    ious = [iou(kbox, b) for b in boxes]
    i = int(np.argmax(ious))
    return i if ious[i] >= FOUND_IOU else None


def compare(frames: list[dict], alphas=(0.05, 0.1)) -> dict:
    """frames: dicts with k_xy [17,2], k_conf [17] (Kinect mapped to COCO 17), k_box,
    y_boxes [N,4], y_xy [N,17,2], y_conf [N,17]. Returns the per joint table."""
    n_frames = len(frames)
    with_kinect = [f for f in frames if f["k_box"] is not None]
    found, err, err_vis, res = [], [], [], []
    for f in with_kinect:
        i = match(f["k_box"], np.asarray(f["y_boxes"]).reshape(-1, 4))
        if i is None:
            continue
        found.append(f)
        kxy, kc = np.asarray(f["k_xy"], float), np.asarray(f["k_conf"], float)
        yxy = np.asarray(f["y_xy"], float)[i]
        side = max(f["k_box"][2] - f["k_box"][0], f["k_box"][3] - f["k_box"][1]) / 1.2   # without the margin
        d = np.linalg.norm(yxy - kxy, axis=1)
        m = (kc > 0) & np.isfinite(d)
        err.append(np.where(m, d / side, np.nan))
        err_vis.append(np.where(m & (kc >= KINECT_VISIBLE), d / side, np.nan))
        hip, ex, ey, L = body_axes(kxy[None])
        if np.isfinite(L[0]) and L[0] > 0:
            dv = yxy - kxy
            res.append(np.where(m[:, None], np.stack([(dv @ ex[0]), (dv @ ey[0])], axis=1) / L[0], np.nan))
        f["_px"] = np.where(m, d, np.nan)
    px = np.array([f["_px"] for f in found]) if found else np.zeros((0, 17))
    rel, rel_vis = (np.array(err) if err else np.zeros((0, 17))), (np.array(err_vis) if err_vis else np.zeros((0, 17)))
    resid = np.array(res) if res else np.zeros((0, 17, 2))

    def pck(r, a):
        tot = np.isfinite(r).sum(axis=0)
        hit = (r <= a).sum(axis=0)
        return [round(float(h / t), 4) if t else None for h, t in zip(hit, tot)], \
            (round(float(hit.sum() / tot.sum()), 4) if tot.sum() else None)
    joints = []
    for j, name in enumerate(COCO17):
        col = px[:, j] if len(px) else np.array([])
        row = {"joint": name, "n": int(np.isfinite(col).sum()) if len(col) else 0,
               "mean_px": _r(np.nanmean(col)) if np.isfinite(col).any() else None,
               "median_px": _r(np.nanmedian(col)) if np.isfinite(col).any() else None}
        for a in alphas:
            row[f"pck_{a}"] = pck(rel, a)[0][j]
            row[f"pck_{a}_visible"] = pck(rel_vis, a)[0][j]
        if len(resid) and np.isfinite(resid[:, j, 0]).any():
            row["offset_torso"] = [_r(np.nanmean(resid[:, j, 0]), 4), _r(np.nanmean(resid[:, j, 1]), 4)]
            row["sd_torso"] = [_r(np.nanstd(resid[:, j, 0]), 4), _r(np.nanstd(resid[:, j, 1]), 4)]
        joints.append(row)
    summary = {"frames": n_frames, "frames_with_kinect": len(with_kinect), "frames_found": len(found),
               "found_share": round(len(found) / len(with_kinect), 4) if with_kinect else None,
               "mean_px": _r(np.nanmean(px)) if np.isfinite(px).any() else None,
               "median_px": _r(np.nanmedian(px)) if np.isfinite(px).any() else None}
    for a in alphas:
        summary[f"pck_{a}"] = pck(rel, a)[1]
        summary[f"pck_{a}_visible"] = pck(rel_vis, a)[1]
    if len(resid):
        summary["residual_sd_torso"] = _r(float(np.nanmean(np.nanstd(resid, axis=0))), 4)
    return {"summary": summary, "joints": joints}


def _r(v, nd: int = 2):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)
