"""Stratified frame sampling for pose labelling, and the crops of the keypoint pass.

Strata per frame, from the run's DET v1 detections (det_conf >= det_conf_min):
  occluded    two boxes overlap (IoU > occl_iou), or the largest person has more
              than occl_joints upper body joints (face, shoulders, elbows, wrists)
              below joint_conf_low (legs out of the picture do not count)
  low_conf    a detection with det_conf < low_det
  none        no detection (to find people the model misses)
  normal      everything else
(On public runs the stricter "any person, any joint" rules put almost every
frame into occluded, since background people have weak joints.)
Frames are spread over shots (fixed and handheld) and strata with a seeded
draw, at least min_gap_s apart within a shot. A small overlap set is labelled
by two labellers (groups A and B both get it) for inter labeller agreement.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..errors import AvError
from ..ingest.ingestor import iter_frames
from ..perception.pose import load_det
from ..storage.files import utc_now

STRATA = ("occluded", "low_conf", "none", "normal")
WEIGHTS = {"occluded": 0.3, "low_conf": 0.3, "none": 0.1, "normal": 0.3}


@dataclass
class Thresholds:
    det_conf_min: float = 0.25    # detections below this are ignored (tracker high band)
    low_det: float = 0.5
    joint_conf_low: float = 0.3
    occl_joints: int = 2
    occl_iou: float = 0.3


UPPER = slice(0, 11)          # COCO 17: face, shoulders, elbows, wrists


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def frame_stratum(boxes: np.ndarray, det_conf: np.ndarray, body_conf: np.ndarray, th: Thresholds) -> str:
    keep = det_conf >= th.det_conf_min
    boxes, det_conf, body_conf = boxes[keep], det_conf[keep], body_conf[keep]
    if len(boxes) == 0:
        return "none"
    main = int(np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
    if (body_conf[main, UPPER] < th.joint_conf_low).sum() > th.occl_joints or any(
            box_iou(boxes[i], boxes[j]) > th.occl_iou for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
        return "occluded"
    if det_conf.min() < th.low_det:
        return "low_conf"
    return "normal"


def frame_table(manifest: dict, run_root: Path, th: Thresholds) -> list[dict]:
    """One record per frame of every shot: frame, shot, shot type, stratum, people."""
    items = {s["shot_id"]: s for s in manifest["shots"]["items"]}
    out = []
    for ps in manifest["pose_body"]["per_shot"]:
        shot = items[ps["shot_id"]]
        d = load_det(run_root / ps["det_path"])
        fi = d["frame_idx"]
        for f in range(shot["start_frame"], shot["end_frame"]):
            m = fi == f
            out.append({"frame_idx": f, "shot_id": shot["shot_id"], "shot_type": shot["shot_type"],
                        "stratum": frame_stratum(d["bbox_xyxy"][m], d["det_conf"][m], d["body_conf"][m], th),
                        "people": int((d["det_conf"][m] >= th.det_conf_min).sum())})
    return out


def _split(total: int, caps: dict, weights: dict) -> dict:
    """Largest remainder split of total over keys by weight, never above a key's cap;
    what a capped key cannot take goes to the others."""
    out = {k: 0 for k in caps}
    left = min(total, sum(caps.values()))
    while left > 0:
        open_ = [k for k in caps if out[k] < caps[k]]
        w = {k: weights[k] for k in open_} if any(weights[k] > 0 for k in open_) else {k: 1.0 for k in open_}
        share = {k: left * w[k] / sum(w.values()) for k in open_}
        add = {k: min(int(share[k]), caps[k] - out[k]) for k in open_}
        if not any(add.values()):
            for k in sorted(open_, key=lambda k: (share[k] - int(share[k]), w[k]), reverse=True)[:left]:
                add[k] = 1
        for k, a in add.items():
            out[k] += a
            left -= a
    return out


def _allocate(counts: dict, n: int, gap_frames: int) -> dict:
    """Picks per (shot, shot type, stratum) cell: first per stratum by WEIGHTS (capped by
    what the min gap leaves), then per stratum over its cells by sqrt(count), at least
    one per shot type where the quota allows."""
    cap = {c: max(1, k // gap_frames) if k else 0 for c, k in counts.items()}
    strata = sorted({c[2] for c in counts})
    s_cap = {st: sum(v for c, v in cap.items() if c[2] == st) for st in strata}
    quota = _split(n, s_cap, {st: WEIGHTS[st] for st in strata})
    alloc = {c: 0 for c in counts}
    for st in strata:
        cells = [c for c in counts if c[2] == st and cap[c] > 0]
        q = quota[st]
        for t in sorted({c[1] for c in cells}):              # one per shot type first
            if q <= 0:
                break
            best = max((c for c in cells if c[1] == t), key=lambda c: counts[c])
            alloc[best] += 1
            q -= 1
        rest = _split(q, {c: cap[c] - alloc[c] for c in cells}, {c: math.sqrt(counts[c]) for c in cells})
        for c in cells:
            alloc[c] += rest[c]
    return alloc


def sample(table: list[dict], n: int, fps: float, seed: int, min_gap_s: float = 1.0,
           overlap: int = 20) -> list[dict]:
    """Pick n frames, stratified over (shot, shot type, stratum) cells; overlap of them
    are marked for both labelling groups."""
    rng = random.Random(seed)
    cells: dict[tuple, list[dict]] = {}
    for r in table:
        cells.setdefault((r["shot_id"], r["shot_type"], r["stratum"]), []).append(r)
    gap = max(1, int(round(min_gap_s * fps)))
    alloc = _allocate({c: len(v) for c, v in cells.items()}, n, gap)
    chosen: list[dict] = []
    taken: dict[str, list[int]] = {}
    pools = {c: rng.sample(v, len(v)) for c, v in sorted(cells.items())}

    def free(r: dict) -> bool:
        return all(abs(r["frame_idx"] - f) >= gap for f in taken.get(r["shot_id"], []))

    def take(r: dict) -> None:
        chosen.append(dict(r))
        taken.setdefault(r["shot_id"], []).append(r["frame_idx"])
    for c in sorted(cells):
        k = 0
        for r in pools[c]:
            if k >= alloc[c]:
                break
            if free(r):
                take(r)
                k += 1
    if len(chosen) < n:                       # gap conflicts between cells of a shot: refill,
        quota = {st: sum(v for c, v in alloc.items() if c[2] == st) for st in STRATA}   # short strata first
        got = {st: sum(1 for r in chosen if r["stratum"] == st) for st in STRATA}
        ids = {r["frame_idx"] for r in chosen}
        rest = [r for c in sorted(cells) for r in pools[c] if r["frame_idx"] not in ids]
        rest.sort(key=lambda r: got[r["stratum"]] - quota[r["stratum"]])
        for r in rest:
            if len(chosen) >= n:
                break
            if free(r):
                take(r)
                got[r["stratum"]] += 1
    chosen.sort(key=lambda r: r["frame_idx"])
    cands = [r for r in chosen if r["stratum"] != "none"]      # frames with people to agree on
    by = {st: [r for r in cands if r["stratum"] == st] for st in STRATA}
    for v in by.values():
        rng.shuffle(v)
    ov: list[dict] = []
    while len(ov) < min(overlap, len(cands)):                 # round robin over the strata
        for st in STRATA:
            if by[st] and len(ov) < overlap:
                ov.append(by[st].pop())
    ov_ids = {r["frame_idx"] for r in ov}
    rest = [r for r in chosen if r["frame_idx"] not in ov_ids]
    for r in chosen:
        r["overlap"] = r["frame_idx"] in ov_ids
        r["group"] = "AB" if r["overlap"] else None
    for i, r in enumerate(rest):
        r["group"] = "A" if i % 2 == 0 else "B"
    return chosen


def frame_name(frame_idx: int) -> str:
    return f"f{frame_idx:07d}.png"


def write_frames(video: Path, picks: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    want = {r["frame_idx"] for r in picks}
    got = set()
    for idx, img in iter_frames(video, want=want.__contains__):
        tmp = out_dir / (frame_name(idx) + ".tmp.png")
        if not cv2.imwrite(str(tmp), img):
            raise AvError("AV-STO-004", what=frame_name(idx), detail="PNG could not be written")
        tmp.replace(out_dir / frame_name(idx))
        got.add(idx)
    if got != want:
        raise AvError("AV-STO-004", what="sampled frames", detail=f"{len(want - got)} frames could not be read")


def box_tasks(picks: list[dict], width: int, height: int, image_base: str) -> list[dict]:
    return [{"data": {"image": f"{image_base.rstrip('/')}/{frame_name(r['frame_idx'])}",
                      "frame_name": frame_name(r["frame_idx"]), "frame_idx": r["frame_idx"], "shot_id": r["shot_id"],
                      "shot_type": r["shot_type"], "stratum": r["stratum"], "group": r["group"],
                      "overlap": r["overlap"], "width": width, "height": height}} for r in picks]


def crop_window(box: list[float], width: int, height: int, margin: float = 0.25,
                out_size: int = 640) -> tuple[int, int, int, int, float]:
    """(x0, y0, x1, y1, scale) of a square crop around a box with margin, inside the frame;
    scale maps frame pixels to crop pixels."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    side = min(max(side, 32.0), float(max(width, height)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    x0 = int(round(min(max(cx - side / 2, 0), max(width - side, 0))))
    y0 = int(round(min(max(cy - side / 2, 0), max(height - side, 0))))
    xe, ye = int(round(min(x0 + side, width))), int(round(min(y0 + side, height)))
    return x0, y0, xe, ye, out_size / max(xe - x0, ye - y0)


def crop_tasks(box_records: list[dict], frames_dir: Path, crops_dir: Path, image_base: str) -> list[dict]:
    """Keypoint pass tasks: one crop per labelled box. For an overlap frame the boxes of the
    first labeller (sorted by name) are used for both, so both label the same crops."""
    crops_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict[str, list[dict]] = {}
    for rec in box_records:
        by_frame.setdefault(rec["frame_name"], []).append(rec)
    tasks = []
    for name in sorted(by_frame):
        rec = sorted(by_frame[name], key=lambda r: r["labeller"])[0]
        img = cv2.imread(str(frames_dir / name))
        if img is None:
            raise AvError("AV-STO-003", file=name, detail="sampled frame missing")
        H, W = img.shape[:2]
        for b, box in enumerate(rec["boxes"]):
            x0, y0, x1, y1, s = crop_window(box, W, H)
            crop = cv2.resize(img[y0:y1, x0:x1], (max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s))),
                              interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
            cname = f"{Path(name).stem}_b{b:02d}.png"
            cv2.imwrite(str(crops_dir / cname), crop)
            tasks.append({"data": {"image": f"{image_base.rstrip('/')}/crops/{cname}", "frame_name": name,
                                   "frame_idx": rec["frame_idx"], "box_id": b, "box": [round(v, 2) for v in box],
                                   "box_labeller": rec["labeller"], "crop_x0": x0, "crop_y0": y0, "crop_scale": s,
                                   "group": rec["group"], "overlap": rec["overlap"]}})
    return tasks


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(path)
    return path


def summary(picks: list[dict], seed: int, th: Thresholds, run_id: str) -> dict:
    count = lambda key: {k: sum(1 for r in picks if r[key] == k) for k in sorted({r[key] for r in picks})}
    return {"run_id": run_id, "seed": seed, "written_utc": utc_now(), "frames": len(picks),
            "overlap_frames": sum(r["overlap"] for r in picks), "by_stratum": count("stratum"),
            "by_shot_type": count("shot_type"), "by_shot": count("shot_id"), "by_group": count("group"),
            "thresholds": th.__dict__, "picks": picks}
