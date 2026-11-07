"""Pose label evaluation: detection precision and recall, OKS AP (pycocotools and our own
implementation as a cross check), PCK@0.1 per joint, and agreement between labellers.

Conventions (stated in every report):
- OKS uses the COCO keypoint sigmas and the area of the labelled box (w x h), since
  there are no segmentation masks; a person without labelled keypoints is ignored
  in OKS AP (as in COCO) but counts for the detection precision and recall.
- Detection precision and recall: detections with det_conf >= det_conf_min against
  every labelled box, greedy by confidence, box IoU >= 0.5.
- PCK@0.1: a labelled joint (visible or occluded) is correct when the predicted
  joint lies within 0.1 x the longer side of the labelled box; over the people
  matched by box IoU >= 0.5.
"""
from __future__ import annotations

import contextlib
import io

import numpy as np

COCO_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
OKS_THRESHOLDS = np.round(np.arange(0.5, 0.951, 0.05), 2)


def oks(gt_xy: np.ndarray, gt_vis: np.ndarray, area: float, pred_xy: np.ndarray) -> float:
    """COCO object keypoint similarity of one prediction to one labelled person."""
    m = gt_vis > 0
    if not m.any():
        return 0.0
    d2 = ((pred_xy - gt_xy) ** 2).sum(axis=1)
    var = (2 * COCO_SIGMAS) ** 2
    e = d2 / var / (area + np.spacing(1)) / 2
    return float(np.exp(-e[m]).sum() / m.sum())


def oks_unlabelled(box, pred_xy: np.ndarray) -> float:
    """COCO's OKS for a person with no labelled keypoints: distance of each predicted
    joint outside the box widened by its size on every side (pycocotools computeOks)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    lo_x, hi_x, lo_y, hi_y = x1 - w, x1 + 2 * w, y1 - h, y1 + 2 * h
    dx = np.maximum(0, lo_x - pred_xy[:, 0]) + np.maximum(0, pred_xy[:, 0] - hi_x)
    dy = np.maximum(0, lo_y - pred_xy[:, 1]) + np.maximum(0, pred_xy[:, 1] - hi_y)
    e = (dx ** 2 + dy ** 2) / (2 * COCO_SIGMAS) ** 2 / (_box_area(box) + np.spacing(1)) / 2
    return float(np.exp(-e).mean())


def box_iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _box_area(b) -> float:
    return float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]))


def match_boxes(gts: list[dict], preds: list[dict], thr: float = 0.5) -> list[tuple[int, int]]:
    """Greedy by prediction score: (gt index, pred index) pairs with IoU >= thr."""
    used, pairs = set(), []
    for pi in sorted(range(len(preds)), key=lambda i: -preds[i]["score"]):
        best, bi = thr, None
        for gi, g in enumerate(gts):
            if gi in used:
                continue
            iou = box_iou(g["box"], preds[pi]["box"])
            if iou >= best:
                best, bi = iou, gi
        if bi is not None:
            used.add(bi)
            pairs.append((bi, pi))
    return pairs


def detection_pr(frames: dict, det_conf_min: float) -> dict:
    """frames: {frame: {"gt": [...], "pred": [...]}}; precision and recall at box IoU 0.5."""
    tp = n_pred = n_gt = 0
    for f in frames.values():
        preds = [p for p in f["pred"] if p["score"] >= det_conf_min]
        tp += len(match_boxes(f["gt"], preds))
        n_pred += len(preds)
        n_gt += len(f["gt"])
    return {"det_conf_min": det_conf_min, "tp": tp, "detections": n_pred, "labelled_people": n_gt,
            "precision": round(tp / n_pred, 4) if n_pred else None, "recall": round(tp / n_gt, 4) if n_gt else None}


def pck(frames: dict, alpha: float = 0.1) -> dict:
    """PCK@alpha per COCO joint over the people matched by box IoU >= 0.5."""
    hit, tot = np.zeros(17), np.zeros(17)
    for f in frames.values():
        for gi, pi in match_boxes(f["gt"], f["pred"]):
            g, p = f["gt"][gi], f["pred"][pi]
            b = g["box"]
            thr = alpha * max(b[2] - b[0], b[3] - b[1])
            m = np.asarray(g["vis"]) > 0
            d = np.linalg.norm(np.asarray(p["xy"]) - np.asarray(g["xy"]), axis=1)
            hit += (m & (d <= thr))
            tot += m
    per = [round(float(h / t), 4) if t else None for h, t in zip(hit, tot)]
    return {"alpha": alpha, "per_joint": per, "joints_labelled": tot.astype(int).tolist(),
            "mean": round(float(hit.sum() / tot.sum()), 4) if tot.sum() else None}


def oks_ap_own(frames: dict, max_dets: int = 20) -> dict:
    """COCO style keypoint AP: per OKS threshold, detections sorted by score over all
    frames, greedy matching per frame, 101 point interpolated precision; GT people
    with no labelled keypoints are ignored (and detections matched to them)."""
    n_valid = 0
    per_thr = {}
    for t in OKS_THRESHOLDS:
        records = []
        n_valid = 0
        for key, f in frames.items():
            gts = f["gt"]
            valid = [int(np.sum(np.asarray(g["vis"]) > 0) > 0) for g in gts]
            n_valid += sum(valid)
            preds = sorted(f["pred"], key=lambda p: -p["score"])[:max_dets]
            used = set()
            order = sorted(range(len(gts)), key=lambda i: -valid[i])        # valid GTs first (COCO)
            for p in preds:
                best, bi = min(t, 1 - 1e-10), None
                for gi in order:
                    if gi in used:
                        continue
                    if bi is not None and valid[bi] and not valid[gi]:
                        break
                    pxy = np.asarray(p["xy"], float)
                    o = (oks(np.asarray(gts[gi]["xy"]), np.asarray(gts[gi]["vis"]), _box_area(gts[gi]["box"]), pxy)
                         if valid[gi] else oks_unlabelled(gts[gi]["box"], pxy))
                    if o >= best:
                        best, bi = o, gi
                if bi is not None:
                    used.add(bi)
                    records.append((p["score"], 1 if valid[bi] else -1))
                else:
                    records.append((p["score"], 0))
        records = [r for r in records if r[1] != -1]
        records.sort(key=lambda r: -r[0])
        if n_valid == 0:
            per_thr[float(t)] = None
            continue
        tp = np.cumsum([r[1] for r in records])
        fp = np.cumsum([1 - r[1] for r in records])
        rec = tp / n_valid
        prec = tp / np.maximum(tp + fp, np.spacing(1))
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        rs = np.linspace(0, 1, 101)
        idx = np.searchsorted(rec, rs, side="left")
        q = np.array([prec[i] if i < len(prec) else 0.0 for i in idx])
        per_thr[float(t)] = float(q.mean())
    vals = [v for v in per_thr.values() if v is not None]
    return {"AP": round(float(np.mean(vals)), 4) if vals else None, "AP50": _r(per_thr.get(0.5)),
            "AP75": _r(per_thr.get(0.75)), "labelled_people_with_keypoints": n_valid}


def _r(v):
    return round(v, 4) if v is not None else None


def oks_ap_coco(frames: dict) -> dict:
    """The same AP with pycocotools (COCOeval, iouType keypoints, GT area = box area)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ..datasets.mapping import COCO17
    images, anns, dts = [], [], []
    aid = 0
    for img_id, (key, f) in enumerate(sorted(frames.items()), 1):
        images.append({"id": img_id})
        for g in f["gt"]:
            aid += 1
            x1, y1, x2, y2 = g["box"]
            kp = []
            for (x, y), v in zip(g["xy"], g["vis"]):
                kp += [float(x), float(y), int(v)] if v > 0 else [0.0, 0.0, 0]
            anns.append({"id": aid, "image_id": img_id, "category_id": 1, "keypoints": kp,
                         "num_keypoints": int(sum(v > 0 for v in g["vis"])), "area": _box_area(g["box"]),
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": 0})
        for p in f["pred"]:
            kp = []
            for x, y in p["xy"]:
                kp += [float(x), float(y), 1]
            dts.append({"image_id": img_id, "category_id": 1, "keypoints": kp, "score": float(p["score"])})
    if not any(a["num_keypoints"] for a in anns):
        return {"AP": None, "AP50": None, "AP75": None}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"info": {}, "licenses": [], "images": images, "annotations": anns,
                      "categories": [{"id": 1, "name": "person", "keypoints": list(COCO17), "skeleton": []}]}
        gt.createIndex()
        dt = gt.loadRes(dts) if dts else None
        if dt is None:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
        ev = COCOeval(gt, dt, "keypoints")
        ev.params.useSegm = None
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {"AP": _r(max(float(s[0]), 0.0)), "AP50": _r(max(float(s[1]), 0.0)), "AP75": _r(max(float(s[2]), 0.0))}


def agreement(pairs: list[tuple[dict, dict]], alpha: float = 0.1) -> dict:
    """Two labellers on the same crops: mean OKS of B against A, PCK@alpha of B against
    A per joint, and how often both marked a joint and agreed on its visibility."""
    ok, hit, tot, both, vis_same = [], np.zeros(17), np.zeros(17), 0, 0
    for a, b in pairs:
        va, vb = np.asarray(a["vis"]), np.asarray(b["vis"])
        xa, xb = np.asarray(a["xy"], float), np.asarray(b["xy"], float)
        if (va > 0).any():
            ok.append(oks(xa, va, _box_area(a["box"]), xb))
        m = (va > 0) & (vb > 0)
        thr = alpha * max(a["box"][2] - a["box"][0], a["box"][3] - a["box"][1])
        d = np.linalg.norm(xa - xb, axis=1)
        hit += m & (d <= thr)
        tot += m
        both += int(m.sum())
        vis_same += int((m & (va == vb)).sum())
    return {"people": len(pairs), "mean_oks": round(float(np.mean(ok)), 4) if ok else None,
            "pck_per_joint": [round(float(h / t), 4) if t else None for h, t in zip(hit, tot)],
            "pck_mean": round(float(hit.sum() / tot.sum()), 4) if tot.sum() else None,
            "joints_marked_by_both": both, "visibility_agreement": round(vis_same / both, 4) if both else None}


def box_agreement(frames_two: list[tuple[list, list]]) -> dict:
    """Box pass on the overlap frames: people found by both (IoU >= 0.5) and their mean IoU."""
    matched = n_a = n_b = 0
    ious = []
    for a, b in frames_two:
        pairs = match_boxes([{"box": x} for x in a], [{"box": y, "score": 1.0} for y in b])
        matched += len(pairs)
        n_a, n_b = n_a + len(a), n_b + len(b)
        ious += [box_iou(a[i], b[j]) for i, j in pairs]
    return {"frames": len(frames_two), "boxes_a": n_a, "boxes_b": n_b, "matched": matched,
            "mean_iou": round(float(np.mean(ious)), 4) if ious else None}
