"""Build ground truth from the two Label Studio passes and compare it with a run's DET v1."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..perception.pose import load_det
from . import metrics


def ground_truth(box_records: list[dict], kp_records: list[dict]) -> tuple[dict, dict]:
    """(gt per frame, agreement inputs). GT people are the boxes of the box labeller the
    crops were made from (first by name); keypoints come from the first keypoint
    labeller of each crop. Second labellers of overlap frames go to the agreement."""
    by_frame: dict[str, list[dict]] = {}
    for r in box_records:
        by_frame.setdefault(r["frame_name"], []).append(r)
    kp: dict[tuple, list[dict]] = {}
    for r in kp_records:
        kp.setdefault((r["frame_name"], r["box_id"]), []).append(r)
    gt, kp_pairs, box_pairs = {}, [], []
    for name, recs in sorted(by_frame.items()):
        recs = sorted(recs, key=lambda r: r["labeller"])
        first = recs[0]
        if len(recs) > 1:
            box_pairs.append((first["boxes"], recs[1]["boxes"]))
        people = []
        for b, box in enumerate(first["boxes"]):
            ks = sorted(kp.get((name, b), []), key=lambda r: r["labeller"])
            if ks:
                people.append({"box": box, "xy": ks[0]["xy"], "vis": ks[0]["vis"], "labeller": ks[0]["labeller"]})
                if len(ks) > 1:
                    kp_pairs.append(({**ks[0], "box": box}, {**ks[1], "box": box}))
            else:
                people.append({"box": box, "xy": [[0.0, 0.0]] * 17, "vis": [0] * 17, "labeller": None})
        gt[name] = {"frame_idx": first["frame_idx"], "gt": people}
    return gt, {"keypoints": kp_pairs, "boxes": box_pairs}


def predictions(manifest: dict, run_root: Path, frame_ids: set[int]) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {f: [] for f in frame_ids}
    for ps in manifest["pose_body"]["per_shot"]:
        d = load_det(run_root / ps["det_path"])
        for i in np.flatnonzero(np.isin(d["frame_idx"], list(frame_ids))):
            out[int(d["frame_idx"][i])].append({"box": d["bbox_xyxy"][i].tolist(), "score": float(d["det_conf"][i]),
                                                "xy": d["body_xy"][i].tolist()})
    return out


def evaluate(gt: dict, preds: dict[int, list[dict]], agree: dict, det_conf_min: float = 0.25) -> dict:
    frames = {name: {"gt": g["gt"], "pred": preds.get(g["frame_idx"], [])} for name, g in gt.items()}
    ap_frames = {k: {"gt": f["gt"], "pred": [p for p in f["pred"] if p["score"] >= det_conf_min]}
                 for k, f in frames.items()}
    own, coco = metrics.oks_ap_own(frames), metrics.oks_ap_coco(frames)     # AP over every stored detection
    return {"frames": len(frames), "labelled_people": sum(len(f["gt"]) for f in frames.values()),
            "labellers": sorted({p["labeller"] for f in frames.values() for p in f["gt"] if p["labeller"]}),
            "detection": metrics.detection_pr(frames, det_conf_min),
            "oks_ap": {"pycocotools": coco, "own": own,
                       "same": all(coco[k] is not None and own[k] is not None and abs(coco[k] - own[k]) <= 1e-3
                                   for k in ("AP", "AP50", "AP75"))},
            "pck": metrics.pck({k: {"gt": f["gt"], "pred": f["pred"]} for k, f in ap_frames.items()}),
            "agreement": {"keypoints": metrics.agreement(agree["keypoints"]),
                          "boxes": metrics.box_agreement(agree["boxes"])},
            "conventions": {"oks_area": "labelled box w x h", "sigmas": "COCO keypoints",
                            "oks_ap_detections": "every stored detection (DET v1 keeps det_conf >= pose.det_conf_min)",
                            "det_conf_min": det_conf_min, "pck": "0.1 x longer side of the labelled box",
                            "detection_match": "box IoU >= 0.5, greedy by det_conf"}}


def report_md(res: dict, run_id: str) -> str:
    from ..datasets.mapping import COCO17
    d, ap = res["detection"], res["oks_ap"]
    lines = [f"# Pose labels evaluation, {run_id}", "",
             f"{res['frames']} frames, {res['labelled_people']} labelled people, labellers: "
             f"{', '.join(res['labellers']) or 'none'}.", "",
             f"Detection (det_conf >= {d['det_conf_min']}, box IoU >= 0.5): precision {d['precision']}, "
             f"recall {d['recall']} ({d['tp']} of {d['labelled_people']} people, {d['detections']} detections).", "",
             "| OKS AP | pycocotools | own |", "| --- | --- | --- |"]
    for k in ("AP", "AP50", "AP75"):
        lines.append(f"| {k} | {ap['pycocotools'][k]} | {ap['own'][k]} |")
    lines += ["", f"PCK@0.1 mean {res['pck']['mean']}.", "", "| joint | PCK@0.1 | labelled | labeller agreement PCK |",
              "| --- | --- | --- | --- |"]
    ag = res["agreement"]["keypoints"]
    for j, name in enumerate(COCO17):
        lines.append(f"| {name} | {res['pck']['per_joint'][j]} | {res['pck']['joints_labelled'][j]} | "
                     f"{ag['pck_per_joint'][j]} |")
    b = res["agreement"]["boxes"]
    lines += ["", f"Agreement between labellers: {ag['people']} people labelled twice, mean OKS {ag['mean_oks']}, "
              f"visibility agreement {ag['visibility_agreement']}; boxes on {b['frames']} frames, {b['matched']} "
              f"matched, mean IoU {b['mean_iou']}.", "",
              "Conventions: " + "; ".join(f"{k} {v}" for k, v in res["conventions"].items()) + ".", ""]
    return "\n".join(lines)
