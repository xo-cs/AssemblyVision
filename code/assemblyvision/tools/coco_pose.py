"""COCO pose check of the trunk and wrist keypoints (rule experiments/w12-coco-pose/rule.md). Public data only.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.coco_pose predict --gpu 0     # YOLO26 x on COCO 2017 val (GPU, under .locks/gpu0.lock)
    python -m assemblyvision.tools.coco_pose score               # writes experiments/w12-coco-pose/results.json and table.md

predict stores the boxes and keypoints in data/public/coco/derived/yolo26x_val2017.npz; score reads them with the COCO
annotations and needs no GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ..storage.files import REPO_ROOT, utc_now

COCO = REPO_ROOT / "data" / "public" / "coco"
ANN = COCO / "annotations" / "person_keypoints_val2017.json"
PRED = COCO / "derived" / "yolo26x_val2017.npz"
OUT = REPO_ROOT / "experiments" / "w12-coco-pose"
L_SHO, R_SHO, L_WRI, R_WRI, L_HIP, R_HIP = 5, 6, 9, 10, 11, 12
JOINTS = {"left shoulder": L_SHO, "right shoulder": R_SHO, "left hip": L_HIP, "right hip": R_HIP,
          "left wrist": L_WRI, "right wrist": R_WRI}
TORSO = (L_SHO, R_SHO, L_HIP, R_HIP)
KP_MIN = 0.30                          # clean.body_conf_min
PCK = 0.1


def predict(gpu: int) -> dict:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import cv2
    import torch
    from ultralytics import YOLO
    from ..config import Config
    from .visual_embed import take_lock
    p = Config.load()["pose"]
    torch.backends.cudnn.allow_tf32 = bool(p["tf32"])
    torch.backends.cuda.matmul.allow_tf32 = False
    ann = json.loads(ANN.read_text())
    ids = sorted({a["image_id"] for a in ann["annotations"]})
    files = {im["id"]: im["file_name"] for im in ann["images"]}
    lock = take_lock(gpu)
    try:
        model = YOLO(str(REPO_ROOT / "models" / f"yolo26{p['variant']}-pose.pt"))
        rows, t0 = [], time.monotonic()
        for i in ids:
            img = cv2.imread(str(COCO / "val2017" / files[i]))
            r = model.predict(img, imgsz=p["imgsz"], conf=p["det_conf_min"], quantize=16 if p["half"] else 32,
                              device="cuda:0", verbose=False, save=False)[0]
            if r.keypoints is None or not len(r.boxes):
                continue
            kd = r.keypoints.data.cpu().numpy()
            for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), kd):
                rows.append((i, *b.tolist(), float(c), k))
        peak = torch.cuda.max_memory_allocated() / 1e6
    finally:
        lock.close()
    PRED.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED, image_id=np.array([r[0] for r in rows], np.int64),
                        box=np.array([r[1:5] for r in rows], np.float32), conf=np.array([r[5] for r in rows], np.float32),
                        kp=np.stack([r[6] for r in rows]).astype(np.float32),
                        meta=json.dumps({"model": f"yolo26{p['variant']}-pose", "imgsz": p["imgsz"], "conf": p["det_conf_min"],
                                         "fp32": not p["half"], "tf32": p["tf32"], "images": len(ids), "date_utc": utc_now()}))
    s = time.monotonic() - t0
    return {"images": len(ids), "detections": len(rows), "seconds": round(s, 1), "peak_mb": round(peak)}


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def match(gts: list[dict], boxes: np.ndarray, conf: np.ndarray) -> dict[int, int]:
    """gt index -> prediction index, greedily by prediction confidence, box IoU >= 0.5."""
    out, used = {}, set()
    for j in np.argsort(-conf):
        best, bi = 0.5, None
        for gi, g in enumerate(gts):
            if gi in out:
                continue
            v = iou(g["box"], boxes[j])
            if v >= best:
                best, bi = v, gi
        if bi is not None and j not in used:
            out[bi] = int(j)
            used.add(int(j))
    return out


def trunk(kp: np.ndarray) -> float:
    from ..analysis.motion import trunk_angle
    sho, hip = (kp[L_SHO, :2] + kp[R_SHO, :2]) / 2, (kp[L_HIP, :2] + kp[R_HIP, :2]) / 2
    return float(trunk_angle(sho, hip))


def band(a: float) -> int:
    from ..analysis.motion import band_of
    return int(band_of(np.array([a]), [20.0, 45.0, 60.0])[0])


def score() -> dict:
    ann = json.loads(ANN.read_text())
    z = np.load(PRED)
    by_img: dict[int, list] = {}
    for a in ann["annotations"]:
        k = np.asarray(a["keypoints"], np.float64).reshape(17, 3)
        if a.get("iscrowd") or not all(k[j, 2] == 2 for j in TORSO):
            continue
        x, y, w, h = a["bbox"]
        by_img.setdefault(a["image_id"], []).append({"kp": k, "box": np.array([x, y, x + w, y + h]), "side": max(w, h)})
    hit = {n: [0, 0, 0] for n in JOINTS}            # correct over all, labelled over all, labelled among matched
    hit_m = {n: 0 for n in JOINTS}
    trunks, persons, matched = [], 0, 0
    pid = z["image_id"]
    for img, gts in by_img.items():
        sel = np.nonzero(pid == img)[0]
        m = match(gts, z["box"][sel], z["conf"][sel]) if len(sel) else {}
        for gi, g in enumerate(gts):
            persons += 1
            pk = z["kp"][sel[m[gi]]] if gi in m else None
            matched += pk is not None
            for n, j in JOINTS.items():
                if g["kp"][j, 2] != 2:
                    continue
                hit[n][1] += 1
                if pk is None:
                    continue
                hit[n][2] += 1
                ok = pk[j, 2] >= KP_MIN and np.hypot(*(pk[j, :2] - g["kp"][j, :2])) <= PCK * g["side"]
                hit[n][0] += ok
                hit_m[n] += ok
            ha = trunk(g["kp"])
            pa = trunk(pk) if pk is not None and all(pk[j, 2] >= KP_MIN for j in TORSO) else None
            trunks.append((ha, pa))

    def trunk_stats(rows):
        both = [(h, p) for h, p in rows if p is not None]
        if not both:
            return {"persons": len(rows), "scored": 0}
        d = [abs(h - p) for h, p in both]
        return {"persons": len(rows), "scored": len(both), "coverage": round(len(both) / len(rows), 3),
                "angle_mae_deg": round(float(np.mean(d)), 2), "angle_median_deg": round(float(np.median(d)), 2),
                "exact_band": round(float(np.mean([band(h) == band(p) for h, p in both])), 3),
                "within_one_band": round(float(np.mean([abs(band(h) - band(p)) <= 1 for h, p in both])), 3)}
    from . import freeze
    res = {"created_utc": utc_now(), "freeze": freeze.record(), "prediction": json.loads(str(z["meta"])),
           "persons": persons, "matched": matched, "match_rate": round(matched / persons, 3),
           "pck": {n: {"all": round(v[0] / v[1], 3) if v[1] else None, "matched": round(hit_m[n] / v[2], 3) if v[2] else None,
                       "labelled": v[1]} for n, v in hit.items()},
           "trunk_all": trunk_stats(trunks), "trunk_bent": trunk_stats([t for t in trunks if t[0] > 45.0]),
           "human_band_counts": [sum(band(h) == b for h, _ in trunks) for b in range(4)]}
    wr = [res["pck"][n]["all"] for n in ("left wrist", "right wrist")]
    res["wrist_pck_all"] = round(float(np.mean(wr)), 3)
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    t, tb = res["trunk_all"], res["trunk_bent"]
    L = ["# w12-coco-pose", "", f"COCO 2017 val, YOLO26 x pose with the flagship's settings (rule.md). Freeze "
         f"{res['freeze']['tag']}, method unchanged: {res['freeze']['method_unchanged']}. {persons} persons with all four "
         f"shoulders and hips labelled visible; {matched} matched to a prediction ({100 * res['match_rate']:.1f} %).", "",
         "**COCO is not factory footage:** these numbers say how well the keypoints are found in everyday photos, not on QiSens.", "",
         "| joint | PCK@0.1, all persons | PCK@0.1, matched persons | labelled visible |", "| --- | --- | --- | --- |"]
    L += [f"| {n} | {v['all']} | {v['matched']} | {v['labelled']} |" for n, v in res["pck"].items()]
    L += ["", "| trunk angle | persons | with a system angle | mean abs difference deg | median deg | exact band | within one band |",
          "| --- | --- | --- | --- | --- | --- | --- |",
          f"| all | {t['persons']} | {t['scored']} | {t.get('angle_mae_deg')} | {t.get('angle_median_deg')} | {t.get('exact_band')} | {t.get('within_one_band')} |",
          f"| bent (human angle over 45 deg) | {tb['persons']} | {tb['scored']} | {tb.get('angle_mae_deg')} | {tb.get('angle_median_deg')} | {tb.get('exact_band')} | {tb.get('within_one_band')} |",
          "", f"Human bands B0 to B3: {res['human_band_counts']}. Targets (frozen 2026-11-26): #3 exact band at least 0.80 and "
          f"within one band at least 0.95; #6 wrist PCK@0.1 at least 0.85 (here {res['wrist_pck_all']}, mean of both wrists, all persons)."]
    (OUT / "table.md").write_text("\n".join(L) + "\n")
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.coco_pose")
    p.add_argument("command", choices=("predict", "score"))
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args(argv)
    if a.command == "predict":
        print(json.dumps(predict(a.gpu)))
        return 0
    r = score()
    print((OUT / "table.md").read_text())
    return 0 if r else 1


if __name__ == "__main__":
    sys.exit(main())
