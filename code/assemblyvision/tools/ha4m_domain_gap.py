"""Domain gap check: YOLO26 x against HA4M's Kinect 2D joints (Week 10, datasets/domain_gap.py).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.ha4m_domain_gap fetch [--videos IDU001V001,IDU023V001] [--seconds 27] [--step 3]
    python -m assemblyvision.tools.ha4m_domain_gap run --gpu 0          # GPU job, ask first
    python -m assemblyvision.tools.ha4m_domain_gap compare              # CPU; writes the experiment

fetch reads a short window of Color frames per video (default: 27 s at every 3rd
frame from 30 % into the video, 270 frames, about 0.7 GB per video; at most
--max-gb 1.5 in total), one member per range request (paced, HTTP 429 handled,
the dataset download lock is taken, so it never runs beside another download).
The PNGs are kept unchanged under data/public/datasets/ha4m/domain_gap/<video>/.
run applies YOLO26 x with the pipeline's settings (imgsz, det_conf_min, FP32 with
the PyTorch default TF32) under the GPU lock and stores every detection. compare
matches YOLO to the Kinect operator and writes experiments/<id>/ (config.toml with
every setting, results.json with provenance, table.md) and INDEX.md.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..datasets import domain_gap, ha4m, ha4m_fetch as hf, remote
from ..errors import AvError
from ..experiments import registry as reg
from ..storage.files import REPO_ROOT, utc_now
from .fetch_dataset import configure_pacer, dataset_root, setup_log, take_lock

EXP_ID = "w10-ha4m-domain-gap"
PILOT = "IDU001V001,IDU023V001"


def gap_dir(root: Path, video: str) -> Path:
    return root / "domain_gap" / video


def window(color: list, n_frames: int, seconds: float, step: int, start_frac: float, fps: float) -> list:
    """Color members of the window: from start_frac of the video, seconds long, every step-th frame."""
    first = int(start_frac * n_frames)
    last = first + int(round(seconds * fps))
    return [i for i in color if (f := ha4m.frame_id(i.filename)) is not None and first <= f < last
            and (f - first) % step == 0]


def fetch(args, cfg, root: Path, lg) -> int:
    src = hf.load_source(root)
    vids = [v.strip() for v in args.videos.split(",") if v.strip()]
    plan = {}
    for v in vids:
        meta = hf.read_meta(root, v)
        if not meta.get("annotations", {}).get("done"):
            lg.error("%s: fetch its annotations first", v)
            return 2
        zf, raw = remote.open_remote_zip(ha4m.DOWNLOAD_URL.format(file_id=src["files"][v]["file_id"]),
                                         buffer_size=1 << 20)
        color = sorted((i for i in zf.infolist() if i.filename.startswith(f"{v}/Color/") and not i.is_dir()),
                       key=lambda i: ha4m.frame_id(i.filename) or -1)
        plan[v] = (raw, window(color, meta["n_frames"], args.seconds, args.step, args.start_frac, ha4m.FPS), meta)
    total = sum(sum(i.compress_size for i in sel) for _, sel, _ in plan.values())
    lg.info("domain gap fetch: %s, %.2f GB in %d frames", ", ".join(f"{v} {len(s)} frames" for v, (_, s, _) in
                                                                   plan.items()), total / 1e9,
            sum(len(s) for _, s, _ in plan.values()))
    if total > args.max_gb * 1e9:
        lg.error("%.2f GB is above --max-gb %.2f; shorten --seconds or raise --step", total / 1e9, args.max_gb)
        return 2
    t0, got = time.monotonic(), 0
    for v, (raw, sel, meta) in plan.items():
        out = gap_dir(root, v) / "frames"
        out.mkdir(parents=True, exist_ok=True)
        rec = {"video": v, "camera": meta.get("camera"), "seconds": args.seconds, "step": args.step,
               "start_frac": args.start_frac, "frames": {}}
        for n, info in enumerate(sel):
            fid = ha4m.frame_id(info.filename)
            path = out / f"FrameID{fid:06d}.png"
            if not (path.is_file() and path.stat().st_size == info.file_size):
                data = remote.read_member(raw, info)
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_bytes(data)
                os.replace(tmp, path)
                got += len(data)
            rec["frames"][str(fid)] = {"member": info.filename, "crc32": f"{info.CRC:08x}", "bytes": info.file_size}
            if n % 30 == 0:
                lg.info("%s domain gap: frame %d/%d, %.0f MB, %.2f MB/s", v, n + 1, len(sel), got / 1e6,
                        got / 1e6 / max(time.monotonic() - t0, 1e-6))
        rec["done_utc"] = utc_now()
        (gap_dir(root, v) / "frames.json").write_text(json.dumps(rec, indent=1) + "\n")
        lg.info("%s domain gap: %d frames stored", v, len(sel))
    return 0


def run_yolo(args, cfg, root: Path, lg) -> int:
    import torch
    from ultralytics import YOLO
    lock_dir = REPO_ROOT / ".locks"
    lock_dir.mkdir(exist_ok=True)
    lock = open(lock_dir / f"gpu{args.gpu}.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lg.error("GPU %d is in use by another job", args.gpu)
        return 2
    p = cfg["pose"]
    model = YOLO(str(REPO_ROOT / cfg["paths"]["models_dir"] / f"yolo26{p['variant']}-pose.pt"), task="pose")
    dev = f"cuda:{args.gpu}"
    for v in [v.strip() for v in args.videos.split(",") if v.strip()]:
        d = gap_dir(root, v)
        rec = json.loads((d / "frames.json").read_text())
        rows = {"frame_id": [], "box": [], "det_conf": [], "xy": [], "conf": []}
        t0 = time.monotonic()
        fids = sorted(int(f) for f in rec["frames"])
        for fid in fids:
            img = cv2.imread(str(d / "frames" / f"FrameID{fid:06d}.png"))
            r = model.predict(img, imgsz=p["imgsz"], conf=p["det_conf_min"], quantize=16 if p["half"] else 32,
                              device=dev, verbose=False, save=False)[0]
            n = len(r.boxes)
            if n:
                k = r.keypoints.data.cpu().numpy()
                rows["frame_id"] += [fid] * n
                rows["box"].append(r.boxes.xyxy.cpu().numpy())
                rows["det_conf"].append(r.boxes.conf.cpu().numpy())
                rows["xy"].append(k[:, :, :2])
                rows["conf"].append(k[:, :, 2])
        cat = lambda a, s: np.concatenate(a).astype(np.float32) if a else np.zeros((0, *s), np.float32)
        np.savez_compressed(d / "yolo.npz", frame_id=np.array(rows["frame_id"], np.int32), box=cat(rows["box"], (4,)),
                            det_conf=cat(rows["det_conf"], ()), xy=cat(rows["xy"], (17, 2)),
                            conf=cat(rows["conf"], (17,)), frames=np.array(fids, np.int32))
        info = {"variant": p["variant"], "imgsz": p["imgsz"], "conf": p["det_conf_min"], "half": p["half"],
                "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32), "device": torch.cuda.get_device_name(args.gpu),
                "frames": len(fids), "detections": len(rows["frame_id"]), "seconds": round(time.monotonic() - t0, 1),
                "utc": utc_now()}
        (d / "yolo.json").write_text(json.dumps(info, indent=1) + "\n")
        lg.info("%s: YOLO26 %s on %d frames, %d detections, %.1f s", v, p["variant"], len(fids), info["detections"],
                info["seconds"])
    return 0


def frames_for(root: Path, v: str) -> list[dict]:
    d = gap_dir(root, v)
    y = dict(np.load(d / "yolo.npz"))
    with np.load(root / "skeleton" / f"{v}.npz") as z:
        k_xy32, k_c32 = z["joints_2d_color"], z["confidence"]
    coco_xy, coco_c = ha4m.kinect_to_coco(k_xy32, k_c32)
    out = []
    for fid in y["frames"]:
        m = y["frame_id"] == fid
        ok = fid < len(k_xy32)
        out.append({"frame_id": int(fid), "k_xy": coco_xy[fid] if ok else np.full((17, 2), np.nan),
                    "k_conf": coco_c[fid] if ok else np.zeros(17),
                    "k_box": domain_gap.kinect_box(k_xy32[fid], k_c32[fid]) if ok else None,
                    "y_boxes": y["box"][m], "y_xy": y["xy"][m], "y_conf": y["conf"][m]})
    return out


def table_md(res: dict) -> str:
    lines = [f"# {res['id']}", "", res["title"], "", "Kinect 2D colour joints mapped to COCO 17 against YOLO26 x on the "
             "same frames. PCK threshold relative to the longer side of the Kinect person box. Found: a YOLO "
             f"box with IoU >= {domain_gap.FOUND_IOU} with the Kinect person box.", ""]
    for v, rec in res["videos"].items():
        s = rec["summary"]
        lines += [f"## {v} ({rec['location']}, camera {rec['camera']})", "",
                  f"{s['frames']} frames, Kinect person in {s['frames_with_kinect']}, YOLO finds it in "
                  f"{s['frames_found']} ({s['found_share']}). Mean error {s['mean_px']} px, median {s['median_px']} px; "
                  f"PCK@0.05 {s['pck_0.05']}, PCK@0.1 {s['pck_0.1']} (Kinect visible joints only: "
                  f"{s['pck_0.05_visible']}, {s['pck_0.1_visible']}); residual sd {s.get('residual_sd_torso')} "
                  f"torso lengths.", "",
                  "| joint | n | mean px | median px | PCK@0.05 | PCK@0.1 | PCK@0.1 visible | offset (torso) | sd (torso) |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for j in rec["joints"]:
            lines.append(f"| {j['joint']} | {j['n']} | {j['mean_px']} | {j['median_px']} | {j['pck_0.05']} | "
                         f"{j['pck_0.1']} | {j['pck_0.1_visible']} | {j.get('offset_torso')} | {j.get('sd_torso')} |")
        lines.append("")
    return "\n".join(lines)


def compare_cmd(args, cfg, root: Path, lg, repo_root: Path = REPO_ROOT) -> int:
    from ..experiments.provenance import git_state, library_versions, model_checksum
    vids = [v.strip() for v in args.videos.split(",") if v.strip()]
    loader = ha4m.HA4MLoader(cfg)
    res = {"id": EXP_ID, "kind": "domain_gap", "source": "public", "title":
           "Domain gap: YOLO26 x against the HA4M Kinect 2D joints, one video per lab", "reference": "",
           "videos": {}}
    for v in vids:
        rec = json.loads((gap_dir(root, v) / "frames.json").read_text())
        yinfo = json.loads((gap_dir(root, v) / "yolo.json").read_text())
        out = domain_gap.compare(frames_for(root, v))
        out.update({"location": loader.video_location(v), "camera": rec["camera"], "window": {
            k: rec[k] for k in ("seconds", "step", "start_frac")}, "frame_ids": [int(f) for f in sorted(
                map(int, rec["frames"]))], "yolo": yinfo})
        res["videos"][v] = out
    weights = repo_root / cfg["paths"]["models_dir"] / f"yolo26{cfg['pose']['variant']}-pose.pt"
    res["provenance"] = {"git": git_state(repo_root), "model": model_checksum(weights) if weights.is_file() else None,
                         "versions": library_versions(), "data_area": "public", "dataset": "HA4M " + ha4m.VERSION,
                         "date_utc": utc_now(), "command": " ".join(sys.argv)}
    res["finished_utc"] = utc_now()
    res["index"] = {"cells": [f"{v} ({r['location']}): found {r['summary']['found_share']}, PCK@0.1 "
                              f"{r['summary']['pck_0.1']}, {r['summary']['mean_px']} px" for v, r in res["videos"].items()],
                    "verdict": "measurement"}
    d = repo_root / "experiments" / EXP_ID
    d.mkdir(parents=True, exist_ok=True)
    meta = {"id": EXP_ID, "title": res["title"], "kind": "domain_gap", "source": "public",
            "videos": [f"data/public/datasets/ha4m/domain_gap/{v}" for v in vids], "repeats": 1, "reference": "",
            "gpu": args.gpu, "variant": cfg["pose"]["variant"]}
    (d / "config.toml").write_text(reg.config_text(meta, cfg.snapshot()))
    (d / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float) + "\n")
    (d / "table.md").write_text(table_md(res))
    reg.write_index(repo_root / "experiments")
    for line in res["index"]["cells"]:
        lg.info("domain gap: %s", line)
    lg.info("wrote %s", d)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.ha4m_domain_gap")
    p.add_argument("what", choices=("fetch", "run", "compare"))
    p.add_argument("--videos", default=PILOT)
    p.add_argument("--seconds", type=float, default=27.0)
    p.add_argument("--step", type=int, default=3)
    p.add_argument("--start-frac", type=float, default=0.3)
    p.add_argument("--max-gb", type=float, default=1.5)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args(argv)
    cfg = Config.load()
    try:
        root = dataset_root(cfg, "ha4m")
        lg = setup_log(REPO_ROOT)
        if args.what == "fetch":
            lock = take_lock(root)
            if lock is None:
                lg.error("another HA4M download is running; the domain gap fetch starts after it")
                return 2
            configure_pacer(cfg)
            return fetch(args, cfg, root, lg)
        if args.what == "run":
            return run_yolo(args, cfg, root, lg)
        return compare_cmd(args, cfg, root, lg)
    except AvError as e:
        print(f"{e.code}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
