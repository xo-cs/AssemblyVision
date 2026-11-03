"""Speed benchmark of YOLO26 pose variants on one public video (one GPU, one process).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.bench_pose --video ../data/public/videos/12981262_1920_1080_25fps.mp4 \
        --gpu 0 --variants n,s,m,l,x --precisions 32,16 [--imgsz 640] [--warmup 20]

Every frame of the video is decoded once per variant and precision and given
to the model as the batch pose stage does (pose.imgsz, pose.det_conf_min).
Reported per combination: model fps and ms per frame (mean and p95, inference
only, the GPU synchronised after each frame), end to end fps with decoding,
peak GPU memory (torch allocator) and the speed relative to the video frame
rate. Only videos under data/public/ are accepted (AV-RUN-001). The GPU lock
.locks/gpu<index>.lock is taken like a run (one job per GPU); a busy GPU is
refused, not waited for. Output: logs/bench_pose_<utc>_gpu<index>.json.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from ..config import VARIANTS as BENCH_VARIANTS, Config
from ..errors import AvError
from ..storage.files import REPO_ROOT, check_video_source, utc_now
from .run_pose import _fail



def summarise(ms: list[float], wall_s: float, n: int, src_fps: float) -> dict:
    a = np.asarray(ms, np.float64)
    fps = 1000.0 / a.mean() if len(a) else None
    return {"frames": n, "ms_mean": round(float(a.mean()), 2), "ms_p95": round(float(np.percentile(a, 95)), 2),
            "fps": round(fps, 1), "fps_e2e": round(n / wall_s, 1) if wall_s > 0 else None,
            "x_realtime": round(fps / src_fps, 2)}


def bench_one(path: Path, weights: Path, quantize: int, cfg: Config, warmup: int, imgsz: int) -> dict:
    import cv2
    import torch
    from ultralytics import YOLO
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(str(weights), task="pose")
    model.to("cuda:0")
    conf = cfg["pose"]["det_conf_min"]
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ms, n, k = [], 0, 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.perf_counter()
        # Ultralytics 8.4.178 replaced half with quantize (32 = FP32, 16 = FP16), as in perception/pose.py
        model.predict(frame, imgsz=imgsz, conf=conf, quantize=quantize, device="cuda:0", verbose=False, save=False)
        torch.cuda.synchronize()
        k += 1
        if k > warmup:
            ms.append(1000 * (time.perf_counter() - t))
            n += 1
        if k == warmup:
            t0 = time.perf_counter()
    wall = time.perf_counter() - t0
    cap.release()
    out = summarise(ms, wall, n, src_fps)
    out["peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    del model
    torch.cuda.empty_cache()
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.bench_pose")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True, help="GPU index in PCI bus order")
    p.add_argument("--variants", default="n,s,m,l,x")
    p.add_argument("--precisions", default="32,16", help="comma list of 32 and 16")
    p.add_argument("--imgsz", type=int, default=None, help="default pose.imgsz")
    p.add_argument("--warmup", type=int, default=20, help="frames not timed at the start of each combination")
    args = p.parse_args(argv)
    variants = args.variants.split(",")
    precisions = [int(x) for x in args.precisions.split(",")]
    if any(v not in BENCH_VARIANTS for v in variants) or any(q not in (16, 32) for q in precisions):
        p.error(f"--variants from {','.join(BENCH_VARIANTS)}, --precisions from 32,16")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        # before torch is imported
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    try:
        cfg = Config.load()
        video = check_video_source(args.video, "public", cfg["paths"])
    except AvError as e:
        return _fail(e)
    models = REPO_ROOT / cfg["paths"]["models_dir"]
    missing = [v for v in variants if not (models / f"yolo26{v}-pose.pt").is_file()]
    if missing:
        print(f"missing weights in {models}: {', '.join(f'yolo26{v}-pose.pt' for v in missing)}", file=sys.stderr)
        return 1
    lock_dir = REPO_ROOT / ".locks"
    lock_dir.mkdir(exist_ok=True)
    lock = open(lock_dir / f"gpu{args.gpu}.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"refused: GPU {args.gpu} is in use (lock held)", file=sys.stderr)
        return 2

    import torch
    props = torch.cuda.get_device_properties(0)
    rec = {"created_utc": utc_now(), "video": video.name, "gpu": args.gpu, "gpu_name": props.name,
           "gpu_cc": f"{props.major}.{props.minor}", "gpu_total_mb": round(props.total_memory / 2**20),
           "torch": torch.__version__, "imgsz": args.imgsz or cfg["pose"]["imgsz"], "warmup": args.warmup,
           "results": []}
    for v in variants:
        for q in precisions:
            r = bench_one(video, models / f"yolo26{v}-pose.pt", q, cfg, args.warmup, rec["imgsz"])
            r.update(variant=v, precision=f"FP{q}")
            rec["results"].append(r)
            print(f"{props.name} {v} FP{q}: {r['fps']} fps, {r['ms_mean']} ms (p95 {r['ms_p95']}), "
                  f"e2e {r['fps_e2e']} fps, peak {r['peak_mb']} MB, {r['x_realtime']}x real time", flush=True)
    out = REPO_ROOT / "logs" / f"bench_pose_{rec['created_utc'].replace(':', '').replace('-', '')}_gpu{args.gpu}.json"
    out.write_text(json.dumps(rec, indent=2))
    print(f"wrote {out}")
    lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
