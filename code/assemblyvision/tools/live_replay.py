"""Offline replay of a public video through the live pipeline, frame by frame,
exactly as the Live tab pushes camera frames (LiveManager.push).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.live_replay --video ../data/public/videos/process_480p30.mp4 \
        [--template combined] [--cycles] [--device gpu|cpu] [--pace realtime|fast] [--fps 10] [--start 0] [--seconds 60]

--pace realtime sends frames at --fps on the wall clock (frames are dropped if
processing is slower, as live); fast waits for each frame to be processed
(throughput, no drops). --device gpu uses GPU live.gpu (the A4000, FP16) and
takes its lock like a live session; cpu runs the model on the CPU.

Only videos under data/public/ are accepted (AV-RUN-001). Output: the session
folder data/public/live/<session_id>/ with summary.json and a few overlay
frames replay_NNN.jpg; the report is printed.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.live_replay")
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--template", default="combined", help="combined (the Live page view) or an old station")
    p.add_argument("--cycles", action="store_true", help="combined: count cycles with the default zones")
    p.add_argument("--device", choices=("gpu", "cpu"), default="cpu")
    p.add_argument("--pace", choices=("realtime", "fast"), default="realtime")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--start", type=float, default=0.0, help="start time in the video (s)")
    p.add_argument("--seconds", type=float, default=60.0, help="length to replay (s)")
    p.add_argument("--snapshots", type=int, default=4, help="overlay frames to save")
    args = p.parse_args(argv)

    from ..config import Config
    from ..storage.files import REPO_ROOT, check_video_source, source_root
    cfg = Config.load()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["live"]["gpu"]) if args.device == "gpu" else ""
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

    from ..errors import AvError
    try:
        video = check_video_source(args.video, "public", cfg["paths"])
    except AvError as e:
        print(f"{e.code}: {e.text} ({e.detail})", file=sys.stderr)
        return 1
    import cv2
    import numpy as np

    from ..live.manager import LiveManager, gpu_free_mb
    from ..live.perception import YoloPose

    live = cfg["live"]
    model = REPO_ROOT / cfg["paths"]["models_dir"] / live["pose_model"]
    device = "cuda:0" if args.device == "gpu" else "cpu"
    factory = lambda: YoloPose(model, device, live["half"], live["input_px"], live["det_conf"])  # noqa: E731
    lock_dir = REPO_ROOT / ".locks" if args.device == "gpu" else Path(tempfile.mkdtemp())
    lm = LiveManager(cfg, source_root("public", cfg["paths"]) / "live", backend_factory=factory,
                     gpu_free=gpu_free_mb if args.device == "gpu" else (lambda i: None), lock_dir=lock_dir)
    cap = cv2.VideoCapture(str(video))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t_load = time.monotonic()
    try:
        sid = lm.start(args.template, False, None, args.cycles)
    except AvError as e:
        print(f"{e.code}: {e.text}" + (f" ({e.detail})" if e.detail else ""), file=sys.stderr)
        return 1
    print(f"session {sid}: model ready in {time.monotonic() - t_load:.1f} s on {args.device}", flush=True)
    s = lm.session
    folder = s.folder
    folder.mkdir(parents=True, exist_ok=True)
    n_send = int(args.seconds * args.fps)
    snap_at = set(np.linspace(n_send // 5, n_send - 1, args.snapshots).astype(int).tolist()) if n_send else set()
    t0 = time.monotonic()
    sent = 0
    pos = int(round(args.start * vfps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)          # one seek, then read in order (a seek per frame is slow)
    for i in range(n_send):
        fidx = int(round((args.start + i / args.fps) * vfps))
        if fidx >= n_total:
            break
        while pos < fidx and cap.grab():
            pos += 1
        ok, bgr = cap.read()
        pos += 1
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if args.pace == "realtime":
            delay = t0 + i / args.fps - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            lm.push(rgb)
        else:
            k = s.processed
            lm.push(rgb)
            while s.processed == k and time.monotonic() - t0 < 3600:
                time.sleep(0.001)
        sent += 1
        if i in snap_at and s.latest_overlay is not None:
            cv2.imwrite(str(folder / f"replay_{i:04d}.jpg"), cv2.cvtColor(s.latest_overlay, cv2.COLOR_RGB2BGR))
    time.sleep(0.5)
    summary = lm.stop()
    cap.release()
    wall = time.monotonic() - t0
    print(f"replay of {video.name} from {args.start:g} s: {sent} frames sent at {args.fps:g} fps ({args.pace}) "
          f"in {wall:.1f} s, processed {summary['processed']}, dropped {summary['dropped']}, "
          f"processed fps {summary['processed'] / wall:.1f}, delay p50 {summary['delay_p50_ms']} ms, "
          f"p95 {summary['delay_p95_ms']} ms")
    print("  time per frame, p50 / p95 ms: " + ", ".join(f"{k} {v['p50']} / {v['p95']}"
                                                         for k, v in summary["stage_ms"].items()))
    for st in summary["stations"]:
        print(f"  {st['title']}:")
        for k, v in st["rows"]:
            print(f"    {k}: {v}")
        for row in st["log"][:12]:
            print(f"    {row}")
    for n in summary["notices"]:
        print(f"  notice {n}")
    print(f"session folder {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
