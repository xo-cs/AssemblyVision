"""The visual stage (Week 11): DINOv2 ViT S/14 embeddings at visual.fps for a stored run
or for a public dataset.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.visual_embed --run RUN_DIR --gpu 0      # writes RUN_DIR/visual.npz
    python -m assemblyvision.tools.visual_embed --dataset ikea_asm --gpu 0 [--limit N]
    python -m assemblyvision.tools.visual_embed --video PUBLIC_VIDEO --out PUBLIC_NPZ --gpu 0

A run's embeddings go into its own folder (a QiSens run's stay under data/qisens; the
stage only prints counts and timings). A dataset's go to
data/public/datasets/<name>/derived/dinov2_vits14/. On a GPU the job takes
.locks/gpu<N>.lock first (one job per GPU). Frames are never written to disk.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

from ..storage.files import REPO_ROOT, data_area, run_source, utc_now

DATASET_VIDEOS = {"ikea_asm": ("rgb_top", "*.avi")}


def take_lock(gpu: int):
    if gpu < 0:
        return None
    d = REPO_ROOT / ".locks"
    d.mkdir(exist_ok=True)
    f = open(d / f"gpu{gpu}.lock", "a+")
    fcntl.flock(f, fcntl.LOCK_EX)
    return f


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.visual_embed")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path)
    g.add_argument("--dataset", choices=sorted(DATASET_VIDEOS))
    g.add_argument("--video", type=Path, help="a public video (with --out)")
    p.add_argument("--out", type=Path)
    p.add_argument("--gpu", type=int, default=0, help="GPU index in PCI bus order, -1 for CPU")
    p.add_argument("--limit", type=int)
    p.add_argument("--force", action="store_true", help="embed again even if the output exists")
    args = p.parse_args(argv)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu >= 0 else ""
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    from ..config import Config
    from ..analysis import visual as V
    from ..experiments.provenance import git_state
    cfg = Config.load()
    vc = cfg["visual"]
    jobs: list[tuple[Path, Path, dict]] = []
    if args.run is not None:
        source, root = run_source(args.run, cfg["paths"])
        manifest = json.loads((root / "manifest.json").read_text())
        jobs.append((root / manifest["ingest"]["cfr"]["path"], root / "visual.npz",
                     {"run_id": manifest.get("run_id"), "source": source}))
    elif args.video is not None:
        if args.out is None:
            p.error("--video needs --out")
        for x in (args.video, args.out):
            if data_area(x.resolve(), cfg["paths"]) != "public":
                p.error(f"{x} is not under data/public")
        jobs.append((args.video, args.out, {"source": "public"}))
    else:
        base = REPO_ROOT / "data/public/datasets" / args.dataset
        sub, pattern = DATASET_VIDEOS[args.dataset]
        vids = sorted((base / sub).rglob(pattern))[:args.limit] if args.limit else sorted((base / sub).rglob(pattern))
        out = base / "derived/dinov2_vits14"
        jobs += [(v, out / v.relative_to(base / sub).with_suffix(".npz"), {"source": "public", "dataset": args.dataset})
                 for v in vids]
    todo = [j for j in jobs if args.force or not j[1].is_file()]
    print(f"visual: {len(jobs)} videos, {len(todo)} to embed")
    if not todo:
        return 0
    lock = take_lock(args.gpu)
    import torch
    device = "cuda" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    if args.gpu >= 0 and device == "cpu":
        print("refused: no CUDA device", file=sys.stderr)
        return 2
    t0 = time.monotonic()
    model = V.load_encoder(REPO_ROOT / "models", device, vc["half"])
    meta0 = {"model": "dinov2_vits14", "weights": V.WEIGHTS, "fps": vc["fps"], "input_px": vc["input_px"],
             "half": vc["half"], "git": git_state(REPO_ROOT).get("commit"), "date_utc": utc_now()}
    n_frames = 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i, (video, out, extra) in enumerate(todo):
        t = time.monotonic()
        res = V.video_embeddings(model, video, vc["fps"], vc["input_px"], vc["batch"], device, vc["half"])
        V.save(out, res, {**meta0, **extra})
        n_frames += len(res["time_s"])
        print(f"  {i + 1}/{len(todo)}: {len(res['time_s'])} frames, {time.monotonic() - t:.1f} s")
    peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    print(f"visual: {n_frames} frames in {time.monotonic() - t0:.1f} s, torch peak {peak:.0f} MB")
    if lock is not None:
        lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
