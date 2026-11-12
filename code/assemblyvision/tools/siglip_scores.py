"""The siglip_scores stage (Week 11): SigLIP 2 zero shot phase scores of a stored run.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.siglip_scores --run RUN_DIR --gpu 0            # writes RUN_DIR/siglip.npz
    python -m assemblyvision.tools.siglip_scores --dataset ikea_asm --gpu 0

A QiSens run is scored with docs_team/phase_descriptions.md, an IKEA ASM run (from
data/public/datasets/ikea_asm/derived/runs.json) with the descriptions of its furniture
type. Full frame and subject crop views at 2 fps (analysis/siglip.py). Output: siglip.npz
in the run folder (a QiSens run's stays under data/qisens); only counts and timings are
printed. On a GPU the job takes .locks/gpu<N>.lock first.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from ..storage.files import REPO_ROOT, run_source, utc_now

FPS, MAX_PX, PAD = 2.0, 1280, 0.2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.siglip_scores")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path)
    g.add_argument("--dataset", choices=("ikea_asm",))
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--furniture", help="ikea_asm: only this furniture type")
    args = p.parse_args(argv)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu >= 0 else ""
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    from ..analysis import phase_text as pt
    from ..analysis import siglip as S
    from ..config import Config
    from .visual_embed import take_lock
    cfg = Config.load()
    jobs = []                                   # (run root, phases, label)
    if args.run is not None:
        source, root = run_source(args.run, cfg["paths"])
        if source == "qisens":
            jobs.append((root, pt.phases("qisens"), "qisens"))
        else:
            ikea = {v: k for k, v in json.loads((REPO_ROOT / "data/public/datasets/ikea_asm/derived/runs.json")
                                                .read_text()).items()}
            if root.name not in ikea:
                p.error("a public run must be an IKEA ASM run (derived/runs.json)")
            furn = ikea[root.name].split("/")[0]
            jobs.append((root, pt.phases("ikea_asm", furn), furn))
    else:
        runs = json.loads((REPO_ROOT / "data/public/datasets/ikea_asm/derived/runs.json").read_text())
        for vid, rid in sorted(runs.items()):
            furn = vid.split("/")[0]
            if args.furniture and furn != args.furniture:
                continue
            root = REPO_ROOT / "data/public/runs" / rid
            if (root / "manifest.json").is_file():
                jobs.append((root, pt.phases("ikea_asm", furn), furn))
    todo = [j for j in jobs if args.force or not (j[0] / "siglip.npz").is_file()]
    print(f"siglip_scores: {len(jobs)} runs, {len(todo)} to score")
    if not todo:
        return 0
    lock = take_lock(args.gpu)
    import torch
    device = "cuda" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    if args.gpu >= 0 and device == "cpu":
        print("refused: no CUDA device", file=sys.stderr)
        return 2
    t0 = time.monotonic()
    model, proc = S.load(REPO_ROOT / "models", device, half=device == "cuda")
    banks: dict[str, tuple] = {}
    n = 0
    for i, (root, phases, label) in enumerate(todo):
        if label not in banks:
            banks[label] = (S.text_bank(model, proc, phases, device), S.text_sha(phases))
        bank, sha = banks[label]
        manifest = json.loads((root / "manifest.json").read_text())
        cfr = manifest["ingest"]["cfr"]
        t = time.monotonic()
        res = S.score_video(model, proc, bank, root / cfr["path"], S.subject_boxes(manifest, root), float(cfr["fps"]),
                            FPS, MAX_PX, PAD, device=device, half=device == "cuda")
        S.save(root / "siglip.npz", res, {"model": S.MODEL_DIR, "revision": S.REVISION, "fps": FPS, "max_px": MAX_PX,
                                          "pad": PAD, "phases": [ph["name"] for ph in phases], "descriptions_sha": sha,
                                          "set": label, "date_utc": utc_now()})
        n += len(res["time_s"])
        print(f"  {i + 1}/{len(todo)}: {len(res['time_s'])} samples ({int(res['crop_ok'].sum())} crops), "
              f"{time.monotonic() - t:.1f} s", flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    print(f"siglip_scores: {n} samples in {time.monotonic() - t0:.1f} s, torch peak {peak:.0f} MB")
    if lock is not None:
        lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
