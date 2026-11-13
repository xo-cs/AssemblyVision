"""The siglip_scores stage (Week 11): SigLIP 2 zero shot phase scores of a stored run.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.siglip_scores --run RUN_DIR --gpu 0            # writes RUN_DIR/siglip.npz
    python -m assemblyvision.tools.siglip_scores --dataset ikea_asm --gpu 0

A QiSens run is scored with docs_team/phase_descriptions.md, an IKEA ASM run (from
data/public/datasets/ikea_asm/derived/runs.json) with the descriptions of its furniture
type. Full frame and subject crop views at zero_shot.siglip_fps (analysis/siglip.py, zero_shot.py). Output: siglip.npz
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

from ..storage.files import REPO_ROOT, run_source


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
    from ..analysis import zero_shot as Z
    from ..config import Config
    from .visual_embed import take_lock
    cfg = Config.load()
    zs = cfg["zero_shot"]
    jobs = []                                   # (run root, phase set)
    ikea_runs = REPO_ROOT / "data/public/datasets/ikea_asm/derived/runs.json"
    if args.run is not None:
        source, root = run_source(args.run, cfg["paths"])
        if source == "qisens":
            jobs.append((root, "qisens"))
        else:
            ikea = {v: k for k, v in json.loads(ikea_runs.read_text()).items()}
            if root.name not in ikea:
                p.error("a public run must be an IKEA ASM run (derived/runs.json)")
            jobs.append((root, "ikea_asm:" + ikea[root.name].split("/")[0]))
    else:
        for vid, rid in sorted(json.loads(ikea_runs.read_text()).items()):
            furn = vid.split("/")[0]
            if args.furniture and furn != args.furniture:
                continue
            root = REPO_ROOT / "data/public/runs" / rid
            if (root / "manifest.json").is_file():
                jobs.append((root, "ikea_asm:" + furn))
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
    from ..analysis import siglip as S
    loaded = S.load(REPO_ROOT / "models", device, half=device == "cuda")
    n = 0
    for i, (root, pset) in enumerate(todo):
        phases, _, label = Z.phase_set(pset)
        manifest = json.loads((root / "manifest.json").read_text())
        t = time.monotonic()
        rec = Z.siglip_run(root, manifest, phases, label, zs, device, loaded)
        n += rec["samples"]
        print(f"  {i + 1}/{len(todo)}: {rec['samples']} samples ({rec['crops']} crops), "
              f"{time.monotonic() - t:.1f} s", flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    print(f"siglip_scores: {n} samples in {time.monotonic() - t0:.1f} s, torch peak {peak:.0f} MB")
    if lock is not None:
        lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
