"""The scene4d stage on a stored run (Week 11; assemblyvision/scene4d, Pi3 in envs/scene4d).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.scene4d --run RUN_DIR [--gpu 0]

Writes RUN_DIR/scene4d/<shot_id>_p<k>.npz (the scene, the person, cameras), .align.json (the GVHMR
skeleton in the scene, when body3d ran first), .player.bin/.json (for the Heatmap page) and summary.json; a QiSens run's files stay under data/qisens. No frame is written. Takes
.locks/gpu<N>.lock first. Prints only counts and timings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import Config
from ..storage.files import run_source


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.scene4d")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--gpu", type=int, help="default scene4d.gpu")
    args = p.parse_args(argv)
    from .. import scene4d as B
    from .visual_embed import take_lock
    cfg = Config.load()
    _, root = run_source(args.run, cfg["paths"])
    need = B.missing()
    if need:
        print("scene4d cannot run yet; missing: " + "; ".join(need), file=sys.stderr)
        return 2
    manifest = json.loads((root / "manifest.json").read_text())
    gpu = cfg["scene4d"]["gpu"] if args.gpu is None else args.gpu
    lock = take_lock(gpu)
    try:
        s = B.run(root, manifest, cfg, gpu)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    finally:
        lock.close()
    samples = sum(x.get("samples", 0) for x in s["pieces"])
    print(f"scene4d: {len(s['pieces'])} parts, {samples} sampled frames in {s['seconds']} s, torch peak "
          f"{s['torch_peak_mb']} MB, {len(s['failed'])} parts failed, {sum(s['aligned'].values())} with the skeleton "
          f"aligned (scene4d/summary.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
