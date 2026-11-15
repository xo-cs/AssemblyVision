"""The body3d stage on a stored run (Week 11; assemblyvision/body3d, GVHMR in envs/gvhmr).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.body3d --run RUN_DIR [--gpu 0]

Writes RUN_DIR/body3d/<shot_id>_p<k>.npz (frames, joints_world [L, 17, 3] in metres, gravity aligned)
and summary.json; a QiSens run's files stay under data/qisens. No frame is written. Takes
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
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.body3d")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--gpu", type=int, help="default body3d.gpu")
    args = p.parse_args(argv)
    from .. import body3d as B
    from .visual_embed import take_lock
    cfg = Config.load()
    _, root = run_source(args.run, cfg["paths"])
    need = B.missing()
    if need:
        print("body3d cannot run yet; missing: " + "; ".join(need), file=sys.stderr)
        return 2
    manifest = json.loads((root / "manifest.json").read_text())
    gpu = cfg["body3d"]["gpu"] if args.gpu is None else args.gpu
    lock = take_lock(gpu)
    try:
        s = B.run(root, manifest, cfg, gpu)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    finally:
        lock.close()
    frames = sum(x["frames"] for x in s["pieces"])
    print(f"body3d: {len(s['pieces'])} pieces, {frames} frames in {s['seconds']} s "
          f"({frames / max(s['seconds'], 1e-6):.1f} frames/s), torch peak {s['torch_peak_mb']} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
