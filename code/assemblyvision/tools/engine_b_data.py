"""Build or extend the Engine B training data cache (datasets/engine_b.py; CPU).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.engine_b_data ha4m

Takes every HA4M video whose annotations are fetched, maps the Kinect 2D colour
joints to COCO 17, computes the body frame features and the boundary targets,
and writes data/public/datasets/ha4m/engine_b/kinect_coco17/<video>.npz plus
index.json. Videos already built (same skeleton file, same settings) are
skipped, so running it again extends the cache as more videos arrive.
"""
from __future__ import annotations

import sys

from ..config import Config
from ..datasets import engine_b, ha4m
from ..errors import AvError


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if args != ["ha4m"]:
        print("usage: python -m assemblyvision.tools.engine_b_data ha4m", file=sys.stderr)
        return 2
    cfg = Config.load()
    try:
        loader = ha4m.HA4MLoader(cfg)
        engine_b.build(loader, cfg, image_h=float(ha4m.RGB_SIZE[1]))
    except AvError as e:
        print(f"{e.code}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
