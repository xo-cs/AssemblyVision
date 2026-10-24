"""Score shot detection against a ground truth JSON from make_test_clips (CPU, writes nothing).

    python -m assemblyvision.tools.eval_shots --video data/public/videos/shots_tune_480p30.mp4 \
        --thresholds 15,20,25,30,35,40

A detected cut matches a ground truth cut within --tolerance frames (one to
one). Cuts in dont_care are neither required nor counted as false. The shot
type check uses the ground truth segment bounds and reports every segment's
background motion; only synthetic segments have a known type.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ..config import Config
from ..storage.files import REPO_ROOT, source_root


def score(detected: list[int], gt: list[int], dont_care: list[int], tol: int) -> dict:
    required = [c for c in gt if c not in dont_care]
    used, tp = set(), 0
    for c in required:
        hit = next((d for d in detected if d not in used and abs(d - c) <= tol), None)
        if hit is not None:
            used.add(hit)
            tp += 1
    fp = [d for d in detected if d not in used and not any(abs(d - c) <= tol for c in dont_care)]
    fn = len(required) - tp
    p = tp / (tp + len(fp)) if tp + len(fp) else 1.0
    r = tp / len(required) if required else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"tp": tp, "fp": len(fp), "fn": fn, "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(f1, 3), "false_cuts": fp}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--thresholds", default="15,20,25,30,35,40")
    ap.add_argument("--tolerance", type=int, default=1)
    ap.add_argument("--types", action="store_true", help="also check fixed or handheld per segment")
    args = ap.parse_args(argv)
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    cfg = Config.load()
    video = Path(os.path.realpath(args.video))
    if not video.is_relative_to(source_root("public", cfg["paths"])):
        print("refused: eval_shots runs on public clips only", file=sys.stderr)
        return 2
    gt = json.loads(video.with_suffix(".json").read_text())

    from ..ingest.shots import ShotDetector, detect_cuts
    from ..types import Shot

    print(f"{video.name}: {len(gt['cuts'])} cuts, dont_care {gt['dont_care']}, tolerance ±{args.tolerance} frames")
    print(f"{'thr':>5} {'tp':>3} {'fp':>3} {'fn':>3} {'prec':>6} {'recall':>6} {'f1':>6}  false cuts")
    for t in (float(x) for x in args.thresholds.split(",")):
        cuts = detect_cuts(video, gt["fps"], t, cfg["shots"]["min_shot_s"])
        s = score(cuts, gt["cuts"], gt["dont_care"], args.tolerance)
        print(f"{t:5.0f} {s['tp']:3d} {s['fp']:3d} {s['fn']:3d} {s['precision']:6.3f} {s['recall']:6.3f} "
              f"{s['f1']:6.3f}  {s['false_cuts']}")
    if args.types:
        sd = ShotDetector(cfg, None)
        shots = [Shot(f"seg{k:02d}", "eval", g["start_frame"], g["start_frame"] + g["frames"], 0, 0, "")
                 for k, g in enumerate(gt["segments"])]
        motion = sd.classify_all(video, shots, gt["fps"])
        thr = cfg["shots"]["fixed_motion_max"]
        print(f"\nfixed if motion p90 < {thr}")
        for sh, g in zip(shots, gt["segments"]):
            m = motion[sh.shot_id]
            got = "fixed" if m is not None and m < thr else "handheld"
            ok = "" if g["shot_type"] is None else ("ok" if got == g["shot_type"] else "WRONG")
            print(f"{sh.shot_id} {g['source']:>13} {g['frames']:4d} fr  motion {m}  -> {got:8s} "
                  f"truth {g['shot_type']}  {ok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
