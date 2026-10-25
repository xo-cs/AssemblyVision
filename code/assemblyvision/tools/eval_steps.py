"""Evaluate the steps of a run, or the agreement of two annotators (CPU).

    python -m assemblyvision.tools.eval_steps --run RUN_DIR --gt GT_CSV
    python -m assemblyvision.tools.eval_steps --kappa A.csv B.csv

--run prints frame accuracy, edit score, F1@10, F1@25, F1@50 and the mean
absolute boundary error, and writes RUN_DIR/eval.json. GT format:
step,label,start_s,end_s,source. --kappa prints Cohen's kappa at frame level
(eval.kappa_fps) of two raw files (step,label,start,end with m:ss times and an
inclusive end second) and writes nothing.

The GT file must be in the same data area (data/qisens or data/public) as the
run, and both kappa files in one area; anything else is refused, so QiSens
labels and results never mix with public ones. Exit codes: 0, 1 error, 2 refused.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from ..config import Config, deep_merge
from ..errors import AvError
from ..storage.files import REPO_ROOT, RunContext, RunPaths, data_area, run_source
from .run_pose import _fail, parse_sets


def kappa_files(a: Path, b: Path, cfg: Config) -> dict:
    from ..evaluation.metrics import cohen_kappa, frames_from_steps
    from ..evaluation.steps_io import read_raw_csv
    sa, sb = read_raw_csv(a), read_raw_csv(b)
    end = max(max(s.end_s for s in sa), max(s.end_s for s in sb))
    fps = cfg["eval"]["kappa_fps"]
    fa, fb = frames_from_steps(sa, end, fps), frames_from_steps(sb, end, fps)
    k, po, pe = cohen_kappa(fa, fb)
    return {"kappa": round(k, 4), "po": round(po, 4), "pe": round(pe, 4), "frames": len(fa), "fps": fps,
            "duration_s": end}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.eval_steps")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path, help="run folder data/<source>/runs/<run_id>")
    g.add_argument("--kappa", nargs=2, type=Path, metavar=("A_CSV", "B_CSV"))
    p.add_argument("--gt", type=Path, help="GT CSV (with --run)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = p.parse_args(argv)
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    try:
        cfg = Config.load(overrides=parse_sets(args.set))
        if args.kappa:
            areas = {data_area(f, cfg["paths"]) for f in args.kappa}
            if len(areas) != 1 or None in areas:
                print("refused: both kappa files must lie in the same data area (data/qisens or data/public)",
                      file=sys.stderr)
                return 2
            r = kappa_files(*args.kappa, cfg)
            print(f"Cohen's kappa {r['kappa']:.4f}  (observed agreement {r['po']:.4f}, chance {r['pe']:.4f}, "
                  f"{r['frames']} frames at {r['fps']} fps)")
            return 0
        if args.gt is None:
            p.error("--run needs --gt")
        source, root = run_source(args.run, cfg["paths"])
        if data_area(args.gt, cfg["paths"]) != source:
            print(f"refused: the GT file must lie in the {source} data area, like the run", file=sys.stderr)
            return 2
        manifest = json.loads((root / "manifest.json").read_text())
        cfg = Config.load(overrides=deep_merge(manifest.get("config", {}), parse_sets(args.set)))
        log = logging.getLogger("assemblyvision.eval_steps")
        log.addHandler(logging.NullHandler())
        log.propagate = False
        ctx = RunContext(manifest["run_id"], source, RunPaths(manifest["run_id"], root), log, manifest)
        from ..evaluation.steps_eval import evaluate_run, format_metrics
        rec = evaluate_run(ctx, args.gt, cfg)
        ctx.write_manifest()
    except AvError as e:
        return _fail(e)
    print(f"{rec['run_id']}: {format_metrics(rec['metrics'])}")
    for b in rec["boundaries"]:
        print(f"  GT boundary {b['gt_s']:8.2f} s  nearest predicted {b['pred_s']}  error {b['error_s']}")
    print(f"wrote {ctx.rel(ctx.paths.eval_json)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
