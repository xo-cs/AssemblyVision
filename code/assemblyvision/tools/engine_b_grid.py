"""Engine B weight and pool window on public runs (Week 10, step 2 of the pool_s plan).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.engine_b_grid --id w10-engine-b-pool --model engine_b/w10-engine-b-v0.pt \\
        --run ../data/public/runs/RUN --gt ../data/public/videos/CLIP_gt.csv [--run ... --gt ...] \\
        [--weights 0,0.2,0.4,0.8] [--pools 0,10,30,60]

CPU only, public runs only (a QiSens run is refused): the compare mode of run_pose
(in memory, nothing in a run folder is written) once per pool window, over the
weights, for every run and its GT. Writes experiments/<id>/ (results.json with
every row, table.md) and INDEX.md. The values are chosen from this table before
any QiSens result (the rule is written in the experiment's note).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only, like the compare mode

from ..config import Config, deep_merge
from ..experiments import registry as reg
from ..storage.files import REPO_ROOT, run_source, source_root, utc_now
from .run_pose import _check_gt, compare_run


def main(argv: list[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.engine_b_grid")
    p.add_argument("--id", required=True)
    p.add_argument("--model", required=True, help="engine_b.model, e.g. engine_b/w10-engine-b-v0.pt")
    p.add_argument("--run", action="append", type=Path, required=True)
    p.add_argument("--gt", action="append", type=Path, required=True)
    p.add_argument("--weights", default="0,0.2,0.4,0.8")
    p.add_argument("--pools", default="0,10,30,60", help="values of --window-key")
    p.add_argument("--weight-key", default="seg_a.engine_b_weight", choices=["seg_a.engine_b_weight", "seg_a.embed_weight"])
    p.add_argument("--window-key", default="engine_b.pool_s", choices=["engine_b.pool_s", "seg_a.embed_window_s"])
    p.add_argument("--note", default="")
    p.add_argument("--summary", default="", help="the INDEX.md cell (the chosen values)")
    args = p.parse_args(argv)
    reg._check_id(args.id)
    if len(args.run) != len(args.gt):
        print("refused: one --gt per --run", file=sys.stderr)
        return 2
    base = Config.load()
    out = []
    public_runs = source_root("public", base["paths"], repo_root) / "runs"
    for run, gt in zip(args.run, args.gt):
        if Path(os.path.realpath(run)).parent != public_runs:      # nothing outside data/public is touched
            print("refused: public runs only", file=sys.stderr)
            return 2
        source, root = run_source(run, base["paths"], repo_root)
        if source != "public":
            print("refused: public runs only", file=sys.stderr)
            return 2
        manifest = json.loads((root / "manifest.json").read_text())
        ws, _, wk = args.window_key.partition(".")
        for pool_s in [float(x) for x in args.pools.split(",")]:
            cfg = Config.load(overrides=deep_merge(manifest.get("config", {}), {"engine_b": {"model": args.model}},
                                                   {ws: {wk: pool_s}}))
            _check_gt(gt, source, cfg)
            for r in compare_run(manifest, root, source, cfg, f"{args.weight_key}={args.weights}", gt):
                out.append({"run": manifest["run_id"], "video": Path(manifest["ingest"]["source_path"]).name
                            if "source_path" in manifest.get("ingest", {}) else manifest["run_id"],
                            "gt": gt.name, "pool_s": pool_s, "weight": float(r["setting"].split("=")[1]),
                            "metrics": r["metrics"], "phases": r["phases"], "engine_b_rate": r["engine_b_rate"]})
    f1 = [k for k in out[0]["metrics"] if k.startswith("f1_")]
    lines = [f"# {args.id}: Engine B {args.weight_key} and {args.window_key} on public runs", "", args.note, "",
             f"Model {args.model}; Engine A compare mode (CPU, in memory). B rate = Engine B peaks per minute, "
             "raw / pooled (as in Engine A's report).", ""]
    for gt in dict.fromkeys(r["gt"] for r in out):
        rows = [r for r in out if r["gt"] == gt]
        lines += [f"## {rows[0]['run']} against {gt}", "",
                  f"| {wk} | {args.weight_key.split('.')[1]} | frame acc | edit | " + " | ".join(f"F1@{k[3:]}" for k in f1)
                  + " | MAE s | phases | B rate |", "|" + " --- |" * (7 + len(f1))]
        for r in rows:
            m = r["metrics"]
            lines.append(f"| {r['pool_s']:g} | {r['weight']:g} | {m['frame_accuracy']:.2f} | {m['edit']:.2f} | "
                         + " | ".join(f"{m[k]:.2f}" for k in f1)
                         + f" | {'n/a' if m['boundary_mae_s'] is None else format(m['boundary_mae_s'], '.2f')} | "
                           f"{r['phases']} | {r['engine_b_rate']} |")
        lines.append("")
    res = {"id": args.id, "kind": "engine_b_grid", "source": "public", "title": f"Engine B {args.weight_key} and {args.window_key}",
           "reference": "", "note": args.note, "model": args.model, "weight_key": args.weight_key, "window_key": args.window_key, "rows": out,
           "provenance": {"date_utc": utc_now(), "command": " ".join(sys.argv), "data_area": "public"},
           "finished_utc": utc_now(),
           "index": {"cells": [args.summary or f"{len(out)} rows over {len(args.run)} public runs; see table"],
                     "verdict": "measurement"}}
    d = repo_root / "experiments" / args.id
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(res, indent=2, default=float) + "\n")
    (d / "table.md").write_text("\n".join(lines))
    reg.write_index(repo_root / "experiments")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
