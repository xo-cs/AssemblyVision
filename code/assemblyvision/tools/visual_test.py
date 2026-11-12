"""The single test evaluation of rule w11-visual-engine-b (Week 11).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.visual_test --id w11-visual-engine-b-test --grid w11-visual-engine-b-cv

CPU only. Rows: Engine A, B v1 as shipped, the setting selected by the grid, and that
setting with intro detection switched. Scored on the HA4M test split (synthetic cuts
p 0.5 and without cuts) and on the IKEA ASM official testing subset of the selection
furniture types, with the final models (v1 and v2 shipped, w11-engine-b-v3.pt).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from ..experiments import registry as reg
from ..storage.files import REPO_ROOT, utc_now
from . import ikea_bench as ib
from . import phase_bench as pb
from .visual_cv_grid import mean, overrides

FINAL = {"v1": "engine_b/w10-engine-b-v1.pt", "v2": "engine_b/w10-engine-b-v2.pt", "v3": "engine_b/w11-engine-b-v3.pt"}
A_ROW = {"seg_a": {"engine": "custom", "engine_b_weight": 0.0, "embed_weight": 0.0, "visual_weight": 0.0}}
B_V1 = {"seg_a": {"engine": "custom", "engine_b_weight": 0.0, "embed_weight": 0.2, "embed_window_s": 30.0,
                  "visual_weight": 0.0, "engine_a_scale": 1.0, "intro_detect": False},
        "engine_b": {"model": FINAL["v1"], "candidates": "engine_a", "video_norm": False}}
CONDS = ("ha4m_p0.5", "ha4m_plain", "Kallax_Shelf_Drawer", "Lack_Coffee_Table")


def score(job: tuple) -> list:
    over, cond, vids = job
    out = []
    if cond.startswith("ha4m"):
        b, cfg = pb.Bench(), pb.bench_cfg(over)
        for v in vids:
            m = b.video(v, cfg, cuts_p=0.5 if cond == "ha4m_p0.5" else None)["metrics"]
            out.append([m["frame_accuracy"], m.get("f1_50"), m.get("boundary_mae_s")])
    else:
        b = ib.IkeaBench()
        for v in vids:
            m = b.video(v, over)["metrics"]
            out.append([m["frame_accuracy"], m.get("f1_50"), m.get("boundary_mae_s")])
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.visual_test")
    p.add_argument("--id", required=True)
    p.add_argument("--grid", required=True)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    g = json.loads((REPO_ROOT / "experiments" / args.grid / "results.json").read_text())
    sel = g["selected"]
    d = dict(sel["decoder"])
    flip = {**d, "intro_detect": not d["intro_detect"]}
    rows = {"Engine A": A_ROW, "B v1 as shipped": B_V1,
            f"selected: {sel['model']}": overrides(FINAL[sel["model"]], d),
            f"selected, intro {'on' if flip['intro_detect'] else 'off'}": overrides(FINAL[sel["model"]], flip)}
    ha = pb.split_videos("test")
    ik = {s: [v for v in ib.split_videos("test") if v.startswith(s + "/")] for s in CONDS[2:]}
    jobs = []
    for name, over in rows.items():
        for c in CONDS:
            vids = ha if c.startswith("ha4m") else ik[c]
            jobs += [(name, c, over, vids[i:i + 8]) for i in range(0, len(vids), 8)]
    with ProcessPoolExecutor(args.workers) as ex:
        parts = list(ex.map(score, [(o, c, v) for _, c, o, v in jobs]))
    res_rows = []
    for name, over in rows.items():
        summ = {}
        for c in CONDS:
            xs = [x for j, part in zip(jobs, parts) if j[0] == name and j[1] == c for x in part]
            summ[c] = {"frame_accuracy": mean([x[0] for x in xs]), "f1_50": mean([x[1] for x in xs]),
                       "boundary_mae_s": mean([x[2] for x in xs]), "videos": len(xs)}
        res_rows.append({"row": name, "sets": over, "summary": summ})
    from ..experiments.provenance import git_state
    res = {"id": args.id, "kind": "visual_test", "source": "public", "reference": "", "grid": args.grid,
           "title": "Visual Engine B: the single test evaluation (rule w11-visual-engine-b)", "rows": res_rows,
           "ikea_videos": ik, "provenance": {"git": git_state(REPO_ROOT), "date_utc": utc_now(),
                                             "command": " ".join(sys.argv)}, "finished_utc": utc_now()}
    res["index"] = {"cells": [f"{r['row']}: " + ", ".join(f"{c} {r['summary'][c]['frame_accuracy']}" for c in CONDS)
                              for r in res_rows], "verdict": "measurement"}
    out = REPO_ROOT / "experiments" / args.id
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    lines = [f"# {args.id}", "", res["title"], "", f"Selected by {args.grid}: {sel['model']} {d}.", "",
             "| row | HA4M p0.5 | HA4M plain | Kallax | Lack Coffee | F1@50 Kallax | MAE s Kallax |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in res_rows:
        s = r["summary"]
        lines.append(f"| {r['row']} | " + " | ".join(str(s[c]["frame_accuracy"]) for c in CONDS)
                     + f" | {s['Kallax_Shelf_Drawer']['f1_50']} | {s['Kallax_Shelf_Drawer']['boundary_mae_s']} |")
    lines.append("")
    (out / "table.md").write_text("\n".join(lines))
    reg.write_index(REPO_ROOT / "experiments")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
