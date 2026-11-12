"""Engine B v2 selection by cross validation on the train split (Week 10).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.engine_b_cv_grid --id w10-engine-b-v2-cv \\
        --models w10-engine-b-v1-cv,w10-engine-b-v2-w4-c05,... [--workers 6]

CPU only, public data only, by the rule in experiments/w10-engine-b-v2/rule.md: for
every model (trained with --folds cv) and every decoder setting of the grid, each of
the 152 train videos is decoded by the model of the fold that held its subject out
(experiments/datasets/ha4m_train_cv.json), with dense candidates, and scored on the
macro phase benchmark without cuts and with the synthetic cuts of each p
(phase_bench.CUT_PS). Settings are ranked by the worst case over p of the mean frame
accuracy over the 152 videos (amendment 2), ties by F1@50 and MAE in that condition,
then simplicity. Writes experiments/<id>/ (results.json, table.md) and
INDEX.md; the benchmark test split is not read.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")    # CPU only

from ..experiments import registry as reg
from ..storage.files import REPO_ROOT, utc_now
from . import phase_bench as pb

CV_FILE = "experiments/datasets/ha4m_train_cv.json"
# rule amendment 2: dense candidates on; Engine B's weight against Engine A's candidate scale
GRID = {"embed_weight": [0.2, 0.5, 1.0, 2.0], "engine_a_scale": [1.0, 0.5, 0.25, 0.0], "embed_window_s": [10.0, 20.0, 30.0],
        "refine_window_s": [0.0, 5.0, 10.0], "video_norm": [False, True]}
CONDITIONS = [None, *pb.CUT_PS]          # without cuts, then the synthetic cuts of each p
CHUNK = 8                                # videos per job


def cond_key(p) -> str:
    return "none" if p is None else f"{p:g}"


def decoders() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(GRID[k] for k in keys))]


def overrides(model_file: str, d: dict) -> dict:
    return {"seg_a": {"engine": "custom", "embed_weight": d["embed_weight"], "embed_window_s": d["embed_window_s"],
                      "engine_a_scale": d["engine_a_scale"], "refine_window_s": d["refine_window_s"]},
            "engine_b": {"model": model_file, "video_norm": d["video_norm"], "candidates": "dense"}}


def score_chunk(args: tuple) -> list[dict]:
    """(model id, fold name, videos) -> per decoder setting and condition the per video metrics."""
    model_id, fold, vids = args
    bench = pb.Bench()
    cfgs = [pb.bench_cfg(overrides(f"engine_b/{model_id}.{fold}.pt", d)) for d in decoders()]
    out = [{cond_key(p): [] for p in CONDITIONS} for _ in cfgs]
    for v in vids:                                       # video outer: Bench keeps one video's base timeline
        for k, cfg in enumerate(cfgs):
            for p in CONDITIONS:
                out[k][cond_key(p)].append(bench.video(v, cfg, cuts_p=p))
    return out


def rank_key(r: dict, models: list[str]):
    """Worst case frame accuracy over the cut conditions, then F1@50 and MAE in that worst
    condition, then simplicity (options off, the shipped scale 1 first, v1 before v2, the
    smaller weight, the longer window)."""
    w = r["cuts"][r["worst_p"]]
    d = r["decoder"]
    return (-round(r["worst_frame_accuracy"], 2), -(w["f1_50"] or 0),
            w["boundary_mae_s"] if w["boundary_mae_s"] is not None else 1e9, d["refine_window_s"], d["video_norm"],
            GRID["engine_a_scale"].index(d["engine_a_scale"]), models.index(r["model"]), d["embed_weight"],
            -d["embed_window_s"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.engine_b_cv_grid")
    p.add_argument("--id", required=True)
    p.add_argument("--models", required=True, help="comma separated, the v1 recipe first")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    folds = json.loads((REPO_ROOT / CV_FILE).read_text())["folds"]
    test = set(pb.split_videos("test"))
    for f in folds:
        if set(f["test_videos"]) & test:
            print("refused: a cv fold holds a benchmark test video", file=sys.stderr)
            return 2
        for m in models:
            if not (REPO_ROOT / "models" / "engine_b" / f"{m}.{f['name']}.pt").is_file():
                print(f"refused: no model engine_b/{m}.{f['name']}.pt", file=sys.stderr)
                return 2
    jobs = [(m, f["name"], f["test_videos"][i:i + CHUNK]) for m in models for f in folds
            for i in range(0, len(f["test_videos"]), CHUNK)]
    with ProcessPoolExecutor(args.workers) as ex:
        parts = list(ex.map(score_chunk, jobs))
    res_rows = []
    for m in models:
        for k, d in enumerate(decoders()):
            per = {cond_key(p): [row for (mm, _, _), part in zip(jobs, parts) if mm == m for row in part[k][cond_key(p)]]
                   for p in CONDITIONS}
            cuts = {c: pb.summarise(rows) for c, rows in per.items() if c != "none"}
            worst = min(cuts, key=lambda c: (cuts[c]["frame_accuracy"], c))
            res_rows.append({"model": m, "decoder": d, "summary": pb.summarise(per["none"]), "cuts": cuts,
                             "worst_p": worst, "worst_frame_accuracy": cuts[worst]["frame_accuracy"]})
    res_rows.sort(key=lambda r: rank_key(r, models))
    v1 = next(r for r in res_rows if r["model"] == models[0])
    v2 = next((r for r in res_rows if r["model"] != models[0]), None)
    from ..experiments.provenance import git_state
    res = {"id": args.id, "kind": "engine_b_cv_grid", "source": "public", "reference": "",
           "title": "Engine B v2 selection: decoder grid by 3 fold CV on the 152 train videos, dense candidates, worst "
                     "case over the synthetic cut conditions (rule: w10-engine-b-v2, amendment 2)",
           "rule": "experiments/w10-engine-b-v2/rule.md", "models": models, "grid": GRID, "candidates": "dense",
           "synthetic_cuts": pb.CUTS, "conditions": [cond_key(p) for p in CONDITIONS],
           "videos": sum(len(f["test_videos"]) for f in folds),
           "v1_selected": {k: v1[k] for k in ("model", "decoder", "summary", "cuts", "worst_p")},
           "v2_selected": {k: v2[k] for k in ("model", "decoder", "summary", "cuts", "worst_p")} if v2 else None,
           "rows": res_rows, "provenance": {"git": git_state(REPO_ROOT), "date_utc": utc_now(),
                                            "command": " ".join(sys.argv)}, "finished_utc": utc_now()}
    res["index"] = {"cells": [f"v1 selected: {v1['decoder']}, CV worst case frame acc {v1['worst_frame_accuracy']}"] +
                    ([f"v2 selected: {v2['model']} {v2['decoder']}, CV worst case frame acc {v2['worst_frame_accuracy']}"]
                     if v2 else []), "verdict": "measurement"}
    d = REPO_ROOT / "experiments" / args.id
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")

    def cells(r):
        dd, c = r["decoder"], r["cuts"]
        return (f"{r['model']} | {dd['embed_weight']} | {dd['engine_a_scale']} | {dd['embed_window_s']} | "
                f"{dd['refine_window_s']} | {dd['video_norm']} | {r['worst_frame_accuracy']} | "
                + " | ".join(str(c[k]["frame_accuracy"]) for k in sorted(c)) + f" | {r['summary']['frame_accuracy']} | "
                f"{c[r['worst_p']]['f1_50']} | {c[r['worst_p']]['boundary_mae_s']}")
    head = ("| rank | model | embed weight | Engine A scale | window s | refine s | video_norm | worst | "
            + " | ".join(f"p {k}" for k in sorted(res_rows[0]["cuts"])) + " | no cuts | F1@50 (worst) | MAE s (worst) |")
    lines = [f"# {args.id}", "", res["title"], "", f"Rule: {res['rule']}. {res['videos']} train videos, each decoded "
             "by the model of the fold that held its subject out, dense candidates. Frame accuracy (mean over videos) "
             f"with the synthetic cuts of each p ({pb.CUTS}) and without cuts. Ranked by the worst case over p, then "
             "F1@50 and MAE in that condition, then simplicity.", "", head, "| --- " * (head.count("|") - 1) + "|"]
    for i, r in enumerate(res_rows[:25]):
        lines.append(f"| {i + 1} | {cells(r)} |")
    lines += ["", "Best per model:", "", head.replace("| rank ", ""), "| --- " * (head.count("|") - 2) + "|"]
    for m in models:
        lines.append(f"| {cells(next(x for x in res_rows if x['model'] == m))} |")
    lines += ["", "Best per Engine A scale (all models):", "", head.replace("| rank ", ""),
              "| --- " * (head.count("|") - 2) + "|"]
    for sc in GRID["engine_a_scale"]:
        lines.append(f"| {cells(next(x for x in res_rows if x['decoder']['engine_a_scale'] == sc))} |")
    lines += ["", f"v1 selected: {v1['decoder']}.", f"v2 selected: {v2['model']} {v2['decoder']}." if v2 else "", ""]
    (d / "table.md").write_text("\n".join(lines))
    reg.write_index(REPO_ROOT / "experiments")
    print("\n".join(lines[-20:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
