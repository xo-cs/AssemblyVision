"""Engine B v2 selection by cross validation on the train split (Week 10).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.engine_b_cv_grid --id w10-engine-b-v2-cv \\
        --models w10-engine-b-v1-cv,w10-engine-b-v2-w4-c05,... [--workers 8]

CPU only, public data only, by the rule in experiments/w10-engine-b-v2/rule.md: for
every model (trained with --folds cv) and every decoder setting of the grid, each of
the 152 train videos is decoded by the model of the fold that held its subject out
(experiments/datasets/ha4m_train_cv.json), and scored on the macro phase benchmark.
Settings are ranked by the mean frame accuracy over the 152 videos, ties by F1@50,
then MAE, then simplicity. Writes experiments/<id>/ (results.json, table.md) and
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
GRID = {"embed_candidates": [False, True], "embed_weight": [0.2, 0.4, 0.8], "embed_window_s": [10.0, 20.0, 30.0],
        "refine_window_s": [0.0, 5.0, 10.0], "video_norm": [False, True]}


def decoders() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(GRID[k] for k in keys))]


def overrides(model_file: str, d: dict) -> dict:
    return {"seg_a": {"engine": "custom", "embed_weight": d["embed_weight"], "embed_window_s": d["embed_window_s"],
                      "embed_candidates": d["embed_candidates"], "refine_window_s": d["refine_window_s"]},
            "engine_b": {"model": model_file, "video_norm": d["video_norm"]}}


def score_fold(args: tuple) -> list[dict]:
    """(model id, fold name, videos) -> per decoder setting the per video metrics."""
    model_id, fold, vids = args
    bench = pb.Bench()
    out = []
    for d in decoders():
        cfg = pb.bench_cfg(overrides(f"engine_b/{model_id}.{fold}.pt", d))
        out.append({"decoder": d, "rows": [bench.video(v, cfg) for v in vids]})
    return out


def rank_key(r: dict, models: list[str]):
    s, d = r["summary"], r["decoder"]
    return (-round(s["frame_accuracy"], 2), -(s["f1_50"] or 0), s["boundary_mae_s"] if s["boundary_mae_s"] is not None
            else 1e9, d["embed_candidates"], d["refine_window_s"], d["video_norm"], models.index(r["model"]),
            d["embed_weight"], -d["embed_window_s"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.engine_b_cv_grid")
    p.add_argument("--id", required=True)
    p.add_argument("--models", required=True, help="comma separated, the v1 recipe first")
    p.add_argument("--workers", type=int, default=8)
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
    jobs = [(m, f["name"], f["test_videos"]) for m in models for f in folds]
    with ProcessPoolExecutor(args.workers) as ex:
        parts = list(ex.map(score_fold, jobs))
    res_rows = []
    for m in models:
        for k, d in enumerate(decoders()):
            per = [row for (mm, _, _), part in zip(jobs, parts) if mm == m for row in part[k]["rows"]]
            res_rows.append({"model": m, "decoder": d, "summary": pb.summarise(per),
                             "per_video": [{"video": r["video"], "frame_accuracy": r["metrics"]["frame_accuracy"]}
                                           for r in per]})
    res_rows.sort(key=lambda r: rank_key(r, models))
    v1 = next(r for r in res_rows if r["model"] == models[0])
    v2 = next((r for r in res_rows if r["model"] != models[0]), None)
    from ..experiments.provenance import git_state
    res = {"id": args.id, "kind": "engine_b_cv_grid", "source": "public", "reference": "",
           "title": "Engine B v2 selection: decoder grid by 3 fold CV on the 152 train videos (rule: w10-engine-b-v2)",
           "rule": "experiments/w10-engine-b-v2/rule.md", "models": models, "grid": GRID,
           "videos": sum(len(f["test_videos"]) for f in folds),
           "v1_selected": {k: v1[k] for k in ("model", "decoder", "summary")},
           "v2_selected": {k: v2[k] for k in ("model", "decoder", "summary")} if v2 else None,
           "rows": res_rows, "provenance": {"git": git_state(REPO_ROOT), "date_utc": utc_now(),
                                            "command": " ".join(sys.argv)}, "finished_utc": utc_now()}
    res["index"] = {"cells": [f"v1 selected: {v1['decoder']}, CV frame acc {v1['summary']['frame_accuracy']}"] +
                    ([f"v2 selected: {v2['model']} {v2['decoder']}, CV frame acc {v2['summary']['frame_accuracy']}"]
                     if v2 else []), "verdict": "measurement"}
    d = REPO_ROOT / "experiments" / args.id
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    lines = [f"# {args.id}", "", res["title"], "", f"Rule: {res['rule']}. {res['videos']} train videos, each decoded "
             "by the model of the fold that held its subject out. Ranked by mean frame accuracy, then F1@50, MAE, "
             "simplicity.", "",
             "| rank | model | candidates | weight | window s | refine s | video_norm | frame acc | median | edit | F1@10 | "
             "F1@25 | F1@50 | MAE s |", "| --- " * 14 + "|"]
    for i, r in enumerate(res_rows[:25]):
        s, dd = r["summary"], r["decoder"]
        lines.append(f"| {i + 1} | {r['model']} | {dd['embed_candidates']} | {dd['embed_weight']} | {dd['embed_window_s']} | "
                     f"{dd['refine_window_s']} | {dd['video_norm']} | {s['frame_accuracy']} | {s['frame_accuracy_median']} | "
                     f"{s['edit']} | {s['f1_10']} | {s['f1_25']} | {s['f1_50']} | {s['boundary_mae_s']} |")
    lines += ["", "Best per model:", "", "| model | decoder | frame acc | F1@50 | MAE s |", "| --- | --- | --- | --- | --- |"]
    for m in models:
        r = next(x for x in res_rows if x["model"] == m)
        lines.append(f"| {m} | {r['decoder']} | {r['summary']['frame_accuracy']} | {r['summary']['f1_50']} | "
                     f"{r['summary']['boundary_mae_s']} |")
    lines += ["", f"v1 selected: {v1['decoder']}.", f"v2 selected: {v2['model']} {v2['decoder']}." if v2 else "", ""]
    (d / "table.md").write_text("\n".join(lines))
    reg.write_index(REPO_ROOT / "experiments")
    print("\n".join(lines[-12:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
