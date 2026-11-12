"""Visual Engine B selection by CV on the train parts (Week 11, rule experiments/w11-visual-engine-b/rule.md).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.visual_cv_grid --id w11-visual-engine-b-cv [--workers 6]

CPU only, public data only. Three selection sets: HA4M macro phases (152 train videos,
synthetic cuts p 0.5, pose only), IKEA ASM Kallax Shelf Drawer and Lack Coffee Table (official
training subset, stored runs with visual.npz). Every decoder setting of GRID with every
model; each held out video is decoded by the model of its fold. Ranked by the worst
case over the sets of the mean frame accuracy. Writes experiments/<id>/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from ..experiments import registry as reg
from ..storage.files import REPO_ROOT, utc_now
from . import ikea_bench as ib
from . import phase_bench as pb

HA4M_CV = "experiments/datasets/ha4m_train_cv.json"
MODELS = {   # model: (HA4M fold model pattern, IKEA model pattern); {fold} is the CV fold
    "v1": ("engine_b/w10-engine-b-v1-cv.{fold}.pt", "engine_b/w10-engine-b-v1.pt"),
    "v2": ("engine_b/w10-engine-b-v2-w4-c10.{fold}.pt", "engine_b/w10-engine-b-v2.pt"),
    "v3": ("engine_b/w11-engine-b-v3.{fold}.pt", "engine_b/w11-engine-b-v3.{fold}.pt"),
}
GRID = {"visual_weight": [0.0, 0.5, 1.0, 2.0], "visual_window_s": [40.0, 20.0, 10.0], "embed_weight": [0.2, 0.0, 1.0],
        "embed_window_s": [30.0, 10.0], "engine_a_scale": [1.0, 0.5, 0.25], "intro_detect": [False, True]}
INTRO = {"intro_k": 1.0, "intro_sustain_s": 20.0, "intro_active_share": 0.5}
SETS = ("ha4m", "Kallax_Shelf_Drawer", "Lack_Coffee_Table")
CHUNK = 6


def decoders() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]


def overrides(model_file: str, d: dict) -> dict:
    return {"seg_a": {"engine": "custom", "engine_b_weight": 0.0, "embed_weight": d["embed_weight"],
                      "embed_window_s": d["embed_window_s"], "visual_weight": d["visual_weight"],
                      "visual_window_s": d["visual_window_s"], "engine_a_scale": d["engine_a_scale"],
                      "refine_window_s": 0.0, "intro_detect": d["intro_detect"], **INTRO},
            "engine_b": {"model": model_file, "video_norm": False, "candidates": "dense"}}


def score_chunk(job: tuple) -> list[list[float]]:
    """(model, set, fold or None, videos) -> [decoder][video] frame accuracy, F1@50, MAE."""
    model, sset, fold, vids = job
    out = [[] for _ in decoders()]
    if sset == "ha4m":
        bench = pb.Bench()
        cfgs = [pb.bench_cfg(overrides(MODELS[model][0].format(fold=fold), d)) for d in decoders()]
        for v in vids:
            for k, cfg in enumerate(cfgs):
                m = bench.video(v, cfg, cuts_p=0.5)["metrics"]
                out[k].append([m["frame_accuracy"], m.get("f1_50"), m.get("boundary_mae_s")])
    else:
        bench = ib.IkeaBench()
        for v in vids:
            f = ib.fold_of(v)
            for k, d in enumerate(decoders()):
                m = bench.video(v, overrides(MODELS[model][1].format(fold=f), d))["metrics"]
                out[k].append([m["frame_accuracy"], m.get("f1_50"), m.get("boundary_mae_s")])
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def rank_key(r: dict):
    d = r["decoder"]
    return (-round(r["worst"], 2), -round(r["mean"], 2), d["intro_detect"], d["visual_weight"],
            GRID["embed_weight"].index(d["embed_weight"]), GRID["engine_a_scale"].index(d["engine_a_scale"]),
            list(MODELS).index(r["model"]), -d["visual_window_s"], -d["embed_window_s"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.visual_cv_grid")
    p.add_argument("--id", required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--models", default=",".join(MODELS))
    args = p.parse_args(argv)
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    models = args.models.split(",")
    for m in models:
        for f in ("cv1", "cv2", "cv3"):
            for pat in MODELS[m]:
                if not (REPO_ROOT / "models" / pat.format(fold=f)).is_file():
                    print(f"refused: no model {pat.format(fold=f)}", file=sys.stderr)
                    return 2
    ha = json.loads((REPO_ROOT / HA4M_CV).read_text())["folds"]
    test = set(pb.split_videos("test"))
    ikea_train = ib.split_videos("train")
    if set(ikea_train) & set(ib.split_videos("test")) or any(set(f["test_videos"]) & test for f in ha):
        print("refused: a train list holds a test video", file=sys.stderr)
        return 2
    ikea = {s: [v for v in ikea_train if v.startswith(s + "/")] for s in SETS[1:]}
    jobs = []
    for m in models:
        for f in ha:
            jobs += [(m, "ha4m", f["name"], f["test_videos"][i:i + CHUNK]) for i in range(0, len(f["test_videos"]), CHUNK)]
        for s, vids in ikea.items():
            jobs += [(m, s, None, vids[i:i + CHUNK]) for i in range(0, len(vids), CHUNK)]
    with ProcessPoolExecutor(args.workers) as ex:
        parts = list(ex.map(score_chunk, jobs))
    rows = []
    for m in models:
        for k, d in enumerate(decoders()):
            per = {s: [x for j, part in zip(jobs, parts) if j[0] == m and j[1] == s for x in part[k]] for s in SETS}
            summ = {s: {"frame_accuracy": mean([x[0] for x in v]), "f1_50": mean([x[1] for x in v]),
                        "boundary_mae_s": mean([x[2] for x in v]), "videos": len(v)} for s, v in per.items()}
            fa = [summ[s]["frame_accuracy"] for s in SETS]
            rows.append({"model": m, "decoder": d, "sets": summ, "worst": min(fa), "mean": round(sum(fa) / 3, 2)})
    rows.sort(key=rank_key)
    best = rows[0]
    from ..experiments.provenance import git_state
    res = {"id": args.id, "kind": "visual_cv_grid", "source": "public", "reference": "",
           "title": "Visual Engine B selection: CV on HA4M (p 0.5) and IKEA ASM Kallax and Lack Coffee Table "
                    "(rule w11-visual-engine-b)", "rule": "experiments/w11-visual-engine-b/rule.md",
           "models": {m: MODELS[m] for m in models}, "grid": GRID, "intro": INTRO, "ikea_videos": ikea,
           "selected": best, "rows": rows,
           "provenance": {"git": git_state(REPO_ROOT), "date_utc": utc_now(), "command": " ".join(sys.argv)},
           "finished_utc": utc_now()}
    res["index"] = {"cells": [f"selected: {best['model']} {best['decoder']}, worst {best['worst']} "
                              + ", ".join(f"{s} {best['sets'][s]['frame_accuracy']}" for s in SETS)],
                    "verdict": "measurement"}
    out = REPO_ROOT / "experiments" / args.id
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")

    def line(r):
        d = r["decoder"]
        return (f"| {r['model']} | {d['visual_weight']} | {d['visual_window_s']} | {d['embed_weight']} | "
                f"{d['embed_window_s']} | {d['engine_a_scale']} | {d['intro_detect']} | {r['worst']} | "
                + " | ".join(str(r["sets"][s]["frame_accuracy"]) for s in SETS) + f" | {r['mean']} |")
    head = ["| model | visual w | visual win s | embed w | embed win s | A scale | intro | worst | HA4M p0.5 | "
            "Kallax | Lack Coffee | mean |", "| --- " * 12 + "|"]
    lines = [f"# {args.id}", "", res["title"], "", f"Videos: HA4M {sum(len(f['test_videos']) for f in ha)}, "
             + ", ".join(f"{s} {len(v)}" for s, v in ikea.items()) + ". Frame accuracy, mean over videos, CV.", "",
             *head, *map(line, rows[:25]), "", "Best per model:", "", *head]
    lines += [line(next(r for r in rows if r["model"] == m)) for m in models]
    lines += ["", "Best per visual weight:", "", *head]
    lines += [line(next(r for r in rows if r["decoder"]["visual_weight"] == w)) for w in GRID["visual_weight"]]
    lines += ["", f"Selected (rule): {best['model']} {best['decoder']}.", ""]
    (out / "table.md").write_text("\n".join(lines))
    reg.write_index(REPO_ROOT / "experiments")
    print("\n".join(lines[-30:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
