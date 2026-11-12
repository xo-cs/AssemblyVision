"""Item 8: the optional intro and outro (seg_a.other_ends) on synthetic intros (Week 10).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.other_ends_cv --id w10-other-ends-cv               # train split, CV
    python -m assemblyvision.tools.other_ends_cv --id w10-other-ends-test --split test --strength 0.3

CPU only, public data only, by amendment 3 of experiments/w10-engine-b-v2/rule.md. Every
video is decoded twice, plain and with a synthetic still intro of 5 to 30 s (seeded per
video), with the v2 selected decoder of the CV grid plus seg_a.other_ends. On the train
split each video uses the CV model of the fold that held its subject out, and the bias
seg_a.other_min_strength is chosen by the mean frame accuracy over all of them (ties: the
larger bias). On the test split (once) the final model is scored with the chosen bias.
Writes experiments/<id>/ (results.json, table.md) and INDEX.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")    # CPU only

from ..experiments import registry as reg
from ..storage.files import REPO_ROOT, utc_now
from . import engine_b_cv_grid as grid
from . import phase_bench as pb

GRID_FROM = "w10-engine-b-v2-cv"
STRENGTHS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
INTRO_S = (5.0, 30.0)
SEED = 42
CHUNK = 8


def intro_length(vid: str) -> float:
    """Seeded per video (its position among all 217 benchmark videos)."""
    vids = sorted(pb.split_videos("train") + pb.split_videos("test"))
    rng = np.random.default_rng([SEED, vids.index(vid)])
    return round(float(rng.uniform(*INTRO_S)), 1)


def overrides(model_file: str, decoder: dict, strength: float | None) -> dict:
    over = grid.overrides(model_file, decoder)
    over["seg_a"].update({"other_ends": strength is not None,
                          **({} if strength is None else {"other_min_strength": strength, "other_max_s": 30.0})})
    return over


def score_chunk(args: tuple) -> dict:
    """(model file, decoder, videos) -> {strength key: {"plain": [...], "intro": [...]}}."""
    model_file, decoder, strengths, vids = args
    bench = pb.Bench()
    cfgs = {key(s): pb.bench_cfg(overrides(model_file, decoder, s)) for s in strengths}
    out = {k: {"plain": [], "intro": []} for k in cfgs}
    for v in vids:
        n = sorted(pb.split_videos("train") + pb.split_videos("test")).index(v)
        for k, cfg in cfgs.items():
            out[k]["plain"].append(bench.video(v, cfg))
        for k, cfg in cfgs.items():
            out[k]["intro"].append(bench.video(v, cfg, intro_s=intro_length(v), seed=n))
    return out


def key(s: float | None) -> str:
    return "off" if s is None else f"{s:g}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.other_ends_cv")
    p.add_argument("--id", required=True)
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--strength", type=float, help="test split: the bias chosen on the train split")
    p.add_argument("--model", default="w10-engine-b-v2", help="test split: the final model id")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    sel = json.loads((REPO_ROOT / "experiments" / GRID_FROM / "results.json").read_text())["v2_selected"]
    decoder = sel["decoder"]
    if args.split == "train":
        strengths = [None, *STRENGTHS]
        folds = json.loads((REPO_ROOT / grid.CV_FILE).read_text())["folds"]
        jobs = [(f"engine_b/{sel['model']}.{f['name']}.pt", decoder, strengths, f["test_videos"][i:i + CHUNK])
                for f in folds for i in range(0, len(f["test_videos"]), CHUNK)]
    else:
        if args.strength is None:
            print("refused: the test split needs --strength (chosen on the train split)", file=sys.stderr)
            return 2
        strengths = [None, args.strength]
        vids = pb.split_videos("test")
        jobs = [(f"engine_b/{args.model}.pt", decoder, strengths, vids[i:i + CHUNK]) for i in range(0, len(vids), CHUNK)]
    with ProcessPoolExecutor(args.workers) as ex:
        parts = list(ex.map(score_chunk, jobs))
    rows = []
    for s in strengths:
        plain = [r for part in parts for r in part[key(s)]["plain"]]
        intro = [r for part in parts for r in part[key(s)]["intro"]]
        rows.append({"other_min_strength": key(s), "plain": pb.summarise(plain), "intro": pb.summarise(intro),
                     "all": pb.summarise(plain + intro)})
    cand = [r for r in rows if r["other_min_strength"] != "off"]
    best = max(cand, key=lambda r: (round(r["all"]["frame_accuracy"], 2), float(r["other_min_strength"])))
    from ..experiments.provenance import git_state
    res = {"id": args.id, "kind": "other_ends_cv", "source": "public", "reference": "", "split": args.split,
           "title": f"Item 8: optional intro and outro on synthetic intros ({args.split} split; rule w10-engine-b-v2, "
                    "amendment 3)",
           "rule": "experiments/w10-engine-b-v2/rule.md", "decoder": decoder,
           "model": sel["model"] + " (CV)" if args.split == "train" else args.model,
           "intro_s": list(INTRO_S), "seed": SEED, "rows": rows, "chosen": best["other_min_strength"],
           "provenance": {"git": git_state(REPO_ROOT), "date_utc": utc_now(), "command": " ".join(sys.argv)},
           "finished_utc": utc_now()}
    res["index"] = {"cells": [f"{r['other_min_strength']}: all {r['all']['frame_accuracy']}, plain "
                              f"{r['plain']['frame_accuracy']}, intro {r['intro']['frame_accuracy']}" for r in rows]
                    + ([f"chosen bias {best['other_min_strength']}"] if args.split == "train" else []),
                    "verdict": "measurement"}
    d = REPO_ROOT / "experiments" / args.id
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    lines = [f"# {args.id}", "", res["title"], "", f"Decoder: {res['model']} with {decoder}, plus seg_a.other_ends. "
             f"Every video plain and with a still intro of {INTRO_S[0]:g} to {INTRO_S[1]:g} s (seeded), no cuts.", "",
             "| other_min_strength | all | plain | intro | intro F1@50 | intro MAE s |", "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['other_min_strength']} | {r['all']['frame_accuracy']} | {r['plain']['frame_accuracy']} | "
                     f"{r['intro']['frame_accuracy']} | {r['intro']['f1_50']} | {r['intro']['boundary_mae_s']} |")
    if args.split == "train":
        lines += ["", f"Chosen bias (rule): {best['other_min_strength']}."]
    lines.append("")
    (d / "table.md").write_text("\n".join(lines))
    reg.write_index(REPO_ROOT / "experiments")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
