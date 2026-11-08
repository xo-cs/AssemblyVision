"""Engine B firing rate on the HA4M test subjects (Week 10, step 1 of the pool_s plan).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.engine_b_firing --id w10-engine-b-firing --model w10-engine-b-v0 \\
        [--pool 0,2,5,10,30,60]

CPU only, public data only. For the held out subjects of the cross subject split
(the model's own test videos), the trained model scores every video at
engine_b.model_fps; then per pool window (s) the probability is pooled by the
centred moving average of Engine A (engine_b.pool_s) and divided by its maximum
in the video (seg_a.engine_b_norm = "max"), as the DP sees it. Reported per window:
peaks per minute (engine_b.peak_threshold, at least max(peak_min_gap_s, pool_s)
apart; the same rule as the rate in Engine A's report) against the true
(action, repetition) changes per minute, and the boundary F1 of those peaks at
engine_b.eval_tols_s and 2 s (information: pooling trades short boundaries for
lasting ones). The model's settings (hip calibration) come from its experiment.
Writes experiments/<id>/ (config.toml, results.json, table.md) and INDEX.md.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tomllib
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")    # CPU only

from ..analysis.segmentation import boundary_tcn as bt
from ..config import Config
from ..datasets import engine_b as eb
from ..datasets.ha4m import RGB_SIZE, HA4MLoader
from ..experiments import registry as reg
from ..storage.files import REPO_ROOT, utc_now
from .run_pose import parse_sets
from .train_engine_b import prepare, video_arrays

log = logging.getLogger("assemblyvision.engine_b_firing")


def firing(probs: list[np.ndarray], items: list[dict], cfg: Config, pool_s: float, tols: list[float]) -> dict:
    e = cfg["engine_b"]
    fps = e["model_fps"]
    gap = int(round(max(e["peak_min_gap_s"], pool_s) * fps))
    vids, n_peaks = [], 0
    for p, it in zip(probs, items):
        q = bt.pool(p, int(round(pool_s * fps)))
        if q.max() > 0:
            q = q / q.max()
        k = bt.peaks(q, e["peak_threshold"], gap)
        n_peaks += len(k)
        vids.append((it["t"][k], it["changes"]))
    minutes = sum(len(it["t"]) for it in items) / fps / 60.0
    f1 = bt.boundary_f1(vids, tols)
    return {"pool_s": pool_s, "peaks": n_peaks, "peaks_per_min": round(n_peaks / minutes, 2),
            "f1": {k: v["f1"] for k, v in f1.items()},
            "precision": {k: v["precision"] for k, v in f1.items()}, "recall": {k: v["recall"] for k, v in f1.items()}}


def main(argv: list[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.engine_b_firing")
    p.add_argument("--id", required=True)
    p.add_argument("--model", required=True, help="the training experiment, e.g. w10-engine-b-v0")
    p.add_argument("--pool", default="0,2,5,10,30,60", help="pool windows (s)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    reg._check_id(args.id)
    exp = repo_root / "experiments" / args.model
    if not (exp / "results.json").is_file():
        print(f"refused: no experiment {args.model}", file=sys.stderr)
        return 2
    tres = json.loads((exp / "results.json").read_text())
    stored = tomllib.loads((exp / "config.toml").read_text())
    cfg = Config.load(overrides={"engine_b": stored["engine_b"], "boundaries": stored["boundaries"],
                                 **parse_sets(args.set)})
    model, _ = bt.load_model(repo_root / tres["model"]["file"])
    offsets = eb.load_calibration(repo_root, cfg["engine_b"]["calibration_from"]) \
        if cfg["engine_b"]["hip_calibration"] else None
    cache = eb.cache_dir(HA4MLoader(cfg, repo_root=repo_root).root)
    rng = np.random.default_rng(0)
    items = [prepare(video_arrays(cache, v), cfg, False, rng, float(RGB_SIZE[1]), offsets)
             for v in tres["data"]["test_video_ids"]]
    probs = [bt.predict(model, it["x"]).astype(np.float64) for it in items]
    minutes = sum(len(it["t"]) for it in items) / cfg["engine_b"]["model_fps"] / 60.0
    n_true = sum(len(it["changes"]) for it in items)
    tols = sorted(set(list(cfg["engine_b"]["eval_tols_s"]) + [2.0]))
    rows = [firing(probs, items, cfg, float(s), tols) for s in args.pool.split(",")]
    res = {"id": args.id, "kind": "engine_b_firing", "source": "public",
           "title": f"Engine B firing rate on the HA4M test subjects ({args.model})", "reference": args.model,
           "data": {"videos": len(items), "minutes": round(minutes, 2), "mean_video_s": round(60 * minutes / len(items), 1),
                    "true_changes": n_true, "true_changes_per_min": round(n_true / minutes, 2),
                    "model": tres["model"], "hip_calibration": cfg["engine_b"]["hip_calibration"]},
           "rows": rows,
           "provenance": {"date_utc": utc_now(), "command": " ".join(sys.argv), "data_area": "public"},
           "finished_utc": utc_now()}
    raw = rows[0]
    res["index"] = {"cells": [f"true {res['data']['true_changes_per_min']} changes/min; peaks/min "
                              + ", ".join(f"pool {r['pool_s']:g} s: {r['peaks_per_min']}" for r in rows)],
                    "verdict": "measurement"}
    head = ["pool_s", "peaks", "peaks / min", *[f"F1@{t:g}" for t in tols], *[f"P@{t:g}" for t in tols]]
    lines = [f"# {args.id}: {res['title']}", "",
             f"{len(items)} videos of the held out subjects ({res['data']['minutes']} min, mean "
             f"{res['data']['mean_video_s']} s a video), {n_true} true (action, repetition) changes = "
             f"**{res['data']['true_changes_per_min']} per minute**. Probability pooled as Engine A pools it "
             f"(engine_b.pool_s), divided by its maximum per video, peaks at {cfg['engine_b']['peak_threshold']}, "
             f"at least max({cfg['engine_b']['peak_min_gap_s']} s, pool_s) apart. Model {tres['model']['file']} "
             f"(hip calibration {'on' if offsets is not None else 'off'}).", "",
             "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    for r in rows:
        lines.append(f"| {r['pool_s']:g} | {r['peaks']} | {r['peaks_per_min']} | "
                     + " | ".join(str(r["f1"][f"{t:g}"]) for t in tols) + " | "
                     + " | ".join(str(r["precision"][f"{t:g}"]) for t in tols) + " |")
    lines += ["", f"Pool 0 is the raw model ({raw['peaks_per_min']} peaks/min). A pooled window much longer than "
                  "the video (HA4M videos are short) leaves about one peak per video.", ""]
    d = repo_root / "experiments" / args.id
    d.mkdir(parents=True, exist_ok=True)
    meta = {"id": args.id, "title": res["title"], "kind": "engine_b_firing", "source": "public",
            "videos": [str(cache.relative_to(repo_root))], "repeats": 1, "reference": args.model, "gpu": -1,
            "variant": cfg["pose"]["variant"]}
    (d / "config.toml").write_text(reg.config_text(meta, cfg.snapshot()))
    (d / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    (d / "table.md").write_text("\n".join(lines))
    reg.write_index(repo_root / "experiments")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
