"""Engine B v3 (Week 11): the v2 recipe (MS TCN, boundary BCE, action head, supervised
contrastive phase objective) on pose plus visual features.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.train_engine_b_v3 --id w11-engine-b-v3 --folds cv --gpu 0     # 3 CV models
    python -m assemblyvision.tools.train_engine_b_v3 --id w11-engine-b-v3 --folds all --gpu 0    # all train videos

Public data only. Input per model step (engine_b.model_fps): the pose features of
datasets/engine_b, the DINOv2 ViT S/14 embedding and a visual mask channel
(boundary_tcn.with_visual).
- HA4M (pose only, visual masked): the Kinect COCO 17 cache with the YOLO like
  augmentation, as v1 and v2; boundaries from its action changes; phases and actions
  of HA4M.
- IKEA ASM (pose and visual): the YOLO26 x keypoints of the subject track of each
  stored run (tools/ikea_runs) and its visual.npz; boundaries from the atomic action
  changes; macro phases (experiments/datasets/ikea_asm_macro_phases.md); 33 actions.
Phase labels are distinct per dataset and furniture type (the contrastive loss never
joins phases of different tasks); the action head has 12 + 33 outputs. Only the
furniture types of the selection set (ikea_bench.SELECTION) are used.
Folds: "cv" trains cv1..cv3, each on the HA4M train subjects outside HA4M cv k and the
IKEA training videos outside IKEA cv k; "all" trains on every train video of both
(the test parts are never used). Models: models/engine_b/<id>.<fold>.pt and <id>.pt.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from ..analysis.segmentation import boundary_tcn as bt
from ..config import Config
from ..datasets import engine_b as eb
from ..datasets import ha4m_phases as hp
from ..datasets import ikea_phases as ip
from ..datasets.ha4m import RGB_SIZE, HA4MLoader
from ..experiments import registry as reg
from ..storage.files import REPO_ROOT, utc_now
from . import ikea_bench as ib
from .run_pose import parse_sets
from .train_engine_b import CV_FILE, prepare, train, video_arrays

log = logging.getLogger("assemblyvision.engine_b_v3")
HA4M_ACTIONS = 12
IKEA_ACTIONS = sorted({lab for groups in ip.MACRO_PHASES.values() for _, labs in groups for lab in labs})
FURNITURE = list(ip.MACRO_PHASES)
PHASE_OFFSET = {"ha4m": 0}
_o = len(hp.NAMES)
for _f in FURNITURE:
    PHASE_OFFSET[_f] = _o
    _o += len(ip.names(_f))
N_ACTIONS = HA4M_ACTIONS + len(IKEA_ACTIONS)


def ha4m_item(d: dict, cfg: Config, train: bool, rng) -> dict:
    it = prepare(d, cfg, train, rng, float(RGB_SIZE[1]))
    it["x"] = bt.with_visual(it["x"], None, None)
    return it


def ikea_item(vid: str, run_id: str, cfg: Config) -> dict | None:
    """Model inputs of one IKEA run at model_fps (subject track keypoints, visual.npz, labels)."""
    from ..analysis import visual
    from ..kts import read_kts
    e, bd = cfg["engine_b"], cfg["boundaries"]
    root = REPO_ROOT / "data/public/runs" / run_id
    manifest = json.loads((root / "manifest.json").read_text())
    cfr = manifest["ingest"]["cfr"]
    fps, n, image_h = float(cfr["fps"]), int(cfr["frame_count"]), float(cfr["height"])
    tracks = [t for t in manifest.get("tracking", {}).get("tracks", []) if t.get("is_subject")]
    if not tracks:
        return None
    series, _ = read_kts(root / max(tracks, key=lambda t: t.get("frames_present", 0))["kts_path"])
    f0 = int(series.frame_idx[0]) if len(series.frame_idx) else 0
    T = min(len(series.present), n - f0)
    idx = bt.sample_index(T, fps, e["model_fps"])
    pres = np.asarray(series.present[:T], bool)[idx]
    xy = np.where(pres[:, None, None], series.body_xy[:T][idx], np.nan)
    conf = np.where(pres[:, None], np.nan_to_num(series.body_conf[:T][idx]), 0.0)
    x = eb.features(xy, conf, image_h)
    vx, vok = visual.to_frames(visual.load(root / "visual.npz"), n, fps)
    x = bt.with_visual(x, vx[f0 + idx], vok[f0 + idx])
    furn = vid.split("/")[0]
    seg = ip.load_annotations(ib.ROOT)[vid]["annotation"]
    n25 = int(round(n / fps * ip.FPS)) + 1
    lab = ip.frame_labels(seg, n25)
    phase25 = ip.phase_index(furn, lab)
    t = (f0 + idx) / fps
    k = np.clip(np.round(t * ip.FPS).astype(np.int64), 0, n25 - 1)
    aid = np.array([IKEA_ACTIONS.index(x) + 1 if x in IKEA_ACTIONS else 0 for x in lab], np.int64)
    labels25 = np.stack([aid, np.zeros_like(aid)], axis=1)
    changes = bt.change_times(labels25, ip.FPS, bd["ignore_dont_care"], 0, np.ones(n25, bool))
    act = aid[k] - 1
    act = np.where(act >= 0, act + HA4M_ACTIONS, -1)
    ph = phase25[k]
    ph = np.where(ph >= 0, ph + PHASE_OFFSET[furn], -1)
    return {"x": x, "y": bt.soft_targets(t, changes, bd["tol_s"]), "m": pres.astype(np.float32), "c": act.astype(np.int64),
            "p": ph.astype(np.int64), "t": t, "changes": changes}


def fold_lists(fold: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """(HA4M train, HA4M held out, IKEA train, IKEA held out) of a fold; held out is empty for "all"."""
    ha = json.loads((REPO_ROOT / "experiments/datasets" / CV_FILE).read_text())["folds"]
    ik = json.loads((REPO_ROOT / ib.FOLDS).read_text())["folds"]
    ikea_train = ib.split_videos("train")
    if fold == "all":
        h = sorted({v for f in ha for v in f["train_videos"]} | {v for f in ha for v in f["test_videos"]})
        return h, [], ikea_train, []
    hf = next(f for f in ha if f["name"] == fold)
    held = set(next(f for f in ik if f["name"] == fold)["test_videos"])
    return hf["train_videos"], hf["test_videos"], [v for v in ikea_train if v not in held], \
        [v for v in ikea_train if v in held]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.train_engine_b_v3")
    p.add_argument("--id", required=True)
    p.add_argument("--folds", choices=("cv", "all"), required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu >= 0 else ""
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    over = parse_sets(args.set)
    cfg = Config.load(overrides=over)
    e = cfg["engine_b"]
    import torch
    device = "cuda" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    lock = None
    if device == "cuda":
        (REPO_ROOT / ".locks").mkdir(exist_ok=True)
        lock = open(REPO_ROOT / ".locks" / f"gpu{args.gpu}.lock", "a+")
        fcntl.flock(lock, fcntl.LOCK_EX)
    cache = eb.cache_dir(HA4MLoader(cfg, repo_root=REPO_ROOT).root)
    runs = ib.runs()
    folds = ["cv1", "cv2", "cv3"] if args.folds == "cv" else ["all"]
    res = {"id": args.id, "kind": "engine_b_v3", "source": "public", "reference": "",
           "title": f"Engine B v3 on pose plus visual features ({args.folds})", "settings": {"engine_b": dict(e)},
           "datasets": {"ha4m": "Kinect COCO 17 cache, visual masked", "ikea_asm": list(ib.SELECTION)},
           "n_actions": N_ACTIONS, "phase_offsets": PHASE_OFFSET, "folds": {}}
    for fold in folds:
        t0 = time.monotonic()
        h_tr, h_te, i_tr, i_te = fold_lists(fold)
        rng = np.random.default_rng(e["train_seed"])
        h_raw = [video_arrays(cache, v) for v in h_tr]
        i_items = [it for it in (ikea_item(v, runs[v], cfg) for v in i_tr) if it is not None]
        log.info("%s: HA4M %d train videos, IKEA %d train videos", fold, len(h_raw), len(i_items))

        def items_fn(r):
            return [ha4m_item(d, cfg, True, r) for d in h_raw] + i_items
        first = items_fn(rng)
        model, info = train(cfg, first, [], device, float(RGB_SIZE[1]), progress=log.info, items_fn=items_fn,
                            n_actions=N_ACTIONS)
        name = f"{args.id}.{fold}.pt" if fold != "all" else f"{args.id}.pt"
        sha = bt.save_model(REPO_ROOT / "models/engine_b" / name, model, bt.model_settings(cfg, N_ACTIONS, visual=True),
                            {"id": args.id, "fold": fold, "ha4m_train": h_tr, "ikea_train": i_tr,
                             "contrast_weight": e["contrast_weight"], "contrast_window_s": e["contrast_window_s"]})
        res["folds"][fold] = {"ha4m_train": len(h_tr), "ha4m_held_out": len(h_te), "ikea_train": len(i_items),
                              "ikea_held_out": len(i_te), "model": f"engine_b/{name}", "sha256": sha,
                              "training": info, "seconds": round(time.monotonic() - t0, 1)}
        log.info("%s: wrote %s (%.0f s)", fold, name, time.monotonic() - t0)
    res["finished_utc"] = utc_now()
    res["index"] = {"cells": [f"{f}: HA4M {r['ha4m_train']} + IKEA {r['ikea_train']} train videos, "
                              f"{r['seconds']} s" for f, r in res["folds"].items()], "verdict": "measurement"}
    d = REPO_ROOT / "experiments" / args.id
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    (d / "table.md").write_text(f"# {args.id}\n\n{res['title']}\n\n" + "\n".join(f"- {c}" for c in res["index"]["cells"])
                                + "\n")
    reg.write_index(REPO_ROOT / "experiments")
    if lock is not None:
        lock.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
