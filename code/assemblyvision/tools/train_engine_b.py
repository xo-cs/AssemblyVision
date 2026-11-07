"""Train Engine B, the boundary scorer (Week 10, analysis/segmentation/boundary_tcn.py).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.train_engine_b --id w10-engine-b-v0 --gpu 0      # GPU job, ask first
        [--set engine_b.KEY=VALUE ...] [--cpu] [--note TEXT]

Data: the Engine B cache of HA4M (datasets/engine_b.py, Kinect 2D colour joints
mapped to COCO 17), split by the fixed subject split
experiments/datasets/ha4m_subject_split.json: train on the train subjects, report
on the held out test subjects. The model is trained for engine_b.epochs and
the last epoch is kept; the test subjects are never used to choose anything
(they are scored once, after training). Augmentation (engine_b.augment) is on
for training only.

Writes the weights to models/engine_b/<id>.pt (not tracked; its sha256 is in
the results) and the experiment experiments/<id>/ (config.toml with every
setting, results.json with the boundary precision, recall and F1 per
tolerance engine_b.eval_tols_s, overall and per lab, and the provenance;
table.md) plus INDEX.md. A GPU run takes ~/AssemblyVision/.locks/gpu<N>.lock
and refuses to start while it is held. --cpu trains on the CPU (tests, smoke runs).
Exit codes: 0, 1 error, 2 refused (GPU busy, no split, no data).
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
from ..datasets.ha4m import RGB_SIZE, HA4MLoader
from ..errors import AvError
from ..experiments import registry as reg
from ..storage.files import REPO_ROOT, utc_now
from .run_pose import _fail, parse_sets

log = logging.getLogger("assemblyvision.train_engine_b")


def video_arrays(cache: Path, vid: str) -> dict:
    with np.load(cache / f"{vid}.npz") as z:
        d = {k: z[k] for k in z.files}
    d["meta"] = json.loads(str(d["meta"]))
    return d


def prepare(d: dict, cfg: Config, train: bool, rng: np.random.Generator, image_h: float) -> dict:
    """Model inputs of one video at model_fps: features [T,F], soft targets [T], mask [T],
    sample times [T] and the true change times. train=True augments (engine_b.augment)."""
    e, bd = cfg["engine_b"], cfg["boundaries"]
    fps = float(d["meta"]["fps"])
    idx = bt.sample_index(len(d["features"]), fps, e["model_fps"])
    if train and e["augment"]:
        xy, conf = eb.augment(d["xy"][idx], d["kinect_conf"][idx], eb.AugmentConfig.from_cfg(cfg), rng)
        feats = eb.features(xy, conf, image_h, float(d["median_torso_px"]))
    else:
        feats = d["features"][idx]
    labelled = np.asarray(d["labelled"], bool)
    changes = bt.change_times(d["labels"], fps, bd["ignore_dont_care"], bd["dont_care_label"], labelled)
    t = idx / fps
    return {"x": feats.astype(np.float32), "y": bt.soft_targets(t, changes, bd["tol_s"]),
            "m": labelled[idx].astype(np.float32), "t": t, "changes": changes}


def batches(items: list[dict], size: int, crop: int, rng: np.random.Generator):
    order = rng.permutation(len(items))
    for k in range(0, len(order), size):
        part = []
        for i in order[k:k + size]:
            it = items[i]
            T = len(it["x"])
            s = int(rng.integers(0, T - crop + 1)) if T > crop else 0
            part.append({n: it[n][s:s + crop] for n in ("x", "y", "m")})
        L = max(len(p["x"]) for p in part)
        X = np.zeros((len(part), eb.N_FEATURES, L), np.float32)
        Y = np.zeros((len(part), L), np.float32)
        M = np.zeros((len(part), L), np.float32)
        for j, p in enumerate(part):
            n = len(p["x"])
            X[j, :, :n], Y[j, :n], M[j, :n] = p["x"].T, p["y"], p["m"]
        yield X, Y, M


def auto_pos_weight(items: list[dict]) -> float:
    pos = sum(float((it["y"] * it["m"]).sum()) for it in items)
    neg = sum(float(((1 - it["y"]) * it["m"]).sum()) for it in items)
    return neg / pos if pos > 0 else 1.0


def evaluate(model, items: list[dict], cfg: Config, device: str) -> dict:
    """Boundary F1 per tolerance over the videos (peaks at engine_b.peak_threshold), plus a
    threshold table for information (not used to choose anything)."""
    e = cfg["engine_b"]
    probs = [bt.predict(model, it["x"], device) for it in items]
    gap = int(round(e["peak_min_gap_s"] * e["model_fps"]))

    def at(th: float) -> dict:
        vids = [(it["t"][bt.peaks(p, th, gap)], it["changes"]) for it, p in zip(items, probs)]
        return bt.boundary_f1(vids, list(e["eval_tols_s"]))
    main = at(e["peak_threshold"])
    sweep = {f"{th:g}": {tol: r["f1"] for tol, r in at(th).items()} for th in (0.3, 0.4, 0.5, 0.6, 0.7)}
    return {"f1": main, "threshold_sweep_f1": sweep, "probs": probs}


def train(cfg: Config, train_items: list[dict], train_raw: list[dict], device: str, image_h: float,
          progress=log.info) -> tuple:
    import torch
    e = cfg["engine_b"]
    torch.manual_seed(e["train_seed"])
    rng = np.random.default_rng(e["train_seed"])
    model = bt.build_model(eb.N_FEATURES, e["stages"], e["layers"], e["channels"], e["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=e["lr"], weight_decay=e["weight_decay"])
    pw = e["pos_weight"] or auto_pos_weight(train_items)
    pos_weight = torch.tensor(pw, device=device)
    crop = max(1, int(round(e["crop_s"] * e["model_fps"])))
    history = []
    for ep in range(e["epochs"]):
        t0 = time.monotonic()
        items = [prepare(d, cfg, True, rng, image_h) for d in train_raw] if e["augment"] else train_items
        model.train()
        tot, n = 0.0, 0.0
        for X, Y, M in batches(items, e["batch_videos"], crop, rng):
            X, Y, M = (torch.from_numpy(a).to(device) for a in (X, Y, M))
            out = model(X, M.unsqueeze(1))
            loss = 0.0
            for s in range(out.shape[0]):
                bce = torch.nn.functional.binary_cross_entropy_with_logits(out[s], Y, pos_weight=pos_weight,
                                                                           reduction="none")
                loss = loss + (bce * M).sum() / M.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * float(M.sum())
            n += float(M.sum())
        history.append({"epoch": ep + 1, "train_loss": round(tot / max(n, 1.0), 5),
                        "seconds": round(time.monotonic() - t0, 2)})
        progress(f"epoch {ep + 1}/{e['epochs']}: train loss {history[-1]['train_loss']:.4f} "
                 f"({history[-1]['seconds']:.1f} s)")
    model.eval()
    return model, {"pos_weight": round(pw, 3), "history": history}


def table_md(res: dict) -> str:
    lines = [f"# {res['id']}: {res['title']}", "", res["note"], "",
             f"Train {res['data']['train_videos']} videos of {res['data']['train_subjects']} subjects, test "
             f"{res['data']['test_videos']} videos of {res['data']['test_subjects']} held out subjects "
             f"({res['data']['split']}); {res['settings']['model_fps']} fps; targets Gaussian sigma "
             f"{res['data']['tol_s']} s; peaks at {res['settings']['peak_threshold']}, "
             f"{res['settings']['peak_min_gap_s']} s apart.", "",
             "| set | tolerance s | precision | recall | F1 | predicted | true |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, block in [("test (all)", res["test"]["f1"])] + [(f"test {k}", v["f1"]) for k, v in res["test"]["per_location"].items()]:
        for tol, r in block.items():
            lines.append(f"| {name} | {tol} | {r['precision']} | {r['recall']} | {r['f1']} | {r['predicted']} | {r['true']} |")
    lines += ["", "F1 by peak threshold (information only, nothing was chosen on the test subjects):", "",
              "| threshold | " + " | ".join(f"F1@{t}" for t in res["test"]["f1"]) + " |",
              "| --- |" + " --- |" * len(res["test"]["f1"])]
    for th, row in res["test"]["threshold_sweep_f1"].items():
        lines.append(f"| {th} | " + " | ".join(str(v) for v in row.values()) + " |")
    lines += ["", f"Model {res['model']['file']} (sha256 {res['model']['sha256'][:16]}...), "
              f"{res['model']['parameters']} parameters, pos_weight {res['training']['pos_weight']}, "
              f"{len(res['training']['history'])} epochs, final train loss {res['training']['history'][-1]['train_loss']}, "
              f"{res['training']['seconds']} s on {res['training']['device']}.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.train_engine_b")
    p.add_argument("--id", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="train on the CPU (no GPU lock)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--title", default="Engine B v0: MS TCN boundary scorer on the HA4M Kinect joints")
    p.add_argument("--note", default="First model, trained on the Engine B cache of HA4M (Kinect 2D colour joints "
                                     "mapped to COCO 17, YOLO like augmentation), cross subject split.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["XDG_CONFIG_HOME"] = str(repo_root / ".config")
    lock = None
    try:
        reg._check_id(args.id)
        cfg = Config.load(overrides=parse_sets(args.set))
        loader = HA4MLoader(cfg, repo_root=repo_root)
        cache = eb.cache_dir(loader.root)
        split_path = repo_root / "experiments" / "datasets" / "ha4m_subject_split.json"
        if not split_path.is_file() or not (cache / "index.json").is_file():
            print("refused: needs experiments/datasets/ha4m_subject_split.json and the Engine B cache", file=sys.stderr)
            return 2
        split = json.loads(split_path.read_text())
        index = json.loads((cache / "index.json").read_text())
        test_s = set(split["test_subjects"])
        vids = sorted(index["videos"])
        tr_ids = [v for v in vids if index["videos"][v]["subject"] not in test_s]
        te_ids = [v for v in vids if index["videos"][v]["subject"] in test_s]
        if not tr_ids or not te_ids:
            print("refused: no train or no test videos in the cache", file=sys.stderr)
            return 2
        if args.cpu:
            device = "cpu"
        else:
            lock_dir = repo_root / ".locks"
            lock_dir.mkdir(exist_ok=True)
            lock = open(lock_dir / f"gpu{args.gpu}.lock", "a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"refused: GPU {args.gpu} is in use by another job", file=sys.stderr)
                return 2
            device = f"cuda:{args.gpu}"
        import torch
        image_h = float(RGB_SIZE[1])                         # the cache's features use the colour image height
        rng = np.random.default_rng(cfg["engine_b"]["train_seed"])
        t0 = time.monotonic()
        train_raw = [video_arrays(cache, v) for v in tr_ids]
        test_raw = [video_arrays(cache, v) for v in te_ids]
        train_items = [prepare(d, cfg, False, rng, image_h) for d in train_raw]
        test_items = [prepare(d, cfg, False, rng, image_h) for d in test_raw]
        log.info("train %d videos, test %d videos (%s)", len(tr_ids), len(te_ids), split_path.name)
        model, info = train(cfg, train_items, train_raw, device, image_h)
        ev = evaluate(model, test_items, cfg, device)
        per_loc = {}
        for loc in sorted({d["meta"]["location"] for d in test_raw}):
            sel = [it for it, d in zip(test_items, test_raw) if d["meta"]["location"] == loc]
            r = evaluate(model, sel, cfg, device)
            per_loc[loc] = {"videos": len(sel), "f1": r["f1"]}
        seconds = round(time.monotonic() - t0, 1)
        e = cfg["engine_b"]
        model_rel = f"engine_b/{args.id}.pt"
        model_path = repo_root / cfg["paths"]["models_dir"] / model_rel
        n_params = int(sum(p.numel() for p in model.parameters()))
        sha = bt.save_model(model_path, model, bt.model_settings(cfg), {"experiment": args.id, "trained_utc": utc_now()})
        from ..experiments.provenance import git_state, library_versions
        res = {"id": args.id, "kind": "engine_b_train", "source": "public", "title": args.title, "note": args.note,
               "reference": "",
               "data": {"dataset": "HA4M", "cache": str(cache.relative_to(repo_root)), "split": split_path.name,
                        "train_videos": len(tr_ids), "test_videos": len(te_ids),
                        "train_subjects": len({index["videos"][v]["subject"] for v in tr_ids}),
                        "test_subjects": len({index["videos"][v]["subject"] for v in te_ids}),
                        "test_video_ids": te_ids, "tol_s": cfg["boundaries"]["tol_s"],
                        "ignore_dont_care": cfg["boundaries"]["ignore_dont_care"],
                        "true_boundaries_test": int(sum(len(it["changes"]) for it in test_items))},
               "settings": {k: e[k] for k in e},
               "training": {**info, "device": torch.cuda.get_device_name(args.gpu) if device != "cpu" else "cpu",
                            "seconds": seconds},
               "test": {"f1": ev["f1"], "threshold_sweep_f1": ev["threshold_sweep_f1"], "per_location": per_loc},
               "model": {"file": f"models/{model_rel}", "sha256": sha, "parameters": n_params},
               "provenance": {"git": git_state(repo_root), "versions": library_versions(), "data_area": "public",
                              "date_utc": utc_now(), "command": " ".join(sys.argv)},
               "finished_utc": utc_now()}
        res["index"] = {"cells": [f"test subjects: boundary F1@{tol} s {r['f1']} (P {r['precision']}, R {r['recall']})"
                                  for tol, r in ev["f1"].items()], "verdict": "measurement"}
        d = repo_root / "experiments" / args.id
        d.mkdir(parents=True, exist_ok=True)
        meta = {"id": args.id, "title": args.title, "kind": "engine_b_train", "source": "public",
                "videos": [str(cache.relative_to(repo_root))], "repeats": 1, "reference": "",
                "gpu": -1 if args.cpu else args.gpu, "variant": cfg["pose"]["variant"]}
        (d / "config.toml").write_text(reg.config_text(meta, cfg.snapshot()))
        (d / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float) + "\n")
        (d / "table.md").write_text(table_md(res))
        reg.write_index(repo_root / "experiments")
        for line in res["index"]["cells"]:
            log.info("%s", line)
        log.info("wrote %s and %s", d, model_path)
        return 0
    except AvError as err:
        return _fail(err)
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    sys.exit(main())
