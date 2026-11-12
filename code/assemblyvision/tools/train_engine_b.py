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
from ..datasets import ha4m_phases as hp
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


def prepare(d: dict, cfg: Config, train: bool, rng: np.random.Generator, image_h: float,
            offsets: np.ndarray | None = None) -> dict:
    """Model inputs of one video at model_fps: features [T,F], soft targets [T], mask [T],
    sample times [T] and the true change times. train=True augments (engine_b.augment);
    offsets [17,2] (engine_b.hip_calibration) move the Kinect joints first."""
    e, bd = cfg["engine_b"], cfg["boundaries"]
    fps = float(d["meta"]["fps"])
    idx = bt.sample_index(len(d["features"]), fps, e["model_fps"])
    xy, kc = d["xy"][idx], d["kinect_conf"][idx]
    if offsets is not None:
        xy = eb.calibrate(xy, kc, offsets)
    a = eb.AugmentConfig.from_cfg(cfg)
    if train and e["augment"]:
        xy, conf = eb.augment(xy, kc, a, rng)
        feats = eb.features(xy, conf, image_h, float(d["median_torso_px"]))
    elif offsets is not None:
        feats = eb.features(xy, eb.yolo_like_conf(kc, a), image_h, float(d["median_torso_px"]))
    else:
        feats = d["features"][idx]
    labelled = np.asarray(d["labelled"], bool)
    changes = bt.change_times(d["labels"], fps, bd["ignore_dont_care"], bd["dont_care_label"], labelled)
    t = idx / fps
    act = np.asarray(d["labels"])[idx, 0].astype(np.int64) - 1          # action 1..K -> 0..K-1
    act[(act < 0) | (act >= e["n_actions"]) | ~labelled[idx]] = -1       # don't care and unlabelled: no action loss
    phase = hp.macro_index(np.asarray(d["labels"])[:, 0])[idx]           # macro phase per step, -1 = other
    phase[~labelled[idx]] = -1
    return {"x": feats.astype(np.float32), "y": bt.soft_targets(t, changes, bd["tol_s"]),
            "m": labelled[idx].astype(np.float32), "c": act, "p": phase.astype(np.int64), "t": t, "changes": changes}


def batches(items: list[dict], size: int, crop: int, rng: np.random.Generator):
    order = rng.permutation(len(items))
    for k in range(0, len(order), size):
        part = []
        for i in order[k:k + size]:
            it = items[i]
            T = len(it["x"])
            s = int(rng.integers(0, T - crop + 1)) if T > crop else 0
            part.append({n: it[n][s:s + crop] for n in ("x", "y", "m", "c", "p") if n in it})
        L = max(len(p["x"]) for p in part)
        X = np.zeros((len(part), part[0]["x"].shape[1], L), np.float32)
        Y = np.zeros((len(part), L), np.float32)
        M = np.zeros((len(part), L), np.float32)
        C = np.full((len(part), L), -1, np.int64)
        P = np.full((len(part), L), -1, np.int64)
        for j, p in enumerate(part):
            n = len(p["x"])
            X[j, :, :n], Y[j, :n], M[j, :n] = p["x"].T, p["y"], p["m"]
            if "c" in p:
                C[j, :n] = p["c"]
            if "p" in p:
                P[j, :n] = p["p"]
        yield X, Y, M, C, P


def window_embeddings(h, P, M, w: int):
    """Mean last stage features over non overlapping windows of w steps -> (unit length
    window embeddings [N,C], macro phase per window [N]); only windows wholly inside one
    macro phase (and real frames) count."""
    import torch
    B, C, L = h.shape
    n = L // w
    if n == 0:
        return h.new_zeros((0, C)), P.new_zeros((0,))
    hw = h[:, :, :n * w].reshape(B, C, n, w).mean(dim=3).permute(0, 2, 1).reshape(B * n, C)
    pw = P[:, :n * w].reshape(B, n, w)
    mw = M[:, :n * w].reshape(B, n, w)
    pure = (pw == pw[:, :, :1]).all(dim=2) & (pw[:, :, 0] >= 0) & (mw > 0).all(dim=2)
    keep = pure.reshape(-1)
    z = torch.nn.functional.normalize(hw[keep], dim=1)
    return z, pw[:, :, 0].reshape(-1)[keep]


def supcon_loss(z, labels, temp: float):
    """Supervised contrastive loss (Khosla et al. 2020): windows of the same macro phase
    close, of different phases far. 0 without an anchor that has a positive."""
    import torch
    if len(z) < 2:
        return z.sum() * 0.0
    sim = z @ z.T / temp
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    has = pos.any(dim=1)
    if not bool(has.any()):
        return z.sum() * 0.0
    return -(logp.masked_fill(~pos, 0.0).sum(dim=1)[has] / pos.sum(dim=1)[has]).mean()


def auto_pos_weight(items: list[dict]) -> float:
    pos = sum(float((it["y"] * it["m"]).sum()) for it in items)
    neg = sum(float(((1 - it["y"]) * it["m"]).sum()) for it in items)
    return neg / pos if pos > 0 else 1.0


def evaluate(model, items: list[dict], cfg: Config, device: str) -> dict:
    """Boundary F1 per tolerance over the videos (peaks at engine_b.peak_threshold), plus a
    threshold table for information (not used to choose anything)."""
    e = cfg["engine_b"]
    outs = [bt.predict_all(model, it["x"], device) for it in items]
    probs = [o["prob"] for o in outs]
    gap = int(round(e["peak_min_gap_s"] * e["model_fps"]))

    def at(th: float) -> dict:
        vids = [(it["t"][bt.peaks(p, th, gap)], it["changes"]) for it, p in zip(items, probs)]
        return bt.boundary_f1(vids, list(e["eval_tols_s"]))
    main = at(e["peak_threshold"])
    sweep = {f"{th:g}": {tol: r["f1"] for tol, r in at(th).items()} for th in (0.3, 0.4, 0.5, 0.6, 0.7)}
    res = {"f1": main, "threshold_sweep_f1": sweep, "probs": probs}
    if outs and outs[0]["actions"] is not None:          # v1: frame accuracy of the action head (labelled frames)
        hit = sum(int((o["actions"].argmax(1) == it["c"])[it["c"] >= 0].sum()) for o, it in zip(outs, items))
        tot = sum(int((it["c"] >= 0).sum()) for it in items)
        res["action_frame_accuracy"] = round(hit / tot, 4) if tot else None
    return res


def train(cfg: Config, train_items: list[dict], train_raw: list[dict], device: str, image_h: float,
          progress=log.info, offsets: np.ndarray | None = None, items_fn=None, n_actions: int | None = None) -> tuple:
    """items_fn(rng) -> the training items of one epoch (v3: several datasets), instead of
    prepare() on train_raw; n_actions overrides engine_b.n_actions (v3: every dataset's actions)."""
    import torch
    e = cfg["engine_b"]
    torch.manual_seed(e["train_seed"])
    rng = np.random.default_rng(e["train_seed"])
    n_act = (n_actions or e["n_actions"]) if e["action_weight"] > 0 else 0
    n_in = train_items[0]["x"].shape[1] if train_items else eb.N_FEATURES
    model = bt.build_model(n_in, e["stages"], e["layers"], e["channels"], e["dropout"], n_act).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=e["lr"], weight_decay=e["weight_decay"])
    pw = e["pos_weight"] or auto_pos_weight(train_items)
    pos_weight = torch.tensor(pw, device=device)
    crop = max(1, int(round(e["crop_s"] * e["model_fps"])))
    history = []
    for ep in range(e["epochs"]):
        t0 = time.monotonic()
        if items_fn is not None:
            items = items_fn(rng)
        else:
            items = [prepare(d, cfg, True, rng, image_h, offsets) for d in train_raw] if e["augment"] else train_items
        model.train()
        tot, n = 0.0, 0.0
        for X, Y, M, C, P in batches(items, e["batch_videos"], crop, rng):
            X, Y, M, C, P = (torch.from_numpy(a).to(device) for a in (X, Y, M, C, P))
            out, act, hid = model(X, M.unsqueeze(1), all_heads=True)
            loss = 0.0
            for s in range(out.shape[0]):
                bce = torch.nn.functional.binary_cross_entropy_with_logits(out[s], Y, pos_weight=pos_weight,
                                                                           reduction="none")
                loss = loss + (bce * M).sum() / M.sum().clamp(min=1.0)
                if act is not None and bool((C >= 0).any()):
                    loss = loss + e["action_weight"] * torch.nn.functional.cross_entropy(act[s], C, ignore_index=-1)
            if e["contrast_weight"] > 0:                  # v2: phase level contrastive objective
                z, lab = window_embeddings(hid, P, M, max(1, int(round(e["contrast_window_s"] * e["model_fps"]))))
                loss = loss + e["contrast_weight"] * supcon_loss(z, lab, e["contrast_temp"])
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


FOLDS = ("cross_subject", "cross_location_test_lab_a", "cross_location_test_lab_b", "cv1", "cv2", "cv3")
SPLITS_FILE = "ha4m_splits.json"
CV_FILE = "ha4m_train_cv.json"          # 3 folds by subject inside the cross subject train split (v2 rule)


def fold_rows(name: str, fold: dict) -> list[str]:
    rows = []
    for setname, block in [("all", fold["test"]["f1"])] + [(k, v["f1"]) for k, v in fold["test"]["per_location"].items()]:
        for tol, r in block.items():
            rows.append(f"| {name} | {setname} | {tol} | {r['precision']} | {r['recall']} | {r['f1']} | "
                        f"{r['predicted']} | {r['true']} |")
    return rows


def table_md(res: dict) -> str:
    cal = res["settings"].get("hip_calibration")
    lines = [f"# {res['id']}: {res['title']}", "", res["note"], "",
             f"Main fold cross_subject: train {res['data']['train_videos']} videos of {res['data']['train_subjects']} "
             f"subjects, test {res['data']['test_videos']} videos of {res['data']['test_subjects']} held out subjects "
             f"({res['data']['split']}); cross location folds from {SPLITS_FILE} (train one lab, test the other). "
             f"{res['settings']['model_fps']} fps; targets Gaussian sigma {res['data']['tol_s']} s; peaks at "
             f"{res['settings']['peak_threshold']}, {res['settings']['peak_min_gap_s']} s apart; hip calibration "
             f"{'on (' + res['settings'].get('calibration_from', '') + ', mean of both labs per joint)' if cal else 'off'}.", "",
             "| fold | test set | tolerance s | precision | recall | F1 | predicted | true |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, fold in res.get("folds", {"cross_subject": res}).items():
        lines += fold_rows(name, fold)
    lines += ["", "F1 by peak threshold, cross_subject (information only, nothing was chosen on the test subjects):", "",
              "| threshold | " + " | ".join(f"F1@{t}" for t in res["test"]["f1"]) + " |",
              "| --- |" + " --- |" * len(res["test"]["f1"])]
    for th, row in res["test"]["threshold_sweep_f1"].items():
        lines.append(f"| {th} | " + " | ".join(str(v) for v in row.values()) + " |")
    lines += [""]
    for name, fold in res.get("folds", {"cross_subject": res}).items():
        lines.append(f"- {name}: model {fold['model']['file']} (sha256 {fold['model']['sha256'][:16]}...), "
                     f"{fold['data']['train_videos']} train / {fold['data']['test_videos']} test videos, pos_weight "
                     f"{fold['training']['pos_weight']}, {len(fold['training']['history'])} epochs, final train loss "
                     f"{fold['training']['history'][-1]['train_loss']}, {fold['training']['seconds']} s.")
    acc = {n: f["test"]["action_frame_accuracy"] for n, f in res.get("folds", {}).items() if "action_frame_accuracy" in f["test"]}
    if acc:
        lines += ["", "Action head (12 HA4M actions, labelled frames, don't care left out), frame accuracy on the test "
                  "videos: " + ", ".join(f"{n} {v}" for n, v in acc.items()) + "."]
    lines += ["", f"{res['model']['parameters']} parameters per model, trained on {res['training']['device']}.", ""]
    return "\n".join(lines)


def run_fold(cfg: Config, cache: Path, tr_ids: list[str], te_ids: list[str], device: str, image_h: float,
             offsets) -> tuple:
    """Train on tr_ids, score te_ids once (overall and per location). Returns (model, fold record)."""
    rng = np.random.default_rng(cfg["engine_b"]["train_seed"])
    t0 = time.monotonic()
    train_raw = [video_arrays(cache, v) for v in tr_ids]
    test_raw = [video_arrays(cache, v) for v in te_ids]
    train_items = [prepare(d, cfg, False, rng, image_h, offsets) for d in train_raw]
    test_items = [prepare(d, cfg, False, rng, image_h, offsets) for d in test_raw]
    model, info = train(cfg, train_items, train_raw, device, image_h, offsets=offsets)
    ev = evaluate(model, test_items, cfg, device)
    per_loc = {}
    for loc in sorted({d["meta"]["location"] for d in test_raw}):
        sel = [it for it, d in zip(test_items, test_raw) if d["meta"]["location"] == loc]
        per_loc[loc] = {"videos": len(sel), "f1": evaluate(model, sel, cfg, device)["f1"]}
    info["seconds"] = round(time.monotonic() - t0, 1)
    rec = {"data": {"train_videos": len(tr_ids), "test_videos": len(te_ids), "test_video_ids": te_ids,
                    "train_locations": sorted({d["meta"]["location"] for d in train_raw}),
                    "true_boundaries_test": int(sum(len(it["changes"]) for it in test_items))},
           "training": info,
           "test": {"f1": ev["f1"], "threshold_sweep_f1": ev["threshold_sweep_f1"], "per_location": per_loc,
                    **({"action_frame_accuracy": ev["action_frame_accuracy"]} if "action_frame_accuracy" in ev else {})}}
    return model, rec


def main(argv: list[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.train_engine_b")
    p.add_argument("--id", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="train on the CPU (no GPU lock)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--folds", default="cross_subject",
                   help="comma list of " + ", ".join(FOLDS) + " (one model each; cross_subject first is the main one)")
    p.add_argument("--title", default="Engine B v0: MS TCN boundary scorer on the HA4M Kinect joints")
    p.add_argument("--note", default="First model, trained on the Engine B cache of HA4M (Kinect 2D colour joints "
                                     "mapped to COCO 17, YOLO like augmentation), cross subject split.")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["XDG_CONFIG_HOME"] = str(repo_root / ".config")
    lock = None
    try:
        reg._check_id(args.id)
        folds = [x for f in args.folds.split(",") if f.strip() for x in (("cv1", "cv2", "cv3") if f.strip() == "cv"
                                                                         else (f.strip(),))]
        if not folds or (folds[0] != "cross_subject" and not all(f.startswith("cv") for f in folds)) \
                or any(f not in FOLDS for f in folds):
            print(f"refused: --folds must start with cross_subject, or be only cv folds (cv = cv1,cv2,cv3); "
                  f"use only {', '.join(FOLDS)}", file=sys.stderr)
            return 2
        cfg = Config.load(overrides=parse_sets(args.set))
        e = cfg["engine_b"]
        loader = HA4MLoader(cfg, repo_root=repo_root)
        cache = eb.cache_dir(loader.root)
        ds = repo_root / "experiments" / "datasets"
        split_path = ds / "ha4m_subject_split.json"
        if not split_path.is_file() or not (cache / "index.json").is_file():
            print("refused: needs experiments/datasets/ha4m_subject_split.json and the Engine B cache", file=sys.stderr)
            return 2
        split = json.loads(split_path.read_text())
        index = json.loads((cache / "index.json").read_text())
        test_s = set(split["test_subjects"])
        vids = sorted(index["videos"])
        lists = {"cross_subject": ([v for v in vids if index["videos"][v]["subject"] not in test_s],
                                   [v for v in vids if index["videos"][v]["subject"] in test_s])}
        for f in folds:
            if f.startswith("cv"):
                cvs = {c["name"]: c for c in json.loads((ds / CV_FILE).read_text())["folds"]}
                lists[f] = ([v for v in cvs[f]["train_videos"] if v in index["videos"]],
                            [v for v in cvs[f]["test_videos"] if v in index["videos"]])
                if set(lists[f][1]) & set(lists["cross_subject"][1]):
                    print("refused: a cv fold holds a benchmark test video", file=sys.stderr)
                    return 2
        if len([f for f in folds if not f.startswith("cv")]) > 1:
            if not (ds / SPLITS_FILE).is_file():
                print(f"refused: cross location folds need experiments/datasets/{SPLITS_FILE}", file=sys.stderr)
                return 2
            stored = {f["name"]: f for f in json.loads((ds / SPLITS_FILE).read_text())["folds"]}
            for f in [f for f in folds[1:] if not f.startswith("cv")]:
                lists[f] = ([v for v in stored[f]["train_videos"] if v in index["videos"]],
                            [v for v in stored[f]["test_videos"] if v in index["videos"]])
        if any(not tr or not te for tr, te in (lists[f] for f in folds)):
            print("refused: no train or no test videos in the cache for a fold", file=sys.stderr)
            return 2
        offsets = None
        if e["hip_calibration"]:
            try:
                offsets = eb.load_calibration(repo_root, e["calibration_from"])
            except FileNotFoundError as err:
                print(f"refused: {err}", file=sys.stderr)
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
        t0 = time.monotonic()
        recs, n_params = {}, 0
        for f in folds:
            tr_ids, te_ids = lists[f]
            log.info("fold %s: train %d videos, test %d videos", f, len(tr_ids), len(te_ids))
            model, rec = run_fold(cfg, cache, tr_ids, te_ids, device, image_h, offsets)
            model_rel = f"engine_b/{args.id}.pt" if f == "cross_subject" else f"engine_b/{args.id}.{f}.pt"
            model_path = repo_root / cfg["paths"]["models_dir"] / model_rel
            n_params = int(sum(q.numel() for q in model.parameters()))
            sha = bt.save_model(model_path, model, bt.model_settings(cfg),
                                {"experiment": args.id, "fold": f, "trained_utc": utc_now(),
                                 **({"contrast": {k: e[k] for k in ("contrast_weight", "contrast_window_s",
                                                                   "contrast_temp")}} if e["contrast_weight"] > 0 else {})})
            rec["model"] = {"file": f"models/{model_rel}", "sha256": sha, "parameters": n_params}
            recs[f] = rec
            for tol, r in rec["test"]["f1"].items():
                log.info("%s: boundary F1@%s s %s (P %s, R %s)", f, tol, r["f1"], r["precision"], r["recall"])
        seconds = round(time.monotonic() - t0, 1)
        main = "cross_subject" if "cross_subject" in recs else folds[0]      # cv only: the first cv fold
        main_rec = recs[main]
        tr_ids, te_ids = lists[main]
        from ..experiments.provenance import git_state, library_versions
        res = {"id": args.id, "kind": "engine_b_train", "source": "public", "title": args.title, "note": args.note,
               "reference": "",
               "data": {"dataset": "HA4M", "cache": str(cache.relative_to(repo_root)), "split": split_path.name,
                        "train_videos": len(tr_ids), "test_videos": len(te_ids),
                        "train_subjects": len({index["videos"][v]["subject"] for v in tr_ids}),
                        "test_subjects": len({index["videos"][v]["subject"] for v in te_ids}),
                        "test_video_ids": te_ids, "tol_s": cfg["boundaries"]["tol_s"],
                        "ignore_dont_care": cfg["boundaries"]["ignore_dont_care"],
                        "true_boundaries_test": main_rec["data"]["true_boundaries_test"]},
               "settings": {k: e[k] for k in e},
               "calibration_offsets_torso": offsets.round(4).tolist() if offsets is not None else None,
               "training": {**main_rec["training"],
                            "device": torch.cuda.get_device_name(args.gpu) if device != "cpu" else "cpu",
                            "seconds_all_folds": seconds},
               "test": main_rec["test"], "model": main_rec["model"], "folds": recs,
               "provenance": {"git": git_state(repo_root), "versions": library_versions(), "data_area": "public",
                              "date_utc": utc_now(), "command": " ".join(sys.argv)},
               "finished_utc": utc_now()}
        cells = [f"{main.replace('cross_subject', 'cross subject')}: boundary F1@{tol} s {r['f1']} "
                 f"(P {r['precision']}, R {r['recall']})"
                 for tol, r in main_rec["test"]["f1"].items()]
        cells += [f"{f.replace('cross_location_test_', 'cross location, test ')}: F1@"
                  + ", F1@".join(f"{tol} s {r['f1']}" for tol, r in recs[f]["test"]["f1"].items()) for f in folds[1:]]
        res["index"] = {"cells": cells, "verdict": "measurement"}
        d = repo_root / "experiments" / args.id
        d.mkdir(parents=True, exist_ok=True)
        meta = {"id": args.id, "title": args.title, "kind": "engine_b_train", "source": "public",
                "videos": [str(cache.relative_to(repo_root))], "repeats": 1, "reference": "",
                "gpu": -1 if args.cpu else args.gpu, "variant": cfg["pose"]["variant"]}
        (d / "config.toml").write_text(reg.config_text(meta, cfg.snapshot()))
        (d / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float) + "\n")
        (d / "table.md").write_text(table_md(res))
        reg.write_index(repo_root / "experiments")
        log.info("wrote %s (%d folds, %.1f s)", d, len(folds), seconds)
        return 0
    except AvError as err:
        return _fail(err)
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    sys.exit(main())
