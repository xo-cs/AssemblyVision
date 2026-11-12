"""Public phase benchmark: the selection set for the step engines (Week 10).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.phase_bench --id w10-phase-bench-baseline \\
        [--split test|train] [--row 'LABEL:KEY=VALUE;KEY=VALUE' ...] [--no-clips]

CPU only, public data only. HA4M videos (the 65 cross subject test videos, or the
152 train videos) with the 12 actions grouped into the 5 macro phases of
datasets/ha4m_phases.py (experiments/datasets/ha4m_macro_phases.md). Each video's
Kinect COCO 17 skeleton (the Engine B cache, YOLO like confidences, no augmentation)
becomes a one shot timeline with one subject, so the step engines decode it exactly
as a run: Engine A's energy and pauses from the cleaned series, Engine B's
probability and embedding from the same model inputs as its test split. The engines
get the 5 phase names in task order. Scored with the QiSens metrics (frame accuracy,
edit, F1@10/25/50, boundary MAE) per video, then the mean over videos.
The two public clips (process_480p30, shots_check_480p30) are scored from their
stored runs of w10-pose-sens-x (YOLO26 x, reference profile) against their GTs.
A row is a label and settings (--row); default rows: Engine A and Engine B v1.
Writes experiments/<id>/ (results.json, table.md) and INDEX.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")    # CPU only

from ..analysis.segmentation import boundary_tcn as bt
from ..analysis.segmentation.base import RunTimeline
from ..analysis.segmentation.rules import RuleBasedEngine
from ..analysis.cleaning import features as clean_features
from ..config import Config, deep_merge, engine_label
from ..datasets import engine_b as eb
from ..datasets import ha4m_phases as hp
from ..datasets.ha4m import RGB_SIZE
from ..evaluation.steps_eval import evaluate
from ..experiments import registry as reg
from ..storage.files import REPO_ROOT, utc_now
from ..types import KeypointSeries
from .train_engine_b import video_arrays

CACHE = "data/public/datasets/ha4m/engine_b/kinect_coco17"
SPLITS = "experiments/datasets/ha4m_splits.json"
CLIPS_FROM = "w10-pose-sens-x"
METRICS = ("frame_accuracy", "edit", "f1_10", "f1_25", "f1_50", "boundary_mae_s")
DEFAULT_ROWS = ['Engine A:seg_a.engine="A"',
                'Engine B v1, engine_a candidates:seg_a.engine="custom";seg_a.embed_weight=0.2;seg_a.embed_window_s=30.0;'
                'engine_b.model="engine_b/w10-engine-b-v1.pt";engine_b.candidates="engine_a"',
                'Engine B v1, dense candidates:seg_a.engine="B v1"']


def parse_row(spec: str) -> tuple[str, dict]:
    """'LABEL:KEY=VALUE;KEY=VALUE' -> (label, overrides)."""
    label, _, sets = spec.partition(":")
    over: dict = {}
    for item in filter(None, (x.strip() for x in sets.split(";"))):
        key, _, value = item.partition("=")
        section, _, name = key.strip().partition(".")
        over = deep_merge(over, {section: {name: tomllib.loads(f"v = {value}")["v"]}})
    return label.strip(), over


def split_videos(split: str, repo_root: Path = REPO_ROOT) -> list[str]:
    folds = json.loads((repo_root / SPLITS).read_text())["folds"]
    fold = next(f for f in folds if f["name"] == "cross_subject")
    return list(fold[f"{split}_videos"])


def video_series(d: dict, cfg: Config) -> KeypointSeries:
    """The cached Kinect COCO 17 joints of one video as a one shot KTS series."""
    xy = np.asarray(d["xy"], np.float32)
    conf = eb.yolo_like_conf(d["kinect_conf"], eb.AugmentConfig.from_cfg(cfg)).astype(np.float32)
    ok = np.isfinite(xy).all(axis=2) & (conf > 0)
    present = ok.any(axis=1)
    xy = np.where(ok[..., None], xy, np.nan).astype(np.float32)
    T, fps = len(xy), float(d["meta"]["fps"])
    lo, hi = np.nanmin(np.where(ok[..., None], xy, np.inf), axis=1), np.nanmax(np.where(ok[..., None], xy, -np.inf), axis=1)
    box = np.where(present[:, None], np.concatenate([lo, hi], axis=1), np.nan).astype(np.float32)
    return KeypointSeries("ha4m", d["meta"]["video"], "subject", fps, RGB_SIZE[0], RGB_SIZE[1],
                          np.arange(T, dtype=np.int32), (np.arange(T) / fps).astype(np.float32), present, box,
                          xy, np.where(ok, conf, 0.0).astype(np.float32),
                          np.full((T, 42, 2), np.nan, np.float32), np.zeros((T, 42), np.float32))


def video_timeline(d: dict, cfg: Config, engine: RuleBasedEngine) -> RunTimeline:
    series = video_series(d, cfg)
    energy = clean_features(engine.cleaner.clean(series))["energy"].astype(np.float32)
    return RunTimeline(series.fps, len(series.present), series.present.copy(), energy, [])


def model_outputs(d: dict, model, settings: dict, present: np.ndarray) -> dict:
    """Engine B probability and embedding per frame from the cached model inputs (the
    test split's own features, no augmentation)."""
    T, fps = len(d["features"]), float(d["meta"]["fps"])
    idx = bt.sample_index(T, fps, settings["model_fps"])
    prob, emb = bt.to_frames(bt.predict_all(model, d["features"][idx]), idx, T, fps, settings["model_fps"], present)
    return {"prob": prob, "embedding": emb, "embedding_valid": present.copy()}


def with_intro(d: dict, seconds: float, rng, cfg: Config) -> dict:
    """A synthetic intro (item 8): the first measured pose of the video held still for
    seconds (Gaussian jitter of 1 px), prepended to the skeleton; its frames are labelled
    don't care, so the GT macro phases make them other. Engine B features are computed
    again for the whole video, as the cache computes them."""
    fps = float(d["meta"]["fps"])
    n = int(round(seconds * fps))
    xy, kc = np.asarray(d["xy"], np.float64), np.asarray(d["kinect_conf"], np.float64)
    first = int(np.argmax((np.isfinite(xy).all(axis=2) & (kc > 0)).sum(axis=1) >= 12))
    still = xy[first][None].repeat(n, axis=0) + rng.normal(0, 1.0, (n, 17, 2))
    xy2 = np.concatenate([still, xy]).astype(np.float32)
    kc2 = np.concatenate([kc[first][None].repeat(n, axis=0), kc]).astype(np.float32)
    labels = np.concatenate([np.zeros((n, 2), np.int64), np.asarray(d["labels"], np.int64)])
    a = eb.AugmentConfig.from_cfg(cfg)
    feats = eb.features(xy2, eb.yolo_like_conf(kc2, a), float(RGB_SIZE[1]), float(d["median_torso_px"]))
    meta = {**d["meta"], "frames": len(xy2), "intro_s": round(n / fps, 3)}
    return {**d, "xy": xy2, "kinect_conf": kc2, "labels": labels, "features": feats.astype(np.float32),
            "labelled": np.concatenate([np.ones(n, bool), np.asarray(d["labelled"], bool)]), "meta": meta}


class Bench:
    """Decodes and scores HA4M videos for any number of settings (models are cached)."""

    def __init__(self, repo_root: Path = REPO_ROOT):
        self.root = repo_root
        self.cache = repo_root / CACHE
        self._models: dict[str, tuple] = {}
        self._outputs: dict[tuple[str, str], dict] = {}

    def model(self, name: str):
        if name not in self._models:
            self._models[name] = bt.load_model(self.root / "models" / name)
        return self._models[name]

    def video(self, vid: str, cfg: Config, intro_s: float = 0.0, seed: int = 0) -> dict:
        d = video_arrays(self.cache, vid)
        if intro_s > 0:
            d = with_intro(d, intro_s, np.random.default_rng(seed), cfg)
            vid = f"{vid}+intro{intro_s:g}"
        engine = RuleBasedEngine(cfg)
        tl = video_timeline(d, cfg, engine)
        if cfg["seg_a"]["engine_b_weight"] > 0 or cfg["seg_a"]["embed_weight"] > 0:
            name = cfg["engine_b"]["model"]
            key = (name, vid)
            if key not in self._outputs:
                m, s = self.model(name)
                self._outputs[key] = model_outputs(d, m, s, tl.person)
            out = self._outputs[key]
            tl.boundary_prob, tl.embedding, tl.embedding_valid = out["prob"], out["embedding"], out["embedding_valid"]
        pred = engine.segment_run(tl)
        gt = hp.gt_steps(d["labels"], tl.fps)
        res, _ = evaluate(pred, gt, tl.duration_s, cfg["eval"]["seg_fps"], cfg["eval"]["overlaps"])
        return {"video": vid, "metrics": res, "in_task_order": hp.in_task_order(gt),
                "phases": f"{engine.report.get('phases_found')}/{engine.report.get('phases_expected')}",
                "engine": engine.report.get("engine")}


def bench_cfg(over: dict) -> Config:
    return Config.load(overrides=deep_merge({"steps": {"use": True, "step": [{"name": n} for n in hp.NAMES]}}, over))


def summarise(rows: list[dict]) -> dict:
    out = {"videos": len(rows)}
    for k in METRICS:
        v = [r["metrics"][k] for r in rows if r["metrics"].get(k) is not None]
        out[k] = round(float(np.mean(v)), 2) if v else None
    out["frame_accuracy_median"] = round(float(np.median([r["metrics"]["frame_accuracy"] for r in rows])), 2)
    out["videos_at_85"] = sum(r["metrics"]["frame_accuracy"] >= 85 for r in rows)
    return out


def clip_rows(over: dict, repo_root: Path = REPO_ROOT) -> dict:
    """The two public clips: their stored x runs scored with these settings."""
    from ..evaluation.steps_eval import score_steps
    from ..evaluation.steps_io import read_steps_csv
    from .run_pose import segment_stored, stored_config, _MemoryCtx
    exp = reg.load(CLIPS_FROM, "public", Config.load()["paths"], repo_root, fill_new=True)
    res = reg.load_results(CLIPS_FROM, "public", Config.load()["paths"], repo_root)
    out = {}
    for v, gt in zip(exp.meta["videos"], exp.meta["gt"]):
        run = repo_root / res["videos"][v]["repeats"][0]["run_dir"]
        manifest = json.loads((run / "manifest.json").read_text())
        cfg = Config.load(overrides=deep_merge(stored_config(manifest), over))
        ctx = _MemoryCtx("public")
        engine, steps = segment_stored(manifest, run, cfg, ctx)
        m, _, _ = score_steps(steps, read_steps_csv(repo_root / gt), Path(gt).name, engine.vocabulary, "public", cfg,
                              float(manifest["ingest"]["cfr"]["duration_s"]))
        out[Path(v).name] = {k: m.get(k) for k in METRICS} | {"engine": engine.report.get("engine"),
                                                             "run": run.name, "notices": ctx.notices}
    return out


def table_md(res: dict) -> str:
    lines = [f"# {res['id']}", "", res["title"], "",
             f"HA4M {res['split']} split of the cross subject fold ({res['n_videos']} videos, "
             f"{res['n_in_order']} in task order), 5 macro phases (experiments/datasets/ha4m_macro_phases.md), Kinect "
             "COCO 17 skeletons with YOLO like confidences. Mean over videos; at 85: videos with frame accuracy >= 85.",
             "", "| row | engine | frame acc | median | at 85 | edit | F1@10 | F1@25 | F1@50 | MAE s |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in res["rows"]:
        s = r["summary"]
        lines.append(f"| {r['label']} | {r['engine']} | {s['frame_accuracy']} | {s['frame_accuracy_median']} | "
                     f"{s['videos_at_85']}/{s['videos']} | {s['edit']} | {s['f1_10']} | {s['f1_25']} | {s['f1_50']} | "
                     f"{s['boundary_mae_s']} |")
    if any(r.get("clips") for r in res["rows"]):
        lines += ["", f"Public clips (stored YOLO26 x runs of {CLIPS_FROM}):", "",
                  "| row | clip | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in res["rows"]:
            for clip, m in (r.get("clips") or {}).items():
                lines.append(f"| {r['label']} | {clip} | " + " | ".join(str(m[k]) for k in METRICS) + " |")
    lines += ["", "Settings per row: results.json (rows[].sets).", ""]
    return "\n".join(lines)


def run_bench(exp_id: str, title: str, split: str, specs: list[str], clips: bool,
              repo_root: Path = REPO_ROOT, videos: list[str] | None = None, log=print) -> dict:
    from ..experiments.provenance import git_state
    vids = videos or split_videos(split, repo_root)
    bench = Bench(repo_root)
    res = {"id": exp_id, "kind": "phase_bench", "source": "public", "title": title, "reference": "",
           "split": split, "n_videos": len(vids), "videos": vids, "rows": []}
    for spec in specs:
        label, over = parse_row(spec)
        cfg = bench_cfg(over)
        per = []
        for v in vids:
            per.append(bench.video(v, cfg))
        row = {"label": label, "sets": over, "engine": engine_label(cfg), "summary": summarise(per), "per_video": per}
        if clips:
            row["clips"] = clip_rows(over, repo_root)
        res["rows"].append(row)
        log(f"{label}: " + ", ".join(f"{k} {row['summary'][k]}" for k in ("frame_accuracy", "edit", "f1_50",
                                                                          "boundary_mae_s")))
    res["n_in_order"] = sum(r["in_task_order"] for r in res["rows"][0]["per_video"]) if res["rows"] else 0
    res["provenance"] = {"git": git_state(repo_root), "cache": CACHE, "splits": SPLITS, "date_utc": utc_now(),
                         "command": " ".join(sys.argv)}
    res["finished_utc"] = utc_now()
    res["index"] = {"cells": [f"{r['label']}: frame acc {r['summary']['frame_accuracy']}, F1@50 {r['summary']['f1_50']}, "
                              f"MAE {r['summary']['boundary_mae_s']} s" for r in res["rows"]], "verdict": "measurement"}
    d = repo_root / "experiments" / exp_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float) + "\n")
    (d / "table.md").write_text(table_md(res))
    reg.write_index(repo_root / "experiments")
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.phase_bench")
    p.add_argument("--id", required=True)
    p.add_argument("--title", default="Public phase benchmark (HA4M macro phases)")
    p.add_argument("--split", choices=("test", "train"), default="test")
    p.add_argument("--row", action="append", default=[])
    p.add_argument("--no-clips", action="store_true")
    args = p.parse_args(argv)
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    res = run_bench(args.id, args.title, args.split, args.row or DEFAULT_ROWS, not args.no_clips)
    print(table_md(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
