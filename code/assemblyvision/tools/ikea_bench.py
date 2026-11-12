"""IKEA ASM macro phase benchmark on the stored public runs (Week 11).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.ikea_bench --make-folds          # experiments/datasets/ikea_asm_cv.json

Each video's run (tools/ikea_runs: YOLO26 x keypoints, shots, visual.npz) is decoded
like a QiSens compare (Engine A's timeline from the stored files, Engine B and the
visual term on the run clock), with the phase names of its furniture type
(experiments/datasets/ikea_asm_macro_phases.md), and scored with the QiSens metrics
against the GT macro phases. Public runs name phases Phase 1..K, so the GT phase k is
Phase k. Timelines, Engine B outputs and visual frames are cached per run.
CV folds: the official training subset, 3 folds by recording (the leading number of a
recording is one assembly session, kept in one fold), stratified by furniture, seed 42.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..config import Config, deep_merge
from ..datasets import ikea_phases as ip
from ..phases import neutral_label
from ..storage.files import REPO_ROOT
from ..types import Step

ROOT = REPO_ROOT / "data/public/datasets/ikea_asm"
FOLDS = "experiments/datasets/ikea_asm_cv.json"
SELECTION = ("Kallax_Shelf_Drawer", "Lack_TV_Bench")      # the furniture types of the selection set


def runs() -> dict:
    return json.loads((ROOT / "derived/runs.json").read_text())


def split_videos(split: str, furniture: tuple[str, ...] = SELECTION) -> list[str]:
    """Official subset ("training" or "testing") videos of these furniture types that have a run with visual.npz."""
    db = ip.load_annotations(ROOT)
    r = runs()
    sub = {"train": "training", "test": "testing"}[split]
    return sorted(v for v, a in db.items() if a["subset"] == sub and v.split("/")[0] in furniture and v in r
                  and (REPO_ROOT / "data/public/runs" / r[v] / "visual.npz").is_file())


def make_folds(seed: int = 42) -> dict:
    db = ip.load_annotations(ROOT)
    train = sorted(v for v, a in db.items() if a["subset"] == "training")
    rng = np.random.default_rng(seed)
    folds = {f"cv{i + 1}": [] for i in range(3)}
    for furn in sorted({v.split("/")[0] for v in train}):
        groups = sorted({v.split("/")[1][:4] for v in train if v.startswith(furn + "/")})
        order = rng.permutation(len(groups))
        for j, gi in enumerate(order):
            g = groups[gi]
            folds[f"cv{j % 3 + 1}"] += [v for v in train if v.startswith(f"{furn}/{g}")]
    out = {"rule": "official training subset; 3 folds by recording number within each furniture type, seed 42",
           "seed": seed, "folds": [{"name": k, "test_videos": sorted(v)} for k, v in folds.items()]}
    (REPO_ROOT / FOLDS).write_text(json.dumps(out, indent=1) + "\n")
    return out


def fold_of(vid: str) -> str:
    for f in json.loads((REPO_ROOT / FOLDS).read_text())["folds"]:
        if vid in f["test_videos"]:
            return f["name"]
    raise KeyError(vid)


def gt_steps(vid: str, duration_s: float) -> list[Step]:
    furn = vid.split("/")[0]
    db = ip.load_annotations(ROOT)
    n = int(round(duration_s * ip.FPS))
    steps = ip.gt_steps(furn, db[vid]["annotation"], n)
    names = ip.names(furn)
    return [Step(s.label if s.label == "other" else neutral_label(names.index(s.label) + 1), s.start_s, s.end_s,
                 s.source) for s in steps]


class IkeaBench:
    def __init__(self):
        self.r = runs()
        self._base: dict[str, tuple] = {}
        self._b: dict[tuple, dict | None] = {}
        self._vis: dict[str, tuple] = {}

    def cfg(self, vid: str, over: dict) -> Config:
        furn = vid.split("/")[0]
        root = REPO_ROOT / "data/public/runs" / self.r[vid]
        from .run_pose import stored_config
        manifest = json.loads((root / "manifest.json").read_text())
        base = stored_config(manifest)
        base["seg_a"]["engine"] = "custom"           # every row sets its keys; the stored run gives the rest
        return Config.load(overrides=deep_merge(base, {"steps": {"use": True, "step": [{"name": n} for n in
                                                                                        ip.names(furn)]}}, over))

    def video(self, vid: str, over: dict) -> dict:
        from ..analysis import visual
        from ..analysis.segmentation import build_timeline, create_engine
        from ..analysis.segmentation.base import RunTimeline
        from ..evaluation.steps_eval import evaluate
        from .run_pose import _MemoryCtx, boundary_probability, engine_b_needed
        cfg = self.cfg(vid, over)
        rid = self.r[vid]
        root = REPO_ROOT / "data/public/runs" / rid
        ctx = _MemoryCtx("public")
        engine = create_engine("A", cfg, ctx)
        if rid not in self._base:
            manifest = json.loads((root / "manifest.json").read_text())
            self._base = {rid: (manifest, build_timeline(manifest, root, engine.cleaner))}
        manifest, b = self._base[rid]
        tl = RunTimeline(b.fps, b.n_frames, b.person.copy(), b.energy, list(b.cuts), b.offset_s)
        if engine_b_needed(cfg):
            key = (rid, cfg["engine_b"]["model"])
            if key not in self._b:
                self._b = {k: v for k, v in self._b.items() if k[0] == rid}
                self._b[key] = boundary_probability(manifest, root, cfg, ctx)
            bp = self._b[key]
            if bp is not None:
                tl.boundary_prob, tl.embedding, tl.embedding_valid = bp["prob"], bp.get("embedding"), bp.get("embedding_valid")
        if cfg["seg_a"]["visual_weight"] > 0:
            if rid not in self._vis:
                self._vis = {rid: visual.to_frames(visual.load(root / "visual.npz"), tl.n_frames, tl.fps)}
            tl.visual, tl.visual_valid = self._vis[rid]
        pred = engine.segment_run(tl)
        dur = float(manifest["ingest"]["cfr"]["duration_s"])
        gt = gt_steps(vid, dur)
        res, _ = evaluate(pred, gt, dur, cfg["eval"]["seg_fps"], cfg["eval"]["overlaps"])
        return {"video": vid, "metrics": res, "in_task_order": ip.in_task_order(vid.split("/")[0], ip.gt_steps(
            vid.split("/")[0], ip.load_annotations(ROOT)[vid]["annotation"], int(round(dur * ip.FPS))))}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.ikea_bench")
    p.add_argument("--make-folds", action="store_true")
    args = p.parse_args(argv)
    if args.make_folds:
        out = make_folds()
        print({f["name"]: len(f["test_videos"]) for f in out["folds"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
