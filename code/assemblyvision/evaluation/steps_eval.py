"""Segmentation evaluation of one run against one GT file (LLD 3.11 eval_segmentation,
without the split, freeze_rule and bootstrap parts, which come with Engine B)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..errors import AvError
from ..storage.files import RunContext, utc_now, write_atomic
from . import metrics as M
from .steps_io import read_steps_csv

DEFINITIONS = {
    "reference": "MS TCN eval.py (Abu Farha and Gall, CVPR 2019); edit and F1@k from Lea et al., CVPR 2017",
    "frame_accuracy": "percent of sampled frames with an equal label, other included",
    "edit": "(1 - Levenshtein(segment labels) / max length) * 100, other removed",
    "f1_k": "segmental F1 at IoU k (same label, each GT segment matched once), other removed, percent",
    "boundary_mae_s": "not MS TCN: mean over GT boundaries (start and end of every non other GT step, "
                      "without 0 and the end) of the distance to the nearest predicted boundary, seconds",
    "sampling": "frame i at time i / fps over [0, duration); frames without a step are other",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(pred: list, gt: list, duration_s: float, fps: float,
             overlaps: list[float]) -> tuple[dict, list[dict]]:
    p = M.frames_from_steps(pred, duration_s, fps)
    g = M.frames_from_steps(gt, duration_s, fps)
    out = {"frames": len(g), "frame_accuracy": round(M.frame_accuracy(p, g), 2),
           "edit": round(M.edit_score(p, g), 2)}
    for k in overlaps:
        out[f"f1_{round(k * 100):d}"] = round(M.f1_overlap(p, g, k), 2)
    mae, rows = M.boundary_mae(pred, gt, duration_s)
    out["boundary_mae_s"] = mae
    return out, rows


def evaluate_run(ctx: RunContext, gt_path: Path, cfg) -> dict:
    """Read <run>/steps.csv and the GT, compute, write <run>/eval.json, return the record."""
    paths = ctx.paths
    if not paths.steps_csv.is_file():
        raise AvError("AV-EVL-001", file="steps.csv", detail="the run has no steps; run the segmentation stage")
    pred = read_steps_csv(paths.steps_csv)
    gt = read_steps_csv(gt_path)
    vocab = ctx.manifest.get("segmentation", {}).get("vocabulary") or [d["name"] for d in cfg["steps"]["step"]]
    allowed = set(vocab) | {"other"}
    for s in gt:
        if s.label not in allowed:
            raise AvError("AV-EVL-001", file=Path(gt_path).name, detail=f"label {s.label!r} is not in the vocabulary")
    if not any(s.label != "other" for s in gt):
        raise AvError("AV-EVL-004", kind="steps", split="whole")
    duration = float(ctx.manifest["ingest"]["cfr"]["duration_s"])
    fps = cfg["eval"]["seg_fps"]
    res, rows = evaluate(pred, gt, duration, fps, cfg["eval"]["overlaps"])
    rec = {"created_utc": utc_now(), "run_id": ctx.run_id, "engine": ctx.manifest.get("segmentation", {}).get("engine"),
           "solver": ctx.manifest.get("segmentation", {}).get("solver_used"),
           "gt_file": Path(gt_path).name, "gt_sha256": sha256(gt_path), "pred_file": "steps.csv",
           "duration_s": duration, "fps": fps, "metrics": res, "boundaries": rows,
           "gt_steps": sum(s.label != "other" for s in gt), "pred_steps": sum(s.label != "other" for s in pred),
           "definitions": DEFINITIONS}
    write_atomic(paths.eval_json, json.dumps(rec, indent=2).encode("utf-8"), paths)
    ctx.manifest["eval"] = {"path": ctx.rel(paths.eval_json), "gt_file": rec["gt_file"], **res}
    return rec


def format_metrics(res: dict) -> str:
    f1 = "  ".join(f"F1@{k[3:]} {v:.2f}" for k, v in res.items() if k.startswith("f1_"))
    mae = "n/a" if res["boundary_mae_s"] is None else f"{res['boundary_mae_s']:.2f} s"
    return (f"frame accuracy {res['frame_accuracy']:.2f}  edit {res['edit']:.2f}  {f1}  "
            f"boundary MAE {mae}  ({res['frames']} frames)")
