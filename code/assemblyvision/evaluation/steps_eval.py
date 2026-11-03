"""Segmentation evaluation of one run against one GT file (LLD 3.11 eval_segmentation,
without the split, freeze_rule and bootstrap parts, which come with Engine B)."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from ..errors import AvError
from ..phases import neutral_label
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


def by_phase_order(steps: list, names: list[str]) -> list:
    """Labels as Phase k by their position: a steps.toml name becomes Phase (its
    index + 1), Phase k and other stay. Public runs label Phase 1..N since
    phases.py, while public GT files and older public runs carry the names."""
    out = []
    for s in steps:
        if s.label in names:
            s = dataclasses.replace(s, label=neutral_label(names.index(s.label) + 1))
        out.append(s)
    return out


def score_steps(pred: list, gt: list, gt_name: str, vocab: list[str], source: str, cfg,
                duration: float) -> tuple[dict, list[dict], str | None]:
    """Metrics of pred against gt (no file is read or written). Public runs are
    scored by phase order (by_phase_order on both sides). Returns (metrics,
    boundary rows, label mapping or None)."""
    mapping = None
    if source == "public":
        names = [d["name"] for d in cfg["steps"]["step"]]
        pred, gt = by_phase_order(pred, names), by_phase_order(gt, names)
        vocab = [neutral_label(k + 1) for k in range(len(names))]
        mapping = "phase order: steps.toml names and Phase k compared by position"
    allowed = set(vocab) | {"other"}
    for s in gt:
        if s.label not in allowed:
            raise AvError("AV-EVL-001", file=gt_name, detail=f"label {s.label!r} is not in the vocabulary")
    if not any(s.label != "other" for s in gt):
        raise AvError("AV-EVL-004", kind="steps", split="whole")
    res, rows = evaluate(pred, gt, duration, cfg["eval"]["seg_fps"], cfg["eval"]["overlaps"])
    return res, rows, mapping


def evaluate_run(ctx: RunContext, gt_path: Path, cfg) -> dict:
    """Read <run>/steps.csv and the GT, compute, write <run>/eval.json, return the record."""
    paths = ctx.paths
    if not paths.steps_csv.is_file():
        raise AvError("AV-EVL-001", file="steps.csv", detail="the run has no steps; run the segmentation stage")
    pred = read_steps_csv(paths.steps_csv)
    gt = read_steps_csv(gt_path)
    vocab = ctx.manifest.get("segmentation", {}).get("vocabulary") or [d["name"] for d in cfg["steps"]["step"]]
    duration = float(ctx.manifest["ingest"]["cfr"]["duration_s"])
    fps = cfg["eval"]["seg_fps"]
    res, rows, mapping = score_steps(pred, gt, Path(gt_path).name, vocab, ctx.source, cfg, duration)
    rec = {"created_utc": utc_now(), "run_id": ctx.run_id, "engine": ctx.manifest.get("segmentation", {}).get("engine"),
           "solver": ctx.manifest.get("segmentation", {}).get("solver_used"),
           "gt_file": Path(gt_path).name, "gt_sha256": sha256(gt_path), "pred_file": "steps.csv",
           "duration_s": duration, "fps": fps, "metrics": res, "boundaries": rows,
           "gt_steps": sum(s.label != "other" for s in gt), "pred_steps": sum(s.label != "other" for s in pred),
           "definitions": DEFINITIONS}
    if mapping:
        rec["label_mapping"] = mapping
    if cfg["eval"]["label"]:
        rec["label"] = cfg["eval"]["label"]
    write_atomic(paths.eval_json, json.dumps(rec, indent=2).encode("utf-8"), paths)
    ctx.manifest["eval"] = {"path": ctx.rel(paths.eval_json), "gt_file": rec["gt_file"], **res,
                            **({"label": rec["label"]} if "label" in rec else {})}
    return rec


def format_metrics(res: dict) -> str:
    f1 = "  ".join(f"F1@{k[3:]} {v:.2f}" for k, v in res.items() if k.startswith("f1_"))
    mae = "n/a" if res["boundary_mae_s"] is None else f"{res['boundary_mae_s']:.2f} s"
    return (f"frame accuracy {res['frame_accuracy']:.2f}  edit {res['edit']:.2f}  {f1}  "
            f"boundary MAE {mae}  ({res['frames']} frames)")
