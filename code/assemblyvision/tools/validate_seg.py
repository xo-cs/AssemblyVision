"""Segmentation validation tables of stored runs (Week 12, docs_team/validation_segmentation.md). CPU.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.validate_seg --run RUN_DIR [--run RUN_DIR ...] --gt GT_CSV \\
        --annotator ANNOTATOR_A.csv --annotator ANNOTATOR_B.csv

For every run: per phase precision, recall and F1, the frame level confusion matrix (the phases and
intro/other) at the video's frame rate, the error of each boundary, and the score against the final labels
and against each annotator alone. Also the agreement of the two annotators (the human ceiling). Reads only
the runs' steps.csv and manifest and the label files; writes validation_segmentation.json and .md in
<data area>/validation/ (a QiSens run's stay under data/qisens) and prints the .md, which holds summary
numbers only: no frame, no keypoint, no boundary time, only errors.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from ..config import Config
from ..errors import AvError
from ..evaluation import metrics as M
from ..evaluation.steps_eval import by_phase_order, evaluate
from ..evaluation.steps_io import read_raw_csv, read_steps_csv
from ..storage.files import run_source, utc_now
from . import freeze

BG = "other"


def read_labels(path: Path) -> list:
    """A step CSV (start_s, end_s) or an annotator's raw CSV (start, end as m:ss, the end second inclusive)."""
    with open(path, newline="", encoding="utf-8") as f:
        head = next(csv.reader(f))
    return read_steps_csv(path) if "start_s" in [h.strip() for h in head] else read_raw_csv(path)


def confusion(pred: list[str], true: list[str], labels: list[str]) -> np.ndarray:
    """Frames, rows true, columns predicted."""
    idx = {x: i for i, x in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), np.int64)
    for t, p in zip(true, pred):
        cm[idx[t], idx[p]] += 1
    return cm


def per_class(cm: np.ndarray, labels: list[str]) -> dict:
    out = {}
    for i, name in enumerate(labels):
        tp, fp, fn = int(cm[i, i]), int(cm[:, i].sum() - cm[i, i]), int(cm[i].sum() - cm[i, i])
        out[name] = {"frames": int(cm[i].sum()), "precision": round(100 * tp / (tp + fp), 2) if tp + fp else None,
                     "recall": round(100 * tp / (tp + fn), 2) if tp + fn else None,
                     "f1": round(200 * tp / (2 * tp + fp + fn), 2) if 2 * tp + fp + fn else None}
    return out


def boundary_errors(pred: list, gt: list, duration: float) -> list[dict]:
    """One row per boundary of the labels: its two sides and the signed error of the nearest predicted boundary
    (positive: the system is late)."""
    bounds = M.step_boundaries(pred, duration, BG, 1e-6)
    named: dict[float, str] = {}
    for s in sorted((x for x in gt if x.label != BG), key=lambda x: x.start_s):
        for t, text in ((round(float(s.start_s), 3), f"start of {s.label}"), (round(float(s.end_s), 3), f"end of {s.label}")):
            if 1e-6 < t < duration - 1e-6:
                named[t] = f"{named[t]} = {text}" if t in named else text
    rows = []
    for g in M.step_boundaries(gt, duration, BG, 1e-6):
        err = None
        if bounds:
            err = round(float(min(bounds, key=lambda b: abs(b - g)) - g), 2)
        rows.append({"between": named[g], "error_s": err})
    return rows


def agreement(a: list, b: list, duration: float, fps: float) -> dict:
    x, y = M.frames_from_steps(a, duration, fps), M.frames_from_steps(b, duration, fps)
    kappa, po, _ = M.cohen_kappa(x, y)
    sa, sb = ({s.label: s for s in z if s.label != BG} for z in (a, b))       # each phase's start and end, by name
    pairs = [(sa[k].start_s, sb[k].start_s) for k in sa if k in sb] + [(sa[k].end_s, sb[k].end_s) for k in sa if k in sb]
    inside = [(p, q) for p, q in pairs if 1e-6 < p < duration - 1e-6 and 1e-6 < q < duration - 1e-6]
    diffs = [abs(p - q) for p, q in inside]
    return {"kappa": round(kappa, 4), "frame_agreement": round(100 * po, 2),
            "mean_boundary_difference_s": round(float(np.mean(diffs)), 2) if diffs else None,
            "max_boundary_difference_s": round(float(np.max(diffs)), 2) if diffs else None, "boundaries": len(diffs)}


def validate_run(root: Path, source: str, gt: list, annotators: list[tuple[str, list]], cfg) -> dict:
    m = json.loads((root / "manifest.json").read_text())
    cfr = m["ingest"]["cfr"]
    duration, fps = float(cfr["duration_s"]), float(cfr["fps"])
    pred = read_steps_csv(root / "steps.csv")
    names = [d["name"] for d in cfg["steps"]["step"]]
    if source == "public":                       # public runs are scored by phase order
        pred, gt = by_phase_order(pred, names), by_phase_order(gt, names)
        annotators = [(n, by_phase_order(a, names)) for n, a in annotators]
    p, g = M.frames_from_steps(pred, duration, fps), M.frames_from_steps(gt, duration, fps)
    labels = list(dict.fromkeys([s.label for s in sorted(gt, key=lambda s: s.start_s) if s.label != BG]))
    labels += sorted({x for x in p + g if x != BG and x not in labels})
    labels = [BG] + labels
    cm = confusion(p, g, labels)
    head, _ = evaluate(pred, gt, duration, cfg["eval"]["seg_fps"], cfg["eval"]["overlaps"])
    out = {"run": root.name, "label": (m.get("milestone") or {}).get("label"), "engine_source": sorted({s.source for s in pred}),
           "duration_s": duration, "video_fps": fps, "frames": len(g), "metrics": head,
           "frame_accuracy_video_fps": round(100 * float(np.trace(cm)) / max(len(g), 1), 2), "labels": labels,
           "confusion": cm.tolist(), "per_phase": per_class(cm, labels), "boundaries": boundary_errors(pred, gt, duration),
           "against_annotator": {}}
    for name, a in annotators:
        out["against_annotator"][name] = evaluate(pred, a, duration, cfg["eval"]["seg_fps"], cfg["eval"]["overlaps"])[0]
    return out


def show(v) -> str:
    return "-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v)


def markdown(res: dict) -> str:
    f = res["freeze"]
    L = ["# Segmentation validation tables", "",
         f"Freeze: {f['tag']} ({f['tag_commit'][:7]}), ran on {f['ran_on'][:7]}, method unchanged: {f['method_unchanged']}.", ""]
    if res.get("annotators"):
        a = res["annotators"]
        L += [f"Human agreement (the two annotators, independent): kappa {a['kappa']:.3f}, frame agreement "
              f"{a['frame_agreement']:.2f} %, mean boundary difference {show(a['mean_boundary_difference_s'])} s, "
              f"largest {show(a['max_boundary_difference_s'])} s.", ""]
    for r in res["runs"]:
        x = r["metrics"]
        L += [f"## {r['label'] or r['run']} ({r['run']})", "",
              f"Step source {', '.join(r['engine_source'])}. {r['duration_s']:.1f} s, {r['frames']} frames at {r['video_fps']:g} fps.", "",
              f"Frame accuracy {x['frame_accuracy']:.2f} (evaluation rate), {r['frame_accuracy_video_fps']:.2f} (every video frame); "
              f"edit {x['edit']:.2f}; F1@10 {x['f1_10']:.2f}, F1@25 {x['f1_25']:.2f}, F1@50 {x['f1_50']:.2f}; boundary MAE "
              f"{show(x['boundary_mae_s'])} s.", "", "| phase | frames | precision | recall | F1 |", "| --- | --- | --- | --- | --- |"]
        for k, v in r["per_phase"].items():
            L.append(f"| {'intro / other' if k == BG else k} | {v['frames']} | {show(v['precision'])} | {show(v['recall'])} | {show(v['f1'])} |")
        names = ["intro / other" if k == BG else k for k in r["labels"]]
        L += ["", "Confusion matrix in frames (rows true, columns predicted):", "", "| | " + " | ".join(names) + " |",
              "| --- |" + " --- |" * len(names)]
        L += [f"| {n} | " + " | ".join(map(str, row)) + " |" for n, row in zip(names, r["confusion"])]
        L += ["", "| boundary | error s (positive: system late) |", "| --- | --- |"]
        L += [f"| {b['between']} | {show(b['error_s'])} |" for b in r["boundaries"]]
        if r["against_annotator"]:
            L += ["", "| scored against | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s |", "| --- | --- | --- | --- | --- | --- | --- |"]
            for n, v in r["against_annotator"].items():
                L.append(f"| {n} | {v['frame_accuracy']:.2f} | {v['edit']:.2f} | {v['f1_10']:.2f} | {v['f1_25']:.2f} | {v['f1_50']:.2f} | {show(v['boundary_mae_s'])} |")
        L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.validate_seg")
    p.add_argument("--run", action="append", type=Path, required=True)
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--annotator", action="append", type=Path, default=[], help="give both annotators' files")
    a = p.parse_args(argv)
    if len(a.annotator) not in (0, 2):
        p.error("--annotator: give none or two files")
    base = Config.load()
    try:
        located = [run_source(r, base["paths"]) for r in a.run]
        if len({s for s, _ in located}) != 1:
            p.error("all runs must be from one data area")
        source = located[0][0]
        gt = read_labels(a.gt)
        ann = [(f"annotator {'AB'[i]}", read_labels(x)) for i, x in enumerate(a.annotator)]
        runs = [validate_run(root, source, gt, ann, base) for _, root in located]
    except (AvError, OSError, ValueError, KeyError) as e:
        print(f"validate_seg: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    res = {"created_utc": utc_now(), "freeze": freeze.record(), "runs": runs}
    if ann:
        res["annotators"] = agreement(ann[0][1], ann[1][1], runs[0]["duration_s"], runs[0]["video_fps"])
    out = located[0][1].parent.parent / "validation"
    out.mkdir(exist_ok=True)
    (out / "validation_segmentation.json").write_text(json.dumps(res, indent=1))
    md = markdown(res)
    (out / "validation_segmentation.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
