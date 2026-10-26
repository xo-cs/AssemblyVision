"""Fake data areas with stored runs, for RunStore, RunManager and the interface."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from assemblyvision.config import Config
from assemblyvision.storage.files import RunPaths, source_root

STEPS_CSV = """step,label,start_s,end_s,source,score
0,other,0.000,2.000,engine_a,
1,Removal,2.000,10.000,engine_a,1.000
0,other,10.000,11.000,engine_a,
2,Cleaning,11.000,20.000,engine_a,1.000
"""
METRIC_ROWS = [("duration_s", 8, "s", 1.0, ""), ("bend_b0_s", 5, "s", 1.0, "view_dependent"),
               ("bend_b1_s", 2, "s", 1.0, "view_dependent"), ("bend_b2_s", 1, "s", 1.0, "view_dependent"),
               ("bend_b3_s", 0, "s", 1.0, "view_dependent"), ("two_hand_simultaneous", None, "ratio", 0.3,
                                                             "low_coverage")]


def make_root(root: Path, cfg: Config | None = None) -> Config:
    cfg = cfg or Config.load()
    for s in ("qisens", "public"):
        for d in ("runs", "videos"):
            (source_root(s, cfg["paths"], root) / d).mkdir(parents=True, exist_ok=True)
    return cfg


def make_run(root: Path, cfg: Config, run_id: str, source: str = "public", status: str = "done",
             stages: dict | None = None, with_results: bool = True, eval_json: bool = True) -> RunPaths:
    p = RunPaths.for_source(source, run_id, cfg["paths"], repo_root=root)
    p.create()
    if stages is None:
        stages = {s: {"status": "done", "seconds": 1.0} for s in
                  ("ingest", "shots", "reference", "pose_body", "tracking", "segmentation", "metrics", "overlay")}
    sid = f"{run_id}-s001"
    tid = f"{sid}-t01"
    m = {"run_id": run_id, "source": source, "status": status, "created_utc": "2026-10-26T00:00:00Z",
         "video": str(source_root(source, cfg["paths"], root) / "videos" / "clip.mp4"), "variant": "m",
         "stages": stages, "notices": [{"stage": "metrics", "code": "AV-MOT-001", "text": "x"}],
         "ingest": {"cfr": {"duration_s": 20.0, "fps": 30.0}},
         "shots": {"items": [{"shot_id": sid, "shot_type": "fixed", "start_frame": 0, "end_frame": 600}]},
         "tracking": {"per_shot": [{"shot_id": sid, "subject": tid, "subject_source": "auto"}],
                      "tracks": [{"track_id": tid, "kts_path": f"keypoints/{tid}.npz"}]},
         "metrics": {"baseline": [{"label": "Removal", "n": 1, "median_s": 8.0, "q1_s": 8.0, "q3_s": 8.0,
                                   "flags": "low_n"}], "params": {"motion": {"angle_bands": [20, 45, 60]}}}}
    p.manifest.write_text(json.dumps(m))
    if with_results:
        p.steps_csv.write_text(STEPS_CSV)
        lines = ["segment_id,step,label,start_s,end_s,metric,value,unit,coverage,flags"]
        for seg, label, a, b in ((2, "Removal", 2, 10), (4, "Cleaning", 11, 20)):
            for name, v, unit, cov, flags in METRIC_ROWS:
                lines.append(f"{run_id}-A-{seg:04d},{seg},{label},{a},{b},{name},{'' if v is None else v},"
                             f"{unit},{cov},{flags}")
        p.metrics_csv.write_text("\n".join(lines) + "\n")
        import cv2
        cv2.imwrite(str(p.ref_png(sid)), np.full((120, 160, 3), 128, np.uint8))
        p.ref_json(sid).write_text(json.dumps({"shot_id": sid, "frame_idx": 5, "boxes": []}))
        T = 10
        np.savez(p.kts(tid), frame_idx=np.arange(T, dtype=np.int32), present=np.ones(T, bool),
                 bbox_xyxy=np.tile(np.array([10, 20, 60, 110], np.float32), (T, 1)))
        if eval_json:
            p.eval_json.write_text(json.dumps({"engine": "A", "gt_file": "gt.csv", "gt_steps": 2, "pred_steps": 2,
                                               "metrics": {"frame_accuracy": 90.0, "edit": 100.0, "f1_10": 100.0,
                                                           "f1_25": 100.0, "f1_50": 50.0,
                                                           "boundary_mae_s": 1.5}}))
    return p
