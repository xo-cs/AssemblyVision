"""A synthetic public run with DET v1 files and two shots, for the labelling tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from assemblyvision.perception.pose import save_det
from helpers import ffmpeg_clip, make_ctx

N_FRAMES = 100       # 4 s at 25 fps
SHOTS = [("s001", 0, 50, "fixed"), ("s002", 50, 100, "handheld")]


def person(x: float, conf: float = 0.9, joint_conf: float = 0.9, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    box = np.array([x, 20, x + 60, 200], np.float32)
    kp = np.stack([rng.uniform(x + 5, x + 55, 17), rng.uniform(25, 195, 17)], axis=1).astype(np.float32)
    return box, conf, kp, np.full(17, joint_conf, np.float32)


def frame_people(f: int) -> list[tuple]:
    """Frame 0..99: every 10th frame empty, frames 20..29 two overlapping people,
    frames 30..39 low confidence, otherwise one or two separate people."""
    if f % 10 == 0:
        return []
    if 20 <= f < 30:
        return [person(50, seed=f), person(70, seed=f + 1000)]
    if 30 <= f < 40:
        return [person(50, conf=0.3, joint_conf=0.4, seed=f)]
    return [person(40, seed=f)] + ([person(200, seed=f + 2000)] if f % 3 == 0 else [])


def make_run(root: Path) -> tuple[Path, dict]:
    cfg, ctx = make_ctx(root)
    ffmpeg_clip(ctx.paths.cfr, "-c:v", "libx264", "-pix_fmt", "yuv420p", seconds=N_FRAMES / 25)
    per_shot = []
    for sid, a, b, _ in SHOTS:
        rows = {"frame_idx": [], "bbox_xyxy": [], "det_conf": [], "body_xy": [], "body_conf": []}
        for f in range(a, b):
            for box, conf, kp, kc in frame_people(f):
                rows["frame_idx"].append(f)
                rows["bbox_xyxy"].append(box)
                rows["det_conf"].append(conf)
                rows["body_xy"].append(kp)
                rows["body_conf"].append(kc)
        arr = {"frame_idx": np.array(rows["frame_idx"], np.int32),
               "bbox_xyxy": np.array(rows["bbox_xyxy"], np.float32).reshape(-1, 4),
               "det_conf": np.array(rows["det_conf"], np.float32),
               "body_xy": np.array(rows["body_xy"], np.float32).reshape(-1, 17, 2),
               "body_conf": np.array(rows["body_conf"], np.float32).reshape(-1, 17)}
        save_det(ctx.paths.det(sid), ctx, arr, a, b - a, "x")
        per_shot.append({"shot_id": sid, "det_path": f"detections/{sid}.npz"})
    manifest = {"run_id": ctx.run_id, "source": "public",
                "ingest": {"cfr": {"path": "video/cfr.mp4", "fps": 25.0, "width": 320, "height": 240,
                                   "frame_count": N_FRAMES}},
                "shots": {"items": [{"shot_id": s, "start_frame": a, "end_frame": b, "shot_type": t}
                                    for s, a, b, t in SHOTS]},
                "pose_body": {"per_shot": per_shot}}
    ctx.paths.manifest.write_text(json.dumps(manifest))
    return ctx.paths.root, manifest


def ls_box_task(frame_idx: int, boxes: list[list[float]], labeller: str, W: int = 320, H: int = 240,
                overlap: bool = False) -> dict:
    """A Label Studio export task of the box pass (percent coordinates)."""
    res = [{"type": "rectanglelabels", "original_width": W, "original_height": H,
            "value": {"x": b[0] / W * 100, "y": b[1] / H * 100, "width": (b[2] - b[0]) / W * 100,
                      "height": (b[3] - b[1]) / H * 100, "rectanglelabels": ["person"]}} for b in boxes]
    return {"data": {"frame_name": f"f{frame_idx:07d}.png", "frame_idx": frame_idx, "group": "A", "overlap": overlap,
                     "width": W, "height": H},
            "annotations": [{"completed_by": {"id": 1, "email": labeller}, "result": res}]}
