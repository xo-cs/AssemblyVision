"""Our pose and tracking on public dataset videos, for Engine B (Week 10).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.dataset_pose --dataset ha4m --videos IDU001V001,IDU023V001 --gpu 0

For each video it runs the batch pipeline (tools/run_pose, default settings, so
YOLO26 x FP32 batch 1 and the GPU lock apply; the overlay is off) on
data/public/datasets/<name>/rgb/<video>.mp4, then writes
data/public/datasets/<name>/derived/yolo26x/<video>/:
    kts/            the KTS v1 files of the run (every track, with is_subject)
    subject.npz     the subject's COCO 17 keypoints over the whole video, row = dataset frame id
    labels.csv      frame, action, repetition (the dataset's labels)
    source.json     run id, git commit, model checksum, configuration hash
Any video or output outside data/public is refused (AV-RUN-001, AV-STO-005).
A video whose source.json exists is skipped (resume). One GPU job; ask first.
Exit codes: 0 success, 1 a video failed, 2 refused or wrong argument.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..config import Config
from ..errors import AvError
from ..experiments.provenance import git_state, model_checksum
from ..kts import read_kts
from ..storage.files import REPO_ROOT, RunPaths, check_video_source, data_area, utc_now

MODEL_TAG = "yolo26x"


def merge_subject(kts_files: list[Path], n_frames: int, first_frame_id: int = 0) -> dict[str, np.ndarray]:
    """COCO 17 keypoints of the subject over the whole video: per shot the is_subject track
    (else the track with most frames present); row r = dataset frame id first_frame_id + mp4 index."""
    xy = np.full((n_frames, 17, 2), np.nan, np.float32)
    conf = np.zeros((n_frames, 17), np.float32)
    box = np.full((n_frames, 4), np.nan, np.float32)
    track = np.full(n_frames, "", dtype="<U48")
    shots: dict[str, list] = {}
    for f in kts_files:
        s, meta = read_kts(f)
        shots.setdefault(s.shot_id, []).append((bool(meta.get("is_subject")), int(s.present.sum()), s))
    for sid, items in sorted(shots.items()):
        _, _, s = max(items, key=lambda t: (t[0], t[1]))
        rows = s.frame_idx.astype(np.int64) + first_frame_id
        ok = s.present & (rows >= 0) & (rows < n_frames)
        xy[rows[ok]] = s.body_xy[ok]
        conf[rows[ok]] = np.nan_to_num(s.body_conf[ok])
        box[rows[ok]] = s.bbox_xyxy[ok]
        track[rows[ok]] = s.track_id
    return {"frame_id": np.arange(n_frames, dtype=np.int32), "body_xy": xy, "body_conf": conf, "bbox_xyxy": box,
            "track_id": track}


def config_hash(manifest: dict) -> str:
    cfg = manifest.get("config", {})
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def run_pipeline(video: Path, gpu: int, repo_root: Path, run=subprocess.run) -> str:
    cmd = [sys.executable, "-m", "assemblyvision.tools.run_pose", "--source", "public", "--video", str(video),
           "--variant", "x", "--gpu", str(gpu), "--set", "overlay.enabled=false"]
    p = run(cmd, cwd=repo_root / "code", capture_output=True, text=True)
    m = re.search(r"^(run_\d{8}_\d{6}_[0-9a-f]{4}) ", p.stdout, re.M)
    if p.returncode != 0 or not m:
        raise AvError("AV-RUN-004", detail=(p.stderr or p.stdout).strip()[-600:])
    return m.group(1)


def derive(dataset_root: Path, video_id: str, run_id: str, cfg: Config, repo_root: Path) -> Path:
    paths = RunPaths.for_source("public", run_id, cfg["paths"], repo_root)
    manifest = json.loads(paths.manifest.read_text())
    frames = json.loads((dataset_root / "rgb" / f"{video_id}.frames.json").read_text())
    out = dataset_root / "derived" / MODEL_TAG / video_id
    if data_area(out, cfg["paths"], repo_root) != "public":
        raise AvError("AV-STO-005", path=str(out), detail="derived keypoints are stored under data/public only")
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "kts").mkdir(parents=True)
    kts = sorted(paths.keypoints_dir.glob("*.npz"))
    for f in kts:
        shutil.copy2(f, tmp / "kts" / f.name)
        shutil.copy2(f.with_suffix(".json"), tmp / "kts" / f.with_suffix(".json").name)
    first = int(frames.get("first_frame_id", 0))
    n = first + int(frames["n_frames"])
    sub = merge_subject(kts, n, first)
    np.savez_compressed(tmp / "subject.npz", **sub)
    shutil.copy2(dataset_root / "labels" / f"{video_id}.csv", tmp / "labels.csv")
    weights = repo_root / cfg["paths"]["models_dir"] / "yolo26x-pose.pt"
    src = {"dataset": dataset_root.name, "video": video_id, "run_id": run_id,
           "run_dir": paths.root.relative_to(repo_root).as_posix(), "git": git_state(repo_root),
           "model": model_checksum(weights) if weights.is_file() else None,
           "pose": {k: manifest.get("pose_body", {}).get(k) for k in ("variant", "precision", "batch", "tf32", "imgsz",
                                                                      "conf", "frames", "pose_fps")},
           "config_sha256_16": config_hash(manifest), "rgb_mp4_sha256": frames.get("mp4_sha256"),
           "frames": n, "frames_with_subject": int(np.isfinite(sub["body_xy"]).all(axis=2).any(axis=1).sum()),
           "tracks": len(kts), "written_utc": utc_now()}
    (tmp / "source.json").write_text(json.dumps(src, indent=1) + "\n")
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    return out


def main(argv: list[str] | None = None, repo_root: Path = REPO_ROOT, run=subprocess.run) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.dataset_pose")
    p.add_argument("--dataset", required=True, choices=("ha4m",))
    p.add_argument("--videos", required=True, help="comma separated video ids")
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args(argv)
    cfg = Config.load()
    root = repo_root / cfg["datasets"]["root"] / args.dataset
    if data_area(root, cfg["paths"], repo_root) != "public":
        print("AV-STO-005: datasets are stored under data/public only", file=sys.stderr)
        return 2
    failed = []
    for vid in [v.strip() for v in args.videos.split(",") if v.strip()]:
        try:
            if (root / "derived" / MODEL_TAG / vid / "source.json").is_file():
                print(f"{vid}: already derived, skipped")
                continue
            video = check_video_source(root / "rgb" / f"{vid}.mp4", "public", cfg["paths"], repo_root)
            if not (root / "rgb" / f"{vid}.frames.json").is_file():
                raise AvError("AV-DAT-003", name=f"{args.dataset} {vid}", detail="its RGB is not fetched yet")
            run_id = run_pipeline(video, args.gpu, repo_root, run)
            out = derive(root, vid, run_id, cfg, repo_root)
            s = json.loads((out / "source.json").read_text())
            print(f"{vid}: {run_id}, {s['frames_with_subject']} of {s['frames']} frames with the subject -> {out}")
        except AvError as e:
            print(f"{vid}: {e.code}: {e}", file=sys.stderr)
            if e.code in ("AV-RUN-001", "AV-STO-005"):
                return 2
            failed.append(vid)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
