"""IKEA ASM videos through the batch pipeline and the visual stage (Week 11).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.ikea_runs --gpu 0 [--max-minutes 55] [--furniture Kallax_Shelf_Drawer]

Public data only. For each downloaded top view video (complete by meta/rgb_top_index.json):
run_pose (YOLO26 x, default settings, ingest.max_fps = 10, no overlay) gives a public
run; tools/visual_embed then writes visual.npz into that run folder. The video to run
mapping is data/public/datasets/ikea_asm/derived/runs.json (resume: a video with a run
that has visual.npz is skipped). No new video is started after --max-minutes, so one
call stays under the hour; call it again to go on. Each subprocess takes the GPU lock.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

from ..storage.files import REPO_ROOT

ROOT = REPO_ROOT / "data/public/datasets/ikea_asm"
RUNS = ROOT / "derived/runs.json"
ORDER = ("Kallax_Shelf_Drawer", "Lack_TV_Bench", "Lack_Coffee_Table", "Lack_Side_Table")
SETS = ["overlay.enabled=false", "ingest.max_fps=10"]


def video_id(path: Path) -> str:
    return f"{path.parent.name}/{path.stem}"


def ready_videos(root: Path = ROOT) -> list[Path]:
    index = json.loads((root / "meta/rgb_top_index.json").read_text())
    out = []
    for item in index:
        p = root / item["path"]
        if p.is_file() and p.stat().st_size == item["size"]:
            out.append(p)
    return sorted(out, key=lambda p: (ORDER.index(p.parent.name), p.name))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.ikea_runs")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--max-minutes", type=float, default=55.0)
    p.add_argument("--furniture", choices=ORDER)
    args = p.parse_args(argv)
    runs = json.loads(RUNS.read_text()) if RUNS.is_file() else {}
    t0 = time.monotonic()
    done = failed = 0
    for video in ready_videos():
        vid = video_id(video)
        if args.furniture and video.parent.name != args.furniture:
            continue
        rid = runs.get(vid)
        if rid and (REPO_ROOT / "data/public/runs" / rid / "visual.npz").is_file():
            continue
        if (time.monotonic() - t0) / 60 > args.max_minutes:
            print(f"ikea_runs: time budget reached after {done} videos")
            break
        if not rid or not (REPO_ROOT / "data/public/runs" / rid / "manifest.json").is_file():
            cmd = [sys.executable, "-m", "assemblyvision.tools.run_pose", "--source", "public", "--video", str(video),
                   "--variant", "x", "--gpu", str(args.gpu)] + [x for s in SETS for x in ("--set", s)]
            r = subprocess.run(cmd, cwd=REPO_ROOT / "code", capture_output=True, text=True)
            m = re.search(r"^(run_\d{8}_\d{6}_[0-9a-f]{4}) done", r.stdout, re.M)
            if r.returncode != 0 or not m:
                print(f"ikea_runs: {vid}: pose failed: {(r.stderr or r.stdout).strip()[-300:]}")
                failed += 1
                continue
            rid = runs[vid] = m.group(1)
            RUNS.parent.mkdir(parents=True, exist_ok=True)
            RUNS.write_text(json.dumps(runs, indent=1, sort_keys=True) + "\n")
        r = subprocess.run([sys.executable, "-m", "assemblyvision.tools.visual_embed", "--run",
                            str(REPO_ROOT / "data/public/runs" / rid), "--gpu", str(args.gpu)],
                           cwd=REPO_ROOT / "code", capture_output=True, text=True)
        if r.returncode != 0:
            print(f"ikea_runs: {vid}: visual failed: {r.stderr.strip()[-300:]}")
            failed += 1
            continue
        done += 1
        print(f"ikea_runs: {vid} -> {rid} ({(time.monotonic() - t0) / 60:.1f} min)", flush=True)
    left = sum(1 for v in ready_videos() if not (runs.get(video_id(v)) and
                                                 (REPO_ROOT / "data/public/runs" / runs[video_id(v)] / "visual.npz").is_file()))
    print(f"ikea_runs: {done} done, {failed} failed, {left} ready videos left")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
