"""Link hand labels to a video by its SHA256, so every run of that video shows its evaluation (CPU).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.link_gt --source qisens --gt ../data/qisens/annotations/steps_gt.csv --run RUN_DIR
    python -m assemblyvision.tools.link_gt --source public --gt GT.csv --video VIDEO     # hashes the video
    python -m assemblyvision.tools.link_gt --source qisens --list

--run takes the video's SHA256 the run recorded at ingest (no video is read). Writes
data/<source>/annotations/gt_links.json; prints only file names and counts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import Config
from ..storage.files import SOURCES, run_source, source_root


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.link_gt")
    p.add_argument("--source", choices=SOURCES, required=True)
    p.add_argument("--gt", type=Path)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--run", type=Path, help="a run of the video (its recorded SHA256 is used)")
    g.add_argument("--video", type=Path, help="the video file (hashed)")
    p.add_argument("--list", action="store_true")
    args = p.parse_args(argv)
    from ..evaluation import gt_links as L
    cfg = Config.load()
    area = source_root(args.source, cfg["paths"])
    if args.list:
        for x in L.load(area):
            print(f"{x['video_name'] or '?'} ({x['video_sha256'][:12]}) -> {x['gt']}")
        return 0
    if args.gt is None or (args.run is None and args.video is None):
        p.error("--gt and --run or --video are needed")
    if args.run is not None:
        src, root = run_source(args.run, cfg["paths"])
        if src != args.source:
            p.error(f"the run belongs to {src}")
        m = json.loads((root / "manifest.json").read_text())
        sha, name = m.get("ingest", {}).get("input_sha256"), Path(m.get("video", "")).name
        if not sha:
            print("this run recorded no video SHA256", file=sys.stderr)
            return 1
    else:
        from ..ingest.ingestor import sha256_file
        from ..storage.files import check_video_source
        video = check_video_source(args.video, args.source, cfg["paths"])
        sha, name = sha256_file(video), video.name
    try:
        rec = L.add(area, sha, args.gt, name)
    except ValueError:
        print(f"the hand labels must lie in {area}", file=sys.stderr)
        return 1
    n = 0
    runs = area / "runs"
    for d in runs.iterdir() if runs.is_dir() else []:
        try:
            n += json.loads((d / "manifest.json").read_text()).get("ingest", {}).get("input_sha256") == sha
        except (OSError, ValueError):
            pass
    print(f"linked {rec['gt']} to {name or 'the video'} ({sha[:12]}); {n} runs of this video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
