"""A labelled copy of a run with another step engine, to show the progress (Week 11). CPU only.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.milestone_run --run RUN_DIR --label "Engine B v1" \\
        --set 'seg_a.engine="B v1"' [--gt GT_CSV] [--config LAYER.toml]

The new run lies next to the original, in the same data area. What does not depend on the step
engine is hard linked (no second copy on disk): the video, detections, keypoints, references,
paths, the SigLIP and VLM scores, body3d and scene4d. Then segmentation, metrics and the overlay
(and the eval with --gt) are made again in the copy with the given settings (tools/run_pose.py
rerun mode). The original run is never changed. manifest.json of the copy records the label and
where it came from; prints only the new run ID and the summary line.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from ..config import Config
from ..storage.files import new_run_id, run_source, utc_now

LINKED = ("video", "detections", "keypoints", "reference", "paths", "body3d", "scene4d")
LINKED_FILES = ("siglip.npz", "vlm.npz", "visual.npz", "run_config.toml")


def link_tree(src: Path, dst: Path) -> int:
    """dst as a copy of src made of hard links (symlinks stay symlinks); returns the file count."""
    n = 0
    for dirpath, _, files in os.walk(src):
        d = dst / Path(dirpath).relative_to(src)
        d.mkdir(parents=True, exist_ok=True)
        for f in files:
            s, t = Path(dirpath) / f, d / f
            if s.is_symlink():
                t.symlink_to(os.readlink(s))
            else:
                try:
                    os.link(s, t)
                except OSError:                      # another file system: a real copy
                    shutil.copy2(s, t)
            n += 1
    return n


def make_copy(src: Path, label: str, sets: list[str], hidden: bool = False) -> Path:
    rid = new_run_id()
    dst = src.parent / rid
    dst.mkdir()
    n = 0
    for name in LINKED:
        if (src / name).exists():
            n += link_tree(src / name, dst / name)
    for name in LINKED_FILES:
        if (src / name).is_file():
            os.link(src / name, dst / name)
            n += 1
    (dst / "log").mkdir()
    m = json.loads((src / "manifest.json").read_text())
    m["run_id"] = rid
    m.pop("worker", None)
    m.pop("eval", None)
    m["milestone"] = {"label": label, "copied_from": src.name, "sets": sets, "created_utc": utc_now(), "files_linked": n}
    if hidden:                                       # a test copy: not in the Results list (Runs page, show all runs)
        m["milestone"]["hidden"] = True
    m["status"] = "done"
    (dst / "manifest.json").write_text(json.dumps(m, indent=2))
    return dst


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.milestone_run")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--label", required=True, help='the name in the Results list, e.g. "Engine B v1"')
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--config", action="append", type=Path, default=[])
    p.add_argument("--gt", type=Path)
    p.add_argument("--hidden", action="store_true", help="a test copy, not listed in Results")
    args = p.parse_args(argv)
    _, src = run_source(args.run, Config.load()["paths"])
    dst = make_copy(src, args.label, args.set, args.hidden)
    cmd = [sys.executable, "-m", "assemblyvision.tools.run_pose", "--run", str(dst),
           "--stages", "segmentation,metrics,overlay" + (",eval" if args.gt else ""),
           "--set", 'overlay.encoder="libx264"', *sum((["--set", s] for s in args.set), []),
           *sum((["--config", str(c)] for c in args.config), []), *(["--gt", str(args.gt)] if args.gt else [])]
    r = subprocess.run(cmd, capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or [""])[-1]
    print(f"{dst.name} ({args.label}) from {src.name}: {last[:300]}")
    if r.returncode != 0:
        print(r.stderr.strip()[-1000:], file=sys.stderr)
        return r.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
