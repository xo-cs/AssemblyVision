"""Pick frames of a run for pose labelling, and make the crops of the keypoint pass.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.sample_frames frames --run RUN_DIR [--n 200] [--seed 42] [--overlap 20]
    python -m assemblyvision.tools.sample_frames crops --run RUN_DIR --boxes BOX_EXPORT.json

frames picks 150 to 300 frames (a short video gives fewer: the frames are at least
--min-gap-s apart within a shot) stratified over shots, fixed and handheld, and the
strata occluded, low_conf, none and normal (labelling/sampling.py), and writes
into RUN_DIR/labelling/: frames/*.png, tasks_boxes.json (Label Studio import),
boxes.xml and keypoints.xml (project configs) and sample.json (strata, seed,
groups). overlap frames go to both groups A and B for labeller agreement. A second
sample of the same run is refused unless --replace (old frames and crops are removed).
crops reads the box pass export and writes crops/*.png and tasks_keypoints.json.
Everything stays inside the run folder, so the frames of a QiSens run stay
under data/qisens/ (Khurshid runs it there; it is tested on public runs only).
The export file must lie in the run's own data area. Nothing is printed about
the content of a frame, only counts.
Exit codes: 0 success, 1 error, 2 refused or wrong argument.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import Config
from ..errors import AvError
from ..labelling import sampling, studio
from ..storage.files import REPO_ROOT, data_area, run_source


def main(argv: list[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.sample_frames")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("--run", type=Path, required=True)
    f.add_argument("--n", type=int, default=200)
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--overlap", type=int, default=20)
    f.add_argument("--min-gap-s", type=float, default=1.0)
    f.add_argument("--image-base", default="http://127.0.0.1:8081")
    f.add_argument("--replace", action="store_true", help="replace an earlier sample (its frames and crops)")
    c = sub.add_parser("crops")
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--boxes", type=Path, required=True, help="Label Studio JSON export of the box pass")
    c.add_argument("--image-base", default="http://127.0.0.1:8081")
    args = p.parse_args(argv)
    cfg = Config.load()
    try:
        source, root = run_source(args.run, cfg["paths"], repo_root)
        manifest = json.loads((root / "manifest.json").read_text())
        out = root / "labelling"
        if args.cmd == "frames":
            if not 1 <= args.n <= 1000:
                print("--n must be 1 to 1000 (150 to 300 for the labelling set)", file=sys.stderr)
                return 2
            if (out / "sample.json").exists():
                if not args.replace:
                    print(f"{out} already holds a sample; labelling may have started. Use --replace to "
                          "discard it (frames and crops).", file=sys.stderr)
                    return 2
                for old in list((out / "frames").glob("*.png")) + list((out / "frames" / "crops").glob("*.png")):
                    old.unlink()
            th = sampling.Thresholds()
            table = sampling.frame_table(manifest, root, th)
            fps = manifest["ingest"]["cfr"]["fps"]
            picks = sampling.sample(table, args.n, fps, args.seed, args.min_gap_s, args.overlap)
            sampling.write_frames(root / manifest["ingest"]["cfr"]["path"], picks, out / "frames")
            cfr = manifest["ingest"]["cfr"]
            sampling.write_json(out / "tasks_boxes.json",
                                sampling.box_tasks(picks, cfr["width"], cfr["height"], args.image_base))
            studio.write_configs(out)
            s = sampling.summary(picks, args.seed, th, manifest["run_id"])
            sampling.write_json(out / "sample.json", s)
            print(f"{s['frames']} frames ({s['overlap_frames']} overlap) in {out}")
            print("strata: " + ", ".join(f"{k} {v}" for k, v in s["by_stratum"].items()))
            print("shot types: " + ", ".join(f"{k} {v}" for k, v in s["by_shot_type"].items()))
            if not 150 <= s["frames"] <= 300:
                print(f"note: {s['frames']} frames, outside 150 to 300", file=sys.stderr)
            return 0
        if data_area(args.boxes, cfg["paths"], repo_root) != source:
            raise AvError("AV-STO-005", path=str(args.boxes), detail=f"the export must lie in the {source} data area")
        boxes = studio.parse_boxes(studio.load_export(args.boxes))
        tasks = sampling.crop_tasks(boxes, out / "frames", out / "frames" / "crops", args.image_base)
        sampling.write_json(out / "tasks_keypoints.json", tasks)
        print(f"{len(tasks)} crops from {len({b['frame_name'] for b in boxes})} frames in {out / 'frames' / 'crops'}")
        return 0
    except AvError as e:
        print(f"{e.code}: {e}", file=sys.stderr)
        return 2 if e.code in ("AV-STO-005", "AV-RUN-001") else 1


if __name__ == "__main__":
    sys.exit(main())
