"""Run tracking again (CPU only) on the DET v1 files of an existing run, for
example to compare duplicate suppression on and off.

    python -m assemblyvision.tools.retrack --run data/public/runs/<run_id> --tag nodup \
        --set track.dup_suppress=false [--overlay]

Output goes to <run>/retrack/<tag>/ (keypoints, assignments, overlay,
manifest.json), so it stays in the data area of the run's source.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from ..config import DEFAULT_TOML, Config
from ..errors import AvError
from ..storage.files import REPO_ROOT, SOURCES, RunContext, RunPaths, source_root, utc_now
from .run_pose import _fail, parse_sets


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.retrack")
    p.add_argument("--run", required=True, type=Path, help="run folder")
    p.add_argument("--tag", required=True, help="name of the output folder under <run>/retrack/")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--overlay", action="store_true")
    args = p.parse_args(argv)
    if not args.tag.replace("_", "").replace("-", "").isalnum():
        print("--tag may only contain letters, digits, '-' and '_'", file=sys.stderr)
        return 2
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")

    try:
        base = Config.load()
        run_root = Path(os.path.realpath(args.run))
        source = next((s for s in SOURCES if run_root.parent == source_root(s, base["paths"]) / "runs"), None)
        if source is None:
            raise AvError("AV-STO-005", path=str(args.run), detail="not a run folder under data/<source>/runs/")
        manifest = json.loads((run_root / "manifest.json").read_text())
        overrides = manifest["config"]
        for section, values in parse_sets(args.set).items():
            overrides.setdefault(section, {}).update(values)
        cfg = Config.load([DEFAULT_TOML], overrides=overrides)
    except (OSError, ValueError, KeyError) as e:
        print(f"cannot read the run: {e}", file=sys.stderr)
        return 1
    except AvError as e:
        return _fail(e)

    from ..output.overlay import render_track_overlay
    from ..perception.tracker import Tracker, summarize
    from ..types import Shot

    run_id = manifest["run_id"]
    src = RunPaths(run_id, run_root)
    out = RunPaths(run_id, run_root / "retrack" / args.tag)
    out.root.parent.mkdir(exist_ok=True)
    try:
        out.create()
    except FileExistsError:
        print(f"{out.root} exists already; choose another --tag", file=sys.stderr)
        return 2
    log = logging.getLogger("assemblyvision.retrack")
    log.setLevel(logging.INFO)
    log.addHandler(logging.FileHandler(out.run_log, encoding="utf-8"))
    log.addHandler(logging.StreamHandler(sys.stderr))
    log.propagate = False
    ctx = RunContext(run_id, source, out, log)
    cfr = manifest["ingest"]["cfr"]
    ctx.manifest.update(run_id=run_id, retrack_of=str(run_root), tag=args.tag, created_utc=utc_now(),
                        sets=args.set, config=cfg.snapshot(), notices=[], ingest={"cfr": cfr},
                        tracking={"per_shot": [], "tracks": []})
    shots = [Shot(i["shot_id"], run_id, i["start_frame"], i["end_frame"], i["start_s"], i["end_s"],
                  i["shot_type"], i.get("scale_m_per_px")) for i in manifest["shots"]["items"]]
    refs = {}
    for s in shots:
        if src.ref_json(s.shot_id).is_file():
            refs[s.shot_id] = json.loads(src.ref_json(s.shot_id).read_text())
    meta = {"pose_model": f"yolo26{manifest['pose_body']['variant']}-pose",
            "pose_variant": manifest["pose_body"]["variant"], "seed": cfg["run"]["seed"]}
    tracker = Tracker(cfg, ctx)
    for s in shots:
        tracker.track(run_id, s, src.det(s.shot_id), refs.get(s.shot_id), fps=cfr["fps"],
                      size=(cfr["width"], cfr["height"]), meta=meta)
    totals = summarize(ctx)
    if args.overlay:
        items = {p["shot_id"]: {"det": src.det(p["shot_id"]), "assign": out.assign(p["shot_id"]),
                                "subject": int(p["subject"].rsplit("-t", 1)[1]) if p["subject"] else None}
                 for p in ctx.manifest["tracking"]["per_shot"]}
        render_track_overlay(src.cfr, shots, items, out.tracks_overlay, cfr["fps"], cfg, ctx)
    ctx.write_manifest()
    print(f"retrack {args.tag}: {totals}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
