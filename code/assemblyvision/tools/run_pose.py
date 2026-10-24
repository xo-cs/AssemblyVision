"""Batch analysis from the command line: ingest, shots, reference frames, YOLO26
pose, tracking (KTS v1) and the track overlay, in LLD order (3.0.6).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.run_pose --source {qisens,public} --video PATH \
        --variant {n,s,m,l} --gpu 0

Source qisens writes everything (run folder, temp files, logs, manifest) under
data/qisens/runs/<run_id>/, source public under data/public/runs/<run_id>/. A
video outside the data area of its source is refused before anything is
opened (AV-RUN-001). Exit codes: 0 success, 1 error, 2 wrong argument (LLD 5.2).
"""
from __future__ import annotations

import argparse
import logging
import os
import platform
import random
import sys
import tempfile
import time
import tomllib
import traceback
from pathlib import Path

from ..config import DEFAULT_TOML, VARIANTS, Config
from ..errors import AvError
from ..storage.files import (REPO_ROOT, SOURCE_TYPE, SOURCES, RunContext, RunPaths, check_video_source,
                             new_run_id, utc_now, write_atomic)

STAGES = ("ingest", "shots", "reference", "pose_body", "tracking", "overlay")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.run_pose",
                                description="Run YOLO26 pose on one video (first pose stage).")
    p.add_argument("--source", required=True, choices=SOURCES)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--gpu", required=True, type=int, help="GPU index in PCI bus order, -1 for CPU")
    p.add_argument("--config", action="append", type=Path, default=[],
                   help="extra TOML layer over configs/default.toml (repeatable)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override one setting, value in TOML syntax (repeatable)")
    p.add_argument("--no-overlay", action="store_true", help="skip the overlay video")
    return p.parse_args(argv)


def parse_sets(items: list[str]) -> dict:
    """--set track.dup_suppress=false -> {"track": {"dup_suppress": False}} (AV-CFG-002 on bad syntax)."""
    out: dict = {}
    for item in items:
        key, sep, value = item.partition("=")
        section, dot, name = key.strip().partition(".")
        try:
            if not sep or not dot:
                raise ValueError("expected SECTION.KEY=VALUE")
            parsed = tomllib.loads(f"v = {value.strip()}")["v"]
        except (ValueError, tomllib.TOMLDecodeError) as e:
            raise AvError("AV-CFG-002", key=key.strip() or item, value=repr(value), range=f"TOML value ({e})") from e
        out.setdefault(section, {})[name] = parsed
    return out


def _fail(err: AvError) -> int:
    print(f"{err.code}: {err.text}", file=sys.stderr)
    if err.detail:
        print(f"  detail: {err.detail}", file=sys.stderr)
    return 1


def _setup_logging(paths: RunPaths) -> logging.Logger:
    log = logging.getLogger("assemblyvision")
    log.setLevel(logging.DEBUG)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(paths.run_log, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def _run_stage(ctx: RunContext, name: str, fn):
    """Run one stage with its own log file; AV errors pass through, anything
    else becomes AV-SYS-001 with the traceback in the stage log only."""
    rec = ctx.manifest["stages"][name]
    rec.update(status="running", started_utc=utc_now())
    handler = logging.FileHandler(ctx.paths.stage_log(name), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    ctx.log.addHandler(handler)
    t0 = time.monotonic()
    try:
        result = fn()
        rec["status"] = "done"
        return result
    except AvError as e:
        rec.update(status="failed", error=e.as_record())
        ctx.log.error("%s %s%s", e.code, e.text, f" ({e.detail})" if e.detail else "")
        raise
    except Exception as e:
        handler.stream.write(traceback.format_exc())
        err = AvError("AV-SYS-001", type=type(e).__name__, stage=name, detail=str(e)[:300])
        rec.update(status="failed", error=err.as_record())
        ctx.log.error("%s %s", err.code, err.text)
        raise err from e
    finally:
        rec["seconds"] = round(time.monotonic() - t0, 2)
        ctx.log.removeHandler(handler)
        handler.close()
        ctx.write_manifest()


def _versions() -> dict:
    import cv2
    import numpy
    import torch
    import ultralytics

    from .. import __version__
    from ..ingest.ingestor import ffmpeg_version
    return {"assemblyvision": __version__, "python": platform.python_version(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__, "opencv": cv2.__version__,
            "numpy": numpy.__version__, "ffmpeg": ffmpeg_version()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Must be set before torch or ultralytics are imported.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) if args.gpu >= 0 else ""
    os.environ["YOLO_OFFLINE"] = "1"              # no analytics, no downloads (FR-POSE-06)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # platformdirs (scenedetect) must not write to ~/.config or ~/.cache outside ~/AssemblyVision
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

    try:
        overrides = parse_sets(args.set)
        overrides.setdefault("pose", {})["variant"] = args.variant
        if args.gpu >= 0:
            overrides["gpu"] = {"pose": args.gpu}
        cfg = Config.load([DEFAULT_TOML, *args.config], overrides=overrides)
        video = check_video_source(args.video, args.source, cfg["paths"])
    except AvError as e:
        return _fail(e)

    while True:
        run_id = new_run_id()
        paths = RunPaths.for_source(args.source, run_id, cfg["paths"])
        try:
            paths.create()
            break
        except FileExistsError:
            continue
    os.environ["TMPDIR"] = str(paths.tmp_dir)     # temp files of this process stay in the run folder
    tempfile.tempdir = None
    log = _setup_logging(paths)
    ctx = RunContext(run_id, args.source, paths, log)
    ctx.manifest.update(
        run_id=run_id, source=args.source, source_type=SOURCE_TYPE[args.source],
        status="running", created_utc=utc_now(), command=" ".join(sys.argv),
        video=str(video), variant=args.variant, gpu=args.gpu, config_sources=cfg.sources,
        config=cfg.snapshot(), stages={s: {"status": "pending"} for s in STAGES}, notices=[])
    write_atomic(paths.run_config, cfg.to_toml().encode("utf-8"), paths)
    log.info("run %s, source %s, video %s, variant %s, gpu %s", run_id, args.source, video.name,
             args.variant, args.gpu)
    log.info("run folder %s", paths.root)

    import numpy as np
    import torch

    from ..ingest.ingestor import VideoIngestor
    from ..ingest.shots import ShotDetector
    from ..output.overlay import render_track_overlay
    from ..perception.pose import PoseEstimator
    from ..perception.tracker import Tracker, summarize

    seed = cfg["run"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    ctx.manifest["versions"] = _versions()
    ctx.write_manifest()

    status, code = "done", 0
    try:
        if args.gpu >= 0 and not torch.cuda.is_available():
            raise AvError("AV-SYS-001", type="CudaUnavailable", stage="pose_body",
                          detail=f"CUDA is not available for GPU {args.gpu}")
        info, _split = _run_stage(ctx, "ingest", lambda: VideoIngestor(cfg, ctx).ingest(
            run_id, video, SOURCE_TYPE[args.source], strict_split=False))

        shots = _run_stage(ctx, "shots", lambda: ShotDetector(cfg, ctx).detect(run_id, paths.cfr, info))

        pose = PoseEstimator(cfg, ctx, device="cuda:0" if args.gpu >= 0 else "cpu")
        refs = _run_stage(ctx, "reference", lambda: (pose.load_models(args.variant),
                                                     pose.reference_frames(run_id, shots))[1])
        ref_by_shot = {r["shot_id"]: r for r in refs}
        for item in ctx.manifest["shots"]["items"]:
            item["ref_frame"] = ref_by_shot.get(item["shot_id"], {}).get("frame_idx")
        det_paths = _run_stage(ctx, "pose_body", lambda: pose.estimate_body_shots(run_id, shots))
        pose_meta = {"pose_model": f"yolo26{pose.variant}-pose", "pose_variant": pose.variant, "seed": seed}

        def track_all() -> dict:
            tracker = Tracker(cfg, ctx)
            ctx.manifest["tracking"] = {"per_shot": [], "tracks": []}
            for s in shots:
                tracker.track(run_id, s, det_paths[s.shot_id], ref_by_shot.get(s.shot_id), fps=info.fps,
                              size=(info.width, info.height), meta=pose_meta)
            return summarize(ctx)
        _run_stage(ctx, "tracking", track_all)

        if args.no_overlay:
            ctx.manifest["stages"]["overlay"]["status"] = "skipped"
        else:
            items = {}
            for p in ctx.manifest["tracking"]["per_shot"]:
                subj = p["subject"]
                items[p["shot_id"]] = {"det": det_paths[p["shot_id"]], "assign": paths.assign(p["shot_id"]),
                                       "subject": int(subj.rsplit("-t", 1)[1]) if subj else None}
            try:
                _run_stage(ctx, "overlay", lambda: render_track_overlay(paths.cfr, shots, items, paths.tracks_overlay,
                                                                        info.fps, cfg, ctx))
            except AvError as e:
                status, code = "partial", _fail(e)
    except AvError as e:
        status, code = "failed", _fail(e)
        ctx.manifest.setdefault("error", e.as_record())
    finally:
        ctx.manifest["status"] = status
        ctx.manifest["finished_utc"] = utc_now()
        ctx.write_manifest()

    pb = ctx.manifest.get("pose_body", {})
    tt = ctx.manifest.get("tracking", {}).get("totals", {})
    ing = ctx.manifest.get("ingest", {}).get("cfr", {})
    summary = (f"{run_id} {status}: variant {pb.get('variant', args.variant)}, "
               f"{pb.get('frames', 0)} frames ({ing.get('width')}x{ing.get('height')} at {ing.get('fps')} fps), "
               f"pose {pb.get('pose_fps')} fps, peak GPU {pb.get('gpu_peak_mb')} MB, "
               f"{pb.get('failed_frames', 0)} failed frames, {ctx.manifest.get('shots', {}).get('count', 0)} shots, "
               f"{tt.get('tracks_kept', 0)} tracks, {tt.get('short_tracks', 0)} short tracks, "
               f"{tt.get('duplicates_suppressed', 0)} duplicates suppressed, run folder {paths.root}")
    log.info(summary)
    print(summary)
    return code


if __name__ == "__main__":
    sys.exit(main())
