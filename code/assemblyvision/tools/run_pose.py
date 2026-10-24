"""First pose stage from the command line: ingest, one shot, YOLO26 pose, overlay.

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
import traceback
from pathlib import Path

from ..config import DEFAULT_TOML, VARIANTS, Config
from ..errors import AvError
from ..storage.files import (SOURCE_TYPE, SOURCES, RunContext, RunPaths, check_video_source,
                             new_run_id, utc_now, write_atomic)

STAGES = ("ingest", "shots", "pose_body", "overlay")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.run_pose",
                                description="Run YOLO26 pose on one video (first pose stage).")
    p.add_argument("--source", required=True, choices=SOURCES)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--gpu", required=True, type=int, help="GPU index in PCI bus order, -1 for CPU")
    p.add_argument("--config", action="append", type=Path, default=[],
                   help="extra TOML layer over configs/default.toml (repeatable)")
    p.add_argument("--no-overlay", action="store_true", help="skip the overlay video")
    return p.parse_args(argv)


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

    try:
        overrides = {"pose": {"variant": args.variant}}
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
    from ..output.overlay import render_pose_overlay
    from ..perception.pose import PoseEstimator
    from ..types import Shot

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

        def one_shot() -> Shot:   # shot detection (LLD 3.3) comes later
            shot = Shot(f"{run_id}_s01", run_id, 0, info.frame_count, 0.0, info.duration_s, "handheld", None)
            ctx.manifest["shots"] = [{**shot.__dict__, "method": "whole video, shot detection not run"}]
            return shot
        shot = _run_stage(ctx, "shots", one_shot)

        pose = PoseEstimator(cfg, ctx, device="cuda:0" if args.gpu >= 0 else "cpu")
        det_path = _run_stage(ctx, "pose_body", lambda: (pose.load_models(args.variant),
                                                         pose.estimate_body(run_id, shot))[1])
        if args.no_overlay:
            ctx.manifest["stages"]["overlay"]["status"] = "skipped"
        else:
            try:
                _run_stage(ctx, "overlay", lambda: render_pose_overlay(paths.cfr, det_path, paths.pose_overlay,
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
    ing = ctx.manifest.get("ingest", {}).get("cfr", {})
    summary = (f"{run_id} {status}: variant {pb.get('variant', args.variant)}, "
               f"{pb.get('frames', 0)} frames ({ing.get('width')}x{ing.get('height')} at {ing.get('fps')} fps), "
               f"pose {pb.get('pose_fps')} fps, peak GPU {pb.get('gpu_peak_mb')} MB, "
               f"{pb.get('failed_frames', 0)} failed frames, run folder {paths.root}")
    log.info(summary)
    print(summary)
    return code


if __name__ == "__main__":
    sys.exit(main())
