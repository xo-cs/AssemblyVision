"""Batch analysis from the command line: ingest, shots, reference frames, YOLO26
pose, tracking (KTS v1), step segmentation (Engine A), motion metrics, the
track overlay and, with --gt, the step evaluation, in LLD order (3.0.6).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.run_pose --source {qisens,public} --video PATH \
        --variant {n,s,m,l} --gpu 0 [--gt GT_CSV]

Rerun stages of an existing run from its stored files (CPU, no pose, no tracking):

    python -m assemblyvision.tools.run_pose --run RUN_DIR --stages segmentation,metrics,overlay,eval \
        [--gt GT_CSV] [--set SECTION.KEY=VALUE] [--config FILE]

(engine is accepted for segmentation.) The rerun takes the configuration stored
in the run's manifest, then --config files, then --set, and overwrites
steps.csv, metrics.csv, eval.json, the overlay and their manifest sections.
metrics alone uses the stored steps.csv; segmentation without metrics marks
the stored metrics as stale.

Source qisens writes everything (run folder, temp files, logs, manifest) under
data/qisens/runs/<run_id>/, source public under data/public/runs/<run_id>/. A
video outside the data area of its source is refused before anything is
opened (AV-RUN-001), and so is a GT file outside the data area of the run.
Exit codes: 0 success, 1 error, 2 wrong argument (LLD 5.2).
"""
from __future__ import annotations

import argparse
import json
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

from ..config import DEFAULT_LAYERS, VARIANTS, Config, _read_toml, deep_merge
from ..errors import AvError
from ..storage.files import (REPO_ROOT, SOURCE_TYPE, SOURCES, RunContext, RunPaths, check_video_source,
                             data_area, new_run_id, run_source, utc_now, write_atomic)

STAGES = ("ingest", "shots", "reference", "pose_body", "tracking", "segmentation", "metrics", "overlay", "eval")
RERUN_STAGES = ("segmentation", "metrics", "overlay", "eval")
_ALIASES = {"engine": "segmentation"}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.run_pose",
                                description="Run the batch analysis on one video, or rerun stages of a run.")
    p.add_argument("--source", choices=SOURCES)
    p.add_argument("--video", type=Path)
    p.add_argument("--variant", choices=VARIANTS)
    p.add_argument("--gpu", type=int, help="GPU index in PCI bus order, -1 for CPU")
    p.add_argument("--run", type=Path, help="existing run folder, with --stages")
    p.add_argument("--stages", help=f"comma list of {', '.join(RERUN_STAGES)} (with --run)")
    p.add_argument("--gt", type=Path, help="step GT CSV (step,label,start_s,end_s,source); adds the eval stage")
    p.add_argument("--config", action="append", type=Path, default=[],
                   help="extra TOML layer over configs/default.toml and steps.toml (repeatable)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override one setting, value in TOML syntax (repeatable)")
    p.add_argument("--no-overlay", action="store_true", help="skip the overlay video")
    args = p.parse_args(argv)
    if args.run is not None:
        if args.stages is None:
            p.error("--run needs --stages")
        if any(v is not None for v in (args.source, args.video, args.variant, args.gpu)):
            p.error("--run cannot be combined with --source, --video, --variant or --gpu")
        try:
            args.stages = parse_stages(args.stages)
        except ValueError as e:
            p.error(str(e))
        if "eval" in args.stages and args.gt is None:
            p.error("the eval stage needs --gt")
    else:
        if args.stages is not None:
            p.error("--stages works only with --run")
        missing = [f"--{n}" for n in ("source", "video", "variant", "gpu") if getattr(args, n) is None]
        if missing:
            p.error(f"the following arguments are required: {', '.join(missing)}")
    return args


def parse_stages(text: str) -> list[str]:
    names = [_ALIASES.get(s.strip(), s.strip()) for s in text.split(",") if s.strip()]
    bad = [s for s in names if s not in RERUN_STAGES]
    if bad or not names:
        raise ValueError(f"--stages takes {', '.join(RERUN_STAGES)} (or engine); got {text!r}")
    return [s for s in RERUN_STAGES if s in names]


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
        h.close()
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
    rec = ctx.manifest["stages"].setdefault(name, {})
    rec.clear()
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


# stages that work from the stored files of a run (used by both modes) -------------

def stage_segmentation(ctx: RunContext, cfg: Config) -> list:
    from ..analysis.segmentation import build_timeline, create_engine, write_steps_csv
    ctx.manifest["notices"] = [n for n in ctx.manifest.get("notices", []) if n.get("stage") != "segmentation"]
    t0 = time.monotonic()
    engine = create_engine("A", cfg, ctx)
    tl = build_timeline(ctx.manifest, ctx.paths.root, engine.cleaner)
    steps = engine.segment_run(tl)
    write_steps_csv(ctx.paths.steps_csv, steps, engine.vocabulary, ctx)
    rep = dict(engine.report)
    rep["steps"] = [{"step": i + 1, "label": s.label, "start_s": s.start_s, "end_s": s.end_s, "source": s.source,
                     "score": s.score} for i, s in enumerate(steps)]
    rep["steps_csv"] = ctx.rel(ctx.paths.steps_csv)
    rep["params"] = {k: cfg[k] for k in ("clean", "seg_a", "steps")}
    rep["seconds"] = round(time.monotonic() - t0, 3)
    ctx.manifest["segmentation"] = rep
    ctx.log.info("segmentation: %s, %d steps, %s", rep.get("mode"), sum(s.label != "other" for s in steps),
                 rep.get("note", ""))
    return steps


def stage_metrics(ctx: RunContext, cfg: Config, steps: list) -> dict:
    from ..analysis.motion import run_metrics
    if steps is None:
        raise AvError("AV-STO-003", file="steps.csv", detail="the run has no steps; run the segmentation stage first")
    return run_metrics(ctx, cfg, steps)


def _shots_from_manifest(manifest: dict) -> list:
    from ..types import Shot
    return [Shot(i["shot_id"], manifest["run_id"], i["start_frame"], i["end_frame"], i["start_s"], i["end_s"],
                 i["shot_type"], i.get("scale_m_per_px")) for i in manifest["shots"]["items"]]


def stage_overlay(ctx: RunContext, cfg: Config, steps: list | None) -> Path:
    from ..output.overlay import render_track_overlay
    paths = ctx.paths
    items = {}
    for p in ctx.manifest["tracking"]["per_shot"]:
        subj = p["subject"]
        items[p["shot_id"]] = {"det": paths.det(p["shot_id"]), "assign": paths.assign(p["shot_id"]),
                               "subject": int(subj.rsplit("-t", 1)[1]) if subj else None}
    fps = float(ctx.manifest["ingest"]["cfr"]["fps"])
    return render_track_overlay(paths.cfr, _shots_from_manifest(ctx.manifest), items, paths.tracks_overlay,
                                fps, cfg, ctx, steps=steps)


def _stored_steps(ctx: RunContext) -> list | None:
    if not ctx.paths.steps_csv.is_file():
        return None
    from ..evaluation.steps_io import read_steps_csv
    return read_steps_csv(ctx.paths.steps_csv)


def _check_gt(gt: Path | None, source: str, cfg: Config) -> None:
    if gt is not None and data_area(gt, cfg["paths"]) != source:
        raise AvError("AV-RUN-001", name=Path(gt).name,
                      detail=f"the GT file must lie in the {source} data area, like the run")


def _summary(ctx: RunContext, status: str) -> str:
    m = ctx.manifest
    seg = m.get("segmentation", {})
    steps = [s for s in seg.get("steps", []) if s["label"] != "other"]
    text = f"{len(steps)} steps"
    if seg.get("mode") == "phases":
        text += f" ({seg.get('phases_found')}/{seg.get('phases_expected')} phases, solver {seg.get('solver_used')})"
    elif seg.get("mode"):
        text += f" ({seg['mode']})"
    mt = m.get("metrics")
    if mt:
        text += (f", metrics: {mt['counts']['records']} records, {mt['counts']['null']} not reported"
                 + (" (stale)" if mt.get("stale") else ""))
    ev = m.get("eval")
    if ev:
        from ..evaluation.steps_eval import format_metrics
        text += f", eval: {format_metrics(ev)}"
    return text


# rerun mode --------------------------------------------------------------------------

def rerun(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # no stage of a rerun uses the GPU
    os.environ["YOLO_OFFLINE"] = "1"
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
    try:
        base = Config.load()
        source, root = run_source(args.run, base["paths"])
        manifest = json.loads((root / "manifest.json").read_text())
        layers = [_read_toml(p) for p in args.config]
        cfg = Config.load(overrides=deep_merge(manifest.get("config", {}), *layers, parse_sets(args.set)))
        _check_gt(args.gt, source, cfg)
        for need in ("ingest", "shots", "tracking"):
            if need not in manifest:
                raise AvError("AV-STO-003", file="manifest.json", detail=f"the run has no {need} section")
    except (OSError, ValueError) as e:
        print(f"cannot read the run: {e}", file=sys.stderr)
        return 1
    except AvError as e:
        return _fail(e)

    run_id = manifest["run_id"]
    paths = RunPaths(run_id, root)
    paths.tmp_dir.mkdir(exist_ok=True)
    paths.log_dir.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(paths.tmp_dir)
    tempfile.tempdir = None
    log = _setup_logging(paths)
    ctx = RunContext(run_id, source, paths, log, manifest)
    manifest.setdefault("stages", {})
    manifest.setdefault("reruns", []).append({"utc": utc_now(), "stages": args.stages, "command": " ".join(sys.argv),
                                              "config_layers": [str(p) for p in args.config], "sets": args.set})
    log.info("rerun of %s: stages %s", run_id, ",".join(args.stages))

    code, steps = 0, None
    try:
        if "segmentation" in args.stages:
            steps = _run_stage(ctx, "segmentation", lambda: stage_segmentation(ctx, cfg))
            if "metrics" not in args.stages and "metrics" in manifest:
                manifest["metrics"]["stale"] = True
                log.warning("the stored metrics belong to the previous steps; rerun with --stages metrics")
        if "metrics" in args.stages:
            if steps is None:
                steps = _stored_steps(ctx)
            _run_stage(ctx, "metrics", lambda: stage_metrics(ctx, cfg, steps))
        if "overlay" in args.stages:
            if steps is None:
                steps = _stored_steps(ctx)
            _run_stage(ctx, "overlay", lambda: stage_overlay(ctx, cfg, steps))
        if "eval" in args.stages:
            from ..evaluation.steps_eval import evaluate_run
            _run_stage(ctx, "eval", lambda: evaluate_run(ctx, args.gt, cfg))
    except AvError as e:
        code = _fail(e)
    finally:
        ctx.write_manifest()
    summary = f"{run_id} rerun {','.join(args.stages)} {'done' if code == 0 else 'failed'}: {_summary(ctx, '')}"
    log.info(summary)
    print(summary)
    return code


# full run ------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run is not None:
        return rerun(args)

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
        cfg = Config.load([*DEFAULT_LAYERS, *args.config], overrides=overrides)
        video = check_video_source(args.video, args.source, cfg["paths"])
        _check_gt(args.gt, args.source, cfg)
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

        steps = None
        try:
            steps = _run_stage(ctx, "segmentation", lambda: stage_segmentation(ctx, cfg))
        except AvError as e:
            status, code = "partial", _fail(e)

        if steps is None:
            ctx.manifest["stages"]["metrics"]["status"] = "skipped"
        else:
            try:
                _run_stage(ctx, "metrics", lambda: stage_metrics(ctx, cfg, steps))
            except AvError as e:
                status, code = "partial", _fail(e)

        if args.no_overlay:
            ctx.manifest["stages"]["overlay"]["status"] = "skipped"
        else:
            try:
                _run_stage(ctx, "overlay", lambda: stage_overlay(ctx, cfg, steps))
            except AvError as e:
                status, code = "partial", _fail(e)

        if args.gt is None or steps is None:
            ctx.manifest["stages"]["eval"]["status"] = "skipped"
        else:
            from ..evaluation.steps_eval import evaluate_run
            try:
                _run_stage(ctx, "eval", lambda: evaluate_run(ctx, args.gt, cfg))
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
               f"{tt.get('duplicates_suppressed', 0)} duplicates suppressed, {_summary(ctx, status)}, "
               f"run folder {paths.root}")
    log.info(summary)
    print(summary)
    return code


if __name__ == "__main__":
    sys.exit(main())
