"""Batch analysis from the command line: ingest, shots, reference frames, YOLO26
pose, tracking (KTS v1), step segmentation (Engine A), motion metrics, the
track overlay and, with --gt, the step evaluation, in LLD order (3.0.6).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.run_pose --source {qisens,public} --video PATH \
        --variant {n,s,m,l,x} --gpu 0 [--gt GT_CSV] [--run-id RUN_ID]

--run-id is used by the interface (RunManager): it creates the run folder with
a queued manifest and starts this command as the worker process. A full run
on a GPU first takes the lock ~/AssemblyVision/.locks/gpu<index>.lock, so a
second run waits (status queued) until the first one has finished (LLD 3.1).

Rerun stages of an existing run from its stored files (CPU, no pose, no tracking):

    python -m assemblyvision.tools.run_pose --run RUN_DIR --stages segmentation,metrics,overlay,eval \
        [--gt GT_CSV] [--set SECTION.KEY=VALUE] [--config FILE]

Compare values of one setting, in memory (CPU, nothing in the run folder is written):

    python -m assemblyvision.tools.run_pose --run RUN_DIR --compare seg_a.activity_change --gt GT_CSV \
        [--set SECTION.KEY=VALUE] [--config FILE]
    python -m assemblyvision.tools.run_pose --run RUN_DIR --compare seg_a.engine_b_weight --gt GT_CSV
    python -m assemblyvision.tools.run_pose --run RUN_DIR --compare engine_b.pool_s=0,10,30 --gt GT_CSV \
        --set seg_a.engine_b_weight=0.4

It segments and evaluates the run once per value on top of the stored
configuration, --config and --set, and prints one table. A boolean alone
compares false and true; seg_a.engine_b_weight alone compares 0, 0.2, 0.4 and
0.8 (Engine B scored once, from engine_b.model under models/); any setting
takes KEY=V1,V2,... in TOML syntax.

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
import fcntl
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

from ..config import DEFAULT_LAYERS, ENGINE_PRESETS, VARIANTS, Config, _read_toml, deep_merge
from ..errors import AvError
from ..storage.files import (REPO_ROOT, SOURCE_TYPE, SOURCES, RunContext, RunPaths, check_video_source,
                             RUN_ID_RE, data_area, new_run_id, run_source, utc_now, write_atomic)

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
    p.add_argument("--compare", metavar="SECTION.KEY[=V1,V2,...]",
                   help="segment and evaluate once per value in memory, write nothing (with --run, --gt); "
                        "a boolean alone compares false and true")
    p.add_argument("--gt", type=Path, help="step GT CSV (step,label,start_s,end_s,source); adds the eval stage")
    p.add_argument("--config", action="append", type=Path, default=[],
                   help="extra TOML layer over configs/default.toml and steps.toml (repeatable)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override one setting, value in TOML syntax (repeatable)")
    p.add_argument("--no-overlay", action="store_true", help="skip the overlay video")
    p.add_argument("--run-id", help="use the queued run folder created by the interface")
    args = p.parse_args(argv)
    if args.compare is not None and (args.run is None or args.stages is not None or args.gt is None):
        p.error("--compare needs --run and --gt and cannot be combined with --stages")
    if args.run is not None:
        if args.stages is None and args.compare is None:
            p.error("--run needs --stages or --compare")
        if any(v is not None for v in (args.source, args.video, args.variant, args.gpu, args.run_id)):
            p.error("--run cannot be combined with --source, --video, --variant, --gpu or --run-id")
        try:
            args.stages = parse_stages(args.stages) if args.stages is not None else None
        except ValueError as e:
            p.error(str(e))
        if args.stages and "eval" in args.stages and args.gt is None:
            p.error("the eval stage needs --gt")
    else:
        if args.stages is not None:
            p.error("--stages works only with --run")
        missing = [f"--{n}" for n in ("source", "video", "variant", "gpu") if getattr(args, n) is None]
        if missing:
            p.error(f"the following arguments are required: {', '.join(missing)}")
        if args.run_id is not None and not RUN_ID_RE.fullmatch(args.run_id):
            p.error(f"--run-id {args.run_id!r} is not a run ID")
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


class ParallelShots:
    """shots.parallel: ShotDetector in a thread while the pose pass runs. The detector
    works on a private context (its own manifest dict and notices), so only the main
    thread ever touches ctx.manifest: result() merges the shot table, the notices and
    the stage record when the pose pass asks for it."""

    def __init__(self, cfg: Config, ctx: RunContext, prof, run_id: str, video: Path, info):
        import threading
        from types import SimpleNamespace

        from ..ingest.shots import ShotDetector
        self.ctx, self.prof, self.out, self.merged = ctx, prof, {}, None
        self.notices: list = []
        self.private = SimpleNamespace(manifest={}, log=ctx.log,
                                       notice=lambda err, stage: self.notices.append((err, stage)))
        self.detector = ShotDetector(cfg, self.private)
        self.thread = threading.Thread(target=self._work, args=(run_id, video, info), name="shots", daemon=True)

    def _work(self, run_id, video, info) -> None:
        a = time.monotonic()
        try:
            self.out["shots"] = self.detector.detect(run_id, video, info)
        except BaseException as e:        # re-raised in the main thread by result()
            self.out["error"] = e
        self.out["span"] = (a, time.monotonic())

    def start(self) -> "ParallelShots":
        rec = self.ctx.manifest["stages"].setdefault("shots", {})
        rec.clear()
        rec.update(status="running", started_utc=utc_now(), progress=0.0, parallel_with="pose_body")
        self.thread.start()
        return self

    def result(self, block: bool = False):
        """The shot list once it is ready (None before, unless block); the first call
        after it is ready merges it into the run (main thread only)."""
        if self.merged is not None:
            return self.merged
        if self.thread.is_alive() and not block:
            return None
        self.thread.join()
        rec = self.ctx.manifest["stages"]["shots"]
        if "error" in self.out:
            e = self.out["error"]
            err = e if isinstance(e, AvError) else AvError("AV-SYS-001", type=type(e).__name__, stage="shots",
                                                           detail=str(e)[:300])
            rec.update(status="failed", error=err.as_record())
            raise err
        self.ctx.manifest["shots"] = self.private.manifest["shots"]
        for err, stage in self.notices:
            self.ctx.notice(err, stage)
        a, b = self.out["span"]
        if self.prof is not None:
            self.prof.marks.append(("shots", a, b))
        rec.update(status="done", progress=1.0, finished_utc=utc_now(), seconds=round(b - a, 3))
        self.merged = self.out["shots"]
        return self.merged


def _run_stage(ctx: RunContext, name: str, fn, prof=None):
    """Run one stage with its own log file; AV errors pass through, anything
    else becomes AV-SYS-001 with the traceback in the stage log only. prof
    (runs.profiler.Profiler) marks the stage for the GPU samples."""
    rec = ctx.manifest["stages"].setdefault(name, {})
    rec.clear()
    rec.update(status="running", started_utc=utc_now(), progress=0.0)
    ctx.write_manifest()
    handler = logging.FileHandler(ctx.paths.stage_log(name), encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    ctx.log.addHandler(handler)
    t0 = time.monotonic()
    mark = prof.stage(name) if prof is not None else None
    if mark is not None:
        mark.__enter__()
    try:
        result = fn()
        rec["status"] = "done"
        rec["progress"] = 1.0
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
        if mark is not None:
            mark.__exit__(None, None, None)
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

def boundary_probability(manifest: dict, root: Path, cfg: Config, ctx) -> dict | None:
    """Engine B probability and embedding on the run clock (CPU) when seg_a.engine_b_weight
    or seg_a.embed_weight > 0, else None. Without weights (engine_b.model empty or missing)
    AV-SEG-001 is noticed and Engine A runs without the terms."""
    if not engine_b_needed(cfg):
        return None
    from ..analysis.segmentation.boundary_tcn import run_probability
    name = cfg["engine_b"]["model"]
    path = REPO_ROOT / cfg["paths"]["models_dir"] / name if name else None
    if path is None or not path.is_file():
        if ctx is not None:
            ctx.notice(AvError("AV-SEG-001", detail=f"engine_b.model = {name!r}"), "segmentation")
        return None
    return run_probability(manifest, root, cfg, path)


def engine_b_needed(cfg: Config) -> bool:
    return cfg["seg_a"]["engine_b_weight"] > 0 or cfg["seg_a"]["embed_weight"] > 0


def segment_stored(manifest: dict, root: Path, cfg: Config, ctx, bprob: dict | None = None) -> tuple:
    """(engine, steps) of Engine A from the stored files of a run; writes nothing.
    bprob: a boundary_probability() result to reuse (compare mode), else computed here."""
    from ..analysis.segmentation import build_timeline, create_engine
    engine = create_engine("A", cfg, ctx)
    tl = build_timeline(manifest, root, engine.cleaner, cfg["motion"] if cfg["seg_a"]["activity_change"] else None)
    if engine_b_needed(cfg):
        bp = bprob if bprob is not None else boundary_probability(manifest, root, cfg, ctx)
        if bp is not None:
            tl.boundary_prob = bp["prob"]
            tl.embedding, tl.embedding_valid = bp.get("embedding"), bp.get("embedding_valid")
            tl.boundary_info = {k: v for k, v in bp.items() if k not in ("prob", "embedding", "embedding_valid")}
    return engine, engine.segment_run(tl)


def stage_segmentation(ctx: RunContext, cfg: Config) -> list:
    from ..analysis.segmentation import write_steps_csv
    ctx.manifest["notices"] = [n for n in ctx.manifest.get("notices", []) if n.get("stage") != "segmentation"]
    t0 = time.monotonic()
    engine, steps = segment_stored(ctx.manifest, ctx.paths.root, cfg, ctx)
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
    shots = _shots_from_manifest(ctx.manifest)
    from ..phases import display_steps
    steps = display_steps(steps, ctx.source)      # older public runs: burn in Phase 1..N, not QiSens names
    from ..analysis.motion import MotionAnalyzer
    from ..output.overlay import overlay_series
    subjects = MotionAnalyzer(cfg).load_subjects(ctx.manifest, ctx.paths.root)
    hud = overlay_series(subjects, max(s.end_frame for s in shots) + 1, fps, cfg)
    return render_track_overlay(paths.cfr, shots, items, paths.tracks_overlay, fps, cfg, ctx, steps=steps, hud=hud)


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


# compare mode ------------------------------------------------------------------------

class _MemoryCtx:
    """What Engine A needs of a RunContext, without files: notices are kept in a list."""

    def __init__(self, source: str):
        self.source = source
        self.notices: list[str] = []

    def notice(self, err: AvError, stage: str) -> None:
        self.notices.append(f"{err.code}: {err.text}")


COMPARE_DEFAULTS = {"seg_a.engine_b_weight": [0.0, 0.2, 0.4, 0.8], "seg_a.embed_weight": [0.0, 0.2, 0.4, 0.8]}    # values when only the key is given


def compare_values(spec: str, cfg: Config) -> tuple[str, list]:
    """"KEY" (a boolean: false, true; or a key of COMPARE_DEFAULTS) or "KEY=V1,V2,..." (TOML values)
    -> (key, values). The key must exist and every value must have its type."""
    key, sep, vals = spec.partition("=")
    key = key.strip()
    section, _, name = key.partition(".")
    current = cfg.snapshot().get(section, {}).get(name, None) if name else None
    if current is None:
        raise AvError("AV-CFG-002", key=key, value="", range="an existing setting")
    if sep:
        try:
            values = tomllib.loads(f"v = [{vals}]")["v"]
        except tomllib.TOMLDecodeError as e:
            raise AvError("AV-CFG-002", key=key, value=repr(vals), range=f"comma separated TOML values ({e})") from e
    elif isinstance(current, bool):
        values = [False, True]
    elif key in COMPARE_DEFAULTS:
        values = list(COMPARE_DEFAULTS[key])
    else:
        raise AvError("AV-CFG-002", key=key, value=repr(current),
                      range="a boolean setting, or KEY=V1,V2,... for any other setting")
    if len(values) < 2:
        raise AvError("AV-CFG-002", key=key, value=repr(vals), range="at least two values")
    return key, values


def _toml_value(v) -> str:
    return str(v).lower() if isinstance(v, bool) else repr(v) if isinstance(v, str) else str(v)


def compare_run(manifest: dict, root: Path, source: str, cfg: Config, spec: str, gt_path: Path) -> list[dict]:
    """Segment and evaluate the stored run once per value of the setting (spec, see
    compare_values), in memory. Returns one row per value; nothing is written."""
    from ..evaluation.steps_eval import score_steps
    from ..evaluation.steps_io import read_steps_csv
    key, values = compare_values(spec, cfg)
    section, _, name = key.partition(".")
    gt = read_steps_csv(gt_path)
    duration = float(manifest["ingest"]["cfr"]["duration_s"])
    rows, bprob = [], None
    if key in ENGINE_PRESETS.get(cfg["seg_a"]["engine"], {}):
        print(f"{key} is set by seg_a.engine = {cfg['seg_a']['engine']!r}; compared with seg_a.engine = \"custom\" "
              "(the preset's other values kept)", file=sys.stderr)
    engine_b_off = []
    for value in values:
        over = {section: {name: value}}
        if key in ENGINE_PRESETS.get(cfg["seg_a"]["engine"], {}):
            over = deep_merge(over, {"seg_a": {"engine": "custom"}})
        c = Config.load(overrides=deep_merge(cfg.snapshot(), over))
        engine_b_off.append(not engine_b_needed(c))
        mctx = _MemoryCtx(source)
        if bprob is None and engine_b_needed(c):
            bprob = boundary_probability(manifest, root, c, mctx) or {}
        engine, steps = segment_stored(manifest, root, c, mctx, bprob or None)
        res, bounds, mapping = score_steps(steps, gt, Path(gt_path).name, engine.vocabulary, source, c, duration)
        rep = engine.report
        rows.append({"setting": f"{key} = {_toml_value(value)}", "metrics": res, "boundaries": bounds,
                     "mapping": mapping, "notices": mctx.notices,
                     "phases": f"{rep.get('phases_found')}/{rep.get('phases_expected')}" if rep.get("mode") == "phases"
                     else rep.get("mode"),
                     "engine_b_rate": (f"{rep['engine_b'].get('peaks_per_min_raw')}/{rep['engine_b'].get('peaks_per_min_pooled')}"
                                       if rep.get("engine_b", {}).get("on") else "-"),
                     "steps": [(s.label, s.start_s, s.end_s) for s in steps if s.label != "other"]})
    if section == "engine_b" and all(engine_b_off):
        print(f"warning: Engine B is off in every compared setting (seg_a.engine_b_weight = 0 and seg_a.embed_weight "
              f"= 0, as stored in this run), so {key} cannot change anything; set a preset, for example "
              "--set 'seg_a.engine=\"B v1\"'", file=sys.stderr)
    return rows


def format_compare(rows: list[dict]) -> str:
    f1 = [k for k in rows[0]["metrics"] if k.startswith("f1_")]
    head = ["setting", "frame acc", "edit", *[f"F1@{k[3:]}" for k in f1], "MAE s", "phases", "B peaks/min raw/pooled"]
    body = [[r["setting"], f"{r['metrics']['frame_accuracy']:.2f}", f"{r['metrics']['edit']:.2f}",
             *[f"{r['metrics'][k]:.2f}" for k in f1],
             "n/a" if r["metrics"]["boundary_mae_s"] is None else f"{r['metrics']['boundary_mae_s']:.2f}",
             str(r["phases"]), r.get("engine_b_rate", "-")] for r in rows]
    w = [max(len(x[i]) for x in [head, *body]) for i in range(len(head))]
    lines = ["  ".join(x[i].ljust(w[i]) for i in range(len(head))) for x in [head, *body]]
    lines.insert(1, "  ".join("-" * n for n in w))
    for r in rows:
        lines.append(f"\n{r['setting']}:")
        lines += [f"  {lab:<28} {a:9.2f} to {b:9.2f} s" for lab, a, b in r["steps"]]
        lines += [f"  GT boundary {b['gt_s']:8.2f} s  nearest predicted {b['pred_s']}  error {b['error_s']}"
                  for b in r["boundaries"]]
        lines += [f"  notice {n}" for n in r["notices"]]
    if rows[0]["mapping"]:
        lines.append(f"\nlabels: {rows[0]['mapping']}")
    lines.append("\nIn memory only: no file of the run folder was written.")
    return "\n".join(lines)


def stored_config(manifest: dict) -> dict:
    """The settings a stored run ran with. A run from before seg_a.engine ran its Engine B
    keys as stored, so it gets seg_a.engine = "custom" (not the current preset)."""
    c = json.loads(json.dumps(manifest.get("config", {})))
    if "seg_a" in c and "engine" not in c["seg_a"]:
        c["seg_a"]["engine"] = "custom"
    return c


def stored_with(manifest: dict, *explicit: dict) -> dict:
    """The stored settings with explicit layers (--config files, --set) on top. When an
    explicit layer sets seg_a.engine to a preset, the preset replaces the run's stored
    values of its keys; only values given explicitly can conflict with it (AV-CFG-002)."""
    from ..config import ENGINE_ALIASES
    stored = stored_config(manifest)
    engine = None
    for layer in explicit:
        engine = layer.get("seg_a", {}).get("engine", engine)
    preset = ENGINE_PRESETS.get(ENGINE_ALIASES.get(engine, engine)) if engine is not None else None
    for key in preset or {}:
        section, name = key.split(".")
        stored.get(section, {}).pop(name, None)
    return deep_merge(stored, *explicit)


def compare(args: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # CPU only, like a rerun
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    try:
        base = Config.load()
        source, root = run_source(args.run, base["paths"])
        manifest = json.loads((root / "manifest.json").read_text())
        layers = [_read_toml(p) for p in args.config]
        cfg = Config.load(overrides=stored_with(manifest, *layers, parse_sets(args.set)))
        _check_gt(args.gt, source, cfg)
        for need in ("ingest", "shots", "tracking"):
            if need not in manifest:
                raise AvError("AV-STO-003", file="manifest.json", detail=f"the run has no {need} section")
        rows = compare_run(manifest, root, source, cfg, args.compare, args.gt)
    except (OSError, ValueError) as e:
        print(f"cannot read the run: {e}", file=sys.stderr)
        return 1
    except AvError as e:
        return _fail(e)
    print(f"{manifest['run_id']}: {args.compare} compared against {Path(args.gt).name}\n")
    print(format_compare(rows))
    return 0


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
        cfg = Config.load(overrides=stored_with(manifest, *layers, parse_sets(args.set)))
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
    manifest["reruns"][-1]["summary"] = summary
    ctx.write_manifest()
    log.info(summary)
    print(summary)
    return code


# full run ------------------------------------------------------------------------------

def _track_worker_init() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""          # tracking is CPU only; never touch the pose GPU


def track_shot_job(job: dict) -> dict:
    """Track one shot in a worker process (track.overlap_pose). The tracker writes
    its KTS and assignment files itself; the manifest records, notices and log
    lines come back so the main process merges them in shot order."""
    from ..perception.tracker import Tracker

    class _Keep(logging.Handler):
        def __init__(self):
            super().__init__()
            self.lines: list[tuple[int, str]] = []

        def emit(self, record):
            self.lines.append((record.levelno, record.getMessage()))
    log = logging.getLogger("assemblyvision.track_worker")
    log.handlers.clear()
    log.propagate = False
    log.setLevel(logging.DEBUG)
    keep = _Keep()
    log.addHandler(keep)
    ctx = RunContext(job["run_id"], job["source"], job["paths"], log,
                     {"ingest": job["ingest"], "tracking": {"per_shot": [], "tracks": []}, "notices": []})
    t0 = time.monotonic()
    Tracker(job["cfg"], ctx).track(job["run_id"], job["shot"], job["det"], job["ref"], fps=job["fps"],
                                   size=job["size"], meta=job["meta"])
    return {"tracking": ctx.manifest["tracking"], "notices": ctx.manifest["notices"], "log": keep.lines,
            "seconds": time.monotonic() - t0}


def _take_gpu_lock(gpu: int, ctx: RunContext):
    """Exclusive flock on .locks/gpu<index>.lock (one job per GPU, LLD 3.1); waits
    with status queued while another run holds it. The system frees the lock
    if the process dies. None on the CPU."""
    if gpu < 0:
        return None
    lock_dir = REPO_ROOT / ".locks"
    lock_dir.mkdir(exist_ok=True)
    f = open(lock_dir / f"gpu{gpu}.lock", "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        ctx.manifest["waiting"] = f"GPU {gpu} is in use by another run; this run starts when it is free"
        ctx.write_manifest()
        ctx.log.info("waiting for GPU %d", gpu)
        fcntl.flock(f, fcntl.LOCK_EX)
        ctx.manifest.pop("waiting", None)
    return f


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare is not None:
        return compare(args)
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

    queued: dict = {}
    if args.run_id is not None:
        run_id = args.run_id
        paths = RunPaths.for_source(args.source, run_id, cfg["paths"])
        try:
            queued = json.loads(paths.manifest.read_text())
        except (OSError, ValueError):
            queued = {}
        if queued.get("status") != "queued" or queued.get("run_id") != run_id:
            print(f"--run-id {run_id}: no queued run folder of source {args.source}", file=sys.stderr)
            return 1
    else:
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
    ctx = RunContext(run_id, args.source, paths, log, progress_every_s=cfg["run"]["progress_every_s"])
    ctx.manifest.update({k: queued[k] for k in ("requested_by",) if k in queued})
    ctx.manifest["worker"] = {"pid": os.getpid(), "started_utc": utc_now()}
    ctx.manifest.update(
        run_id=run_id, source=args.source, source_type=SOURCE_TYPE[args.source],
        status="queued", created_utc=queued.get("created_utc", utc_now()), command=" ".join(sys.argv),
        video=str(video), variant=args.variant, half=cfg["pose"]["half"], gpu=args.gpu, config_sources=cfg.sources,
        config=cfg.snapshot(), stages={s: {"status": "pending"} for s in STAGES},
        notices=list(queued.get("notices", [])))
    write_atomic(paths.run_config, cfg.to_toml().encode("utf-8"), paths)
    ctx.write_manifest()
    gpu_lock = _take_gpu_lock(args.gpu, ctx)                # held until this process exits
    ctx.manifest["status"] = "running"
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

    from ..runs.profiler import Profiler
    prof = Profiler(cfg["profile"]["enabled"], args.gpu, cfg["profile"]["gpu_sample_s"]).start()

    def stage(name: str, fn):
        return _run_stage(ctx, name, fn, prof)

    track_pool = None
    status, code = "done", 0
    try:
        if args.gpu >= 0 and not torch.cuda.is_available():
            raise AvError("AV-SYS-001", type="CudaUnavailable", stage="pose_body",
                          detail=f"CUDA is not available for GPU {args.gpu}")
        info, _split = stage("ingest", lambda: VideoIngestor(cfg, ctx).ingest(
            run_id, video, SOURCE_TYPE[args.source], strict_split=False))

        parallel = cfg["shots"]["parallel"]
        if parallel:        # shots.parallel: the shot table is made in a thread during the pose pass
            shots_job = ParallelShots(cfg, ctx, prof, run_id, paths.cfr, info).start()
            shots = None
        else:
            shots = stage("shots", lambda: ShotDetector(cfg, ctx).detect(run_id, paths.cfr, info))

        pose = PoseEstimator(cfg, ctx, device="cuda:0" if args.gpu >= 0 else "cpu")
        fused = cfg["pose"]["reference_in_pass"]
        jobs: dict = {}
        if cfg["track"]["overlap_pose"]:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            track_pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                                             initializer=_track_worker_init)

        def set_ref_frames(refs: list[dict]) -> dict:
            by_shot = {r["shot_id"]: r for r in refs}
            for item in ctx.manifest["shots"]["items"]:
                item["ref_frame"] = by_shot.get(item["shot_id"], {}).get("frame_idx")
            return by_shot

        if fused:       # the model is loaded here; the reference frames come from the pose pass
            stage("reference", lambda: pose.load_models(args.variant))
            ctx.manifest["stages"]["reference"]["fused"] = "pose_body"
            ref_by_shot = {}
        else:
            ref_by_shot = set_ref_frames(stage("reference", lambda: (pose.load_models(args.variant),
                                                                     pose.reference_frames(run_id, shots))[1]))
        pose_meta = {"pose_model": f"yolo26{args.variant}-pose", "pose_variant": args.variant, "seed": seed}

        def submit(shot, det, ref) -> None:
            jobs[shot.shot_id] = track_pool.submit(track_shot_job, {
                "run_id": run_id, "source": args.source, "paths": paths, "cfg": cfg, "shot": shot, "det": det,
                "ref": ref, "fps": info.fps, "size": (info.width, info.height),
                "meta": {**pose_meta, "pose_model": f"yolo26{pose.variant}-pose", "pose_variant": pose.variant},
                "ingest": ctx.manifest.get("ingest", {})})
        det_paths = stage("pose_body", lambda: pose.estimate_body_shots(
            run_id, shots, with_reference=fused, on_shot_done=submit if track_pool is not None else None,
            shots_later=shots_job.result if parallel else None, n_total=info.frame_count))
        if parallel:
            shots = shots_job.result(True)
        if fused:
            ref_by_shot = set_ref_frames(pose.pass_refs)
        pose_meta = {"pose_model": f"yolo26{pose.variant}-pose", "pose_variant": pose.variant, "seed": seed}

        def track_all() -> dict:
            ctx.manifest["tracking"] = {"per_shot": [], "tracks": []}
            if track_pool is not None:          # merge the worker results in shot order
                worker_s = 0.0
                for s in shots:
                    r = jobs[s.shot_id].result()
                    ctx.manifest["tracking"]["per_shot"].extend(r["tracking"]["per_shot"])
                    ctx.manifest["tracking"]["tracks"].extend(r["tracking"]["tracks"])
                    ctx.manifest.setdefault("notices", []).extend(r["notices"])
                    for level, msg in r["log"]:
                        ctx.log.log(level, "%s", msg)
                    worker_s += r["seconds"]
                ctx.manifest["tracking"]["overlap_pose"] = {"worker_seconds": round(worker_s, 2)}
                return summarize(ctx)
            tracker = Tracker(cfg, ctx)
            for s in shots:
                tracker.track(run_id, s, det_paths[s.shot_id], ref_by_shot.get(s.shot_id), fps=info.fps,
                              size=(info.width, info.height), meta=pose_meta)
            return summarize(ctx)
        stage("tracking", track_all)

        steps = None
        try:
            steps = stage("segmentation", lambda: stage_segmentation(ctx, cfg))
        except AvError as e:
            status, code = "partial", _fail(e)

        if steps is None:
            ctx.manifest["stages"]["metrics"]["status"] = "skipped"
        else:
            try:
                stage("metrics", lambda: stage_metrics(ctx, cfg, steps))
            except AvError as e:
                status, code = "partial", _fail(e)

        if args.no_overlay or not cfg["overlay"]["enabled"]:
            ctx.manifest["stages"]["overlay"]["status"] = "skipped"
        else:
            try:
                stage("overlay", lambda: stage_overlay(ctx, cfg, steps))
            except AvError as e:
                status, code = "partial", _fail(e)

        if args.gt is None or steps is None:
            ctx.manifest["stages"]["eval"]["status"] = "skipped"
        else:
            from ..evaluation.steps_eval import evaluate_run
            try:
                stage("eval", lambda: evaluate_run(ctx, args.gt, cfg))
            except AvError as e:
                status, code = "partial", _fail(e)
    except AvError as e:
        status, code = "failed", _fail(e)
        ctx.manifest.setdefault("error", e.as_record())
    finally:
        if track_pool is not None:
            track_pool.shutdown(wait=True, cancel_futures=True)
        prof.stop()
        if prof.enabled:
            ctx.manifest["profile"] = prof.summary(ctx.manifest)
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
    ctx.manifest["summary"] = summary
    ctx.write_manifest()
    log.info(summary)
    print(summary)
    if gpu_lock is not None:
        gpu_lock.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
