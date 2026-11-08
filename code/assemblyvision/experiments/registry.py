"""Experiment registry (Week 10): every result of the paper comes from a stored config.

An experiment is a folder <area>/<id>/ with
- config.toml: an [experiment] section (id, title, kind, source, videos, repeats,
  reference, gpu, variant, gt) and the FULL resolved configuration (every
  setting of configs/default.toml and steps.toml), so it reruns the same way;
- results.json: provenance (git commit, model checksum, GPU, driver, library
  versions, data area, date), every repeat and the medians, the equivalence
  with the reference experiment;
- table.md: the same as tables.

Areas: public experiments live in experiments/ (tracked in git) and are listed
in experiments/INDEX.md. A QiSens experiment lives only in
data/qisens/experiments/<id>/ (config, results, tables, temp files) and is never
listed in experiments/; a summary row is added there by hand once approved.
Nothing under data/qisens is opened unless the source qisens is named.

A full_run repeat is one run_pose process (model load included) with the
profiler on; the run folder goes to data/<source>/runs/ as usual. Before every
repeat the GPU must be free of other processes (gpu_check), so no time is
measured on a shared GPU.
"""
from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..config import Config, _toml_value, deep_merge
from ..storage.files import REPO_ROOT, RUN_ID_RE, RunPaths, check_video_source, source_root, utc_now

ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
KINDS = ("full_run",)
STAGE_ORDER = ("ingest", "shots", "reference", "pose_body", "tracking", "segmentation", "metrics", "overlay", "eval")
NOISE_FLOOR_FILE = "noise_floor.json"      # experiments/noise_floor.json (tracked, public measurement)
INDEX_HEAD = ("# Experiments\n\nOne line per experiment (public data only; QiSens rows only as approved summaries).\n"
              "Regenerate any table with `python -m assemblyvision.tools.exp table <id> ...`.\n\n"
              "{noise}"
              "| id | date | title | per video: full run median s, pose fps, mean GPU util %, peak GPU MB | verdict |\n"
              "| --- | --- | --- | --- | --- |\n")


class ExperimentError(Exception):
    pass


def area_dir(source: str, paths_cfg: dict, repo_root: Path = REPO_ROOT) -> Path:
    if source == "public":
        return repo_root / "experiments"
    if source == "qisens":
        return source_root("qisens", paths_cfg, repo_root) / "experiments"
    raise ExperimentError(f"unknown source {source}")


@dataclass
class Experiment:
    id: str
    dir: Path
    meta: dict          # the [experiment] section
    settings: dict      # every configuration section

    @property
    def source(self) -> str:
        return self.meta["source"]


def _check_id(exp_id: str) -> None:
    if not ID_RE.fullmatch(exp_id):
        raise ExperimentError(f"experiment id {exp_id!r}: lower case letters, digits, '.', '_' and '-' only")


def config_text(meta: dict, settings: dict) -> str:
    head = [f"# Experiment {meta['id']}: every setting is stored, so it reruns the same way.", "[experiment]"]
    head += [f"{k} = {_toml_value(v)}" for k, v in meta.items()]
    return "\n".join(head) + "\n\n" + Config(settings, []).to_toml()


def load(exp_id: str, source: str, paths_cfg: dict, repo_root: Path = REPO_ROOT) -> Experiment:
    _check_id(exp_id)
    d = area_dir(source, paths_cfg, repo_root) / exp_id
    path = d / "config.toml"
    if not path.is_file():
        raise ExperimentError(f"no experiment {exp_id} in the {source} area")
    data = tomllib.loads(path.read_text())
    meta = data.pop("experiment", None)
    if not meta or meta.get("source") != source or meta.get("id") != exp_id:
        raise ExperimentError(f"{path}: the [experiment] section does not match id {exp_id}, source {source}")
    default = Config.load().snapshot()
    missing = [f"{s}.{k}" for s, vals in default.items() for k in vals if k not in data.get(s, {})]
    if missing:
        raise ExperimentError(f"{path}: not every setting is stored, missing {', '.join(missing[:8])}")
    Config.load(overrides=data)          # AV-CFG-001/002 on unknown keys or bad values
    return Experiment(exp_id, d, meta, data)


def create(exp_id: str, *, source: str, videos: list[str], paths_cfg: dict, title: str = "",
           repeats: int = 3, reference: str = "", gpu: int = 0, variant: str | None = None, gt: str = "",
           base: Experiment | None = None, sets: dict | None = None, kind: str = "full_run",
           repo_root: Path = REPO_ROOT, overwrite: bool = False) -> Experiment:
    """Write <area>/<id>/config.toml with the full configuration: the base
    experiment's settings (or the defaults), then sets."""
    _check_id(exp_id)
    if kind not in KINDS:
        raise ExperimentError(f"kind must be one of {', '.join(KINDS)}")
    if repeats < 1:
        raise ExperimentError("repeats must be at least 1")
    if not videos:
        raise ExperimentError("at least one video is needed")
    rel_videos = []
    for v in videos:
        real = check_video_source(Path(v) if Path(v).is_absolute() else repo_root / v, source, paths_cfg, repo_root)
        rel_videos.append(real.relative_to(repo_root).as_posix())
    if gt:
        check_video_source(repo_root / gt, source, paths_cfg, repo_root)     # same data area rule
    settings = deep_merge(base.settings if base is not None else Config.load().snapshot(), sets or {})
    if variant:
        settings["pose"]["variant"] = variant
    cfg = Config.load(overrides=settings)
    d = area_dir(source, paths_cfg, repo_root) / exp_id
    if (d / "config.toml").exists() and not overwrite:
        raise ExperimentError(f"experiment {exp_id} exists already ({d})")
    meta = {"id": exp_id, "title": title, "kind": kind, "source": source, "videos": rel_videos,
            "repeats": repeats, "reference": reference, "gpu": gpu, "variant": cfg["pose"]["variant"], "gt": gt,
            "from": base.id if base is not None else "", "created_utc": utc_now()}
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.toml").write_text(config_text(meta, cfg.snapshot()))
    return Experiment(exp_id, d, meta, cfg.snapshot())


# running ------------------------------------------------------------------------------

def default_runner(cmd: list[str], cwd: Path) -> tuple[int, str, str, float]:
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr, time.monotonic() - t0


def default_gpu_check(gpu: int) -> list[dict]:
    from ..runs.profiler import gpu_processes
    return gpu_processes(gpu)


def _median(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 3) if xs else None


def _repeat_record(manifest: dict, wall_s: float, run_dir: Path, repo_root: Path) -> dict:
    prof = manifest.get("profile", {})
    pb = manifest.get("pose_body", {})
    bd = pb.get("breakdown_s", {})
    frames = pb.get("frames") or 0
    stages = {k: v.get("seconds") for k, v in manifest.get("stages", {}).items() if v.get("seconds") is not None}
    ev = manifest.get("eval")
    return {"run_id": manifest["run_id"], "run_dir": run_dir.relative_to(repo_root).as_posix(),
            "status": manifest.get("status"), "wall_s": round(wall_s, 3), "stages_s": stages,
            "eval": {k: v for k, v in ev.items() if k != "path"} if ev else None,
            "stage_sum_s": round(sum(stages.values()), 3),
            "pose": {"frames": frames, "pose_fps": pb.get("pose_fps"),
                     "inference_fps": round(frames / bd["inference"], 1) if bd.get("inference") else None,
                     "batch": pb.get("batch"), "batch_used": pb.get("batch_used"), "precision": pb.get("precision"),
                     "tf32": pb.get("tf32"), "decode": pb.get("decode"), "torch_peak_mb": pb.get("gpu_peak_mb"), "breakdown_s": bd},
            "gpu": {"run": prof.get("run"), "stages": {k: {x: v.get(x) for x in ("gpu_util_mean", "gpu_util_max",
                                                                               "gpu_mem_peak_mb")}
                                                       for k, v in prof.get("stages", {}).items()}},
            "breakdown_s": prof.get("breakdown", {}),
            "overlay": {k: manifest.get("overlay", {}).get(k) for k in ("encoder", "workers", "frames")}}


def summarise(repeats: list[dict]) -> dict:
    stages = {s: _median([r["stages_s"].get(s) for r in repeats]) for s in STAGE_ORDER}
    runs = [r["gpu"]["run"] or {} for r in repeats]
    pose_gpu = [r["gpu"]["stages"].get("pose_body", {}) for r in repeats]
    return {"repeats": len(repeats),
            "wall_s": {"median": _median([r["wall_s"] for r in repeats]), "min": min(r["wall_s"] for r in repeats),
                       "max": max(r["wall_s"] for r in repeats)},
            "stage_sum_s": _median([r["stage_sum_s"] for r in repeats]),
            "stages_s": {k: v for k, v in stages.items() if v is not None},
            "pose_fps": _median([r["pose"]["pose_fps"] for r in repeats]),
            "inference_fps": _median([r["pose"]["inference_fps"] for r in repeats]),
            "gpu_util_mean": _median([g.get("gpu_util_mean") for g in runs]),
            "gpu_mem_peak_mb": max([g.get("gpu_mem_peak_mb") or 0 for g in runs]) or None,
            "pose_gpu_util_mean": _median([g.get("gpu_util_mean") for g in pose_gpu]),
            "torch_peak_mb": max([r["pose"]["torch_peak_mb"] or 0 for r in repeats]) or None,
            "pose_breakdown_s": {k: _median([r["pose"]["breakdown_s"].get(k) for r in repeats])
                                 for k in repeats[0]["pose"]["breakdown_s"]},
            "eval": repeats[0].get("eval"),
            "eval_same_in_repeats": len({json.dumps(r.get("eval"), sort_keys=True) for r in repeats}) == 1}


def load_noise_floor(repo_root: Path = REPO_ROOT) -> dict | None:
    """The noise floor of the "equivalent within numerical noise" rule, or None if not measured yet."""
    f = repo_root / "experiments" / NOISE_FLOOR_FILE
    return json.loads(f.read_text())["floor"] if f.is_file() else None


def add_noise(rec: dict, repo_root: Path = REPO_ROOT) -> None:
    """rec["noise"] = the noise check of rec["equivalence"] when the strict check fails."""
    from ..evaluation.equivalence import noise_check
    eq, floor = rec.get("equivalence"), load_noise_floor(repo_root)
    rec.pop("noise", None)
    if eq and eq["verdict"] != "equivalent" and floor:
        rec["noise"] = noise_check(eq, floor)


def write_noise_floor(exp_id: str, paths_cfg: dict, repo_root: Path = REPO_ROOT) -> Path:
    """experiments/noise_floor.json from the equivalence records of exp_id (strict FP32 against TF32)."""
    from ..evaluation.equivalence import noise_floor, noise_measures
    res = load_results(exp_id, "public", paths_cfg, repo_root)
    eqs = {v: rec["equivalence"] for v, rec in res["videos"].items() if rec.get("equivalence")}
    if not eqs:
        raise ExperimentError(f"{exp_id} has no equivalence records")
    out = {"source": f"{exp_id} against {res['reference']}", "measured_utc": res.get("rechecked_utc")
           or res.get("finished_utc"), "floor": noise_floor(list(eqs.values())),
           "per_video": {v: noise_measures(e) for v, e in eqs.items()}}
    path = repo_root / "experiments" / NOISE_FLOOR_FILE
    path.write_text(json.dumps(out, indent=1) + "\n")
    return path


def video_verdict(rec: dict) -> str | None:
    eq = rec.get("equivalence")
    if not eq:
        return None
    if eq["verdict"] == "equivalent":
        return "equivalent"
    return "within noise" if rec.get("noise", {}).get("within_noise") else "different"


def _settings_file(exp: Experiment, paths_cfg: dict, repo_root: Path) -> Path:
    """The settings as a config layer, in the temp folder of the experiment's data area."""
    tmp = source_root(exp.source, paths_cfg, repo_root) / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / f"exp_{exp.id}.toml"
    path.write_text(Config(exp.settings, []).to_toml())
    return path


def run(exp: Experiment, paths_cfg: dict, *, repo_root: Path = REPO_ROOT,
        runner: Callable[[list[str], Path], tuple[int, str, str, float]] = default_runner,
        gpu_check: Callable[[int], list[dict]] = default_gpu_check,
        provenance: Callable[[Experiment], dict] | None = None,
        equivalence: Callable[[Path, Path], dict] | None = None,
        log: Callable[[str], None] = print) -> dict:
    """Run every repeat of every video, then write results.json and table.md
    (and INDEX.md for a public experiment). Returns the results."""
    if equivalence is None:
        from ..evaluation.equivalence import compare_runs as equivalence
    gpu, source = int(exp.meta["gpu"]), exp.source
    for v in exp.meta["videos"]:
        check_video_source(repo_root / v, source, paths_cfg, repo_root)
    layer = _settings_file(exp, paths_cfg, repo_root)
    results = {"id": exp.id, "title": exp.meta.get("title", ""), "kind": exp.meta["kind"], "source": source,
               "reference": exp.meta.get("reference", ""), "started_utc": utc_now(),
               "provenance": provenance(exp) if provenance else collect_provenance(exp, repo_root),
               "videos": {}}
    try:
        for v in exp.meta["videos"]:
            reps = []
            for k in range(int(exp.meta["repeats"])):
                foreign = gpu_check(gpu)
                if foreign:
                    names = ", ".join(f"{p['name']} (pid {p['pid']}, {p['used_mb']} MB)" for p in foreign)
                    raise ExperimentError(f"GPU {gpu} is in use by {names}; no time is measured on a shared GPU")
                cmd = [sys.executable, "-m", "assemblyvision.tools.run_pose", "--source", source, "--video",
                       str(repo_root / v), "--variant", exp.meta["variant"], "--gpu", str(gpu),
                       "--config", str(layer), "--set", "profile.enabled=true"]
                if exp.meta.get("gt"):
                    cmd += ["--gt", str(repo_root / exp.meta["gt"])]
                log(f"{exp.id}: {Path(v).name} repeat {k + 1}/{exp.meta['repeats']}")
                code, out, err, wall = runner(cmd, repo_root / "code")
                m = re.search(r"^(run_\d{8}_\d{6}_[0-9a-f]{4}) ", out, re.M)
                if code not in (0,) or m is None or not RUN_ID_RE.fullmatch(m.group(1)):
                    raise ExperimentError(f"run_pose failed (exit {code}): {err.strip()[-600:]}")
                paths = RunPaths.for_source(source, m.group(1), paths_cfg, repo_root)
                manifest = json.loads(paths.manifest.read_text())
                reps.append(_repeat_record(manifest, wall, paths.root, repo_root))
                log(f"  {m.group(1)}: {wall:.1f} s, pose {reps[-1]['pose']['pose_fps']} fps")
            rec = {"repeats": reps, "summary": summarise(reps)}
            first = repo_root / reps[0]["run_dir"]
            rec["repeat_consistency"] = [equivalence(first, repo_root / r["run_dir"])["verdict"] for r in reps[1:]]
            ref = _reference_run(exp, v, paths_cfg, repo_root)
            if ref is not None:
                rec["equivalence"] = equivalence(ref, first)
                add_noise(rec, repo_root)
            results["videos"][v] = rec
    finally:
        layer.unlink(missing_ok=True)
    results["finished_utc"] = utc_now()
    (exp.dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (exp.dir / "table.md").write_text(format_table(results))
    if source == "public":
        write_index(area_dir("public", paths_cfg, repo_root))
    return results


def recheck(exp: Experiment, paths_cfg: dict, *, repo_root: Path = REPO_ROOT,
            equivalence: Callable[[Path, Path], dict] | None = None) -> dict:
    """Compare the stored runs again with the reference experiment's current runs
    (CPU only, after the reference was rerun) and rewrite results.json and table.md."""
    if equivalence is None:
        from ..evaluation.equivalence import compare_runs as equivalence
    results = load_results(exp.id, exp.source, paths_cfg, repo_root)
    for v, rec in results["videos"].items():
        ref = _reference_run(exp, v, paths_cfg, repo_root)
        if ref is not None:
            rec["equivalence"] = equivalence(ref, repo_root / rec["repeats"][0]["run_dir"])
            add_noise(rec, repo_root)
    results["rechecked_utc"] = utc_now()
    (exp.dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (exp.dir / "table.md").write_text(format_table(results))
    if exp.source == "public":
        write_index(area_dir("public", paths_cfg, repo_root))
    return results


def _reference_run(exp: Experiment, video: str, paths_cfg: dict, repo_root: Path) -> Path | None:
    ref = exp.meta.get("reference", "")
    if not ref or ref == exp.id:
        return None
    path = area_dir(exp.source, paths_cfg, repo_root) / ref / "results.json"
    if not path.is_file():
        raise ExperimentError(f"reference experiment {ref} has no results yet ({path})")
    res = json.loads(path.read_text())
    if video not in res["videos"]:
        raise ExperimentError(f"reference experiment {ref} has no run of {video}")
    return repo_root / res["videos"][video]["repeats"][0]["run_dir"]


def collect_provenance(exp: Experiment, repo_root: Path = REPO_ROOT) -> dict:
    from ..runs.profiler import gpu_info
    from .provenance import git_state, library_versions, model_checksum, sha256_file
    models = repo_root / exp.settings["paths"]["models_dir"]
    weights = models / f"yolo26{exp.meta['variant']}-pose.pt"
    return {"git": git_state(repo_root), "model": model_checksum(weights) if weights.is_file() else None,
            "gpu_index": int(exp.meta["gpu"]), "gpu": gpu_info(int(exp.meta["gpu"])), "versions": library_versions(),
            "data_area": exp.source,
            "videos": {v: {"name": Path(v).name, "sha256": sha256_file(repo_root / v)} for v in exp.meta["videos"]},
            "date_utc": utc_now(), "command": " ".join(sys.argv)}


# tables -------------------------------------------------------------------------------

def _f(x, nd: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def format_table(res: dict) -> str:
    """table.md of one experiment: per video the stage medians, the cost and the equivalence."""
    lines = [f"# {res['id']}: {res.get('title', '')}".rstrip(": "), "",
             f"Source {res['source']}, reference {res.get('reference') or 'none'}, "
             f"commit {(res['provenance'].get('git') or {}).get('commit', 'n/a')}, "
             f"GPU {(res['provenance'].get('gpu') or {}).get('name', 'n/a')}, "
             f"date {res.get('finished_utc', '')}.", ""]
    for v, rec in res["videos"].items():
        s = rec["summary"]
        lines += [f"## {Path(v).name} ({s['repeats']} repeats, medians)", "",
                  "| stage | median s |", "| --- | --- |"]
        lines += [f"| {k} | {_f(x, 2)} |" for k, x in s["stages_s"].items()]
        lines += [f"| full run (process wall) | {_f(s['wall_s']['median'], 2)} "
                  f"(min {_f(s['wall_s']['min'], 2)}, max {_f(s['wall_s']['max'], 2)}) |", "",
                  "| cost | value |", "| --- | --- |",
                  f"| pose only fps (pose stage) | {_f(s['pose_fps'])} |",
                  f"| inference fps (model only) | {_f(s['inference_fps'])} |",
                  f"| mean GPU util, run / pose stage (%) | {_f(s['gpu_util_mean'])} / {_f(s['pose_gpu_util_mean'])} |",
                  f"| peak GPU memory, nvidia-smi / torch (MB) | {_f(s['gpu_mem_peak_mb'])} / {_f(s['torch_peak_mb'])} |",
                  "", "| pose breakdown | median s |", "| --- | --- |"]
        lines += [f"| {k} | {_f(x, 2)} |" for k, x in s["pose_breakdown_s"].items()]
        lines += ["", f"Repeat consistency against repeat 1: {', '.join(rec['repeat_consistency']) or 'n/a'}."]
        ev = s.get("eval")
        if ev:
            f1 = ", ".join(f"F1@{k[3:]} {x}" for k, x in ev.items() if k.startswith("f1_"))
            lines += ["", f"Engine A eval against {ev.get('gt_file')}: frame accuracy {ev.get('frame_accuracy')}, edit "
                          f"{ev.get('edit')}, {f1}, boundary MAE {ev.get('boundary_mae_s')} s"
                          f"{'' if s.get('eval_same_in_repeats', True) else ' (differs between repeats)'}."]
        eq = rec.get("equivalence")
        if eq:
            p = eq.get("pose", {})
            lines += ["", f"Equivalence with {res['reference']}: **{eq['verdict']}** "
                          f"(frames within 1 px {_f(p.get('share_within_1px'), 4)}, identical {_f(p.get('share_identical'), 4)}, "
                          f"box max {p.get('box_max_px')} px, keypoint max {p.get('kp_max_px')} px, "
                          f"keypoint mean {p.get('kp_mean_px')} px, conf max {p.get('det_conf_max')} / {p.get('kp_conf_max')})."]
            lines += [f"- {d}" for d in eq["differences"]]
            nz = rec.get("noise")
            if nz:
                lines += ["", f"Within numerical noise: **{'yes' if nz['within_noise'] else 'no'}** (measures "
                              + ", ".join(f"{k} {v}" for k, v in nz["measures"].items()) + "; floor "
                              + ", ".join(f"{k} {v}" for k, v in nz["floor"].items()) + ")."]
                lines += [f"- {d}" for d in nz["failed"]]
                from ..evaluation.equivalence import change_text
                lines += [f"- above the floor: {change_text(c)}" for c in nz.get("metrics_above_floor", [])]
        lines.append("")
    return "\n".join(lines)


def _index_line(res: dict) -> str:
    if "index" in res:                     # other kinds (domain_gap, ...) give their own cells and verdict
        return (f"| [{res['id']}]({res['id']}/table.md) | {res.get('finished_utc', '')[:10]} | {res.get('title', '')} | "
                f"{'<br>'.join(res['index']['cells'])} | {res['index']['verdict']} |")
    cells = []
    for v, rec in res["videos"].items():
        s = rec["summary"]
        cells.append(f"{Path(v).name}: {_f(s['wall_s']['median'])} s, {_f(s['pose_fps'])} fps, "
                     f"{_f(s['gpu_util_mean'])} %, {_f(s['gpu_mem_peak_mb'], 0)} MB")
    verdicts = {video_verdict(rec) for rec in res["videos"].values()} - {None}
    verdict = "reference" if not res.get("reference") or res["reference"] == res["id"] else \
        ("equivalent" if verdicts == {"equivalent"} else "within noise" if verdicts <= {"equivalent", "within noise"}
         and verdicts else "different" if verdicts else "n/a")
    return (f"| [{res['id']}]({res['id']}/table.md) | {res.get('finished_utc', '')[:10]} | {res.get('title', '')} | "
            f"{'<br>'.join(cells)} | {verdict} |")


def write_index(root: Path) -> Path:
    """experiments/INDEX.md from every results.json (and approved summary.json) under root."""
    lines = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for name in ("results.json", "summary.json"):
            f = d / name
            if f.is_file():
                lines.append(_index_line(json.loads(f.read_text())))
                break
    path = root / "INDEX.md"
    path.write_text(INDEX_HEAD.format(noise=noise_text(root)) + "\n".join(lines) + "\n")
    return path


DEFAULT_NOTE = ("Default profile (decision of 2026-11-08): the w10-best-b8 settings (pose batch 8, TF32), recorded "
                "as **above the noise floor; Engine A steps, NULL decisions and flags identical** (the metric changes "
                "above the floor are listed with their absolute values in w10_summary.md). Accuracy checks use "
                "configs/reference.toml (batch 1, strict FP32, the w10-ref-notf32 settings). The QiSens run "
                "w10-qisens-b8 decides whether this default stays.")


def noise_text(root: Path) -> str:
    """The rule "equivalent within numerical noise" and the measured floor, for INDEX.md and the w10 table."""
    from ..evaluation.equivalence import NOISE_RULE
    f = root / NOISE_FLOOR_FILE
    if not f.is_file():
        return ""
    nf = json.loads(f.read_text())
    rows = "".join(f"| {k} | {v} | {' / '.join(str(x[k]) for x in nf['per_video'].values())} |\n"
                   for k, v in nf["floor"].items())
    name, _, rule = NOISE_RULE.partition(": ")
    return (f"Verdicts: **equivalent** = every strict check of evaluation/equivalence.py holds; **{name}** "
            f"(within noise) = {rule}; **different** otherwise.\n\n"
            f"Noise floor ({nf['source']}, measured {nf['measured_utc'][:10]}):\n\n"
            f"| measure | floor | {' / '.join(Path(v).name for v in nf['per_video'])} |\n| --- | --- | --- |\n{rows}\n"
            f"{DEFAULT_NOTE}\n\n")


def compare_table(results: list[dict]) -> str:
    """One table per video over several experiments: stage medians, full run, pose fps, cost, verdict."""
    from ..evaluation.equivalence import noise_measures
    videos = []
    for r in results:
        videos += [v for v in r["videos"] if v not in videos]
    out = []
    short = {"pose_body": "pose", "segmentation": "seg", "reference": "ref"}
    for v in videos:
        stages = [s for s in STAGE_ORDER if any(s in r["videos"].get(v, {}).get("summary", {}).get("stages_s", {})
                                                for r in results)]
        head = ["experiment", *[short.get(s, s) for s in stages], "full run s", "pose fps", "infer fps",
                "GPU util %", "pose util %", "peak MB", "torch MB", "within 1 px", "track changes / frame", "metric max rel delta", "verdict"]
        rows = []
        for r in results:
            rec = r["videos"].get(v)
            if rec is None:
                continue
            s, eq = rec["summary"], rec.get("equivalence") or {}
            nm = noise_measures(eq) if eq else {}
            rows.append([r["id"], *[_f(s["stages_s"].get(x), 1) for x in stages], _f(s["wall_s"]["median"], 1),
                         _f(s["pose_fps"]), _f(s["inference_fps"]), _f(s["gpu_util_mean"]),
                         _f(s["pose_gpu_util_mean"]), _f(s["gpu_mem_peak_mb"], 0), _f(s["torch_peak_mb"], 0),
                         _f(eq.get("pose", {}).get("share_within_1px"), 4) if eq else "ref",
                         _f(nm.get("track_changes_per_frame"), 4) if eq else "ref",
                         _f(nm.get("metric_max_rel_delta"), 4) if eq else "ref",
                         video_verdict(rec) or "reference"])
        out += [f"### {Path(v).name}", "", "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
        out += ["| " + " | ".join(row) + " |" for row in rows]
        from ..evaluation.equivalence import change_text
        notes = [f"- {r['id']}: {change_text(c)}" for r in results if r["videos"].get(v)
                 for c in (r["videos"][v].get("noise") or {}).get("metrics_above_floor", [])]
        if notes:
            out += ["", "Metrics above the noise floor (reference to candidate, absolute change, relative change):", *notes]
        out.append("")
    return "\n".join(out)


def load_results(exp_id: str, source: str, paths_cfg: dict, repo_root: Path = REPO_ROOT) -> dict:
    _check_id(exp_id)
    path = area_dir(source, paths_cfg, repo_root) / exp_id / "results.json"
    if not path.is_file():
        raise ExperimentError(f"experiment {exp_id} has no results.json in the {source} area")
    return json.loads(path.read_text())

