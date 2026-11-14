"""Report export of one run (Week 11): report.json, steps.csv, metrics.csv and report.pdf.

Everything is read from the run folder (manifest.json, steps.csv, metrics.csv, eval.json,
siglip.npz, vlm.npz) and written into <run>/export/rev<N>/, so a QiSens run's export stays
under data/qisens. Revision 0 is the run as the pipeline made it; a correction made in the
interface (ui/corrections.py) is a later revision with its own steps.csv and metrics.csv,
and the earlier revisions are never changed.

- Timestamps: the run's created and finished times, the export time.
- Phase confidence per step: the decoder's combination of the zero shot scores per frame,
  posterior(k) proportional to exp(sum over the sources of weight * log p_source(k)) (SigLIP
  through softmax(cosine / siglip_temp), the VLM probabilities as given), averaged over the
  frames of the step for its own phase. None when the run has no zero shot scores.
- Provenance: the git commit of the run (and of the export), the model files with SHA256,
  the configuration with its SHA256, the GPU, the library versions.
- Notices: every notice of the run.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from ..storage.files import REPO_ROOT, utc_now

SHA_CACHE = REPO_ROOT / "models" / ".sha256_cache.json"


def file_sha256(path: Path) -> str | None:
    """SHA256 of a file, cached by (path, size, mtime) in models/.sha256_cache.json."""
    if not path.is_file():
        return None
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    try:
        cache = json.loads(SHA_CACHE.read_text())
    except (OSError, ValueError):
        cache = {}
    if key not in cache:
        from ..experiments.provenance import sha256_file
        cache[key] = sha256_file(path)
        try:
            SHA_CACHE.write_text(json.dumps(cache, indent=1))
        except OSError:
            pass
    return cache[key]


def models_used(manifest: dict) -> list[dict]:
    cfg = manifest.get("config", {})
    seg = cfg.get("seg_a", {})
    md = REPO_ROOT / cfg.get("paths", {}).get("models_dir", "models")
    out = [("pose", md / f"yolo26{manifest.get('variant', 'x')}-pose.pt")]
    if seg.get("embed_weight", 0) > 0 or seg.get("engine_b_weight", 0) > 0:
        out.append(("engine_b", md / cfg.get("engine_b", {}).get("model", "")))
    if seg.get("siglip_weight", 0) > 0 or seg.get("siglip_crop_weight", 0) > 0:
        out.append(("siglip", md / "siglip2/so400m-patch14-384/model.safetensors"))
    if seg.get("vlm_weight", 0) > 0:
        from ..analysis import vlm as V
        out += [("vlm", md / V.MODEL), ("vlm_mmproj", md / V.MMPROJ)]
    return [{"role": r, "file": str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else p.name,
             "sha256": file_sha256(p)} for r, p in out]


def provenance(manifest: dict) -> dict:
    from ..experiments.provenance import git_state
    cfg = manifest.get("config", {})
    run_prov = manifest.get("provenance", {})
    return {"run_commit": run_prov.get("git", {}).get("commit"), "run_dirty": run_prov.get("git", {}).get("dirty"),
            "export_commit": git_state(REPO_ROOT).get("commit"),
            "models": models_used(manifest),
            "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            "config": cfg, "gpu": {"index": manifest.get("gpu"), "name": run_prov.get("gpu_name"),
                                   "vlm_index": cfg.get("zero_shot", {}).get("vlm_gpu")
                                   if cfg.get("seg_a", {}).get("vlm_weight", 0) > 0 else None,
                                   "vlm_name": run_prov.get("vlm_gpu_name")},
            "versions": manifest.get("versions", {})}


def frame_posterior(manifest: dict, root: Path, K: int) -> np.ndarray | None:
    """[n_frames, K] posterior of the phases from the run's zero shot scores, or None."""
    from ..analysis.siglip import run_phase_scores
    cfg = manifest.get("config", {})
    seg = cfg.get("seg_a", {})
    if not any(seg.get(k, 0) > 0 for k in ("siglip_weight", "siglip_crop_weight", "vlm_weight")):
        return None
    cfr = manifest["ingest"]["cfr"]
    n, fps = int(cfr["frame_count"]), float(cfr["fps"])
    seg_k = {**seg, "other_class": False}
    scores, _ = run_phase_scores(root, n, fps, seg_k, cfg.get("zero_shot", {}).get("vlm_file", "vlm.npz"))
    weights = {"siglip_full": seg["siglip_weight"], "siglip_crop": seg["siglip_crop_weight"], "vlm": seg["vlm_weight"]}
    total, any_src = np.zeros((n, K)), False
    for k, x in scores.items():
        if weights.get(k, 0) <= 0 or x.shape[1] != K:
            continue
        ok = np.isfinite(x).all(axis=1)
        if k.startswith("siglip"):
            z = np.where(ok[:, None], x, 0.0) / seg.get("siglip_temp", 0.01)
            z = np.exp(z - z.max(axis=1, keepdims=True))
            p = z / z.sum(axis=1, keepdims=True)
        else:
            p = np.where(ok[:, None], x, 1.0 / K)
            p = p / p.sum(axis=1, keepdims=True)
        total += weights[k] * np.log(np.clip(np.where(ok[:, None], p, 1.0 / K), 1e-9, 1.0))
        any_src = True
    if not any_src:
        return None
    total = np.exp(total - total.max(axis=1, keepdims=True))
    return total / total.sum(axis=1, keepdims=True)


def read_steps(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return [{"step": int(r["step"]), "label": r["label"], "start_s": float(r["start_s"]), "end_s": float(r["end_s"]),
                 "source": r.get("source", ""), "score": None if r.get("score") in (None, "") else float(r["score"])}
                for r in csv.DictReader(f)]


def read_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def step_confidence(steps: list[dict], post: np.ndarray | None, vocab: list[str], fps: float) -> None:
    """Adds "phase_confidence" (0 to 1, or None) to each step in place. A step is matched to
    its phase by its label (a vocabulary name, or "Phase k" by position)."""
    for s in steps:
        k = vocab.index(s["label"]) if s["label"] in vocab else None
        if k is None and s["label"].startswith("Phase ") and s["label"][6:].isdigit():
            k = int(s["label"][6:]) - 1
        if post is None or k is None or not 0 <= k < post.shape[1]:
            s["phase_confidence"] = None
            continue
        a, b = int(round(s["start_s"] * fps)), int(round(s["end_s"] * fps))
        seg = post[a:max(b, a + 1), k]
        s["phase_confidence"] = round(float(seg.mean()), 3) if len(seg) else None


def next_revision(root: Path) -> int:
    d = root / "export"
    revs = [int(p.name[3:]) for p in d.glob("rev*") if p.name[3:].isdigit()] if d.is_dir() else []
    return max(revs) + 1 if revs else 0


def build(root: Path, steps_csv: Path | None = None, metrics_csv: Path | None = None,
          revision: dict | None = None) -> dict:
    """The report record of a run (revision: {"number", "based_on", "by", "note", "created_utc"})."""
    manifest = json.loads((root / "manifest.json").read_text())
    cfr = manifest.get("ingest", {}).get("cfr", {})
    fps = float(cfr.get("fps", 30.0))
    vocab = manifest.get("segmentation", {}).get("vocabulary") or \
        [d["name"] for d in manifest.get("config", {}).get("steps", {}).get("step", [])]
    steps = read_steps(steps_csv or root / "steps.csv")
    phases = [s for s in steps if s["label"] != "other"]
    post = frame_posterior(manifest, root, len(vocab)) if vocab else None
    step_confidence(steps, post, vocab, fps)
    ev = None
    if (root / "eval.json").is_file() and (revision is None or revision.get("number") == 0):   # eval of the run as computed
        e = json.loads((root / "eval.json").read_text())
        ev = {"gt_file": e.get("gt_file"), "metrics": e.get("metrics")}
    return {"report_version": 1, "run_id": manifest.get("run_id"), "source": manifest.get("source"),
            "video": Path(manifest.get("video", "")).name, "duration_s": cfr.get("duration_s"),
            "timestamps": {"run_created_utc": manifest.get("created_utc"), "run_finished_utc": manifest.get("finished_utc"),
                           "exported_utc": utc_now()},
            "revision": revision or {"number": 0, "based_on": None, "by": "pipeline", "note": "as computed"},
            "status": manifest.get("status"), "step_engine": manifest.get("config", {}).get("seg_a", {}).get("engine"),
            "steps": steps, "phases_found": len(phases), "phases_expected": len(vocab),
            "metrics": read_metrics(metrics_csv or root / "metrics.csv"), "eval": ev,
            "notices": manifest.get("notices", []), "provenance": provenance(manifest)}


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: "" if r.get(c) is None else r.get(c) for c in cols})


def write_pdf(path: Path, rep: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    def text_page(pdf, title: str, lines: list[str], per_page: int = 52) -> None:
        for i in range(0, max(len(lines), 1), per_page):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(0.07, 0.96, title + (" (continued)" if i else ""), fontsize=13, weight="bold", va="top")
            fig.text(0.07, 0.93, "\n".join(lines[i:i + per_page]), fontsize=7.5, family="monospace", va="top")
            pdf.savefig(fig)
            plt.close(fig)
    rev = rep["revision"]
    with PdfPages(path, metadata={"Title": f"AssemblyVision report {rep['run_id']} rev {rev['number']}",
                                  "Creator": "AssemblyVision"}) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.96, "AssemblyVision run report", fontsize=16, weight="bold", va="top")
        ts = rep["timestamps"]
        head = [f"Run {rep['run_id']} ({rep['source']}), video {rep['video']}, {rep['duration_s']} s",
                f"Revision {rev['number']} ({rev['by']}: {rev['note']})" + (f", based on rev {rev['based_on']}"
                                                                           if rev.get("based_on") is not None else ""),
                f"Run created {ts['run_created_utc']}, finished {ts['run_finished_utc']}, exported {ts['exported_utc']}",
                f"Step engine {rep['step_engine']}, status {rep['status']}, "
                f"{rep['phases_found']} of {rep['phases_expected']} phases"]
        if rep["eval"]:
            m = rep["eval"]["metrics"]
            head.append(f"Eval against {rep['eval']['gt_file']}: frame accuracy {m.get('frame_accuracy')}, "
                        f"edit {m.get('edit')}, F1@50 {m.get('f1_50')}, boundary MAE {m.get('boundary_mae_s')} s")
        fig.text(0.07, 0.92, "\n".join(head), fontsize=9, va="top")
        ax = fig.add_axes([0.07, 0.66, 0.86, 0.12])
        cmap = plt.get_cmap("tab10")
        for s in rep["steps"]:
            col = (0.85, 0.85, 0.85) if s["label"] == "other" else cmap(s["step"] % 10)
            ax.barh(0, s["end_s"] - s["start_s"], left=s["start_s"], color=col, edgecolor="white")
            if s["label"] != "other":
                c = s.get("phase_confidence")
                ax.text((s["start_s"] + s["end_s"]) / 2, 0, f"{s['step']}" + ("" if c is None else f"\n{c:.2f}"),
                        ha="center", va="center", fontsize=7)
        ax.set_yticks([])
        ax.set_xlabel("time (s); label: step and phase confidence")
        ax.set_xlim(0, float(rep["duration_s"] or max(s["end_s"] for s in rep["steps"])))
        rows = [[s["step"], s["label"], f"{s['start_s']:.1f}", f"{s['end_s']:.1f}", f"{s['end_s'] - s['start_s']:.1f}",
                 "-" if s.get("phase_confidence") is None else f"{s['phase_confidence']:.2f}"] for s in rep["steps"]]
        tax = fig.add_axes([0.07, 0.08, 0.86, 0.52])
        tax.axis("off")
        if rows:
            t = tax.table(cellText=rows[:40], colLabels=["step", "label", "start s", "end s", "duration s", "confidence"],
                          loc="upper center", cellLoc="left")
            t.auto_set_font_size(False)
            t.set_fontsize(8)
        pdf.savefig(fig)
        plt.close(fig)
        mlines = [f"{'step':>4}  {'label':<28} {'metric':<22} {'value':>10} {'unit':<8} {'cov':>6}  flags"]
        for r in rep["metrics"]:
            mlines.append(f"{r.get('step', ''):>4}  {r.get('label', '')[:28]:<28} {r.get('metric', '')[:22]:<22} "
                          f"{r.get('value', ''):>10} {r.get('unit', ''):<8} {r.get('coverage', ''):>6}  {r.get('flags', '')}")
        text_page(pdf, "Motion metrics", mlines)
        nlines = [f"[{n.get('stage', '')}] {n.get('code', '')} {n.get('text', '')}"
                  + (f" ({n['detail']})" if n.get("detail") else "") for n in rep["notices"]] or ["none"]
        pv = rep["provenance"]
        plines = [f"run commit {pv['run_commit']}, export commit {pv['export_commit']}",
                  f"config sha256 {pv['config_sha256']}",
                  f"GPU {pv['gpu']}", f"versions {pv['versions']}", "models:"]
        plines += [f"  {m['role']:<11} {m['file']}  sha256 {m['sha256']}" for m in pv["models"]]
        wrapped = []
        for line in ["Notices:", *nlines, "", "Provenance:", *plines]:
            while len(line) > 98:
                wrapped.append(line[:98])
                line = "    " + line[98:]
            wrapped.append(line)
        text_page(pdf, "Notices and provenance", wrapped)


STEP_COLS = ["step", "label", "start_s", "end_s", "source", "score", "phase_confidence"]
METRIC_COLS = ["segment_id", "step", "label", "start_s", "end_s", "metric", "value", "unit", "coverage", "flags"]


def export_run(root: Path, number: int | None = None, steps_csv: Path | None = None,
               metrics_csv: Path | None = None, revision: dict | None = None) -> Path:
    """Writes <root>/export/rev<N>/ (report.json, steps.csv, metrics.csv, report.pdf);
    an existing revision folder is never overwritten (FileExistsError)."""
    number = next_revision(root) if number is None else number
    out = root / "export" / f"rev{number}"
    out.mkdir(parents=True, exist_ok=False)
    rev = dict(revision or {"number": number, "based_on": None, "by": "pipeline", "note": "as computed"})
    rev.setdefault("created_utc", utc_now())
    rev["number"] = number
    rep = build(root, steps_csv, metrics_csv, rev)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    write_csv(out / "steps.csv", rep["steps"], STEP_COLS)
    write_csv(out / "metrics.csv", rep["metrics"], METRIC_COLS)
    write_pdf(out / "report.pdf", rep)
    return out
