"""The robustness study of a stored run (Week 12, rule experiments/w12-robustness/rule.md). CPU.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.robustness run --run RUN_DIR --gt GT_CSV [--groups a,b,c,d,e]
        [--vlm-file vlm_3s2f.npz] [--l-run L_COPY_DIR] [--reference 93.20]
    python -m assemblyvision.tools.robustness prepare-pose --l-run L_RUN_DIR --x-run X_RUN_DIR [--gpu -1]

Every variant is segmented and scored in memory with the frozen decoder. Description variants score the run's
stored SigLIP image embeddings with other text; a temporary score file lies in the run folder during the
run and is removed. prepare-pose makes a hidden copy of the l run with its own SigLIP scores and the x run's
VLM scores. Output: <data area>/validation/robustness.json and .md (summary numbers only), also printed.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np

from ..config import Config, deep_merge
from ..errors import AvError
from ..storage.files import REPO_ROOT, run_source, utc_now
from . import freeze

AUTHORS = ("asel", "kazi", "huzaifa")
AUTHOR_DIR = REPO_ROOT / "docs_team" / "independent_descriptions"
SEED = 20261121
N_SUBSETS = 20
TOLERANCE = 5.0
SHARE = 0.8
TEMP_FILE = "_robustness_siglip.npz"
WEIGHTS = ("vlm_weight", "siglip_weight", "siglip_crop_weight")


def weight_variants(seg: dict) -> list[tuple[str, dict]]:
    out = []
    for k in WEIGHTS:
        for f in (0.5, 2.0):
            out.append((f"{k} x{f:g}", {k: seg[k] * f}))
    for f in (0.5, 2.0):
        out.append((f"siglip_temp x{f:g}", {"siglip_temp": seg["siglip_temp"] * f}))
    return out


def subsets(counts: list[int], i: int) -> list[np.ndarray]:
    """For each phase a random half of its descriptions (variant i, seeded)."""
    rng = np.random.default_rng(SEED + i)
    return [np.sort(rng.choice(n, n // 2, replace=False)) for n in counts]


def bank_from(emb: list[np.ndarray], pick: list[np.ndarray] | None = None) -> np.ndarray:
    """[K, D] phase embeddings: the normalised mean of the chosen unit description embeddings (siglip.text_bank)."""
    rows = []
    for k, e in enumerate(emb):
        m = (e if pick is None else e[pick[k]]).mean(0)
        rows.append(m / np.linalg.norm(m))
    return np.stack(rows).astype(np.float32)


def summarise(values: list[float], reference: float) -> dict:
    """n, min, median, max, the share within the tolerance and the verdict of the rule."""
    if not values:
        return {"n": 0, "verdict": "not tested"}
    within = sum(abs(v - reference) <= TOLERANCE for v in values) / len(values)
    need = 1.0 if len(values) == 1 else SHARE
    return {"n": len(values), "min": round(min(values), 2), "median": round(statistics.median(values), 2),
            "max": round(max(values), 2), "share_within": round(within, 3),
            "verdict": "robust" if within >= need else "sensitive"}


def overall(groups: dict) -> str:
    v = [g["summary"]["verdict"] for g in groups.values()]
    if "sensitive" in v:
        return "sensitive to: " + ", ".join(k for k, g in groups.items() if g["summary"]["verdict"] == "sensitive")
    if "not tested" in v:
        return "not complete: " + ", ".join(k for k, g in groups.items() if g["summary"]["verdict"] == "not tested") + " not tested"
    return "robust overall"


class Study:
    def __init__(self, root: Path, source: str, gt_path: Path, layers: list[dict]):
        from ..evaluation.steps_io import read_steps_csv
        from .run_pose import stored_with
        self.root, self.source, self.gt_path = root, source, gt_path
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.cfg = Config.load(overrides=stored_with(self.manifest, *layers, {"seg_a": {"engine": "B zero shot"}}))
        self.gt = read_steps_csv(gt_path)
        self.duration = float(self.manifest["ingest"]["cfr"]["duration_s"])
        self.bprob = None

    def score(self, over: dict | None = None, zero: dict | None = None) -> dict:
        from ..evaluation.steps_eval import score_steps
        from .run_pose import _MemoryCtx, boundary_probability, engine_b_needed, segment_stored
        layer = {}
        if over:
            layer["seg_a"] = {**over, "engine": "custom"}
        if zero:
            layer["zero_shot"] = zero
        c = Config.load(overrides=deep_merge(self.cfg.snapshot(), layer))
        ctx = _MemoryCtx(self.source)
        if self.bprob is None and engine_b_needed(c):
            self.bprob = boundary_probability(self.manifest, self.root, c, ctx) or {}
        engine, steps = segment_stored(self.manifest, self.root, c, ctx, self.bprob or None)
        if any("AV-SEG-00" in n for n in ctx.notices):
            raise ValueError(f"{self.root.name}: scores missing: {ctx.notices}")
        res, _, _ = score_steps(steps, self.gt, self.gt_path.name, engine.vocabulary, self.source, c, self.duration)
        return res

    def with_bank(self, bank: np.ndarray) -> dict:
        """The score with SigLIP cosines of the stored image embeddings against this text bank."""
        from ..analysis.siglip import load_scores
        z = load_scores(self.root / self.cfg["zero_shot"]["siglip_file"])
        tmp = self.root / TEMP_FILE
        try:
            np.savez(tmp, time_s=z["time_s"], full_cos=z["full_emb"].astype(np.float32) @ bank.T,
                     crop_cos=np.where(z["crop_ok"][:, None], z["crop_emb"].astype(np.float32) @ bank.T, np.nan).astype(np.float32),
                     meta=json.dumps(z["meta"]))
            return self.score(zero={"siglip_file": TEMP_FILE})
        finally:
            tmp.unlink(missing_ok=True)


def phase_set_of(root: Path, cfg) -> list[dict]:
    from ..analysis import zero_shot as Z
    from ..analysis.siglip import load_scores
    name = load_scores(root / cfg["zero_shot"]["siglip_file"])["meta"].get("set", "qisens")
    return Z.phase_set(name if name == "qisens" or ":" in name else f"ikea_asm:{name}")[0]


def encode(phases: list[dict]) -> list[np.ndarray]:
    """Per phase the unit embedding of each description, on the CPU (SigLIP 2, offline)."""
    from ..analysis import siglip as S
    model, proc = S.load(REPO_ROOT / "models", "cpu", half=False)
    return [S.text_bank(model, proc, [{"descriptions": [d]} for d in p["descriptions"]], "cpu") for p in phases]


def author_phases(name: str, wanted: list[str]) -> list[dict] | None:
    from ..analysis import phase_text as pt
    path = AUTHOR_DIR / f"{name}.md"
    if not path.is_file():
        return None
    got = pt.parse(path).get(None, [])
    if [p["name"] for p in got] != wanted or not all(p["descriptions"] for p in got):
        raise ValueError(f"{path.name}: needs the phases {wanted} in this order, each with \"- \" lines")
    return got


def row(name: str, r: dict) -> dict:
    return {"variant": name, **{k: r[k] for k in ("frame_accuracy", "edit", "f1_10", "f1_25", "f1_50", "boundary_mae_s")}}


def run_study(args) -> dict:
    base = Config.load()
    source, root = run_source(args.run, base["paths"])
    layers = []
    st = Study(root, source, args.gt, layers)
    groups: dict = {}
    want = args.groups.split(",")
    base_row = st.score()
    ref = args.reference if args.reference is not None else base_row["frame_accuracy"]
    add = lambda k, rows: groups.__setitem__(k, {"rows": rows, "summary": summarise([x["frame_accuracy"] for x in rows], ref)})
    if "a" in want:
        add("a. weights", [row(n, st.score(o)) for n, o in weight_variants(st.cfg["seg_a"])])
    emb, phases = None, None
    if "b" in want or "c" in want:
        phases = phase_set_of(root, st.cfg)
        emb = encode(phases)
    rebuilt = None
    if emb is not None:
        rebuilt = row("all descriptions, from the stored embeddings", st.with_bank(bank_from(emb)))
    if "b" in want:
        counts = [len(e) for e in emb]
        add("b. description subsets", [row(f"subset {i}", st.with_bank(bank_from(emb, subsets(counts, i))))
                                       for i in range(N_SUBSETS)])
    if "c" in want:
        rows = []
        for a in AUTHORS:
            got = author_phases(a, [p["name"] for p in phases])
            if got is not None:
                rows.append(row(a, st.with_bank(bank_from(encode(got)))))
        add("c. independent descriptions", rows)
    if "d" in want:
        ok = args.vlm_file and (root / args.vlm_file).is_file()
        add("d. VLM sampling", [row(f"VLM file {args.vlm_file}", st.score(zero={"vlm_file": args.vlm_file}))] if ok else [])
    if "e" in want:
        rows = []
        if args.l_run:
            _, lr = run_source(args.l_run, base["paths"])
            rows = [row("pose YOLO26 l", Study(lr, source, args.gt, layers).score())]
        add("e. pose model", rows)
    return {"created_utc": utc_now(), "freeze": freeze.record(), "run": root.name, "reference": ref,
            "base": row("the run as stored", base_row), "rebuilt": rebuilt, "tolerance": TOLERANCE, "groups": groups,
            "overall": overall(groups), "area": str(root.parent.parent)}


def markdown(res: dict) -> str:
    f = res["freeze"]
    L = ["# Robustness study (rule w12-robustness)", "",
         f"Freeze: {f['tag']} ({f['tag_commit'][:7]}), ran on {f['ran_on'][:7]}, method unchanged: {f['method_unchanged']}. "
         f"Run {res['run']}; reference {res['reference']:.2f}; base row {res['base']['frame_accuracy']:.2f}"
         + (f"; all descriptions from the stored embeddings {res['rebuilt']['frame_accuracy']:.2f}" if res.get("rebuilt") else "") + ".", "",
         f"**Overall: {res['overall']}.**", "", "| group | n | min | median | max | within 5 points | verdict |",
         "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, g in res["groups"].items():
        s = g["summary"]
        L.append(f"| {k} | {s['n']} | - | - | - | - | {s['verdict']} |" if not s["n"] else
                 f"| {k} | {s['n']} | {s['min']:.2f} | {s['median']:.2f} | {s['max']:.2f} | {100 * s['share_within']:.0f} % | {s['verdict']} |")
    L += ["", "| group | variant | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, g in res["groups"].items():
        for r in g["rows"]:
            mae = "n/a" if r["boundary_mae_s"] is None else f"{r['boundary_mae_s']:.2f}"
            L.append(f"| {k[0]} | {r['variant']} | {r['frame_accuracy']:.2f} | {r['edit']:.2f} | {r['f1_10']:.2f} | {r['f1_25']:.2f} | {r['f1_50']:.2f} | {mae} |")
    return "\n".join(L) + "\n"


def prepare_pose(args) -> int:
    from ..analysis.siglip import load_scores
    from .milestone_run import make_copy
    base = Config.load()
    _, l_root = run_source(args.l_run, base["paths"])
    _, x_root = run_source(args.x_run, base["paths"])
    dst = make_copy(l_root, "Robustness pose l (test copy)", [], hidden=True)
    for f in ("vlm.npz", "siglip.npz"):              # never write through a hard link into the l run's own files
        (dst / f).unlink(missing_ok=True)
    os.link(x_root / "vlm.npz", dst / "vlm.npz")
    fps = load_scores(x_root / "siglip.npz")["meta"]["fps"]
    r = subprocess.run([sys.executable, "-m", "assemblyvision.tools.siglip_scores", "--run", str(dst), "--gpu", str(args.gpu),
                        "--force", "--fps", str(fps)], env={**os.environ, "CUDA_VISIBLE_DEVICES": "" if args.gpu < 0 else os.environ.get("CUDA_VISIBLE_DEVICES", "")})
    if r.returncode != 0:
        return r.returncode
    print(f"l copy for group e: {dst.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.robustness")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--gt", type=Path, required=True)
    r.add_argument("--groups", default="a,b,c,d,e")
    r.add_argument("--vlm-file")
    r.add_argument("--l-run", type=Path)
    r.add_argument("--reference", type=float, help="the frame accuracy the tolerance is around (the rule: 93.20)")
    q = sub.add_parser("prepare-pose")
    q.add_argument("--l-run", type=Path, required=True)
    q.add_argument("--x-run", type=Path, required=True)
    q.add_argument("--gpu", type=int, default=-1)
    a = p.parse_args(argv)
    if a.cmd == "prepare-pose":
        return prepare_pose(a)
    try:
        res = run_study(a)
    except (AvError, OSError, ValueError, KeyError) as e:
        print(f"robustness: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    out = Path(res.pop("area")) / "validation"
    out.mkdir(exist_ok=True)
    (out / "robustness.json").write_text(json.dumps(res, indent=1))
    md = markdown(res)
    (out / "robustness.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
