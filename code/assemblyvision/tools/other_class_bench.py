"""seg_a.other_class validation (Week 11, rule experiments/w11-other-class/rule.md).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.other_class_bench --id w11-other-class --vlm-file vlm.npz

CPU only, public data only: the 40 IKEA ASM training videos of w11-vlm-speed (videos.json),
untuned "B zero shot" weights, synthetic cuts p 0.5, other_class off and on. Two conditions:
- natural: the videos as they are (their own frames outside the annotated actions);
- synthetic intro and outro: before and after each video, "no work" material spliced from
  the frames outside the annotated actions of the other videos of the same furniture type
  (seeded lengths, intro 20 to 60 s, outro 10 to 30 s, a cut at every junction). Every
  per frame evidence (person, motion, the B v1 embedding, the phase scores) is spliced the
  same way, and the spliced frames are "other" in the GT.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from ..storage.files import REPO_ROOT, utc_now
from . import ikea_bench as ib

SETS = ("Kallax_Shelf_Drawer", "Lack_Coffee_Table")
INTRO_S, OUTRO_S, SEED = (20.0, 60.0), (10.0, 30.0), 42


def overrides(vlm_file: str, other: bool) -> dict:
    from .zero_shot_grid import UNTUNED, overrides as zo
    o = zo(UNTUNED)
    o["seg_a"]["other_class"] = other
    o["zero_shot"] = {"vlm_file": vlm_file}
    return o


def no_work(vid: str, n: int, fps: float) -> list[tuple[int, int]]:
    """The frame ranges [s, e) of a video outside its annotated actions, at its start and end."""
    steps = ib.gt_steps(vid, n / fps)
    out = []
    for st in (steps[0], steps[-1]):
        if st.label == "other":
            s, e = int(round(st.start_s * fps)), min(int(round(st.end_s * fps)), n)
            if e - s >= int(fps):                   # at least 1 s
                out.append((s, e))
    return out


def _cut(tl, s: int, e: int) -> dict:
    """The per frame arrays of tl over [s, e)."""
    return {"person": tl.person[s:e], "energy": tl.energy[s:e],
            "prob": None if tl.boundary_prob is None else tl.boundary_prob[s:e],
            "emb": None if tl.embedding is None else tl.embedding[s:e],
            "emb_ok": None if tl.embedding_valid is None else tl.embedding_valid[s:e],
            "scores": {k: v[s:e] for k, v in tl.phase_scores.items()},
            "cuts": [c - s for c in tl.cuts if s < c < e]}


def splice(pieces: list[dict], fps: float):
    """One RunTimeline from pieces in order, a cut at every junction."""
    from ..analysis.segmentation.base import RunTimeline
    cuts, off = [], 0
    for i, p in enumerate(pieces):
        if i > 0:
            cuts.append(off)
        cuts += [off + c for c in p["cuts"]]
        off += len(p["person"])
    tl = RunTimeline(fps, off, np.concatenate([p["person"] for p in pieces]),
                     np.concatenate([p["energy"] for p in pieces]), sorted(set(cuts)))
    if all(p["prob"] is not None for p in pieces):
        tl.boundary_prob = np.concatenate([p["prob"] for p in pieces])
    if all(p["emb"] is not None for p in pieces):
        tl.embedding = np.concatenate([p["emb"] for p in pieces])
        tl.embedding_valid = np.concatenate([p["emb_ok"] for p in pieces])
    keys = set.intersection(*(set(p["scores"]) for p in pieces))
    tl.phase_scores = {k: np.concatenate([p["scores"][k] for p in pieces]) for k in sorted(keys)}
    return tl


def plan(vids: dict) -> dict:
    """Per target video: intro and outro lengths and the donor order (seeded)."""
    out = {}
    for s, vs in vids.items():
        for i, v in enumerate(vs):
            rng = np.random.default_rng([SEED, SETS.index(s), i])
            donors = [d for d in vs if d != v]
            out[v] = {"intro_s": round(float(rng.uniform(*INTRO_S)), 1), "outro_s": round(float(rng.uniform(*OUTRO_S)), 1),
                      "donors": [donors[j] for j in rng.permutation(len(donors))]}
    return out


def score(job: tuple) -> list:
    """(vids, plan, vlm_file) -> per video [natural off, natural on, synthetic off, synthetic on] metrics."""
    from ..analysis.segmentation import create_engine
    from ..evaluation.steps_eval import evaluate
    from ..types import Step
    from .run_pose import _MemoryCtx
    vids, pl, vlm_file = job
    b = ib.IkeaBench()
    res = []
    for v in vids:
        row = []
        for other in (False, True):
            row.append(b.video(v, overrides(vlm_file, other), cuts_p=0.5)["metrics"])
        for other in (False, True):
            over = overrides(vlm_file, other)
            cfg, _, tl, manifest = b.timeline(v, over, cuts_p=0.5)
            fps = tl.fps
            parts = {}

            def fill(sec: float) -> list[dict]:
                need, got = int(round(sec * fps)), []
                for d in pl[v]["donors"] * 4:
                    if need <= 0:
                        break
                    if d not in parts:
                        _, _, dtl, _ = b.timeline(d, over, cuts_p=None)
                        parts[d] = [_cut(dtl, s, e) for s, e in no_work(d, dtl.n_frames, fps)]
                    for p in parts[d]:
                        if need <= 0:
                            break
                        k = min(need, len(p["person"]))
                        got.append({kk: (vv[:k] if isinstance(vv, np.ndarray) else
                                         {a: x[:k] for a, x in vv.items()} if isinstance(vv, dict) else
                                         [c for c in vv if c < k] if isinstance(vv, list) else vv)
                                    for kk, vv in p.items()})
                        need -= k
                return got
            intro, outro = fill(pl[v]["intro_s"]), fill(pl[v]["outro_s"])
            n_in = sum(len(p["person"]) for p in intro)
            main = _cut(tl, 0, tl.n_frames)
            main["cuts"] = list(tl.cuts)
            stl = splice([*intro, main, *outro], fps)
            engine = create_engine("A", cfg, _MemoryCtx("public"))
            pred = engine.segment_run(stl)
            dur0 = float(manifest["ingest"]["cfr"]["duration_s"])
            sh = n_in / fps
            gt = [Step("other", 0.0, round(sh, 3), "gt")] if n_in else []
            gt += [Step(s.label, round(s.start_s + sh, 3), round(s.end_s + sh, 3), s.source) for s in ib.gt_steps(v, dur0)]
            dur = stl.n_frames / fps
            if gt[-1].end_s < dur:
                gt.append(Step("other", gt[-1].end_s, round(dur, 3), "gt"))
            m, _ = evaluate(pred, gt, dur, cfg["eval"]["seg_fps"], cfg["eval"]["overlaps"])
            m["intro_s"], m["outro_s"] = round(sh, 2), round(dur - sh - dur0, 2)
            row.append(m)
        res.append(row)
    return res


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 2) if xs else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.other_class_bench")
    p.add_argument("--id", required=True)
    p.add_argument("--vlm-file", default="vlm.npz", help="the VLM file of the setting (its _other name for on)")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    from ..experiments import registry as reg
    from ..experiments.provenance import git_state
    sel = json.loads((REPO_ROOT / "experiments/w11-vlm-speed/videos.json").read_text())
    vids = {s: sel["sets"][s] for s in SETS}
    pl = plan(vids)
    jobs = [(s, vs[i:i + 4]) for s, vs in vids.items() for i in range(0, len(vs), 4)]
    with ProcessPoolExecutor(args.workers) as ex:
        parts = list(ex.map(score, [(v, pl, args.vlm_file) for _, v in jobs]))
    names = ["natural, off", "natural, on", "synthetic intro and outro, off", "synthetic intro and outro, on"]
    summ = {n: {} for n in names}
    for k, n in enumerate(names):
        for s in SETS:
            xs = [row[k] for (ss, _), part in zip(jobs, parts) if ss == s for row in part]
            summ[n][s] = {"frame_accuracy": mean([x["frame_accuracy"] for x in xs]), "f1_50": mean([x.get("f1_50") for x in xs]),
                          "boundary_mae_s": mean([x.get("boundary_mae_s") for x in xs]), "videos": len(xs)}
    worst = {n: min(summ[n][s]["frame_accuracy"] for s in SETS) for n in names}
    d_nat = round(worst[names[1]] - worst[names[0]], 2)
    d_syn = round(worst[names[3]] - worst[names[2]], 2)
    on = d_nat >= -1.0 and d_syn >= 2.0
    res = {"id": args.id, "kind": "other_class_bench", "source": "public", "reference": "",
           "title": "seg_a.other_class on IKEA ASM: natural ends and synthetic intros and outros (rule w11-other-class)",
           "rule": "experiments/w11-other-class/rule.md", "vlm_file": args.vlm_file, "videos": vids, "plan": pl,
           "summary": summ, "worst": worst, "delta_natural": d_nat, "delta_synthetic": d_syn,
           "decision": "on" if on else "off",
           "provenance": {"git": git_state(REPO_ROOT), "date_utc": utc_now(), "command": " ".join(sys.argv)},
           "finished_utc": utc_now()}
    lines = [f"# {args.id}", "", res["title"] + ".", "",
             f"40 IKEA ASM training videos (w11-vlm-speed/videos.json), untuned B zero shot weights, synthetic cuts "
             f"p 0.5, VLM file {args.vlm_file} (on: its _other version). Frame accuracy, mean over videos.", "",
             "| condition | Kallax | Lack Coffee | worst | F1@50 Kallax | MAE s Kallax |", "| --- | --- | --- | --- | --- | --- |"]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(str(summ[n][s]["frame_accuracy"]) for s in SETS)
                     + f" | {worst[n]} | {summ[n][SETS[0]]['f1_50']} | {summ[n][SETS[0]]['boundary_mae_s']} |")
    lines += ["", f"Change of the worst case with other_class on: natural {d_nat:+.2f}, synthetic {d_syn:+.2f}.",
              f"Decision (rule: natural at least -1.0 and synthetic at least +2.0): **{res['decision']}**.", ""]
    res["index"] = {"cells": [f"natural {d_nat:+.2f}, synthetic {d_syn:+.2f}: {res['decision']}"], "verdict": "measurement"}
    out = REPO_ROOT / "experiments" / args.id
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    (out / "table.md").write_text("\n".join(lines))
    reg.write_index(REPO_ROOT / "experiments")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
