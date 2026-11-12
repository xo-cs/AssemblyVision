"""Zero shot phase scoring selection (Week 11, rule experiments/w11-zero-shot/rule.md).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.zero_shot_grid --id w11-zero-shot-train                     # the grid
    python -m assemblyvision.tools.zero_shot_grid --id w11-zero-shot-test --test w11-zero-shot-train

CPU only, public data only. IKEA ASM Kallax and Lack Coffee Table runs with siglip.npz
and vlm.npz, the synthetic cuts p 0.5 (ikea_bench.CUTS), the B v1 boundary evidence plus
dense candidates, and the phase likelihood settings of GRID. The train mode scores the
official training subset and selects by the worst case over the two sets; the test mode
scores the rows of the rule on the testing subset once (p 0.5 and without cuts).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from ..experiments import registry as reg
from ..storage.files import REPO_ROOT, utc_now
from . import ikea_bench as ib
from .visual_cv_grid import mean

SETS = ib.SELECTION
GRID = {"siglip_weight": [0.0, 0.5, 1.0, 2.0, 4.0], "siglip_crop_weight": [0.0, 0.5, 1.0, 2.0, 4.0],
        "vlm_weight": [0.0, 0.5, 1.0, 2.0, 4.0], "siglip_temp": [0.01, 0.03], "shot_smooth": [False, True]}
UNTUNED = {"siglip_weight": 1.0, "siglip_crop_weight": 1.0, "vlm_weight": 1.0, "siglip_temp": 0.01, "shot_smooth": False}
B_V1 = {"engine": "custom", "engine_b_weight": 0.0, "embed_weight": 0.2, "embed_window_s": 30.0, "visual_weight": 0.0,
        "engine_a_scale": 1.0, "refine_window_s": 0.0, "intro_detect": False}


def decoders() -> list[dict]:
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]


def overrides(d: dict | None, candidates: str = "dense") -> dict:
    seg = dict(B_V1, **({"siglip_weight": 0.0, "siglip_crop_weight": 0.0, "vlm_weight": 0.0} if d is None else d))
    return {"seg_a": seg, "engine_b": {"model": "engine_b/w10-engine-b-v1.pt", "candidates": candidates,
                                       "video_norm": False}}


ENGINE_A = {"seg_a": {"engine": "custom", "engine_b_weight": 0.0, "embed_weight": 0.0, "visual_weight": 0.0,
                      "siglip_weight": 0.0, "siglip_crop_weight": 0.0, "vlm_weight": 0.0}}


def score(job: tuple) -> list:
    """(list of overrides, videos, cuts_p) -> [setting][video] (frame acc, F1@50, MAE)."""
    overs, vids, cuts_p = job
    b = ib.IkeaBench()
    out = [[] for _ in overs]
    for v in vids:
        for k, o in enumerate(overs):
            m = b.video(v, o, cuts_p=cuts_p)["metrics"]
            out[k].append([m["frame_accuracy"], m.get("f1_50"), m.get("boundary_mae_s")])
    return out


def summarise(rows: list) -> dict:
    return {"frame_accuracy": mean([x[0] for x in rows]), "f1_50": mean([x[1] for x in rows]),
            "boundary_mae_s": mean([x[2] for x in rows]), "videos": len(rows)}


def videos(split: str) -> dict:
    ok = [v for v in ib.split_videos(split, SETS) if all((REPO_ROOT / "data/public/runs" / ib.runs()[v] / f).is_file()
                                                        for f in ("siglip.npz", "vlm.npz"))]
    return {s: [v for v in ok if v.startswith(s + "/")] for s in SETS}


def run(overs: list[dict], vids: dict, cuts_p, workers: int) -> list[dict]:
    jobs = [(s, vs[i:i + 6]) for s, vs in vids.items() for i in range(0, len(vs), 6)]
    with ProcessPoolExecutor(workers) as ex:
        parts = list(ex.map(score, [(overs, v, cuts_p) for _, v in jobs]))
    res = []
    for k in range(len(overs)):
        res.append({s: summarise([x for (ss, _), part in zip(jobs, parts) if ss == s for x in part[k]]) for s in SETS})
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.zero_shot_grid")
    p.add_argument("--id", required=True)
    p.add_argument("--test", help="the train grid experiment whose selection is scored once on the testing subset")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    from ..experiments.provenance import git_state
    out = REPO_ROOT / "experiments" / args.id
    out.mkdir(parents=True, exist_ok=True)
    head = ["| row | siglip | crop | vlm | temp | shot smooth | worst | Kallax | Lack Coffee | F1@50 Kallax | MAE s Kallax |",
            "| --- " * 11 + "|"]

    def line(name, d, r):
        d = d or {}
        fa = [r[s]["frame_accuracy"] for s in SETS]
        return (f"| {name} | {d.get('siglip_weight', '-')} | {d.get('siglip_crop_weight', '-')} | "
                f"{d.get('vlm_weight', '-')} | {d.get('siglip_temp', '-')} | {d.get('shot_smooth', '-')} | {min(fa)} | "
                + " | ".join(str(x) for x in fa) + f" | {r[SETS[0]]['f1_50']} | {r[SETS[0]]['boundary_mae_s']} |")
    if args.test is None:
        vids = videos("train")
        decs = decoders()
        res = run([overrides(d) for d in decs] + [overrides(UNTUNED)], vids, 0.5, args.workers)
        rows = [{"decoder": d, "sets": r, "worst": min(r[s]["frame_accuracy"] for s in SETS),
                 "mean": round(sum(r[s]["frame_accuracy"] for s in SETS) / len(SETS), 2)} for d, r in zip(decs, res)]
        untuned = res[-1]

        def key(r):
            d = r["decoder"]
            return (-round(r["worst"], 2), -round(r["mean"], 2), d["shot_smooth"],
                    d["siglip_weight"] + d["siglip_crop_weight"] + d["vlm_weight"], d["siglip_temp"])
        rows.sort(key=key)
        best = rows[0]
        alone = {}
        for src in ("siglip_weight", "siglip_crop_weight", "vlm_weight"):
            others = [k for k in ("siglip_weight", "siglip_crop_weight", "vlm_weight") if k != src]
            alone[src] = next((r for r in rows if r["decoder"][src] > 0 and all(r["decoder"][o] == 0 for o in others)), None)
        none = next(r for r in rows if all(r["decoder"][k] == 0 for k in ("siglip_weight", "siglip_crop_weight",
                                                                         "vlm_weight")))
        res_d = {"id": args.id, "kind": "zero_shot_grid", "source": "public", "reference": "", "split": "train",
                 "title": "Zero shot phase scoring: likelihood weights on the IKEA ASM training subset (p 0.5; rule "
                          "w11-zero-shot)", "rule": "experiments/w11-zero-shot/rule.md", "grid": GRID, "videos": vids,
                 "selected": best, "untuned": {"decoder": UNTUNED, "sets": untuned}, "alone": alone,
                 "no_likelihood": none, "rows": rows}
        lines = [f"# {args.id}", "", res_d["title"], "", f"Videos: " + ", ".join(f"{s} {len(v)}" for s, v in vids.items())
                 + ". B v1 boundary evidence plus dense candidates in every row. Frame accuracy, mean over videos.", "",
                 *head, line("selected (rule)", best["decoder"], best["sets"]),
                 line("untuned equal weights", UNTUNED, untuned), line("no likelihood (B v1 dense)", none["decoder"], none["sets"])]
        lines += [line(f"best {k.replace('_weight', '')} alone", r["decoder"], r["sets"]) for k, r in alone.items() if r]
        lines += ["", "Top 20:", "", *head] + [line(str(i + 1), r["decoder"], r["sets"]) for i, r in enumerate(rows[:20])]
        lines += ["", f"Selected (rule): {best['decoder']}.", ""]
    else:
        g = json.loads((REPO_ROOT / "experiments" / args.test / "results.json").read_text())
        sel = g["selected"]["decoder"]
        vids = videos("test")
        names = ["Engine A", "B v1 as shipped", "untuned equal weights", "selected (rule)"]
        overs = [ENGINE_A, overrides(None, "engine_a"), overrides(UNTUNED), overrides(sel)]
        res_p = run(overs, vids, 0.5, args.workers)
        res_0 = run(overs, vids, None, args.workers)
        res_d = {"id": args.id, "kind": "zero_shot_test", "source": "public", "reference": "", "split": "test",
                 "title": "Zero shot phase scoring: the single test evaluation (rule w11-zero-shot)",
                 "selected_from": args.test, "selected": sel, "videos": vids,
                 "rows": [{"row": n, "p0.5": a, "no_cuts": c} for n, a, c in zip(names, res_p, res_0)]}
        lines = [f"# {args.id}", "", res_d["title"], "", f"Selected on {args.test}: {sel}.", "", "With synthetic cuts p 0.5:",
                 "", *head] + [line(n, sel if "selected" in n else (UNTUNED if "untuned" in n else None), r)
                               for n, r in zip(names, res_p)]
        lines += ["", "Without cuts:", "", *head] + [line(n, sel if "selected" in n else (UNTUNED if "untuned" in n else None), r)
                                                   for n, r in zip(names, res_0)]
        lines.append("")
    res_d.update(provenance={"git": git_state(REPO_ROOT), "date_utc": utc_now(), "command": " ".join(sys.argv)},
                 finished_utc=utc_now())
    res_d["index"] = {"cells": [lines[8] if len(lines) > 8 else ""], "verdict": "measurement"}
    (out / "results.json").write_text(json.dumps(res_d, indent=1, default=float) + "\n")
    (out / "table.md").write_text("\n".join(lines))
    reg.write_index(REPO_ROOT / "experiments")
    print("\n".join(lines[:14]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
