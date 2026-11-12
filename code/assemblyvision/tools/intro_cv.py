"""The intro detector on public data (Week 11, rule experiments/w11-intro-detector/rule.md).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.intro_cv --id w11-intro-detector-cv                       # train, CV
    python -m assemblyvision.tools.intro_cv --id w11-intro-detector-test --split test --setting K,SUSTAIN,SHARE

CPU only. Every HA4M benchmark video plain, with a still intro and with a sway intro
(5 to 30 s, seeded per video), decoded by the v2 selected decoder of w10-engine-b-v2-cv
with seg_a.intro_detect on (each grid setting) or off. Writes experiments/<id>/.
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
from . import engine_b_cv_grid as grid
from . import phase_bench as pb
from .other_ends_cv import CHUNK, GRID_FROM, intro_length

GRID = {"intro_k": [0.5, 1.0, 1.5], "intro_sustain_s": [5.0, 10.0, 20.0], "intro_active_share": [0.5, 0.6, 0.8]}
MODES = ("plain", "still", "sway")


def settings() -> list[dict | None]:
    keys = list(GRID)
    return [None] + [dict(zip(keys, v)) for v in itertools.product(*(GRID[k] for k in keys))]


def key(st: dict | None) -> str:
    return "off" if st is None else ",".join(f"{st[k]:g}" for k in GRID)


def overrides(model_file: str, decoder: dict, st: dict | None) -> dict:
    over = grid.overrides(model_file, decoder)
    over["seg_a"].update({"intro_detect": st is not None, **(st or {})})
    return over


def score_chunk(args: tuple) -> dict:
    model_file, decoder, sts, vids = args
    bench = pb.Bench()
    cfgs = {key(st): pb.bench_cfg(overrides(model_file, decoder, st)) for st in sts}
    allv = sorted(pb.split_videos("train") + pb.split_videos("test"))
    out = {k: {m: [] for m in MODES} for k in cfgs}
    for v in vids:
        n = allv.index(v)
        for mode in MODES:
            for k, cfg in cfgs.items():
                if mode == "plain":
                    out[k][mode].append(bench.video(v, cfg))
                else:
                    out[k][mode].append(bench.video(v, cfg, intro_s=intro_length(v), seed=n, intro_mode=mode))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.intro_cv")
    p.add_argument("--id", required=True)
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--setting", help="test split: K,SUSTAIN,SHARE chosen on the train split")
    p.add_argument("--model", default="w10-engine-b-v2")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    sel = json.loads((REPO_ROOT / "experiments" / GRID_FROM / "results.json").read_text())["v2_selected"]
    dec = sel["decoder"]
    if args.split == "train":
        sts = settings()
        folds = json.loads((REPO_ROOT / grid.CV_FILE).read_text())["folds"]
        jobs = [(f"engine_b/{sel['model']}.{f['name']}.pt", dec, sts, f["test_videos"][i:i + CHUNK])
                for f in folds for i in range(0, len(f["test_videos"]), CHUNK)]
    else:
        if not args.setting:
            print("refused: the test split needs --setting (chosen on the train split)", file=sys.stderr)
            return 2
        k, su, sh = (float(x) for x in args.setting.split(","))
        sts = [None, {"intro_k": k, "intro_sustain_s": su, "intro_active_share": sh}]
        vids = pb.split_videos("test")
        jobs = [(f"engine_b/{args.model}.pt", dec, sts, vids[i:i + CHUNK]) for i in range(0, len(vids), CHUNK)]
    with ProcessPoolExecutor(args.workers) as ex:
        parts = list(ex.map(score_chunk, jobs))
    rows = []
    for st in sts:
        per = {m: [r for part in parts for r in part[key(st)][m]] for m in MODES}
        rows.append({"setting": key(st), **{m: pb.summarise(per[m]) for m in MODES},
                     "all": pb.summarise([r for m in MODES for r in per[m]])})
    off = rows[0]
    gated = [r for r in rows[1:] if r["plain"]["frame_accuracy"] >= off["plain"]["frame_accuracy"] - 0.5]
    best = max(gated, key=lambda r: (round(r["all"]["frame_accuracy"], 2), float(r["setting"].split(",")[0]))) \
        if gated else None
    from ..experiments.provenance import git_state
    res = {"id": args.id, "kind": "intro_cv", "source": "public", "reference": "", "split": args.split,
           "title": f"Intro detector on HA4M with still and sway intros ({args.split} split; rule w11-intro-detector)",
           "rule": "experiments/w11-intro-detector/rule.md", "decoder": dec, "grid": GRID, "rows": rows,
           "chosen": (best["setting"] if best else None) if args.split == "train" else None,
           "provenance": {"git": git_state(REPO_ROOT), "date_utc": utc_now(), "command": " ".join(sys.argv)},
           "finished_utc": utc_now()}
    res["index"] = {"cells": [f"off: all {off['all']['frame_accuracy']} (plain {off['plain']['frame_accuracy']}, still "
                              f"{off['still']['frame_accuracy']}, sway {off['sway']['frame_accuracy']})"] +
                    ([f"{best['setting']}: all {best['all']['frame_accuracy']} (plain {best['plain']['frame_accuracy']}, "
                      f"still {best['still']['frame_accuracy']}, sway {best['sway']['frame_accuracy']})"] if best else
                     ["no setting passes the plain gate"]), "verdict": "measurement"}
    d = REPO_ROOT / "experiments" / args.id
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    lines = [f"# {args.id}", "", res["title"], "", "Setting = intro_k, intro_sustain_s, intro_active_share. Gate: plain "
             "within 0.5 of off.", "", "| setting | all | plain | still | sway | gate |", "| --- | --- | --- | --- | --- | --- |"]
    for r in sorted(rows, key=lambda r: -r["all"]["frame_accuracy"]):
        g = "ref" if r is off else ("pass" if r in gated else "fail")
        lines.append(f"| {r['setting']} | {r['all']['frame_accuracy']} | {r['plain']['frame_accuracy']} | "
                     f"{r['still']['frame_accuracy']} | {r['sway']['frame_accuracy']} | {g} |")
    if args.split == "train":
        lines += ["", f"Chosen (rule): {best['setting'] if best else 'none, the detector stays off'}.", ""]
    else:
        lines += ["", "Test split, scored once with the setting chosen on the train split; nothing is chosen here "
                  "(gate: plain within 0.5 of off, for information).", ""]
    (d / "table.md").write_text("\n".join(lines))
    reg.write_index(REPO_ROOT / "experiments")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
