"""The stored eval of several runs, and its min, median and max (Week 11; CPU only).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.run_evals --source qisens --runs RUN_ID [RUN_ID ...]

Reads only <run>/eval.json (written by the eval stage) and prints the summary metrics:
frame accuracy, edit, F1@10/25/50, boundary MAE, and min, median and max over the runs.
No step, boundary, label or time of the video is printed, so the output is a summary
fit for an approved QiSens table.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys

from ..config import Config
from ..storage.files import RUN_ID_RE, RunPaths, SOURCES

KEYS = ("frame_accuracy", "edit", "f1_10", "f1_25", "f1_50", "boundary_mae_s")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.run_evals")
    p.add_argument("--source", choices=SOURCES, required=True)
    p.add_argument("--runs", nargs="+", required=True)
    args = p.parse_args(argv)
    cfg = Config.load()
    rows = []
    for rid in args.runs:
        if not RUN_ID_RE.fullmatch(rid):
            p.error(f"{rid!r} is not a run ID")
        path = RunPaths.for_source(args.source, rid, cfg["paths"]).eval_json
        try:
            m = json.loads(path.read_text())["metrics"]
        except (OSError, ValueError, KeyError) as e:
            print(f"{rid}: no eval ({type(e).__name__})", file=sys.stderr)
            return 1
        rows.append((rid, [m.get(k) for k in KEYS]))
    head = "run                         frame acc  edit    F1@10   F1@25   F1@50   MAE s"
    print(head)

    def line(name, xs):
        return f"{name:<27} " + "  ".join("   -  " if x is None else f"{x:6.2f}" for x in xs)
    for rid, xs in rows:
        print(line(rid, xs))
    for name, fn in (("min", min), ("median", statistics.median), ("max", max)):
        print(line(name, [None if any(r[1][i] is None for r in rows) else fn(r[1][i] for r in rows)
                          for i in range(len(KEYS))]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
