"""The vlm_scores stage (Week 11): Qwen3 VL 8B zero shot phase scores of a stored run.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.vlm_scores --run RUN_DIR --gpu 1          # writes RUN_DIR/vlm.npz
    python -m assemblyvision.tools.vlm_scores --dataset ikea_asm --gpu 1 [--videos FILE]

Starts its own llama-server on 127.0.0.1:--port on the GPU (default GPU1) under
.locks/gpu<N>.lock, scores every 2 s window of 4 full frames (analysis/vlm.py) and stops
the server. A QiSens run uses docs_team/phase_descriptions.md, an IKEA ASM run the
descriptions of its furniture type. Output: vlm.npz in the run folder (window start
times, probabilities [N, K], reasons, parse failures); only counts and timings are printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ..storage.files import REPO_ROOT, run_source, utc_now

WINDOW_S, N_FRAMES, PX = 2.0, 4, 448
PROCESS = {"qisens": "a video of a worker maintaining a railway vehicle (bogie maintenance) in a workshop",
           "ikea": "a top view video of a person assembling an IKEA {name}"}
IKEA_NAMES = {"Kallax_Shelf_Drawer": "Kallax shelf drawer", "Lack_TV_Bench": "Lack TV bench",
              "Lack_Coffee_Table": "Lack coffee table", "Lack_Side_Table": "Lack side table"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.vlm_scores")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path)
    g.add_argument("--dataset", choices=("ikea_asm",))
    p.add_argument("--videos", type=Path, help="ikea_asm: a JSON list of video ids to score (default: all runs)")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--workers", type=int, default=2, help="parallel requests (llama-server slots)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    from ..analysis import phase_text as pt
    from ..analysis import vlm as V
    from ..config import Config
    from .visual_embed import take_lock
    cfg = Config.load()
    jobs = []
    if args.run is not None:
        source, root = run_source(args.run, cfg["paths"])
        if source == "qisens":
            jobs.append((root, pt.phases("qisens"), PROCESS["qisens"]))
        else:
            ikea = {v: k for k, v in json.loads((REPO_ROOT / "data/public/datasets/ikea_asm/derived/runs.json")
                                                .read_text()).items()}
            if root.name not in ikea:
                p.error("a public run must be an IKEA ASM run (derived/runs.json)")
            f = ikea[root.name].split("/")[0]
            jobs.append((root, pt.phases("ikea_asm", f), PROCESS["ikea"].format(name=IKEA_NAMES[f])))
    else:
        runs = json.loads((REPO_ROOT / "data/public/datasets/ikea_asm/derived/runs.json").read_text())
        want = set(json.loads(args.videos.read_text())) if args.videos else set(runs)
        for vid, rid in sorted(runs.items()):
            root = REPO_ROOT / "data/public/runs" / rid
            f = vid.split("/")[0]
            if vid in want and (root / "manifest.json").is_file():
                jobs.append((root, pt.phases("ikea_asm", f), PROCESS["ikea"].format(name=IKEA_NAMES[f])))
    todo = [j for j in jobs if args.force or not (j[0] / "vlm.npz").is_file()]
    print(f"vlm_scores: {len(jobs)} runs, {len(todo)} to score")
    if not todo:
        return 0
    lock = take_lock(args.gpu)
    # the server log of a QiSens run stays in its run folder (data/qisens), public ones go to logs/
    log = jobs[0][0] / "vlm_server.log" if args.run is not None and source == "qisens" else \
        REPO_ROOT / "logs/vlm_scores_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    server = V.start_server(args.gpu, args.port, log, parallel=args.workers)
    print(f"vlm_scores: server ready in {time.monotonic() - t0:.0f} s")
    total = 0
    try:
        for i, (root, phases, process) in enumerate(todo):
            names = [ph["name"] for ph in phases]
            text = V.prompt(phases, process)
            manifest = json.loads((root / "manifest.json").read_text())
            t = time.monotonic()
            wins = list(V.windows(root / manifest["ingest"]["cfr"]["path"], WINDOW_S, N_FRAMES, 1280))

            def one(w):
                try:
                    return V.parse(V.ask(args.port, text, w[1], PX), names)
                except Exception as e:                 # a failed request counts as unparsed
                    return np.full(len(names), 1.0 / len(names), np.float32), f"request failed: {e}"[:200], False
            with ThreadPoolExecutor(args.workers) as ex:
                res = list(ex.map(one, wins))
            probs = np.stack([r[0] for r in res]) if res else np.zeros((0, len(names)), np.float32)
            tmp = root / "vlm.part.npz"
            np.savez_compressed(tmp, time_s=np.array([w[0] for w in wins], np.float32), probs=probs,
                                reasons=np.array([r[1] for r in res]), parsed=np.array([r[2] for r in res]),
                                meta=json.dumps({"model": V.MODEL, "mmproj": V.MMPROJ, "llama_cpp": "b12005",
                                                 "window_s": WINDOW_S, "n_frames": N_FRAMES, "px": PX,
                                                 "phases": names, "prompt": text, "date_utc": utc_now()}))
            tmp.replace(root / "vlm.npz")
            total += len(wins)
            print(f"  {i + 1}/{len(todo)}: {len(wins)} windows, {sum(not r[2] for r in res)} unparsed, "
                  f"{time.monotonic() - t:.1f} s", flush=True)
    finally:
        server.terminate()
        server.wait(timeout=60)
        lock.close()
    print(f"vlm_scores: {total} windows in {time.monotonic() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
