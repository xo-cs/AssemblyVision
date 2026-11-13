"""The vlm_scores stage (Week 11): Qwen3 VL 8B zero shot phase scores of a stored run.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.vlm_scores --run RUN_DIR --gpu 1          # writes RUN_DIR/vlm.npz
    python -m assemblyvision.tools.vlm_scores --dataset ikea_asm --gpu 1 [--videos FILE]
        [--window-s 2 --frames 4 --slots 2 --out vlm.npz] [--other]

Starts its own llama-server on 127.0.0.1:--port on the GPU (default GPU1) under
.locks/gpu<N>.lock, scores every --window-s window of --frames full frames (analysis/vlm.py,
analysis/zero_shot.py; defaults from [zero_shot]) and stops the server. A QiSens run uses docs_team/phase_descriptions.md, an IKEA ASM run the
descriptions of its furniture type. Output: vlm.npz in the run folder (window start
times, probabilities [N, K], reasons, parse failures); only counts and timings are printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from ..storage.files import REPO_ROOT, run_source


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.vlm_scores")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", type=Path)
    g.add_argument("--dataset", choices=("ikea_asm",))
    p.add_argument("--videos", type=Path, help="ikea_asm: a JSON list of video ids to score (default: all runs)")
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--slots", "--workers", type=int, dest="slots", help="parallel llama-server slots and requests")
    p.add_argument("--window-s", type=float, help="seconds per request (default zero_shot.vlm_window_s)")
    p.add_argument("--frames", type=int, help="frames per request (default zero_shot.vlm_frames)")
    p.add_argument("--out", default="vlm.npz", help="file name in the run folder")
    p.add_argument("--other", action="store_true", help="add the \"other\" class (seg_a.other_class); writes the "
                                                       "_other name of --out (vlm_other.npz)")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if not args.out.endswith(".npz") or "/" in args.out:
        p.error("--out is a file name *.npz")
    if args.other:
        from ..analysis.siglip import other_file
        args.out = other_file(args.out)
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    from ..analysis import vlm as V
    from ..analysis import zero_shot as Z
    from ..config import Config
    from .visual_embed import take_lock
    cfg = Config.load()
    zs = dict(cfg["zero_shot"])
    for k, v in (("vlm_slots", args.slots), ("vlm_window_s", args.window_s), ("vlm_frames", args.frames)):
        if v is not None:
            zs[k] = v
    jobs = []
    ikea_runs = REPO_ROOT / "data/public/datasets/ikea_asm/derived/runs.json"
    if args.run is not None:
        source, root = run_source(args.run, cfg["paths"])
        if source == "qisens":
            jobs.append((root, "qisens"))
        else:
            ikea = {v: k for k, v in json.loads(ikea_runs.read_text()).items()}
            if root.name not in ikea:
                p.error("a public run must be an IKEA ASM run (derived/runs.json)")
            jobs.append((root, "ikea_asm:" + ikea[root.name].split("/")[0]))
    else:
        runs = json.loads(ikea_runs.read_text())
        want = set(json.loads(args.videos.read_text())) if args.videos else set(runs)
        for vid, rid in sorted(runs.items()):
            root = REPO_ROOT / "data/public/runs" / rid
            if vid in want and (root / "manifest.json").is_file():
                jobs.append((root, "ikea_asm:" + vid.split("/")[0]))
    todo = [j for j in jobs if args.force or not (j[0] / args.out).is_file()]
    print(f"vlm_scores: {len(jobs)} runs, {len(todo)} to score; window {zs['vlm_window_s']} s, "
          f"{zs['vlm_frames']} frames, {zs['vlm_slots']} slots -> {args.out}")
    if not todo:
        return 0
    lock = take_lock(args.gpu)
    # the server log of a QiSens run stays in its run folder (data/qisens), public ones go to logs/
    log = jobs[0][0] / "vlm_server.log" if args.run is not None and source == "qisens" else \
        REPO_ROOT / "logs/vlm_scores_server.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    server = V.start_server(args.gpu, args.port, log, ctx=V.context_size(zs), parallel=zs["vlm_slots"])
    print(f"vlm_scores: server ready in {time.monotonic() - t0:.0f} s")
    total, t_score = 0, 0.0
    try:
        for i, (root, pset) in enumerate(todo):
            phases, process, _ = Z.phase_set(pset)
            other = Z.other_class(pset) if args.other else None
            manifest = json.loads((root / "manifest.json").read_text())
            t = time.monotonic()
            rec = Z.vlm_score_video(root / manifest["ingest"]["cfr"]["path"], root / args.out, phases, process, zs,
                                    args.port, other=other)
            t_score += time.monotonic() - t
            total += rec["windows"]
            print(f"  {i + 1}/{len(todo)}: {rec['windows']} windows, {rec['unparsed']} unparsed, "
                  f"{time.monotonic() - t:.1f} s", flush=True)
    finally:
        server.terminate()
        server.wait(timeout=60)
        lock.close()
    print(f"vlm_scores: {total} windows in {time.monotonic() - t0:.1f} s (scoring {t_score:.1f} s, "
          f"{t_score / max(total, 1):.3f} s per window)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
