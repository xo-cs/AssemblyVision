"""VLM speed selection (Week 11, rule experiments/w11-vlm-speed/rule.md).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.vlm_speed pick                 # the 40 videos (seed 42) and the 3 slot videos
    python -m assemblyvision.tools.vlm_speed slots --gpu 1        # step 1: slots 1, 2, 4 (speed only)
    python -m assemblyvision.tools.vlm_speed grid --gpu 1 --slots N [--only W,F]   # step 2: score the settings
    python -m assemblyvision.tools.vlm_speed eval                 # CPU: accuracy, speed, the choice

Public data only (IKEA ASM training subset). Every setting is scored by tools/vlm_scores.py
into its own file in the run folders (vlm_w<W>f<F>s<N>.npz), so the stored vlm.npz (the
reference) is never overwritten. Timings and the scoring seconds per window go to
experiments/w11-vlm-speed/timing.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from ..storage.files import REPO_ROOT, utc_now

EXP = REPO_ROOT / "experiments" / "w11-vlm-speed"
SETS = ("Kallax_Shelf_Drawer", "Lack_Coffee_Table")
WINDOWS, FRAMES, SLOTS = (2.0, 3.0, 4.0), (4, 2), (1, 2, 4)
CURRENT = (2.0, 4)


def fname(w: float, f: int, s: int) -> str:
    return f"vlm_w{w:g}f{f}s{s}.npz"


def _timing() -> dict:
    p = EXP / "timing.json"
    return json.loads(p.read_text()) if p.is_file() else {"slots": {}, "grid": {}}


def _save_timing(t: dict) -> None:
    (EXP / "timing.json").write_text(json.dumps(t, indent=1) + "\n")


def _duration_s(vids: list[str]) -> float:
    from . import ikea_bench as ib
    r = ib.runs()
    return sum(float(json.loads((REPO_ROOT / "data/public/runs" / r[v] / "manifest.json").read_text())
                     ["ingest"]["cfr"]["duration_s"]) for v in vids)


def score(vids: list[str], w: float, f: int, s: int, gpu: int, out: str) -> dict:
    """Runs tools/vlm_scores.py on vids; {"windows", "scoring_s", "s_per_window", "wall_s"}."""
    lst = EXP / f".videos_{out}.json"
    lst.write_text(json.dumps(vids))
    cmd = [sys.executable, "-m", "assemblyvision.tools.vlm_scores", "--dataset", "ikea_asm", "--videos", str(lst),
           "--gpu", str(gpu), "--slots", str(s), "--window-s", str(w), "--frames", str(f), "--out", out, "--force"]
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=REPO_ROOT / "code", capture_output=True, text=True)
    lst.unlink()
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    m = re.search(r"vlm_scores: (\d+) windows in [\d.]+ s \(scoring ([\d.]+) s, ([\d.]+) s per window\)", p.stdout)
    return {"windows": int(m.group(1)), "scoring_s": float(m.group(2)), "s_per_window": float(m.group(3)),
            "wall_s": round(time.monotonic() - t0, 1), "date_utc": utc_now()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.vlm_speed")
    p.add_argument("mode", choices=("pick", "slots", "grid", "eval"))
    p.add_argument("--gpu", type=int, default=1)
    p.add_argument("--slots", type=int, help="grid: the slots chosen in step 1")
    p.add_argument("--only", help="grid: one setting W,F")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    EXP.mkdir(parents=True, exist_ok=True)
    if args.mode == "pick":
        import numpy as np
        from .zero_shot_grid import videos
        vids = videos("train")
        rng = np.random.default_rng(42)
        pick = {s: sorted(rng.choice(vids[s], size=20, replace=False).tolist()) for s in SETS}
        slot_vids = [pick[SETS[0]][0], pick[SETS[1]][0], pick[SETS[0]][1]]
        (EXP / "videos.json").write_text(json.dumps({"seed": 42, "rule": "rule.md", "sets": pick,
                                                     "slots_videos": slot_vids}, indent=1) + "\n")
        print({s: len(v) for s, v in pick.items()}, f"{_duration_s(sum(pick.values(), [])) / 60:.1f} min of video;",
              f"slot videos {_duration_s(slot_vids):.0f} s")
        return 0
    sel = json.loads((EXP / "videos.json").read_text())
    vids = sum((sel["sets"][s] for s in SETS), [])
    if args.mode == "slots":
        t = _timing()
        for s in SLOTS:
            r = score(sel["slots_videos"], *CURRENT, s, args.gpu, f"vlm_slots{s}.npz")
            t["slots"][str(s)] = r
            _save_timing(t)
            print(f"slots {s}: {r}", flush=True)
        return 0
    if args.mode == "grid":
        if args.slots is None:
            p.error("grid needs --slots")
        todo = [(w, f) for w in WINDOWS for f in FRAMES]
        if args.only:
            w, f = args.only.split(",")
            todo = [(float(w), int(f))]
        t = _timing()
        for w, f in todo:
            r = score(vids, w, f, args.slots, args.gpu, fname(w, f, args.slots))
            t["grid"][fname(w, f, args.slots)] = {"window_s": w, "frames": f, "slots": args.slots, **r}
            _save_timing(t)
            print(f"window {w:g} s, {f} frames: {r}", flush=True)
        return 0
    return evaluate(sel, args.workers)


def evaluate(sel: dict, workers: int) -> int:
    from ..experiments import registry as reg
    from ..experiments.provenance import git_state
    from .zero_shot_grid import UNTUNED, overrides, run
    t = _timing()
    minutes = _duration_s(sum(sel["sets"].values(), [])) / 60
    files = {"vlm.npz": None, **{k: v for k, v in t["grid"].items()}}
    overs = []
    for f in files:
        o = overrides(UNTUNED)
        o["zero_shot"] = {"vlm_file": f}
        overs.append(o)
    res = run(overs, {s: sel["sets"][s] for s in SETS}, 0.5, workers)
    rows = []
    for (f, tim), r in zip(files.items(), res):
        fa = [r[s]["frame_accuracy"] for s in SETS]
        rows.append({"file": f, "setting": tim and {k: tim[k] for k in ("window_s", "frames", "slots")},
                     "sets": r, "worst": min(fa),
                     "s_per_video_min": tim and round(tim["scoring_s"] / minutes, 2),
                     "s_per_window": tim and tim["s_per_window"]})
    ref = rows[0]
    ok = [r for r in rows[1:] if r["worst"] >= ref["worst"] - 1.0]
    best = min(ok, key=lambda r: r["s_per_video_min"]) if ok else None
    cur = next((r for r in rows[1:] if (r["setting"]["window_s"], r["setting"]["frames"]) == CURRENT), None)
    res_d = {"id": "w11-vlm-speed", "kind": "vlm_speed", "source": "public", "reference": "",
             "title": "VLM speed: window and frames per request on IKEA ASM training videos (rule w11-vlm-speed)",
             "rule": "experiments/w11-vlm-speed/rule.md", "videos": sel, "video_minutes": round(minutes, 1),
             "slots": t["slots"], "rows": rows, "selected": best and best["setting"],
             "provenance": {"git": git_state(REPO_ROOT), "date_utc": utc_now(), "command": " ".join(sys.argv)},
             "finished_utc": utc_now()}
    lines = ["# w11-vlm-speed", "", res_d["title"] + ".", "",
             f"40 videos ({minutes:.1f} min), untuned equal weights, synthetic cuts p 0.5; frame accuracy, mean over "
             "videos. Speed: VLM scoring seconds per minute of video on GPU1 (server start not included).", "",
             "Step 1, slots (3 videos, window 2 s, 4 frames):", "", "| slots | windows | s per window |",
             "| --- | --- | --- |"]
    lines += [f"| {s} | {r['windows']} | {r['s_per_window']} |" for s, r in sorted(t["slots"].items())]
    lines += ["", "Step 2:", "", "| file | window s | frames | slots | Kallax | Lack Coffee | worst | s per video min |"
              " s per window |", "| --- " * 9 + "|"]
    for r in rows:
        st = r["setting"] or {"window_s": 2.0, "frames": 4, "slots": 2}
        lines.append(f"| {r['file']}{' (reference)' if r is ref else ''} | {st['window_s']:g} | {st['frames']} | "
                     f"{st['slots']} | " + " | ".join(str(r['sets'][s]['frame_accuracy']) for s in SETS)
                     + f" | {r['worst']} | {r['s_per_video_min'] or '-'} | {r['s_per_window'] or '-'} |")
    lines += ["", f"Selected (rule): {best['setting'] if best else 'none passes, the current setting stays'}."]
    if best and cur:
        lines.append(f"Speed up against the current setting at the same slots: "
                     f"{cur['s_per_video_min'] / best['s_per_video_min']:.2f}x.")
    lines.append("")
    res_d["index"] = {"cells": [f"selected {best['setting'] if best else 'none'}: worst {best['worst'] if best else '-'}"
                                f" (reference {ref['worst']}), {best['s_per_video_min'] if best else '-'} s per video min"],
                      "verdict": "measurement"}
    (EXP / "results.json").write_text(json.dumps(res_d, indent=1, default=float) + "\n")
    (EXP / "table.md").write_text("\n".join(lines))
    reg.write_index(REPO_ROOT / "experiments")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
