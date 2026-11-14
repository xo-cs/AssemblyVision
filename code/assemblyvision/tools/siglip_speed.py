"""SigLIP at 1 fps (Week 11, rule experiments/w11-siglip-speed/rule.md). CPU only, public data.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.siglip_speed

Makes siglip_f1.npz in each of the 40 IKEA ASM videos of w11-vlm-speed from its siglip.npz
(every other 2 fps sample, each covering 1 s), then scores both with the untuned "B zero
shot" weights and the chosen VLM file (vlm_w3f2s4.npz), synthetic cuts p 0.5.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from ..storage.files import REPO_ROOT, utc_now

EXP = REPO_ROOT / "experiments" / "w11-siglip-speed"
SETS = ("Kallax_Shelf_Drawer", "Lack_Coffee_Table")
VLM = "vlm_w3f2s4.npz"


def make_1fps(root) -> None:
    from ..analysis import siglip as S
    z = S.load_scores(root / "siglip.npz")
    meta = z.pop("meta")
    t = z["time_s"]
    keep = np.flatnonzero(np.isclose(np.mod(t + 1e-3, 1.0), 1e-3, atol=0.05))   # the samples at whole seconds
    S.save(root / "siglip_f1.npz", {k: v[keep] for k, v in z.items()}, {**meta, "fps": 1.0, "from": "siglip.npz"})


def main(argv: list[str] | None = None) -> int:
    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    from ..experiments import registry as reg
    from ..experiments.provenance import git_state
    from . import ikea_bench as ib
    from .zero_shot_grid import UNTUNED, overrides, run
    sel = json.loads((REPO_ROOT / "experiments/w11-vlm-speed/videos.json").read_text())
    vids = {s: sel["sets"][s] for s in SETS}
    r = ib.runs()
    for v in sum(vids.values(), []):
        make_1fps(REPO_ROOT / "data/public/runs" / r[v])
    overs = []
    for f in ("siglip.npz", "siglip_f1.npz"):
        o = overrides(UNTUNED)
        o["zero_shot"] = {"vlm_file": VLM, "siglip_file": f}
        overs.append(o)
    res = run(overs, vids, 0.5, 6)
    worst = [min(x[s]["frame_accuracy"] for s in SETS) for x in res]
    delta = round(worst[1] - worst[0], 2)
    chosen = 1.0 if delta >= -1.0 else 2.0
    lines = ["# w11-siglip-speed", "", "SigLIP 2 at 1 fps against 2 fps (rule w11-siglip-speed): 40 IKEA ASM training "
             f"videos, untuned B zero shot weights, VLM {VLM}, synthetic cuts p 0.5. Frame accuracy, mean over videos.", "",
             "| SigLIP fps | Kallax | Lack Coffee | worst |", "| --- | --- | --- | --- |"]
    for name, x, w in zip(("2 (siglip.npz)", "1 (siglip_f1.npz)"), res, worst):
        lines.append(f"| {name} | " + " | ".join(str(x[s]["frame_accuracy"]) for s in SETS) + f" | {w} |")
    lines += ["", f"Change of the worst case at 1 fps: {delta:+.2f}. Decision (at most -1.0): **{chosen:g} fps**.", ""]
    out = {"id": "w11-siglip-speed", "kind": "siglip_speed", "source": "public", "reference": "",
           "title": "SigLIP 2 at 1 fps against 2 fps (rule w11-siglip-speed)", "rule": "experiments/w11-siglip-speed/rule.md",
           "videos": vids, "rows": {"2 fps": res[0], "1 fps": res[1]}, "worst": worst, "delta": delta,
           "chosen_fps": chosen, "provenance": {"git": git_state(REPO_ROOT), "date_utc": utc_now()},
           "finished_utc": utc_now(), "index": {"cells": [f"1 fps {delta:+.2f} worst case: {chosen:g} fps"],
                                                "verdict": "measurement"}}
    EXP.mkdir(parents=True, exist_ok=True)
    (EXP / "results.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    (EXP / "table.md").write_text("\n".join(lines))
    reg.write_index(REPO_ROOT / "experiments")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
