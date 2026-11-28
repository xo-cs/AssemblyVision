"""Black and white figures of the validation report (docs_team/validation_report.md). CPU.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.report_figures f1                       # per phase F1 from the numbers in the docs
    python -m assemblyvision.tools.report_figures confusion --run-id RUN   # from <area>/validation/validation_segmentation.json

f1 writes docs_team/figures/per_phase_f1.png. confusion reads the summary file validate_seg wrote (frame counts only) and
writes the figure next to it, in the run's data area (data/qisens/validation/figures/ for a QiSens run); it prints only the
path. Copy it into docs_team/figures/ once you approve it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..storage.files import REPO_ROOT

# the flagship's per phase F1 (docs_team/validation_segmentation.md, section 3; approved summary numbers)
FLAGSHIP_F1 = {"intro / other": 23.08, "Removal": 86.36, "Disassembly": 88.60, "Cleaning": 93.33,
               "Adjustment and Inspection": 98.42, "Assembly and Installation": 99.79}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "black", "axes.linewidth": 0.8})
    return plt


def per_phase_f1(out: Path, values: dict = FLAGSHIP_F1) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=150)
    names = list(values)
    bars = ax.barh(range(len(names)), [values[n] for n in names], color="white", edgecolor="black", hatch="///", linewidth=0.8)
    for b, n in zip(bars, names):
        ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2, f"{values[n]:.1f}", va="center", fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("F1 per phase, every video frame (%)")
    ax.axvline(100, color="black", linewidth=0.5, linestyle=":")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def confusion(json_path: Path, run_id: str, out: Path) -> Path:
    import numpy as np
    plt = _plt()
    res = json.loads(json_path.read_text())
    r = next(x for x in res["runs"] if x["run"] == run_id)
    cm = np.asarray(r["confusion"], float)
    names = ["intro / other" if k == "other" else k for k in r["labels"]]
    share = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.6, 5.6), dpi=150)
    ax.imshow(share, cmap="Greys", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", fontsize=8,
                        color="white" if share[i, j] > 0.5 else "black")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (final hand labels)")
    ax.set_title("Frames per cell; grey level = share of the true row", fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.report_figures")
    p.add_argument("figure", choices=("f1", "confusion"))
    p.add_argument("--run-id", default="run_20261108_133251_1550")
    p.add_argument("--area", choices=("qisens", "public"), default="qisens")
    a = p.parse_args(argv)
    if a.figure == "f1":
        print(per_phase_f1(REPO_ROOT / "docs_team" / "figures" / "per_phase_f1.png"))
        return 0
    from ..config import Config
    from ..storage.files import source_root
    v = source_root(a.area, Config.load()["paths"]) / "validation"
    print(confusion(v / "validation_segmentation.json", a.run_id, v / "figures" / f"confusion_{a.run_id}.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
