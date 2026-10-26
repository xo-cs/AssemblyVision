"""Charts and pictures of the interface, black, white and greys only, sized for
a projector. Each function returns an RGB array [H, W, 3] (uint8); nothing is
written to disk here. Matplotlib runs with the Agg backend."""
from __future__ import annotations

import io

import cv2
import numpy as np

BAND_GREYS = ("#ffffff", "#c8c8c8", "#6e6e6e", "#000000")     # B0 to B3
FONT = 13


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": FONT, "font.family": "DejaVu Sans", "axes.edgecolor": "black",
                         "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                         "text.color": "black", "figure.facecolor": "white", "axes.facecolor": "white"})
    return plt


def _to_rgb(fig) -> np.ndarray:
    plt = _plt()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _clock(s: float) -> str:
    return f"{int(s // 60)}:{int(round(s % 60)):02d}"


def timeline(steps: list[dict], duration_s: float) -> np.ndarray:
    """Process timeline: one row per phase (in order of first appearance), black
    bars where Engine A found it, time axis over the whole video."""
    plt = _plt()
    labels = []
    for s in steps:
        if s["label"] != "other" and s["label"] not in labels:
            labels.append(s["label"])
    fig, ax = plt.subplots(figsize=(13, 0.62 * max(len(labels), 1) + 1.3))
    for row, lab in enumerate(labels):
        spans = [(s["start_s"], s["end_s"] - s["start_s"]) for s in steps if s["label"] == lab]
        ax.broken_barh(spans, (row - 0.36, 0.72), facecolors="black", edgecolors="black")
        for x, w in spans:
            if w > 0.06 * duration_s:
                ax.text(x + w / 2, row, f"{w:.0f} s", ha="center", va="center", color="white", fontsize=FONT - 1)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(0, max(duration_s, 1e-6))
    ticks = np.linspace(0, duration_s, 9)
    ax.set_xticks(ticks, [_clock(t) for t in ticks])
    ax.set_xlabel("video time (min:s)")
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if not labels:
        ax.text(0.5, 0.5, "no steps stored", transform=ax.transAxes, ha="center", va="center")
    return _to_rgb(fig)


def bending_bars(rows: list[tuple[str, list[float | None]]], bands: list[float]) -> np.ndarray:
    """Stacked horizontal bar per step of the seconds in back bending bands B0
    to B3. rows: [(step name, [b0, b1, b2, b3] seconds or None when not reported)]."""
    plt = _plt()
    names = [f"B0 < {bands[0]:g}°", f"B1 {bands[0]:g} to {bands[1]:g}°", f"B2 {bands[1]:g} to {bands[2]:g}°",
             f"B3 ≥ {bands[2]:g}°"]
    fig, ax = plt.subplots(figsize=(13, 0.62 * max(len(rows), 1) + 1.6))
    for y, (_, vals) in enumerate(rows):
        if any(v is None for v in vals):
            ax.text(0, y, "  not reported (low coverage)", va="center", fontsize=FONT - 1)
            continue
        left = 0.0
        for b, v in enumerate(vals):
            ax.barh(y, v, left=left, height=0.62, color=BAND_GREYS[b], edgecolor="black", linewidth=0.8)
            left += v
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    longest = max((sum(v) for _, v in rows if all(x is not None for x in v)), default=1.0)
    ax.set_xlim(0, 1.05 * max(longest, 1.0))
    ax.set_xlabel("time in band (s)")
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="black") for c in BAND_GREYS]
    ax.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False)
    return _to_rgb(fig)


def mark_subject(png_path, box: list[float] | None, label: str = "S") -> np.ndarray:
    """The reference frame (with its numbered boxes) and the subject box drawn
    thick in white over black, with label S."""
    img = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if img is None:
        return np.full((240, 320, 3), 255, np.uint8)
    if box is not None:
        h = img.shape[0]
        t = max(3, round(h / 160))
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), t + 4)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), t)
        scale = max(0.8, h / 600)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
        ty = max(y1, th + 8)
        cv2.rectangle(img, (x1, ty - th - 8), (x1 + tw + 10, ty), (0, 0, 0), -1)
        cv2.putText(img, label, (x1 + 5, ty - 5), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
