"""Charts and pictures of the interface, black, white and greys; the trunk
bands alone use their traffic light colours (motion.BAND_COLORS). Each
function returns an RGB array [H, W, 3] (uint8); nothing is written to disk
here. Matplotlib runs with the Agg backend.

Style (same as the page): system sans font, no top and right spines, thin
light grey gridlines, grey tick labels, black and grey fills, time axes with
round ticks (every 10 s, 30 s, 1 min and so on).
"""
from __future__ import annotations

import io

import cv2
import numpy as np

from ..analysis.motion import BAND_COLORS

INK, GREY, GRID = "#111111", "#6b6b6b", "#e5e5e5"
FONT = 12
SANS = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"]
TICK_STEPS = (5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600)


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": FONT, "font.family": "sans-serif", "font.sans-serif": SANS,
        "axes.edgecolor": GRID, "axes.linewidth": 1.0, "axes.labelcolor": GREY, "axes.labelsize": FONT - 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": GREY, "ytick.color": GREY, "xtick.labelcolor": GREY, "ytick.labelcolor": INK,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 8,
        "text.color": INK, "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": False, "legend.fontsize": FONT - 1})
    return plt


def _to_rgb(fig) -> np.ndarray:
    plt = _plt()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=144, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _clock(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def time_ticks(duration_s: float, max_ticks: int = 12) -> list[float]:
    """Round tick positions over [0, duration]: the smallest step of TICK_STEPS
    that gives at most max_ticks ticks (every 30 s for 5 min, every minute for 10 min)."""
    d = max(float(duration_s), 1.0)
    step = next((st for st in TICK_STEPS if d / st + 1 <= max_ticks), TICK_STEPS[-1])
    return [float(t) for t in np.arange(0.0, d + 1e-6, step)]


def _style(ax, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)


def timeline(steps: list[dict], duration_s: float) -> np.ndarray:
    """Process timeline: one row per phase (in order of first appearance), black
    bars where Engine A found it, time axis over the whole video."""
    plt = _plt()
    labels = []
    for s in steps:
        if s["label"] != "other" and s["label"] not in labels:
            labels.append(s["label"])
    fig, ax = plt.subplots(figsize=(11, 0.5 * max(len(labels), 1) + 0.9))
    for row, lab in enumerate(labels):
        spans = [(s["start_s"], s["end_s"] - s["start_s"]) for s in steps if s["label"] == lab]
        ax.broken_barh(spans, (row - 0.32, 0.64), facecolors=INK, edgecolors="none")
        for x, w in spans:
            if w > 0.06 * duration_s:
                ax.text(x + w / 2, row, f"{w:.0f} s", ha="center", va="center", color="white", fontsize=FONT - 2)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.set_xlim(0, max(duration_s, 1e-6))
    ticks = time_ticks(duration_s)
    ax.set_xticks(ticks, [_clock(t) for t in ticks])
    ax.set_xlabel("video time (min:s)")
    _style(ax)
    if not labels:
        ax.text(0.5, 0.5, "no steps stored", transform=ax.transAxes, ha="center", va="center")
    return _to_rgb(fig)


def bending_bars(rows: list[tuple[str, list[float | None]]], bands: list[float]) -> np.ndarray:
    """Stacked horizontal bar per step of the seconds in back bending bands B0
    to B3. rows: [(step name, [b0, b1, b2, b3] seconds or None when not reported)]."""
    plt = _plt()
    names = [f"B0 under {bands[0]:g}°", f"B1 {bands[0]:g} to {bands[1]:g}°", f"B2 {bands[1]:g} to {bands[2]:g}°",
             f"B3 {bands[2]:g}° or more"]
    fig, ax = plt.subplots(figsize=(11, 0.5 * max(len(rows), 1) + 1.3))
    for y, (_, vals) in enumerate(rows):
        if any(v is None for v in vals):
            ax.text(0, y, "  not reported (low coverage)", va="center", fontsize=FONT - 1, color=GREY)
            continue
        left = 0.0
        for b, v in enumerate(vals):
            ax.barh(y, v, left=left, height=0.6, color=BAND_COLORS[b], edgecolor="white", linewidth=1.0)
            left += v
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    longest = max((sum(v) for _, v in rows if all(x is not None for x in v)), default=1.0)
    ax.set_xlim(0, 1.05 * max(longest, 1.0))
    ax.set_xlabel("time in band (s)")
    _style(ax)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c in BAND_COLORS]
    ax.legend(handles, names, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, handlelength=1.0,
              handleheight=1.0, columnspacing=1.6, labelcolor=GREY)
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
