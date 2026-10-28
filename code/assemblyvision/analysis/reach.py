"""Hand reach: where the subject's wrists are relative to their own body.

A floor heatmap needs a fixed camera; most of the footage is handheld. This
measure does not depend on the camera: every frame is put into a body frame
with the mid hip at the origin, the mid hip to mid shoulder axis pointing up
(+y) and the torso length (mid hip to mid shoulder of that frame) as the unit,
so x is the body's sideways axis in the picture and y its up axis.

A frame is used when both shoulders and both hips are measured (the rule of the
bending metrics, clean.body_conf_min, filled gaps do not count); each wrist
counts only when it is measured too. Coverage is the share of the frames of the
selection (one phase, or all phases) that gave at least one wrist.

Computed on demand from the stored KTS v1 files (no pipeline stage).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cleaning import L_HIP, L_SHO, R_HIP, R_SHO
from .motion import L_EAR, L_EYE, L_WRI, NOSE, R_EAR, R_EYE, R_WRI

HEAD = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)
TORSO = (L_SHO, R_SHO, L_HIP, R_HIP)


def body_axes(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Origin (mid hip), unit axes ex, ey and torso length L per frame from [T,17,2]
    image coordinates. ey points from the hip to the shoulders; ex is ey turned so
    that an upright person's ex points to the right of the picture."""
    xy = np.asarray(xy, np.float64)
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    up = (xy[:, L_SHO] + xy[:, R_SHO]) / 2 - hip
    L = np.linalg.norm(up, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ey = up / L[:, None]
    ex = np.stack([-ey[:, 1], ey[:, 0]], axis=1)
    return hip, ex, ey, L


def to_body(points: np.ndarray, hip: np.ndarray, ex: np.ndarray, ey: np.ndarray, L: np.ndarray) -> np.ndarray:
    """[T,2] image points -> [T,2] body frame points (torso lengths)."""
    d = np.asarray(points, np.float64) - hip
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([(d * ex).sum(1) / L, (d * ey).sum(1) / L], axis=1)


@dataclass
class Reach:
    """Normalized samples of a selection of frames."""
    points: np.ndarray                    # [M,2] wrist positions (both hands), torso lengths
    frames: int                           # frames in the selection
    used: int                             # frames that gave at least one wrist
    skeleton: dict = field(default_factory=dict)   # head [2], sho_half and hip_half [half width, 0]

    @property
    def coverage(self) -> float:
        return self.used / self.frames if self.frames else 0.0


def reach_samples(xy: np.ndarray, measured: np.ndarray, rows: np.ndarray | None = None) -> tuple[np.ndarray, int, dict]:
    """Wrist points, the number of frames used and the body reference sums of one
    track. xy [T,17,2], measured bool [T,17]; rows (bool [T]) limits the frames."""
    xy = np.asarray(xy, np.float64)
    measured = np.asarray(measured, bool)
    sel = np.ones(len(xy), bool) if rows is None else np.asarray(rows, bool)
    torso = measured[:, list(TORSO)].all(1) & sel
    hip, ex, ey, L = body_axes(xy)
    torso &= np.isfinite(L) & (L > 0)
    pts, used = [], np.zeros(len(xy), bool)
    for j in (L_WRI, R_WRI):
        ok = torso & measured[:, j]
        if ok.any():
            pts.append(to_body(xy[ok, j], hip[ok], ex[ok], ey[ok], L[ok]))
            used |= ok
    ref = {}
    if torso.any():
        t = torso
        # left and right swap in the picture as the worker turns, so the figure
        # uses half widths: shoulders at y = 1 and hips at y = 0 by construction
        for name, (a, b) in (("sho_half", (L_SHO, R_SHO)), ("hip_half", (L_HIP, R_HIP))):
            pa = to_body(xy[t, a], hip[t], ex[t], ey[t], L[t])
            pb = to_body(xy[t, b], hip[t], ex[t], ey[t], L[t])
            half = np.abs(pa[:, 0] - pb[:, 0]) / 2
            ref[name] = (np.array([half.sum(), 0.0]), len(half))
        head = torso & measured[:, list(HEAD)].any(1)
        if head.any():
            m = measured[head][:, list(HEAD)]
            hx = np.where(m[..., None], xy[head][:, list(HEAD)], np.nan)
            c = np.nanmean(hx, axis=1)
            p = to_body(c, hip[head], ex[head], ey[head], L[head])
            ref["head"] = (p.sum(0), len(p))
    points = np.concatenate(pts) if pts else np.zeros((0, 2))
    return points, int(used.sum()), ref


def combine(parts: list[tuple[np.ndarray, int, dict]], frames: int) -> Reach:
    """One Reach from the samples of several tracks; frames = frames of the selection."""
    pts = [p for p, _, _ in parts if len(p)]
    sums: dict[str, list] = {}
    for _, _, ref in parts:
        for k, (s, n) in ref.items():
            acc = sums.setdefault(k, [np.zeros(2), 0])
            acc[0] = acc[0] + s
            acc[1] += n
    skeleton = {k: s / n for k, (s, n) in sums.items() if n}
    return Reach(np.concatenate(pts) if pts else np.zeros((0, 2)), frames, sum(u for _, u, _ in parts), skeleton)


def run_reach(subjects: list, steps: list[dict], fps: float, labels: set[str] | None = None) -> Reach:
    """Reach of the subject tracks (motion.Subject) over the frames of the chosen
    steps: labels None means every step except other. steps: dicts with label,
    start_s and end_s (RunStore.get_segments). Without steps, the whole run."""
    chosen = [s for s in steps if s["label"] != "other" and (labels is None or s["label"] in labels)]
    if steps:
        spans = [(int(round(s["start_s"] * fps)), int(round(s["end_s"] * fps))) for s in chosen]
    else:
        spans = [(min(int(sub.cs.frame_idx[0]) for sub in subjects), 1 + max(int(sub.cs.frame_idx[-1])
                                                                             for sub in subjects))] if subjects else []
    frames = sum(b - a for a, b in spans)
    parts = []
    for sub in subjects:
        f = np.asarray(sub.cs.frame_idx, np.int64)
        rows = np.zeros(len(f), bool)
        for a, b in spans:
            rows |= (f >= a) & (f < b)
        if rows.any():
            parts.append(reach_samples(sub.cs.xy, sub.cs.measured, rows))
    return combine(parts, frames)


def density(points: np.ndarray, extent: tuple[float, float, float, float], cell: float = 0.08,
            sigma_cells: float = 1.5) -> np.ndarray:
    """2D histogram [ny, nx] (row 0 at the bottom) with light Gaussian smoothing,
    normalised to max 1 (all zero without points)."""
    x0, x1, y0, y1 = extent
    nx, ny = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    if not len(points):
        return np.zeros((ny, nx))
    h, _, _ = np.histogram2d(points[:, 1], points[:, 0], bins=[ny, nx], range=[[y0, y1], [x0, x1]])
    r = max(1, int(round(3 * sigma_cells)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2)
    k /= k.sum()
    h = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, h)
    h = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 1, h)
    return h / h.max() if h.max() > 0 else h
