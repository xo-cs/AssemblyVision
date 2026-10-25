"""SignalCleaner (LLD 3.6): the same keypoint preparation for both engines and the metrics.

1. Keypoints below clean.body_conf_min become NaN.
2. Gaps up to clean.gap_max_s are interpolated linearly; they are valid but
   never measured (NFR-REL-06).
3. A Savitzky Golay filter (clean.sg_window_s, order clean.sg_order) smooths
   every valid run. SciPy is not installed, so the filter is the least squares
   projection written in NumPy; at the ends of a run it fits the polynomial to
   the first or last window, like scipy.signal.savgol_filter(mode="interp").
4. The median torso length (mid shoulder to mid hip, measured frames only)
   normalises all features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..types import KeypointSeries

L_SHO, R_SHO, L_HIP, R_HIP = 5, 6, 11, 12


@dataclass
class CleanSeries:
    xy: np.ndarray          # float32 [T,17,2], NaN where not valid
    measured: np.ndarray    # bool [T,17], seen by the model at or above the threshold
    valid: np.ndarray       # bool [T,17], measured or interpolated
    torso: float | None     # median torso length (px), None if never measured
    fps: float
    frame_idx: np.ndarray   # int32 [T]


def savgol_coeffs(window: int, order: int) -> np.ndarray:
    """Projection matrix P [window, window]: smoothed = P @ y for one window.
    The middle row is the usual Savitzky Golay kernel."""
    half = window // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    A = np.vander(x, order + 1, increasing=True)
    return A @ np.linalg.pinv(A)


def savgol(y: np.ndarray, window: int, order: int) -> np.ndarray:
    """Savitzky Golay smoothing of a 1 D array without NaN (mode 'interp')."""
    n = len(y)
    if window % 2 == 0:
        window -= 1
    window = min(window, n if n % 2 else n - 1)
    if window <= order or window < 3:
        return y.astype(np.float64)
    P = savgol_coeffs(window, order)
    half = window // 2
    out = np.convolve(y, P[half][::-1], mode="same").astype(np.float64)
    out[:half] = P[:half] @ y[:window]
    out[n - half:] = P[half + 1:] @ y[n - window:]
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of the True runs."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


class SignalCleaner:
    def __init__(self, cfg):
        self.c = cfg["clean"]

    def clean(self, series: KeypointSeries) -> CleanSeries:
        fps = float(series.fps)
        xy = np.array(series.body_xy, np.float64)
        conf = np.nan_to_num(np.asarray(series.body_conf, np.float64), nan=0.0)
        measured = (np.asarray(series.present, bool)[:, None] & (conf >= self.c["body_conf_min"])
                    & np.isfinite(xy).all(-1))
        xy[~measured] = np.nan
        valid = measured.copy()
        gap_max = int(round(self.c["gap_max_s"] * fps))
        window = int(round(self.c["sg_window_s"] * fps)) | 1
        T = len(xy)
        for j in range(xy.shape[1]):
            m = measured[:, j]
            idx = np.flatnonzero(m)
            if len(idx) >= 2:
                for a, b in zip(idx[:-1], idx[1:]):
                    if 1 < b - a <= gap_max + 1:
                        w = (np.arange(a + 1, b) - a) / (b - a)
                        xy[a + 1:b, j] = xy[a, j] + w[:, None] * (xy[b, j] - xy[a, j])
                        valid[a + 1:b, j] = True
            if self.c["sg_window_s"] > 0:
                for a, b in _runs(valid[:, j]):
                    for c in range(2):
                        xy[a:b, j, c] = savgol(xy[a:b, j, c], window, self.c["sg_order"])
        both = measured[:, [L_SHO, R_SHO, L_HIP, R_HIP]].all(1)
        torso = None
        if both.any():
            raw = np.asarray(series.body_xy, np.float64)[both]
            sho = (raw[:, L_SHO] + raw[:, R_SHO]) / 2
            hip = (raw[:, L_HIP] + raw[:, R_HIP]) / 2
            t = float(np.median(np.linalg.norm(sho - hip, axis=1)))
            torso = t if t > 0 else None
        return CleanSeries(xy.astype(np.float32), measured, valid, torso, fps,
                           np.asarray(series.frame_idx, np.int32)[:T])


def features(cs: CleanSeries, min_points: int = 4) -> dict[str, np.ndarray]:
    """Per frame motion energy (mean keypoint speed) and mid hip speed, both in
    torso lengths per second; NaN where too few points are valid or the torso is unknown."""
    T = len(cs.xy)
    energy = np.full(T, np.nan, np.float32)
    hip = np.full(T, np.nan, np.float32)
    if cs.torso is None or T < 2:
        return {"energy": energy, "hip_speed": hip}
    scale = cs.fps / cs.torso
    d = np.linalg.norm(np.diff(cs.xy.astype(np.float64), axis=0), axis=-1)       # [T-1,17]
    ok = cs.valid[1:] & cs.valid[:-1]
    n = ok.sum(1)
    s = np.where(ok, d, 0.0).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        e = np.where(n >= min_points, s / np.maximum(n, 1) * scale, np.nan)
    energy[1:] = e
    mid = (cs.xy[:, L_HIP] + cs.xy[:, R_HIP]) / 2
    hv = cs.valid[:, L_HIP] & cs.valid[:, R_HIP]
    dh = np.linalg.norm(np.diff(mid.astype(np.float64), axis=0), axis=-1) * scale
    hip[1:] = np.where(hv[1:] & hv[:-1], dh, np.nan)
    return {"energy": energy, "hip_speed": hip}
