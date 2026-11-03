"""Activity change across a boundary candidate (Engine A evidence, seg_a.activity_change).

Per frame activity of the subject (NaN without subject or where the keypoints
are not valid), from the same helpers as the motion metrics (analysis/motion.py):
  wrist      mean of the left and right wrist speed relative to the mid hip (torso lengths/s)
  trunk      trunk angle against the vertical (deg), shoulders and hips measured
  hand_h     mean wrist height above the mid hip (torso lengths, up is positive)
  both       1 if both wrists move at or above motion.hand_active, else 0
  idle       1 if the motion energy is below motion.idle_energy, else 0

Window features over a stretch of frames (FEATURES): the mean and spread (std)
of the wrist speed, the mean trunk angle, the mean hand height, the share of
both hands moving and the idle share. A feature is NaN if fewer than
MIN_COVERAGE of the window frames carry it.

Normalisation over the whole video: the features of windows of the same
length every HOP_S seconds across the active span give, per feature, the
median and a robust spread (1.4826 * MAD, else the std; a feature that never
varies is left out). activity_change(c) compares the window that ends at the
candidate with the one that starts there: the mean over features known in both
of |z_before - z_after|, mapped to [0, 1) by 1 - exp(-d). 0 when no feature is known.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ..cleaning import L_HIP, R_HIP, CleanSeries
from ..motion import L_WRI, R_WRI, subject_trunk, wrist_speeds

FEATURES = ("wrist_mean", "wrist_spread", "trunk_mean", "hand_height", "both_share", "idle_share")
CHANNELS = ("wrist", "trunk", "hand_h", "both", "idle")
MIN_COVERAGE = 0.25      # share of window frames a feature needs
HOP_S = 1.0              # step of the normalisation windows (s)


def empty_activity(n: int) -> dict[str, np.ndarray]:
    return {k: np.full(n, np.nan, np.float32) for k in CHANNELS}


def frame_activity(cs: CleanSeries, energy: np.ndarray, hand_active: float,
                   idle_energy: float) -> dict[str, np.ndarray]:
    """Per row activity channels of one cleaned subject track (see the module docstring)."""
    out = empty_activity(len(cs.xy))
    e = np.asarray(energy, np.float64)
    fin = np.isfinite(e)
    out["idle"][fin] = (e[fin] < idle_energy).astype(np.float32)
    out["trunk"][:] = subject_trunk(cs)[1]
    if cs.torso is None or len(cs.xy) < 2:
        return out
    wl, wr = wrist_speeds(cs.xy, cs.valid, cs.torso, cs.fps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # frames with both wrists unknown
        out["wrist"][:] = np.nanmean(np.stack([wl, wr]), axis=0)
    both_ok = np.isfinite(wl) & np.isfinite(wr)
    out["both"][both_ok] = ((wl[both_ok] >= hand_active) & (wr[both_ok] >= hand_active)).astype(np.float32)
    xy = cs.xy.astype(np.float64)
    hip_y = (xy[:, L_HIP, 1] + xy[:, R_HIP, 1]) / 2
    ok = cs.valid[:, [L_WRI, R_WRI, L_HIP, R_HIP]].all(1)
    wri_y = (xy[:, L_WRI, 1] + xy[:, R_WRI, 1]) / 2
    out["hand_h"][ok] = ((hip_y - wri_y) / cs.torso)[ok]          # image y grows downwards
    return out


def window_features(act: dict[str, np.ndarray], lo: int, hi: int) -> np.ndarray:
    """FEATURES of frames [lo, hi), NaN where a channel covers less than MIN_COVERAGE."""
    lo, hi = max(lo, 0), max(hi, lo)
    n = hi - lo
    out = np.full(len(FEATURES), np.nan)
    if n <= 0:
        return out

    def vals(ch):
        v = np.asarray(act[ch][lo:hi], np.float64)
        v = v[np.isfinite(v)]
        return v if len(v) >= max(2, MIN_COVERAGE * n) else None
    w = vals("wrist")
    if w is not None:
        out[0], out[1] = w.mean(), w.std()
    for i, ch in ((2, "trunk"), (3, "hand_h"), (4, "both"), (5, "idle")):
        v = vals(ch)
        if v is not None:
            out[i] = v.mean()
    return out


@dataclass
class Normaliser:
    center: np.ndarray
    scale: np.ndarray        # inf for a feature that never varies (z = 0)

    def z(self, f: np.ndarray) -> np.ndarray:
        return (f - self.center) / self.scale


def video_stats(act: dict[str, np.ndarray], a: int, b: int, win: int, hop: int) -> Normaliser:
    """Median and robust spread of the window features over the active span [a, b)."""
    starts = range(a, max(a + 1, b - win + 1), max(1, hop))
    F = np.array([window_features(act, s, s + win) for s in starts]).reshape(-1, len(FEATURES))
    center, scale = np.zeros(len(FEATURES)), np.full(len(FEATURES), np.inf)
    for i in range(len(FEATURES)):
        v = F[:, i][np.isfinite(F[:, i])]
        if len(v) == 0:
            continue
        center[i] = np.median(v)
        s = 1.4826 * np.median(np.abs(v - center[i]))
        if not s > 1e-9:
            s = v.std()
        if s > 1e-9:
            scale[i] = s
    return Normaliser(center, scale)


def activity_change(act: dict[str, np.ndarray], start: int, end: int, a: int, b: int, win: int,
                    norm: Normaliser) -> float:
    """Change between [start - win, start) and [end, end + win), both clipped to [a, b)."""
    zb = norm.z(window_features(act, max(a, start - win), start))
    za = norm.z(window_features(act, end, min(b, end + win)))
    ok = np.isfinite(zb) & np.isfinite(za) & np.isfinite(norm.scale)
    if not ok.any():
        return 0.0
    d = float(np.abs(zb[ok] - za[ok]).mean())
    return 1.0 - float(np.exp(-d))
