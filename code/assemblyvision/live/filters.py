"""One Euro filter (Casiez, Roussel, Vogel, CHI 2012) on the 17 COCO keypoints.

Each keypoint is smoothed on its own with an adaptive low pass: a low cutoff
while it is still (no jitter), a higher one while it moves fast (little lag).
Time stamps may be irregular. A keypoint below the confidence threshold is not
updated and not returned (NaN); one unseen for longer than gap_s starts again
from its next measurement.
"""
from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2 * math.pi * np.asarray(cutoff, np.float64))
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0, gap_s: float = 1.0,
                 n: int = 17):
        self.min_cutoff, self.beta, self.d_cutoff, self.gap_s = min_cutoff, beta, d_cutoff, gap_s
        self.x = np.full((n, 2), np.nan)
        self.dx = np.zeros((n, 2))
        self.t_seen = np.full(n, -np.inf)

    def __call__(self, t: float, kp: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """kp [17, 2] pixels, ok [17] bool; returns the smoothed [17, 2], NaN where not ok."""
        kp = np.asarray(kp, np.float64)
        ok = np.asarray(ok, bool) & np.isfinite(kp).all(1)
        dt = t - self.t_seen
        fresh = ok & ((dt > self.gap_s) | (dt <= 0) | ~np.isfinite(self.x).all(1))
        go = ok & ~fresh
        self.x[fresh], self.dx[fresh] = kp[fresh], 0.0
        if go.any():
            d = dt[go][:, None]
            dx = (kp[go] - self.x[go]) / d
            self.dx[go] = self.dx[go] + _alpha(self.d_cutoff, d) * (dx - self.dx[go])
            speed = np.linalg.norm(self.dx[go], axis=1, keepdims=True)
            a = _alpha(self.min_cutoff + self.beta * speed, d)
            self.x[go] = self.x[go] + a * (kp[go] - self.x[go])
        self.t_seen[ok] = t
        out = np.full_like(kp, np.nan)
        out[ok] = self.x[ok]
        return out
