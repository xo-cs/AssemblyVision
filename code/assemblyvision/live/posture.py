"""Side work station: sustained awkward posture timer and rough gaze shares.

A stretch is time spent continuously in band posture.awkward_band (B1) or
worse; going back to a better band for posture.release_s ends it (the short
return is not counted as a break, so a stretch is not split by jitter). Shown:
the current stretch, the longest one and the total awkward time.

Gaze uses the batch rule (analysis/motion.gaze_class) per frame, in image
terms: front, side (left and right of the image together) and away, as shares
of the frames where the head direction could be told.
"""
from __future__ import annotations

import numpy as np

from ..analysis.motion import band_of, gaze_class
from .body import trunk


class PostureTimer:
    def __init__(self, params: dict, bands: list[float], gap_s: float):
        self.awkward = params["awkward_band"]
        self.release_s = params["release_s"]
        self.bands, self.gap_s = bands, gap_s
        self.last_t = None
        self.stretch_start: float | None = None
        self.better_since: float | None = None
        self.longest_s = 0.0
        self.total_s = 0.0
        self.stretches = 0
        self.band = None

    def update(self, t: float, kp: np.ndarray, ok: np.ndarray) -> dict:
        dt = 0.0 if self.last_t is None else t - self.last_t
        gap = not (0 < dt <= self.gap_s)
        self.last_t = t
        a = trunk(kp, ok)
        if gap and self.stretch_start is not None:          # no time across a gap: the stretch ends
            self._end(self.better_since or t)
        if a is None:
            self.band = None
            return self.state(t)
        self.band = int(band_of(a, self.bands))
        bad = self.band >= self.awkward
        if bad:
            if not gap:
                self.total_s += dt
            self.better_since = None
            if self.stretch_start is None:
                self.stretch_start = t
                self.stretches += 1
        elif self.stretch_start is not None:
            self.better_since = self.better_since or t
            if t - self.better_since >= self.release_s:
                self._end(self.better_since)
        return self.state(t)

    def _end(self, t: float) -> None:
        self.longest_s = max(self.longest_s, t - self.stretch_start)
        self.stretch_start = self.better_since = None

    def current_s(self, t: float) -> float:
        if self.stretch_start is None:
            return 0.0
        return (self.better_since or t) - self.stretch_start

    def state(self, t: float) -> dict:
        cur = self.current_s(t)
        return {"band": self.band, "current_s": round(cur, 1), "longest_s": round(max(self.longest_s, cur), 1),
                "total_s": round(self.total_s, 1), "stretches": self.stretches}


class GazeShare:
    NAMES = ("front", "side", "away")

    def __init__(self, conf_min: float, yaw_max: float, gap_s: float):
        self.conf_min, self.yaw_max, self.gap_s = conf_min, yaw_max, gap_s
        self.time = np.zeros(3)
        self.last_t = None
        self.now = None

    def update(self, t: float, kp: np.ndarray, conf: np.ndarray) -> str | None:
        dt = 0.0 if self.last_t is None else t - self.last_t
        dt = dt if 0 < dt <= self.gap_s else 0.0
        self.last_t = t
        valid = (np.nan_to_num(conf) >= self.conf_min) & np.isfinite(kp).all(1)
        c = int(gaze_class(kp[None], valid[None], self.yaw_max)[0])
        self.now = None if c < 0 else self.NAMES[{0: 0, 1: 1, 2: 1, 3: 2}[c]]
        if self.now is not None:
            self.time[self.NAMES.index(self.now)] += dt
        return self.now

    def shares(self) -> dict:
        tot = self.time.sum()
        return {n: (round(float(self.time[i] / tot), 3) if tot else None) for i, n in enumerate(self.NAMES)}
