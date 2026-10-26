"""Body measures for live frames, tolerant of side views where one shoulder,
hip or leg is hidden: the visible side is used when the other is missing."""
from __future__ import annotations

import numpy as np

from ..analysis.cleaning import L_HIP, L_SHO, R_HIP, R_SHO
from ..analysis.motion import trunk_angle

L_KNEE, R_KNEE, L_ANK, R_ANK = 13, 14, 15, 16
L_WRI, R_WRI = 9, 10


def _mid(kp: np.ndarray, ok: np.ndarray, a: int, b: int) -> np.ndarray | None:
    pts = [kp[i] for i in (a, b) if ok[i]]
    return np.mean(pts, axis=0) if pts else None


def mid_hip(kp: np.ndarray, ok: np.ndarray) -> np.ndarray | None:
    return _mid(kp, ok, L_HIP, R_HIP)


def trunk(kp: np.ndarray, ok: np.ndarray) -> float | None:
    """Trunk angle (deg, 0 = upright) from the visible shoulders and hips (LLD 3.7 definition)."""
    sho, hip = _mid(kp, ok, L_SHO, R_SHO), mid_hip(kp, ok)
    if sho is None or hip is None or np.allclose(sho, hip):
        return None
    return float(trunk_angle(sho, hip))


def knee_angle(kp: np.ndarray, conf: np.ndarray, ok: np.ndarray) -> float | None:
    """Angle at the knee between thigh and shank (deg, 180 = straight leg), on the
    side whose hip, knee and ankle are all seen with the highest confidence."""
    best, val = -1.0, None
    for hip, knee, ank in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        if not ok[[hip, knee, ank]].all():
            continue
        c = float(min(conf[hip], conf[knee], conf[ank]))
        u, v = kp[hip] - kp[knee], kp[ank] - kp[knee]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0 or c <= best:
            continue
        best = c
        val = float(np.degrees(np.arccos(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))))
    return val


class Hold:
    """A boolean that changes only after the new value held for hold_s (irregular time)."""

    def __init__(self, hold_s: float, value: bool = False):
        self.hold_s, self.value, self._since = hold_s, value, None
        self.changed, self.changed_at = False, None

    def update(self, t: float, raw: bool) -> bool:
        """New debounced value. On the frame where it changes, changed is True and
        changed_at is the time the new value began (not the end of the hold)."""
        self.changed = False
        if raw == self.value:
            self._since = None
        elif self._since is None:
            self._since = t
        if self._since is not None and t - self._since >= self.hold_s - 1e-9:
            self.value, self.changed, self.changed_at, self._since = raw, True, self._since, None
        return self.value
