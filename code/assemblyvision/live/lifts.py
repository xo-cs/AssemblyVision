"""LiftDetector for the floor pick station (side view).

A lift starts when the trunk angle reaches lift.trunk_deg or the knee angle
(hip, knee, ankle; 180 deg = straight) drops to lift.knee_bend_deg, each held
lift.hold_s, and ends when the trunk is below trunk_deg and the knee above
lift.knee_straight_deg again for hold_s. The knee condition is there because a
good squat keeps the back fairly upright and would be missed by the trunk
angle alone. Dips shorter than lift.min_lift_s are not lifts.

At the lowest point of the lift (the frame with the mid hip lowest in the
image) the knee angle decides the type: below lift.squat_knee_max_deg it is a
squat, otherwise a stoop. Per lift: peak trunk angle, knee angle at the lowest
point, seconds in bands B2 and B3 (motion.angle_bands) and duration.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from ..analysis.motion import band_of
from .body import Hold, knee_angle, mid_hip, trunk


@dataclass
class Lift:
    n: int
    kind: str                 # stoop or squat
    start_s: float
    end_s: float
    peak_trunk_deg: float
    knee_deg: float | None    # at the lowest hip point
    b2_s: float
    b3_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


class LiftDetector:
    def __init__(self, params: dict, bands: list[float], gap_s: float):
        self.p = params
        self.bands = bands
        self.gap_s = gap_s
        self.trunk_hold = Hold(params["hold_s"])
        self.knee_hold = Hold(params["hold_s"])
        self.end_hold = Hold(params["hold_s"])
        self.lifts: list[Lift] = []
        self.cur: dict | None = None
        self.last_t: float | None = None
        self.now = {"trunk_deg": None, "knee_deg": None, "in_lift": False}
        self._hist: deque = deque(maxlen=64)

    def update(self, t: float, kp: np.ndarray, conf: np.ndarray, ok: np.ndarray) -> Lift | None:
        """One frame; returns the lift that ended on this frame, if any."""
        dt = 0.0 if self.last_t is None else t - self.last_t
        dt = dt if 0 < dt <= self.gap_s else 0.0
        self.last_t = t
        a, k = trunk(kp, ok), knee_angle(kp, conf, ok)
        hip = mid_hip(kp, ok)
        self.now = {"trunk_deg": a, "knee_deg": k, "in_lift": self.cur is not None}
        p = self.p
        if self.cur is None:
            self._hist.append((t, a, dt))
            down_t = self.trunk_hold.update(t, a is not None and a >= p["trunk_deg"])
            down_k = self.knee_hold.update(t, k is not None and k <= p["knee_bend_deg"])
            if down_t or down_k:
                hs = [h.changed_at for h in (self.trunk_hold, self.knee_hold) if h.value and h.changed_at is not None]
                start = min(hs) if hs else t - p["hold_s"]
                self.cur = {"start": start, "peak": a or 0.0, "low_y": -np.inf, "knee": k, "b2": 0.0, "b3": 0.0}
                for tt, aa, dd in self._hist:          # the frames of the hold belong to the lift
                    if tt > start and aa is not None:
                        b = int(band_of(aa, self.bands))
                        if b >= 2:
                            self.cur["b2" if b == 2 else "b3"] += dd
                        self.cur["peak"] = max(self.cur["peak"], aa)
                self._hist.clear()
                self.end_hold = Hold(p["hold_s"])
                self.now["in_lift"] = True
            return None
        c = self.cur
        if a is not None:
            c["peak"] = max(c["peak"], a)
            b = int(band_of(a, self.bands))
            if b == 2:
                c["b2"] += dt
            elif b == 3:
                c["b3"] += dt
        if hip is not None and hip[1] > c["low_y"]:
            c["low_y"] = float(hip[1])
            if k is not None:
                c["knee"] = k
        elif c["knee"] is None and k is not None:
            c["knee"] = k
        up = (a is None or a < p["trunk_deg"]) and (k is None or k > p["knee_straight_deg"]) \
            and (a is not None or k is not None)
        if self.end_hold.update(t, up):
            return self._finish(self.end_hold.changed_at)
        return None

    def _finish(self, end: float) -> Lift | None:
        c, self.cur = self.cur, None
        self.trunk_hold, self.knee_hold = Hold(self.p["hold_s"]), Hold(self.p["hold_s"])
        self.now["in_lift"] = False
        if end - c["start"] < self.p["min_lift_s"]:
            return None
        kind = "squat" if c["knee"] is not None and c["knee"] < self.p["squat_knee_max_deg"] else "stoop"
        lift = Lift(len(self.lifts) + 1, kind, round(c["start"], 3), round(end, 3), round(c["peak"], 1),
                    None if c["knee"] is None else round(c["knee"], 1), round(c["b2"], 2), round(c["b3"], 2))
        self.lifts.append(lift)
        return lift

    def abort(self, t: float) -> Lift | None:
        """Subject or camera lost: a lift in progress is ended at t."""
        return self._finish(t) if self.cur is not None else None

    def comparison(self) -> dict | None:
        """Means per type once there is at least one stoop and one squat."""
        out = {}
        for kind in ("stoop", "squat"):
            ls = [x for x in self.lifts if x.kind == kind]
            if not ls:
                return None
            knees = [x.knee_deg for x in ls if x.knee_deg is not None]
            out[kind] = {"n": len(ls), "peak_trunk_deg": round(float(np.mean([x.peak_trunk_deg for x in ls])), 1),
                         "knee_deg": round(float(np.mean(knees)), 1) if knees else None,
                         "b2_b3_s": round(float(np.mean([x.b2_s + x.b3_s for x in ls])), 2),
                         "duration_s": round(float(np.mean([x.duration_s for x in ls])), 2)}
        return out

    def rows(self) -> list[dict]:
        return [asdict(x) | {"duration_s": round(x.duration_s, 2)} for x in self.lifts]
