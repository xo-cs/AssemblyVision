"""Streaming motion metrics for the Live mode (LLD 3.12): back bending, two
hand coordination, repetition and idle, updated per frame from past frames only.

Each class takes update(t, kp, conf) with the time stamp t (s), the COCO 17
keypoints kp [17, 2] in image pixels and their confidences conf [17], and
returns its current state; summary() gives the totals so far. Live keypoints
are already smoothed by the One Euro filter of LiveSession, so no Savitzky
Golay filter is used here (it needs future frames). Frames arrive at about
10 fps and may be dropped: an interval longer than clean.gap_max_s adds no
time and resets the speeds. The torso length is a running median over the
last motion.stream_torso_s. Definitions and thresholds are the batch ones,
through the functions shared with analysis/motion.py.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .cleaning import L_HIP, L_SHO, R_HIP, R_SHO
from .motion import L_WRI, R_WRI, acf_period, band_of, trunk_angle

GRID_HZ = 15.0      # repetition: the irregular frames are resampled to this rate


class _Stream:
    def __init__(self, cfg):
        self.c = cfg["clean"]
        self.m = cfg["motion"]
        self.last_t: float | None = None
        self._torso: deque = deque()

    def _ok(self, kp: np.ndarray, conf: np.ndarray) -> np.ndarray:
        """bool [17]: keypoint seen at or above clean.body_conf_min."""
        return (np.nan_to_num(np.asarray(conf, np.float64)) >= self.c["body_conf_min"]) \
            & np.isfinite(np.asarray(kp, np.float64)).all(-1)

    def _dt(self, t: float) -> float:
        """Time since the previous frame, 0 for the first frame and after a gap."""
        dt = 0.0 if self.last_t is None else t - self.last_t
        self.last_t = t
        return dt if 0 < dt <= self.c["gap_max_s"] else 0.0

    def _torso_len(self, t: float, kp: np.ndarray, ok: np.ndarray) -> float | None:
        if ok[[L_SHO, R_SHO, L_HIP, R_HIP]].all():
            v = (kp[L_SHO] + kp[R_SHO]) / 2 - (kp[L_HIP] + kp[R_HIP]) / 2
            self._torso.append((t, float(np.hypot(*v))))
        while self._torso and self._torso[0][0] < t - self.m["stream_torso_s"]:
            self._torso.popleft()
        if not self._torso:
            return None
        med = float(np.median([x for _, x in self._torso]))
        return med if med > 0 else None


class BendingStream(_Stream):
    """Trunk angle and band now, seconds per band so far (FR-MOTION-02)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.band_s = np.zeros(4)

    def update(self, t: float, kp, conf) -> dict:
        kp = np.asarray(kp, np.float64)
        ok = self._ok(kp, conf)
        dt = self._dt(t)
        angle = band = None
        if ok[[L_SHO, R_SHO, L_HIP, R_HIP]].all():
            angle = float(trunk_angle((kp[L_SHO] + kp[R_SHO]) / 2, (kp[L_HIP] + kp[R_HIP]) / 2))
            band = int(band_of(angle, self.m["angle_bands"]))
            self.band_s[band] += dt
        return {"angle_deg": angle, "band": band, "band_s": self.band_s.round(3).tolist()}

    def summary(self) -> dict:
        return {f"bend_b{i}_s": round(float(s), 3) for i, s in enumerate(self.band_s)}


class IdleStream(_Stream):
    """Motion energy now and idle time so far (FR-MOTION-10). A still run counts
    once it lasts idle_min_s, and then from its start, as in batch mode."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.prev = None
        self.run_s = 0.0
        self.counted = False
        self.idle_s = 0.0
        self.seen_s = 0.0

    def update(self, t: float, kp, conf) -> dict:
        kp = np.asarray(kp, np.float64)
        ok = self._ok(kp, conf)
        dt = self._dt(t)
        torso = self._torso_len(t, kp, ok)
        energy = None
        if dt > 0 and torso and self.prev is not None:
            both = ok & self.prev[1]
            if both.sum() >= 4:
                energy = float(np.linalg.norm(kp[both] - self.prev[0][both], axis=1).mean() / dt / torso)
        self.prev = (kp, ok)
        if energy is not None and energy < self.m["idle_energy"]:
            self.run_s += dt
            if self.counted:
                self.idle_s += dt
            elif self.run_s >= self.m["idle_min_s"] - 1e-9:
                self.idle_s += self.run_s
                self.counted = True
        else:
            self.run_s, self.counted = 0.0, False
        if energy is not None:
            self.seen_s += dt
        return {"energy": energy, "idle_now": self.counted, "idle_s": round(self.idle_s, 3)}

    def summary(self) -> dict:
        return {"idle_time_s": round(self.idle_s, 3),
                "idle_share": round(self.idle_s / self.seen_s, 4) if self.seen_s else None}


class TwoHandStream(_Stream):
    """Active hands now and the Bailey and Lang measures so far (FR-MOTION-07)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.prev = None
        self.l_s = self.r_s = self.both_s = self.either_s = 0.0
        self.n, self.mx, self.my, self.sxx, self.syy, self.sxy = 0, 0.0, 0.0, 0.0, 0.0, 0.0

    def update(self, t: float, kp, conf) -> dict:
        kp = np.asarray(kp, np.float64)
        ok = self._ok(kp, conf)
        dt = self._dt(t)
        torso = self._torso_len(t, kp, ok)
        hip_ok = ok[L_HIP] and ok[R_HIP]
        cur = None
        if torso and hip_ok and ok[L_WRI] and ok[R_WRI]:
            hip = (kp[L_HIP] + kp[R_HIP]) / 2
            cur = ((kp[L_WRI] - hip) / torso, (kp[R_WRI] - hip) / torso)
        active = None
        if cur is not None and self.prev is not None and dt > 0:
            sl = float(np.hypot(*(cur[0] - self.prev[0]))) / dt
            sr = float(np.hypot(*(cur[1] - self.prev[1]))) / dt
            al, ar = sl >= self.m["hand_active"], sr >= self.m["hand_active"]
            self.l_s += dt * al
            self.r_s += dt * ar
            self.both_s += dt * (al and ar)
            self.either_s += dt * (al or ar)
            self.n += 1                                   # Welford, for the speed correlation
            dx, dy = sl - self.mx, sr - self.my
            self.mx += dx / self.n
            self.my += dy / self.n
            self.sxx += dx * (sl - self.mx)
            self.syy += dy * (sr - self.my)
            self.sxy += dx * (sr - self.my)
            active = (bool(al), bool(ar))
        self.prev = cur
        return {"active": active, **self.summary()}

    def summary(self) -> dict:
        most = max(self.l_s, self.r_s)
        corr = self.sxy / np.sqrt(self.sxx * self.syy) if self.n >= 3 and self.sxx > 0 and self.syy > 0 else None
        return {"two_hand_simultaneous": round(self.both_s / self.either_s, 4) if self.either_s else None,
                "two_hand_use_ratio": round(min(self.l_s, self.r_s) / most, 4) if most else None,
                "two_hand_speed_corr": None if corr is None else round(float(corr), 4)}


class RepetitionStream(_Stream):
    """Period of the wrist motion over the last rep_window_s, recomputed every
    stream_rep_every_s with the batch autocorrelation (FR-MOTION-03)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.buf: deque = deque()
        self.next_t: float | None = None
        self.state = {"period_s": None, "peak": None, "periodic": False}

    def update(self, t: float, kp, conf) -> dict:
        kp = np.asarray(kp, np.float64)
        ok = self._ok(kp, conf)
        self._dt(t)
        torso = self._torso_len(t, kp, ok)
        row = np.full(4, np.nan)
        if torso and ok[L_HIP] and ok[R_HIP]:
            hip = (kp[L_HIP] + kp[R_HIP]) / 2
            for i, w in enumerate((L_WRI, R_WRI)):
                if ok[w]:
                    row[2 * i:2 * i + 2] = (kp[w] - hip) / torso
        self.buf.append((t, row))
        while self.buf and self.buf[0][0] < t - self.m["rep_window_s"]:
            self.buf.popleft()
        if self.next_t is None:
            self.next_t = t
        if t >= self.next_t:
            self.next_t = t + self.m["stream_rep_every_s"]
            self._compute()
        return dict(self.state)

    def _grid(self) -> np.ndarray | None:
        ts = np.array([b[0] for b in self.buf])
        if len(ts) < 3 or ts[-1] - ts[0] < 2 * self.m["rep_lag_max_s"]:
            return None
        g = np.arange(ts[0], ts[-1], 1.0 / GRID_HZ)
        vals = np.stack([b[1] for b in self.buf])
        out = np.full((len(g), 4), np.nan)
        for c in range(4):
            ok = np.isfinite(vals[:, c])
            if ok.sum() < 2:
                continue
            tv, v = ts[ok], vals[ok, c]
            j = np.clip(np.searchsorted(tv, g), 1, len(tv) - 1)
            close = (tv[j] - tv[j - 1] <= self.c["gap_max_s"]) & (g >= tv[0]) & (g <= tv[-1])
            out[:, c] = np.where(close, np.interp(g, tv, v), np.nan)
        return out

    def _compute(self) -> None:
        grid = self._grid()
        if grid is None:
            return
        m = self.m
        best = (None, 0.0)
        for sig in (grid[:, 0:2], grid[:, 2:4]):
            if np.isfinite(sig).all(1).mean() < m["cov_min"]:
                continue
            r = acf_period(sig, GRID_HZ, m["rep_lag_min_s"], m["rep_lag_max_s"])
            if r[0] is not None and r[1] > best[1]:
                best = r
        periodic = best[0] is not None and best[1] >= m["rep_acf_min"]
        self.state = {"period_s": round(best[0], 3) if periodic else None,
                      "peak": round(best[1], 4) if best[0] is not None else None, "periodic": periodic}

    def summary(self) -> dict:
        return dict(self.state)


class StreamingMotion:
    """The four streams together, as the Live panel uses them."""

    def __init__(self, cfg):
        self.bending, self.idle = BendingStream(cfg), IdleStream(cfg)
        self.two_hand, self.repetition = TwoHandStream(cfg), RepetitionStream(cfg)

    def update(self, t: float, kp, conf) -> dict:
        return {"bending": self.bending.update(t, kp, conf), "idle": self.idle.update(t, kp, conf),
                "two_hand": self.two_hand.update(t, kp, conf), "repetition": self.repetition.update(t, kp, conf)}

    def summary(self) -> dict:
        return {**self.bending.summary(), **self.idle.summary(), **self.two_hand.summary(),
                **{f"repetition_{k}": v for k, v in self.repetition.summary().items()}}
