"""MotionAnalyzer (LLD 3.7): per step motion metrics of the subject worker.

Every record carries value, unit, coverage and flags (HLD 3.3.7). Coverage is
the share of the step's frames in which the keypoints a metric needs were
measured; frames filled by the SignalCleaner are used but never counted
(NFR-REL-06). Below motion.cov_min the value is NULL with the flag
low_coverage and AV-MOT-001 is logged; nothing is estimated from missing data.

Deviation from LLD 3.7: there analyse() works on the subject series of one
shot. Engine A phases cross many cuts, so every step is split into pieces
(step, shot and the subject track of that shot) and each metric combines its
pieces. Frames of a step without a subject (title cards, close ups) count as
missing. Distances in px from different shots (zoom, view) are not comparable,
so hand travel is also given in torso lengths (unit "torso", not in LLD 4.6).

Two hand coordination (FR-MOTION-07) uses the accelerometry measures of
Bailey RR, Lang CE, "Upper-limb activity in adults: referent values using
accelerometry", J Rehabil Res Dev 50(9), 2013 (PMID 24458962), and Bailey RR,
Klaesner JW, Lang CE, "Quantifying real-world upper-limb activity in
nondisabled adults and adults with chronic stroke", Neurorehabil Neural Repair
29(10), 2015 (PMID 25896988), with wrist keypoints in place of wrist sensors.
A hand is active while its wrist moves relative to the mid hip at least
motion.hand_active torso lengths/s, on frames where both wrists are valid:
  two_hand_simultaneous  time both hands active / time at least one is active
  two_hand_use_ratio     active time of the less active hand / of the more
                         active hand (their use ratio; hand dominance is
                         unknown, so min over max, 0 to 1)
  two_hand_speed_corr    Pearson correlation of the two wrist speeds (our own
                         addition, for timing coordination)
"""
from __future__ import annotations

import csv
import io
import math
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import AvError
from ..kts import read_kts
from ..storage.files import RunContext, write_atomic
from ..types import KeypointSeries, MetricRecord, Shot, Step
from .cleaning import L_HIP, L_SHO, R_HIP, R_SHO, CleanSeries, SignalCleaner, _runs, features

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_WRI, R_WRI = 9, 10
OTHER = "other"
BANDS = ("b0", "b1", "b2", "b3")
GAZE = ("front", "left", "right", "away")
_STRUCTURAL = ("handheld", "too_short")   # NULL for a reason other than coverage


# shared with the streaming version (motion_stream.py) -----------------------------

def trunk_angle(sho, hip) -> np.ndarray:
    """Angle of v = mid shoulder - mid hip against the upward image vertical (deg),
    atan2(|v.x|, -v.y) as in LLD 3.7; inputs [..., 2] in image coordinates."""
    v = np.asarray(sho, np.float64) - np.asarray(hip, np.float64)
    return np.degrees(np.arctan2(np.abs(v[..., 0]), -v[..., 1]))


# traffic light colours of the trunk bands B0 to B3, the same on every chart,
# table and live tile: green, yellow, orange, red
BAND_COLORS = ("#22C55E", "#EAB308", "#F97316", "#DC2626")


def band_of(angle, edges) -> np.ndarray:
    """0 for B0 (< edges[0]), 1 for B1, 2 for B2, 3 for B3 (>= edges[2])."""
    return np.searchsorted(np.asarray(edges, np.float64), angle, side="right")


def subject_trunk(cs: CleanSeries) -> tuple[np.ndarray, np.ndarray]:
    """Run frame and trunk angle (deg) of every row of a cleaned subject track,
    NaN where back_bending would not count the frame: shoulders and hips must
    be measured (filled gaps do not count), positions are the cleaned ones."""
    ok = cs.measured[:, [L_SHO, R_SHO, L_HIP, R_HIP]].all(1)
    xy = cs.xy.astype(np.float64)
    ang = np.full(len(ok), np.nan)
    if ok.any():
        ang[ok] = trunk_angle(((xy[:, L_SHO] + xy[:, R_SHO]) / 2)[ok], ((xy[:, L_HIP] + xy[:, R_HIP]) / 2)[ok])
    return np.asarray(cs.frame_idx, np.int64), ang


def gaze_class(raw: np.ndarray, valid: np.ndarray, yaw_max: float) -> np.ndarray:
    """Rough head direction per frame in image terms (FR-MOTION-08), from the
    nose, eye and ear keypoints [T, 17, 2] and their validity [T, 17]:
    -1 unknown, 0 front, 1 left, 2 right, 3 away (indices of GAZE).
      nose and both eyes: yaw = (nose.x - eye midpoint.x) / eye distance,
                          left below -yaw_max, right above, front between
      nose, one eye at most, one ear: profile, left or right of that ear
      no nose but an ear: away"""
    raw = np.asarray(raw, np.float64)
    v = np.asarray(valid, bool)
    nose, eyes, el, er = v[:, NOSE], v[:, L_EYE] & v[:, R_EYE], v[:, L_EAR], v[:, R_EAR]
    cls = np.full(len(raw), -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ed = np.linalg.norm(raw[:, L_EYE] - raw[:, R_EYE], axis=1)
        yaw = (raw[:, NOSE, 0] - (raw[:, L_EYE, 0] + raw[:, R_EYE, 0]) / 2) / ed
    fr = nose & eyes & (ed > 0)
    cls[fr] = np.where(yaw[fr] < -yaw_max, 1, np.where(yaw[fr] > yaw_max, 2, 0))
    prof = nose & ~eyes & (el ^ er) & (cls < 0)
    ear_x = np.where(el, raw[:, L_EAR, 0], raw[:, R_EAR, 0])
    cls[prof] = np.where(raw[prof, NOSE, 0] < ear_x[prof], 1, 2)
    cls[~nose & (el | er) & (cls < 0)] = 3
    return cls


def acf_period(sig: np.ndarray, fps: float, lag_min_s: float, lag_max_s: float) -> tuple[float | None, float]:
    """Period (s) and peak of the normalised autocorrelation of sig [T, C].

    NaN marks missing frames, which are left out of the sums; the channels are
    pooled and each is detrended linearly first. Of the local maxima between
    lag_min_s and lag_max_s the highest is found, then the shortest lag whose
    maximum reaches 0.9 of it, so a multiple of the period is not taken. The
    lag is refined by a parabola through the peak. (None, peak) if no positive
    local maximum exists."""
    x = np.array(sig, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    T = len(x)
    m = np.isfinite(x).all(1)
    lo = max(2, int(math.floor(lag_min_s * fps)))
    hi = int(math.ceil(lag_max_s * fps))
    if T <= hi + 2 or m.sum() < 3:
        return None, 0.0
    t = np.arange(T, dtype=np.float64)
    A = np.stack([t[m], np.ones(int(m.sum()))], 1)
    for c in range(x.shape[1]):
        coef = np.linalg.lstsq(A, x[m, c], rcond=None)[0]
        x[:, c] -= t * coef[0] + coef[1]
    x[~m] = 0.0
    lags = np.arange(lo - 1, hi + 2)
    r = np.full(len(lags), np.nan)
    for k, L in enumerate(lags):
        mm = (m[:-L] & m[L:])[:, None]
        if mm.sum() < L:
            continue
        a, b = x[:-L] * mm, x[L:] * mm
        den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
        if den > 0:
            r[k] = float((a * b).sum()) / den
    peaks = [k for k in range(1, len(lags) - 1)
             if np.isfinite(r[k - 1:k + 2]).all() and r[k] > r[k - 1] and r[k] >= r[k + 1]]
    if not peaks:
        return None, 0.0
    best = max(r[k] for k in peaks)
    if best <= 0:
        return None, float(best)
    k = next(k for k in peaks if r[k] >= 0.9 * best)
    curv = r[k - 1] - 2 * r[k] + r[k + 1]
    off = 0.5 * (r[k - 1] - r[k + 1]) / curv if curv < 0 else 0.0
    return float((lags[k] + off) / fps), float(r[k])


def speed(p: np.ndarray, ok: np.ndarray, fps: float) -> np.ndarray:
    """Per frame speed (units/s) of points p [T, 2]; NaN where this or the previous frame is not ok."""
    s = np.full(len(p), np.nan)
    if len(p) > 1:
        d = np.linalg.norm(np.diff(np.asarray(p, np.float64), axis=0), axis=-1) * fps
        s[1:] = np.where(ok[1:] & ok[:-1], d, np.nan)
    return s


def wrist_speeds(xy: np.ndarray, valid: np.ndarray, torso: float, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """Left and right wrist speed relative to the mid hip (torso lengths/s) per row
    of cleaned keypoints [T,17,2]; NaN where the wrist or a hip is not valid in
    this or the previous row. The input of two_hand (FR-MOTION-07)."""
    xy = np.asarray(xy, np.float64)
    hip = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    a = speed((xy[:, L_WRI] - hip) / torso, valid[:, [L_WRI, L_HIP, R_HIP]].all(1), fps)
    b = speed((xy[:, R_WRI] - hip) / torso, valid[:, [R_WRI, L_HIP, R_HIP]].all(1), fps)
    return a, b


def rolling_both_share(al: np.ndarray, ar: np.ndarray, ok: np.ndarray, window: int,
                       min_valid: float = 0.5) -> np.ndarray:
    """two_hand_simultaneous over the last `window` frames, for every frame:
    frames with both hands active / frames with at least one active, counted on
    frames where both wrist speeds are known (ok). NaN when fewer than min_valid
    of the window (of the frames so far, at the start) is ok or no hand was active."""
    def last(x):
        c = np.concatenate([[0], np.cumsum(x, dtype=np.int64)])
        i = np.arange(1, len(x) + 1)
        return c[i] - c[np.maximum(i - window, 0)]
    al, ar, ok = (np.asarray(v, bool) for v in (al, ar, ok))
    both, either, n_ok = last(al & ar & ok), last((al | ar) & ok), last(ok)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = both / either
    span = np.minimum(np.arange(1, len(ok) + 1), window)
    out[(either == 0) | (n_ok < min_valid * span)] = np.nan
    return out


# pieces of a step --------------------------------------------------------------------

@dataclass
class Subject:
    """Cleaned subject track of one shot."""
    shot: Shot
    series: KeypointSeries
    cs: CleanSeries
    energy: np.ndarray
    f0: int                       # run frame of row 0
    torso_ref: float | None       # 90th percentile torso length of the shot (LLD torso_ref)


@dataclass
class Piece:
    """Rows [lo, hi) of one subject track that lie inside a step."""
    sub: Subject
    lo: int
    hi: int

    shot = property(lambda self: self.sub.shot)
    fixed = property(lambda self: self.sub.shot.shot_type == "fixed")
    torso = property(lambda self: self.sub.cs.torso)
    n = property(lambda self: self.hi - self.lo)

    def xy(self, j: int) -> np.ndarray:
        return self.sub.cs.xy[self.lo:self.hi, j].astype(np.float64)

    def meas(self, *js: int) -> np.ndarray:
        return self.sub.cs.measured[self.lo:self.hi][:, list(js)].all(1)

    def val(self, *js: int) -> np.ndarray:
        return self.sub.cs.valid[self.lo:self.hi][:, list(js)].all(1)

    def hip(self) -> np.ndarray:
        return (self.xy(L_HIP) + self.xy(R_HIP)) / 2


@dataclass
class StepView:
    segment_id: str
    idx: int
    step: Step
    n: int                        # frames of the step
    fps: float
    pieces: list[Piece]


def _flags(*items) -> str:
    out = []
    for f in items:
        if f and f not in out:
            out.append(f)
    return ",".join(out)


def cycle_baseline(steps: list[Step]) -> list[dict]:
    """FR-SEG-03: n, median and quartiles of the duration per label (other
    excluded); flag low_n if n < 3. Percentiles are linear (NumPy default)."""
    by: dict[str, list[float]] = {}
    for s in steps:
        if s.label != OTHER:
            by.setdefault(s.label, []).append(s.end_s - s.start_s)
    rows = []
    for label, d in by.items():
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        rows.append({"label": label, "n": len(d), "median_s": round(float(med), 3), "q1_s": round(float(q1), 3),
                     "q3_s": round(float(q3), 3), "flags": "low_n" if len(d) < 3 else ""})
    return rows


class MotionAnalyzer:
    def __init__(self, cfg, ctx: RunContext | None = None):
        self.cfg = cfg
        self.m = cfg["motion"]
        self.ctx = ctx
        self.cleaner = SignalCleaner(cfg)
        self.notices: list[AvError] = []
        self.paths: dict[str, dict] = {}       # segment_id -> stored mid hip path
        self._metrics = {"duration": self.duration, "path": self.movement_path, "back_bending": self.back_bending,
                         "repetition": self.repetition, "hand_travel": self.hand_travel,
                         "two_hand": self.two_hand, "idle": self.idle_time, "gaze": self.gaze}

    # inputs ------------------------------------------------------------------------

    def subject(self, series: KeypointSeries, shot: Shot) -> Subject:
        cs = self.cleaner.clean(series)
        ok = cs.measured[:, [L_SHO, R_SHO, L_HIP, R_HIP]].all(1)
        ref = None
        if ok.any():
            raw = np.asarray(series.body_xy, np.float64)[ok]
            t = np.linalg.norm((raw[:, L_SHO] + raw[:, R_SHO]) / 2 - (raw[:, L_HIP] + raw[:, R_HIP]) / 2, axis=1)
            ref = float(np.percentile(t, 90)) or None
        f0 = int(series.frame_idx[0]) if len(series.frame_idx) else shot.start_frame
        return Subject(shot, series, cs, features(cs)["energy"], f0, ref)

    def load_subjects(self, manifest: dict, run_root: Path) -> list[Subject]:
        """Subject tracks of all shots from the stored KTS files."""
        shots = {i["shot_id"]: Shot(i["shot_id"], manifest["run_id"], i["start_frame"], i["end_frame"],
                                    i["start_s"], i["end_s"], i["shot_type"], i.get("scale_m_per_px"))
                 for i in manifest["shots"]["items"]}
        out = []
        for tr in manifest.get("tracking", {}).get("tracks", []):
            if tr.get("is_subject"):
                series, _ = read_kts(run_root / tr["kts_path"])
                out.append(self.subject(series, shots[tr["shot_id"]]))
        return sorted(out, key=lambda s: s.f0)

    # analysis ------------------------------------------------------------------------

    def analyse(self, run_id: str, subjects: list[Subject], steps: list[Step], fps: float,
                ) -> tuple[list[tuple[StepView, list[MetricRecord]]], list[dict]]:
        """Records per step (label != other) and the cycle baseline."""
        baseline = cycle_baseline(steps)
        self._base = {b["label"]: b for b in baseline}
        enabled = [m for m in self.m["enabled"] if m != "duration"]
        out = []
        for idx, step in enumerate(steps, 1):
            if step.label == OTHER:
                continue
            a, b = round(step.start_s * fps), round(step.end_s * fps)
            pieces = []
            for s in subjects:
                lo, hi = max(a, s.f0), min(b, s.f0 + len(s.series.present))
                if hi > lo:
                    pieces.append(Piece(s, lo - s.f0, hi - s.f0))
            sv = StepView(f"{run_id}-A-{idx:04d}", idx, step, max(b - a, 1), fps, pieces)
            recs = self.duration(sv)          # always, as in the LLD pseudocode
            for name in enabled:
                try:
                    got = self._metrics[name](sv)
                except Exception as e:        # one metric never stops the others (TC-MOT-15)
                    if self.ctx is not None:
                        self.ctx.log.debug("metric %s failed for %s:\n%s", name, sv.segment_id, traceback.format_exc())
                    self._notice(AvError("AV-MOT-004", metric=name, step=step.label,
                                         detail=f"{type(e).__name__}: {e}"[:300]))
                    continue
                self._coverage(got, name, step)
                recs += got
            out.append((sv, recs))
        return out, baseline

    def _notice(self, err: AvError) -> None:
        self.notices.append(err)
        if self.ctx is not None:
            self.ctx.notice(err, "metrics")

    def _coverage(self, recs: list[MetricRecord], name: str, step: Step) -> None:
        low = [r for r in recs if r.coverage < self.m["cov_min"]
               and not any(f in r.flags.split(",") for f in _STRUCTURAL)]
        for r in low:
            r.value = None
            r.flags = _flags(*r.flags.split(","), "low_coverage")
        if low:
            cov = min(r.coverage for r in low)
            self._notice(AvError("AV-MOT-001", metric=name, step=step.label, cov=f"{100 * cov:.0f}"))

    def _rec(self, sv: StepView, name: str, value, unit: str, cov: float, flags: str = "") -> MetricRecord:
        v = None if value is None or not np.isfinite(value) else float(value)
        return MetricRecord(sv.segment_id, name, v, unit, round(min(float(cov), 1.0), 4), flags)

    # metrics -------------------------------------------------------------------------

    def duration(self, sv: StepView) -> list[MetricRecord]:
        """FR-SEG-02, 03: duration and its deviation from the label baseline median (TC-MOT-01)."""
        d = sv.step.end_s - sv.step.start_s
        b = self._base.get(sv.step.label)
        dev = (d - b["median_s"]) / b["median_s"] if b and b["median_s"] > 0 else None
        return [self._rec(sv, "duration_s", d, "s", 1.0),
                self._rec(sv, "duration_dev", dev, "ratio", 1.0, b["flags"] if b else "low_n")]

    def movement_path(self, sv: StepView) -> list[MetricRecord]:
        """FR-MOTION-01: mid hip path at motion.path_hz on fixed shots only."""
        fixed = [p for p in sv.pieces if p.fixed]
        if not fixed:
            return [self._rec(sv, "movement_path", None, "px", 0.0, "handheld")]
        scaled = all(p.shot.scale_m_per_px for p in fixed)
        every = max(1, round(sv.fps / self.m["path_hz"]))
        total, meas = 0.0, 0
        t, x, y, k = [], [], [], []
        for i, p in enumerate(fixed):
            meas += int(p.meas(L_HIP, R_HIP).sum())
            rows = np.arange(0, p.n, every)
            q, ok = p.hip()[rows], p.val(L_HIP, R_HIP)[rows]
            d = np.linalg.norm(np.diff(q, axis=0), axis=-1)
            total += float(d[ok[1:] & ok[:-1]].sum()) * (p.shot.scale_m_per_px if scaled else 1.0)
            t.append((p.sub.f0 + p.lo + rows[ok]) / sv.fps)
            x.append(q[ok, 0])
            y.append(q[ok, 1])
            k.append(np.full(int(ok.sum()), i, np.int16))
        self.paths[sv.segment_id] = {
            "t_s": np.concatenate(t).astype(np.float32), "x_px": np.concatenate(x).astype(np.float32),
            "y_px": np.concatenate(y).astype(np.float32), "shot": np.concatenate(k),
            "shot_ids": np.array([p.shot.shot_id for p in fixed]), "path_hz": np.float32(self.m["path_hz"])}
        flags = _flags("partial_fixed" if len(fixed) < len(sv.pieces) else "",
                       "px_mixed_shots" if not scaled and len(fixed) > 1 else "")
        return [self._rec(sv, "movement_path", total, "m" if scaled else "px", meas / sv.n, flags)]

    def back_bending(self, sv: StepView) -> list[MetricRecord]:
        """FR-MOTION-02: time per trunk angle band, frames with shoulders and hips measured."""
        counts, meas, fore, total = np.zeros(4), 0, 0, 0.0
        for p in sv.pieces:
            ok = p.meas(L_SHO, R_SHO, L_HIP, R_HIP)
            if not ok.any():
                continue
            sho, hip = ((p.xy(L_SHO) + p.xy(R_SHO)) / 2)[ok], p.hip()[ok]
            ang = trunk_angle(sho, hip)
            counts += np.bincount(band_of(ang, self.m["angle_bands"]), minlength=4)
            total += float(ang.sum())
            meas += int(ok.sum())
            if p.sub.torso_ref:
                fore += int((np.linalg.norm(sho - hip, axis=1) < self.m["foreshort_ratio"] * p.sub.torso_ref).sum())
        flags = _flags("view_dependent",
                       "view_limited" if meas and fore / meas > self.m["view_limited_share"] else "")
        cov = meas / sv.n
        recs = [self._rec(sv, f"bend_{b}_s", counts[i] / sv.fps if meas else None, "s", cov, flags)
                for i, b in enumerate(BANDS)]
        return recs + [self._rec(sv, "bend_mean_deg", total / meas if meas else None, "deg", cov, flags)]

    def repetition(self, sv: StepView) -> list[MetricRecord]:
        """FR-MOTION-03: autocorrelation of the wrist trajectories relative to the
        mid hip (torso lengths) in windows of rep_window_s, moved by rep_hop_s."""
        m, fps = self.m, sv.fps
        win, hop = round(m["rep_window_s"] * fps), max(1, round(m["rep_hop_s"] * fps))
        min_len = math.ceil(2 * m["rep_lag_max_s"] * fps)
        analysed, periods, periodic_frames, cov_frames, long_enough = 0, [], 0, 0, False
        for p in sv.pieces:
            cov_frames += int((p.meas(L_HIP, R_HIP) & (p.meas(L_WRI) | p.meas(R_WRI))).sum())
            if p.torso is None or p.n < min_len:
                continue
            long_enough = True
            hip, hv = p.hip(), p.val(L_HIP, R_HIP)
            sigs = []
            for w in (L_WRI, R_WRI):
                s = (p.xy(w) - hip) / p.torso
                s[~(p.val(w) & hv)] = np.nan
                sigs.append(s)
            ln = win if p.n >= win else p.n
            mask = np.zeros(p.n, bool)
            for s0 in range(0, p.n - ln + 1, hop):
                if max(np.isfinite(s[s0:s0 + ln]).all(1).mean() for s in sigs) < m["cov_min"]:
                    continue
                analysed += 1
                best = max((acf_period(s[s0:s0 + ln], fps, m["rep_lag_min_s"], m["rep_lag_max_s"]) for s in sigs),
                           key=lambda r: r[1])
                if best[0] is not None and best[1] >= m["rep_acf_min"]:
                    periods.append(best[0])
                    mask[s0:s0 + ln] = True
            periodic_frames += int(mask.sum())
        names = (("repetition_share", "ratio"), ("repetition_period_s", "s"), ("repetition_count", "count"))
        cov = cov_frames / sv.n
        if sv.pieces and max(p.n for p in sv.pieces) < min_len:
            dur = sv.step.end_s - sv.step.start_s
            self._notice(AvError("AV-MOT-002", metric="repetition", step=sv.step.label, dur=f"{dur:.1f}"))
            return [self._rec(sv, n, None, u, cov, "too_short") for n, u in names]
        if analysed == 0:   # no subject, or no window with enough valid frames
            return [self._rec(sv, n, None, u, cov if long_enough else 0.0, "no_window") for n, u in names]
        if not periods:
            return [self._rec(sv, "repetition_share", 0.0, "ratio", cov, "not_periodic"),
                    self._rec(sv, "repetition_period_s", None, "s", cov, "not_periodic"),
                    self._rec(sv, "repetition_count", 0.0, "count", cov, "not_periodic")]
        period = float(np.median(periods))
        return [self._rec(sv, "repetition_share", len(periods) / analysed, "ratio", cov),
                self._rec(sv, "repetition_period_s", period, "s", cov),
                self._rec(sv, "repetition_count", periodic_frames / fps / period, "count", cov)]

    def hand_travel(self, sv: StepView) -> list[MetricRecord]:
        """FR-MOTION-04: wrist path length; relative to the mid hip on handheld shots."""
        scaled = bool(sv.pieces) and all(p.fixed and p.shot.scale_m_per_px for p in sv.pieces)
        recs = []
        for w, side in ((L_WRI, "l"), (R_WRI, "r")):
            total, total_t, meas, rel = 0.0, 0.0, 0, False
            for p in sv.pieces:
                if p.fixed:
                    pos, ok, mm = p.xy(w), p.val(w), p.meas(w)
                else:
                    pos, ok, mm = p.xy(w) - p.hip(), p.val(w, L_HIP, R_HIP), p.meas(w, L_HIP, R_HIP)
                    rel = True
                meas += int(mm.sum())
                if p.n < 2:
                    continue
                d = np.linalg.norm(np.diff(pos, axis=0), axis=-1)
                d = np.where(ok[1:] & ok[:-1], d, 0.0)
                if p.torso:
                    d[d < self.m["jitter_torso"] * p.torso] = 0.0
                    total_t += float(d.sum()) / p.torso
                total += float(d.sum()) * (p.shot.scale_m_per_px if scaled else 1.0)
            flags = _flags("body_relative" if rel else "",
                           "px_mixed_shots" if not scaled and len(sv.pieces) > 1 else "")
            cov = meas / sv.n
            recs += [self._rec(sv, f"hand_travel_{side}", total, "m" if scaled else "px", cov, flags),
                     self._rec(sv, f"hand_travel_{side}_torso", total_t, "torso", cov, flags)]
        return recs

    def two_hand(self, sv: StepView) -> list[MetricRecord]:
        """FR-MOTION-07, see the module docstring for the definitions and sources."""
        sl, sr, meas = [], [], 0
        for p in sv.pieces:
            meas += int(p.meas(L_WRI, R_WRI, L_HIP, R_HIP).sum())
            if p.torso is None:
                continue
            cs = p.sub.cs
            a, b = wrist_speeds(cs.xy[p.lo:p.hi], cs.valid[p.lo:p.hi], p.torso, sv.fps)
            both = np.isfinite(a) & np.isfinite(b)
            sl.append(a[both])
            sr.append(b[both])
        sl = np.concatenate(sl) if sl else np.zeros(0)
        sr = np.concatenate(sr) if sr else np.zeros(0)
        al, ar = sl >= self.m["hand_active"], sr >= self.m["hand_active"]
        either, most = int((al | ar).sum()), max(int(al.sum()), int(ar.sum()))
        simul = (al & ar).sum() / either if either else None
        use = min(int(al.sum()), int(ar.sum())) / most if most else None
        corr = None
        if len(sl) >= 3 and sl.std() > 0 and sr.std() > 0:
            corr = float(np.corrcoef(sl, sr)[0, 1])
        flags = _flags("view_dependent", "no_activity" if len(sl) and not either else "")
        cov = meas / sv.n
        return [self._rec(sv, "two_hand_simultaneous", simul, "ratio", cov, flags),
                self._rec(sv, "two_hand_use_ratio", use, "ratio", cov, flags),
                self._rec(sv, "two_hand_speed_corr", corr, "ratio", cov, flags)]

    def idle_time(self, sv: StepView) -> list[MetricRecord]:
        """FR-MOTION-10: runs of motion energy below idle_energy lasting at least idle_min_s."""
        min_run = max(1, round(self.m["idle_min_s"] * sv.fps))
        idle, seen, meas = 0, 0, 0
        for p in sv.pieces:
            e = p.sub.energy[p.lo:p.hi]
            fin = np.isfinite(e)
            seen += int(fin.sum())
            meas += int((fin & (p.sub.cs.measured[p.lo:p.hi].sum(1) >= 4)).sum())
            idle += sum(b - a for a, b in _runs(fin & (np.nan_to_num(e, nan=np.inf) < self.m["idle_energy"]))
                        if b - a >= min_run)
        cov = meas / sv.n
        return [self._rec(sv, "idle_time_s", idle / sv.fps if seen else None, "s", cov),
                self._rec(sv, "idle_share", idle / seen if seen else None, "ratio", cov)]

    def gaze(self, sv: StepView) -> list[MetricRecord]:
        """FR-MOTION-08 (Want), rough head direction in image terms from the raw
        nose, eye and ear keypoints at or above gaze_conf_min:
          nose and both eyes: yaw = (nose.x - eye midpoint.x) / eye distance,
                              left below -gaze_yaw, right above, front between
          nose, one eye at most, one ear: profile, left or right of that ear
          no nose but an ear: away
          down: nose below the mean visible ear by more than gaze_down_torso torso lengths
        """
        m = self.m
        counts, down, down_n, used_conf = np.zeros(4), 0, 0, []
        for p in sv.pieces:
            raw = np.asarray(p.sub.series.body_xy[p.lo:p.hi], np.float64)
            c = np.nan_to_num(np.asarray(p.sub.series.body_conf[p.lo:p.hi], np.float64))
            v = (c >= m["gaze_conf_min"]) & np.asarray(p.sub.series.present[p.lo:p.hi], bool)[:, None] \
                & np.isfinite(raw).all(-1)
            nose, el, er = v[:, NOSE], v[:, L_EAR], v[:, R_EAR]
            cls = gaze_class(raw, v, m["gaze_yaw"])
            ok = cls >= 0
            counts += np.bincount(cls[ok], minlength=4)
            face = v[ok][:, :5]
            used_conf.extend(c[ok][:, :5][face].tolist())
            if p.torso:
                ears = el | er
                with np.errstate(invalid="ignore"):
                    ear_y = np.nanmean(np.where(np.stack([el, er], 1), raw[:, [L_EAR, R_EAR], 1], np.nan), axis=1)
                dn = nose & ears
                down_n += int(dn.sum())
                down += int((raw[dn, NOSE, 1] - ear_y[dn] > m["gaze_down_torso"] * p.torso).sum())
        n = counts.sum()
        flags = _flags("rough", "view_dependent",
                       "low_confidence" if used_conf and float(np.mean(used_conf)) < 0.6 else "")
        cov = n / sv.n
        recs = [self._rec(sv, f"gaze_{g}_share", counts[i] / n if n else None, "ratio", cov, flags)
                for i, g in enumerate(GAZE)]
        return recs + [self._rec(sv, "gaze_down_share", down / down_n if down_n else None, "ratio",
                                 down_n / sv.n, flags)]

    # heatmap -------------------------------------------------------------------------

    def heatmap(self, sub: Subject) -> dict:
        """FR-MOTION-01: grid of mid hip samples at path_hz over the whole shot;
        fixed shots only, at least heatmap.min_samples samples."""
        h = self.cfg["heatmap"]
        rec = {"shot_id": sub.shot.shot_id, "shot_type": sub.shot.shot_type}
        if sub.shot.shot_type != "fixed":
            return {**rec, "status": "skipped", "reason": "handheld shot: picture positions are not room positions"}
        every = max(1, round(float(sub.series.fps) / self.m["path_hz"]))
        rows = np.arange(0, len(sub.series.present), every)
        hip = ((sub.cs.xy[:, L_HIP] + sub.cs.xy[:, R_HIP]) / 2)[rows].astype(np.float64)
        ok = sub.cs.valid[rows][:, [L_HIP, R_HIP]].all(1)
        if ok.sum() < h["min_samples"]:
            return {**rec, "status": "skipped", "samples": int(ok.sum()),
                    "reason": f"{int(ok.sum())} path samples, fewer than {h['min_samples']}"}
        scale = sub.shot.scale_m_per_px
        cell = h["cell_m"] / scale if scale else float(h["cell_px"])
        W, H = int(sub.series.width), int(sub.series.height)
        nx, ny = max(1, math.ceil(W / cell)), max(1, math.ceil(H / cell))
        grid, _, _ = np.histogram2d(hip[ok, 1], hip[ok, 0], bins=[ny, nx], range=[[0, ny * cell], [0, nx * cell]])
        return {**rec, "status": "done", "samples": int(ok.sum()), "cell_px": round(cell, 3),
                "cell_m": h["cell_m"] if scale else None, "grid": grid.astype(np.int32), "size": (W, H)}


# storing -------------------------------------------------------------------------------

CSV_COLUMNS = ["segment_id", "step", "label", "start_s", "end_s", "metric", "value", "unit", "coverage", "flags"]


def metrics_csv_bytes(results) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLUMNS)
    for sv, recs in results:
        for r in recs:
            w.writerow([r.segment_id, sv.idx, sv.step.label, f"{sv.step.start_s:.3f}", f"{sv.step.end_s:.3f}",
                        r.metric_name, "" if r.value is None else f"{r.value:.6g}", r.unit, f"{r.coverage:.4f}",
                        r.flags])
    return buf.getvalue().encode("utf-8")


def _npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def heatmap_png(hm: dict, background: np.ndarray | None) -> np.ndarray:
    """Heat colours over the reference frame of the shot (OpenCV only)."""
    import cv2
    W, H = hm["size"]
    img = np.zeros((H, W, 3), np.uint8) if background is None else cv2.resize(background, (W, H))
    grid = hm["grid"].astype(np.float64)
    cell = hm["cell_px"]
    big = cv2.resize(grid / max(grid.max(), 1.0), (max(1, round(grid.shape[1] * cell)),
                                                   max(1, round(grid.shape[0] * cell))),
                     interpolation=cv2.INTER_NEAREST)
    big = np.pad(big, ((0, max(0, H - big.shape[0])), (0, max(0, W - big.shape[1]))))[:H, :W]
    color = cv2.applyColorMap((255 * big).astype(np.uint8), cv2.COLORMAP_JET)
    on = big > 0
    img[on] = (0.4 * img[on] + 0.6 * color[on]).astype(np.uint8)
    return img


def run_metrics(ctx: RunContext, cfg, steps: list[Step]) -> dict:
    """The metrics stage: records per step, metrics.csv, paths/*.npz, heatmaps
    and the manifest section (CPU only, from the stored KTS files)."""
    import cv2
    t0 = time.monotonic()
    ctx.manifest["notices"] = [n for n in ctx.manifest.get("notices", []) if n.get("stage") != "metrics"]
    an = MotionAnalyzer(cfg, ctx)
    fps = float(ctx.manifest["ingest"]["cfr"]["fps"])
    subjects = an.load_subjects(ctx.manifest, ctx.paths.root)
    results, baseline = an.analyse(ctx.run_id, subjects, steps, fps)
    paths = ctx.paths
    write_atomic(paths.metrics_csv, metrics_csv_bytes(results), paths)
    paths.paths_dir.mkdir(exist_ok=True)
    for f in paths.paths_dir.glob("*.npz"):            # paths of an earlier segmentation
        f.unlink()
    for sid, arrays in an.paths.items():
        write_atomic(paths.path_npz(sid), _npz_bytes(arrays), paths)
    heat = []
    if "path" in cfg["motion"]["enabled"]:
        paths.out_dir.mkdir(exist_ok=True)
        for s in subjects:
            hm = an.heatmap(s)
            if hm["status"] == "done":
                ref = paths.ref_png(s.shot.shot_id)
                bg = cv2.imread(str(ref)) if ref.is_file() else None
                write_atomic(paths.heatmap(s.shot.shot_id, "npz"),
                             _npz_bytes({"grid": hm["grid"], "cell_px": np.float32(hm["cell_px"])}), paths)
                ok, png = cv2.imencode(".png", heatmap_png(hm, bg))
                write_atomic(paths.heatmap(s.shot.shot_id), png.tobytes(), paths)
                hm = {k: v for k, v in hm.items() if k not in ("grid", "size")}
                hm["png"] = ctx.rel(paths.heatmap(s.shot.shot_id))
            heat.append(hm)

    def r4(v):
        return None if v is None else round(v, 4)
    rows, n_null, n_low = [], 0, 0
    for sv, recs in results:
        n_null += sum(r.value is None for r in recs)
        n_low += sum("low_coverage" in r.flags.split(",") for r in recs)
        rows.append({"segment_id": sv.segment_id, "step": sv.idx, "label": sv.step.label,
                     "start_s": sv.step.start_s, "end_s": sv.step.end_s,
                     "pieces": [{"shot_id": p.shot.shot_id, "shot_type": p.shot.shot_type, "frames": p.n}
                                for p in sv.pieces],
                     "records": [{"metric": r.metric_name, "value": r4(r.value), "unit": r.unit,
                                  "coverage": r.coverage, "flags": r.flags} for r in recs]})
    sec = {"engine": "A", "stale": False, "steps": rows, "baseline": baseline,
           "counts": {"records": sum(len(r) for _, r in results), "null": n_null, "low_coverage": n_low},
           "metrics_csv": ctx.rel(paths.metrics_csv),
           "paths": [ctx.rel(paths.path_npz(s)) for s in an.paths], "heatmaps": heat,
           "params": {k: cfg[k] for k in ("motion", "heatmap")}, "seconds": round(time.monotonic() - t0, 3)}
    ctx.manifest["metrics"] = sec
    ctx.log.info("metrics: %d steps, %d records, %d not reported", len(rows), sec["counts"]["records"], n_null)
    return sec
