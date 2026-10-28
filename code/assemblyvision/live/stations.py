"""One station of a live session: the template's logic plus what every station
shows (trunk angle and band now, seconds per band). Kinds:
  workbench  OnlineRuleEngine steps and cycles, two hand coordination
  lift       LiftDetector, lift log and the stoop vs squat comparison
  posture    PostureTimer, idle time (IdleStream), rough gaze shares
  combined   the one view of the Live page (CombinedStation): all of the above
             on every frame except gaze; lifts only count when the knees are
             seen, cycles only while cycle counting is switched on
The streaming classes of analysis/motion_stream.py are reused; their torso
length comes from the visible side here, so they also work in a side view.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..analysis.motion import band_of
from ..analysis.motion_stream import IdleStream, TwoHandStream
from .body import knee_conf, trunk
from .engine import OnlineRuleEngine
from .lifts import LiftDetector
from .posture import GazeShare, PostureTimer
from .templates import ProcessTemplate, load_template

COMBINED = "combined"
KNEE_WINDOW_S = 3.0      # lifts are possible when the knees were seen in this recent window


class _SideTorso:
    """Torso length from the visible shoulders and hips, running median over stream_torso_s."""

    def _torso_len(self, t, kp, ok):
        from .body import _mid, mid_hip
        sho, hip = _mid(kp, ok, 5, 6), mid_hip(kp, ok)
        if sho is not None and hip is not None:
            self._torso.append((t, float(np.hypot(*(sho - hip)))))
        while self._torso and self._torso[0][0] < t - self.m["stream_torso_s"]:
            self._torso.popleft()
        if not self._torso:
            return None
        med = float(np.median([x for _, x in self._torso]))
        return med if med > 0 else None


class LiveIdle(_SideTorso, IdleStream):
    pass


class LiveTwoHand(_SideTorso, TwoHandStream):
    pass


class Station:
    def __init__(self, template: ProcessTemplate | None, cfg, number: int, started_s: float):
        self.t = template
        self.kind = template.kind if template else "workbench"
        self.title = template.title if template else "Segments (no template)"
        self.number = number
        self.started_s = started_s
        self.ended_s: float | None = None
        live, m = cfg["live"], cfg["motion"]
        self.kp_min = live["kp_conf_min"]
        self.bands = m["angle_bands"]
        self.gap_s = max(cfg["clean"]["gap_max_s"], live["lost_s"] / 2)
        self.band_s = np.zeros(4)
        self.band_now: int | None = None
        self.trunk_now: float | None = None
        self.last_t: float | None = None
        self.seen_s = 0.0
        self.engine = self.lifts = self.posture = self.gaze = self.idle = self.two_hand = None
        if self.kind == "workbench":
            self.engine = OnlineRuleEngine(template, self.kp_min, live["zone_hold_s"], m["idle_energy"],
                                           station=self.title)
            self.two_hand = LiveTwoHand(cfg)
        elif self.kind == "lift":
            self.lifts = LiftDetector(template.params, self.bands, self.gap_s)
        else:
            self.posture = PostureTimer(template.params, self.bands, self.gap_s)
            self.idle = LiveIdle(cfg)
            self.gaze = GazeShare(m["gaze_conf_min"], m["gaze_yaw"], self.gap_s)
        self.new_lift = None

    def set_template(self, template: ProcessTemplate) -> None:
        self.t = template
        if self.engine is not None:
            self.engine.set_template(template)

    def draw_zones(self) -> dict | None:
        return self.t.zones if self.t is not None and self.kind == "workbench" else None

    def update(self, t: float, kp: np.ndarray, conf: np.ndarray, size: tuple[int, int]) -> None:
        """One frame of the subject: kp [17, 2] smoothed (NaN = unseen), conf [17]."""
        ok = (np.nan_to_num(conf) >= self.kp_min) & np.isfinite(kp).all(1)
        dt = 0.0 if self.last_t is None else t - self.last_t
        dt = dt if 0 < dt <= self.gap_s else 0.0
        self.last_t = t
        a = trunk(kp, ok)
        self.trunk_now = a
        self.band_now = None if a is None else int(band_of(a, self.bands))
        if self.band_now is not None:
            self.band_s[self.band_now] += dt
            self.seen_s += dt
        kp_raw = np.where(ok[:, None], kp, np.nan)
        if self.engine is not None:
            self.engine.update(t, kp, conf, size)
            self.hands_now = self.two_hand.update(t, kp_raw, conf)["active"]   # (left, right) active, or None
        elif self.lifts is not None:
            self.new_lift = self.lifts.update(t, kp, conf, ok) or self.new_lift
        else:
            self.posture.update(t, kp, ok)
            self.idle.update(t, kp_raw, conf)
            self.gaze.update(t, kp, conf)

    def on_lost(self, t: float) -> None:
        """Subject or camera lost for live.lost_s: open steps and lifts end."""
        if self.engine is not None:
            self.engine.close_all(t, "lost")
        elif self.lifts is not None:
            self.lifts.abort(t)
        self.last_t = None

    def close(self, t: float, how: str) -> None:
        if self.engine is not None:
            self.engine.close_all(t, how)
        elif self.lifts is not None:
            self.lifts.abort(t)
        self.ended_s = t

    # what the page shows ---------------------------------------------------------------

    def panel(self, t: float) -> dict:
        d = {"kind": self.kind, "title": self.title, "trunk_deg": None if self.trunk_now is None else
             round(self.trunk_now), "band": self.band_now, "band_s": [round(float(x), 1) for x in self.band_s]}
        if self.engine is not None:
            st = self.engine.state
            ct = st.cycle_times
            d.update(step=st.open_step, step_s=None if st.open_since is None else round(t - st.open_since, 1),
                     cycles=st.cycles, last_cycle_s=None if st.last_cycle_s is None else round(st.last_cycle_s, 1),
                     mean_cycle_s=round(float(np.mean(ct)), 1) if ct else None, **self.two_hand.summary())
        elif self.lifts is not None:
            d.update(lifts=self.lifts.rows(), in_lift=self.lifts.cur is not None,
                     knee_deg=None if self.lifts.now["knee_deg"] is None else round(self.lifts.now["knee_deg"]),
                     comparison=self.lifts.comparison())
        else:
            d.update(posture=self.posture.state(t), gaze_now=self.gaze.now, gaze=self.gaze.shares(),
                     **self.idle.summary())
        return d

    def log_rows(self) -> list[dict]:
        if self.engine is not None:
            return [{"step": s.name, "start_s": round(s.start_s - self.started_s, 1),
                     "duration_s": round(s.end_s - s.start_s, 1), "closed_by": s.closed_by}
                    for s in self.engine.steps]
        if self.lifts is not None:
            return self.lifts.rows()
        return []

    def summary(self, t: float) -> dict:
        """Numbers of the station for the Stop table and summary.json."""
        p = self.panel(t)
        end = self.ended_s if self.ended_s is not None else t
        rows = [("Time at station", f"{end - self.started_s:.0f} s")]
        bs = p["band_s"]
        rows.append(("Trunk bands B0 / B1 / B2 / B3", " / ".join(f"{x:.0f} s" for x in bs)))
        if self.kind == "workbench":
            rows += [("Cycles", str(p["cycles"])),
                     ("Last cycle", _s(p["last_cycle_s"])), ("Mean cycle", _s(p["mean_cycle_s"])),
                     ("Both hands active together", _pct(p.get("two_hand_simultaneous"))),
                     ("Use ratio of the hands", _pct(p.get("two_hand_use_ratio")))]
        elif self.kind == "lift":
            for kind in ("stoop", "squat"):
                ls = [x for x in p["lifts"] if x["kind"] == kind]
                rows.append((f"{kind.capitalize()} lifts", str(len(ls))))
            cmp_ = p["comparison"]
            if cmp_:
                for kind in ("stoop", "squat"):
                    c = cmp_[kind]
                    rows.append((f"{kind.capitalize()}: mean peak trunk / knee / B2+B3",
                                 f"{c['peak_trunk_deg']:.0f}° / {_deg(c['knee_deg'])} / {c['b2_b3_s']:.1f} s"))
        else:
            ps = p["posture"]
            g = p["gaze"]
            rows += [("Longest bent stretch (B1 or worse)", _s(ps["longest_s"])),
                     ("Total bent time", _s(ps["total_s"])), ("Idle time", _s(p.get("idle_time_s"))),
                     ("Gaze front / side / away", " / ".join(_pct(g[k]) for k in ("front", "side", "away")))]
        return {"number": self.number, "title": self.title, "kind": self.kind, "rows": rows,
                "log": self.log_rows()}


class CombinedStation(Station):
    """The one live view: trunk band timers, time bent and the longest bent
    stretch (PostureTimer), idle time, both hands together and the hand use
    ratio, lifts with stoop or squat (only while the knees are seen, live.
    lift_knee_conf and lift_knee_seen) and, with cycle counting on, the cycles
    of the workbench zones. The settings come from the three templates."""

    def __init__(self, cfg, number: int, started_s: float, cycles: bool = False, borders=None):
        live, m = cfg["live"], cfg["motion"]
        self.kind, self.title = COMBINED, "Live analysis"
        self.number, self.started_s, self.ended_s = number, started_s, None
        self.kp_min, self.bands = live["kp_conf_min"], m["angle_bands"]
        self.gap_s = max(cfg["clean"]["gap_max_s"], live["lost_s"] / 2)
        self.zone_hold_s, self.idle_energy = live["zone_hold_s"], m["idle_energy"]
        self.knee_min, self.knee_share = live["lift_knee_conf"], live["lift_knee_seen"]
        self.band_s = np.zeros(4)
        self.band_now = self.trunk_now = self.last_t = None
        self.seen_s = 0.0
        self.lifts = LiftDetector({**load_template("floor_pick").params, "knee_conf_min": self.knee_min,
                                   "knee_seen_min": self.knee_share}, self.bands, self.gap_s)
        self.posture = PostureTimer(load_template("side_work").params, self.bands, self.gap_s)
        self.idle, self.two_hand = LiveIdle(cfg), LiveTwoHand(cfg)
        self.gaze = None
        self.bench = load_template("workbench")
        self.t = self.bench
        self.engine = None
        self.cycles_on = False
        self.knee_recent: deque = deque()
        self.knee_frames = self.frames = 0
        self.new_lift = None
        if cycles:
            self.set_cycles(True, borders)

    def set_cycles(self, on: bool, borders=None) -> None:
        """Cycle counting on (with the zone borders) or off; the counts so far are kept."""
        if on:
            self.t = self.bench.with_borders(*borders) if borders else self.bench
            if self.engine is None:
                self.engine = OnlineRuleEngine(self.t, self.kp_min, self.zone_hold_s, self.idle_energy,
                                               station=self.title)
            else:
                self.engine.set_template(self.t)
        elif self.cycles_on and self.engine is not None:
            self.engine.close_all(self.last_t or 0.0, "off")
        self.cycles_on = on

    def set_template(self, template: ProcessTemplate) -> None:
        self.t = template
        if self.engine is not None:
            self.engine.set_template(template)

    def draw_zones(self) -> dict | None:
        return self.t.zones if self.cycles_on else None

    def lifts_ready(self) -> bool:
        """Whether the knees were seen in most frames of the last KNEE_WINDOW_S."""
        if not self.knee_recent:
            return False
        return sum(s for _, s in self.knee_recent) >= self.knee_share * len(self.knee_recent)

    def update(self, t: float, kp: np.ndarray, conf: np.ndarray, size: tuple[int, int]) -> None:
        ok = (np.nan_to_num(conf) >= self.kp_min) & np.isfinite(kp).all(1)
        dt = 0.0 if self.last_t is None else t - self.last_t
        dt = dt if 0 < dt <= self.gap_s else 0.0
        self.last_t = t
        a = trunk(kp, ok)
        self.trunk_now = a
        self.band_now = None if a is None else int(band_of(a, self.bands))
        if self.band_now is not None:
            self.band_s[self.band_now] += dt
            self.seen_s += dt
        kp_raw = np.where(ok[:, None], kp, np.nan)
        self.posture.update(t, kp, ok)
        self.idle.update(t, kp_raw, conf)
        self.hands_now = self.two_hand.update(t, kp_raw, conf)["active"]   # (left, right) active, or None
        self.new_lift = self.lifts.update(t, kp, conf, ok) or self.new_lift
        seen = self.lifts.now["knee_deg"] is not None and knee_conf(conf) >= self.knee_min
        self.knee_frames += int(seen)
        self.frames += 1
        self.knee_recent.append((t, seen))
        while self.knee_recent and self.knee_recent[0][0] < t - KNEE_WINDOW_S:
            self.knee_recent.popleft()
        if self.cycles_on:
            self.engine.update(t, kp, conf, size)

    def on_lost(self, t: float) -> None:
        if self.engine is not None and self.cycles_on:
            self.engine.close_all(t, "lost")
        self.lifts.abort(t)
        self.knee_recent.clear()
        self.last_t = None

    def close(self, t: float, how: str) -> None:
        if self.engine is not None and self.cycles_on:
            self.engine.close_all(t, how)
        self.lifts.abort(t)
        self.ended_s = t

    def panel(self, t: float) -> dict:
        ready = self.lifts_ready()
        knee = self.lifts.now["knee_deg"]
        d = {"kind": self.kind, "title": self.title, "trunk_deg": None if self.trunk_now is None else
             round(self.trunk_now), "band": self.band_now, "band_s": [round(float(x), 1) for x in self.band_s],
             "posture": self.posture.state(t), **self.idle.summary(), **self.two_hand.summary(),
             "lifts": self.lifts.rows(), "lifts_ready": ready, "in_lift": ready and self.lifts.cur is not None,
             "knee_deg": None if knee is None or not ready else round(knee), "comparison": self.lifts.comparison(),
             "cycles_on": self.cycles_on}
        if self.engine is not None:
            st = self.engine.state
            ct = st.cycle_times
            d.update(step=st.open_step if self.cycles_on else None,
                     step_s=None if st.open_since is None or not self.cycles_on else round(t - st.open_since, 1),
                     cycles=st.cycles, last_cycle_s=None if st.last_cycle_s is None else round(st.last_cycle_s, 1),
                     mean_cycle_s=round(float(np.mean(ct)), 1) if ct else None)
        return d

    def log_rows(self) -> list[dict]:
        return self.lifts.rows()

    def step_rows(self) -> list[dict]:
        if self.engine is None:
            return []
        return [{"step": s.name, "start_s": round(s.start_s - self.started_s, 1),
                 "duration_s": round(s.end_s - s.start_s, 1), "closed_by": s.closed_by} for s in self.engine.steps]

    def summary(self, t: float) -> dict:
        """Every measure of the combined run for the Stop table and summary.json."""
        p = self.panel(t)
        end = self.ended_s if self.ended_s is not None else t
        ps = p["posture"]
        rows = [("Time analysed", f"{end - self.started_s:.0f} s"),
                ("Trunk bands B0 / B1 / B2 / B3", " / ".join(f"{x:.0f} s" for x in p["band_s"])),
                ("Time bent (B1 or worse)", _s(ps["total_s"])),
                ("Longest bent stretch", _s(ps["longest_s"])),
                ("Idle time", _s(p.get("idle_time_s"))),
                ("Both hands active together", _pct(p.get("two_hand_simultaneous"))),
                ("Hand use ratio", _pct(p.get("two_hand_use_ratio")))]
        if p["lifts"] or (self.frames and self.knee_frames >= self.knee_share * self.frames):
            for kind in ("stoop", "squat"):
                rows.append((f"{kind.capitalize()} lifts", str(sum(1 for x in p["lifts"] if x["kind"] == kind))))
            cmp_ = p["comparison"]
            if cmp_:
                for kind in ("stoop", "squat"):
                    c = cmp_[kind]
                    rows.append((f"{kind.capitalize()}: mean peak trunk / knee / B2+B3",
                                 f"{c['peak_trunk_deg']:.0f}° / {_deg(c['knee_deg'])} / {c['b2_b3_s']:.1f} s"))
        else:
            rows.append(("Lifts", "n/a, lifts need a side view with the whole body"))
        if self.engine is not None:
            rows += [("Cycles", str(p["cycles"])), ("Last cycle", _s(p["last_cycle_s"])),
                     ("Mean cycle", _s(p["mean_cycle_s"]))]
        return {"number": self.number, "title": self.title, "kind": self.kind, "rows": rows,
                "log": self.log_rows(), "steps": self.step_rows()}


def _s(v) -> str:
    return "n/a" if v is None else f"{v:.1f} s"


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.0f} %"


def _deg(v) -> str:
    return "n/a" if v is None else f"{v:.0f}°"
