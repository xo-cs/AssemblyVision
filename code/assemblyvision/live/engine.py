"""OnlineRuleEngine (LLD 3.12): Engine A run on past frames only, for the
workbench station.

Events from the wrists (COCO 9 and 10) seen at or above live.kp_conf_min:
zone_enter:<zone> when a wrist has stayed inside for live.zone_hold_s,
zone_exit:<zone> when it has stayed outside as long, both_in:<zone> when both
wrists are inside (once per entry). They drive the ordered steps of the
template: a step opens on its start_event and closes on its end_event once it
lasted min_s; one longer than max_s is closed as timeout. An event that closes
a step may also open the next one. In a cyclic template the close of the last
step counts one cycle, timed from the opening of the first step.

Deviations from the LLD pseudocode, for a robust demo: after a timeout or a
lost subject the cycle starts again at the first step (an incomplete cycle is
never counted), and the start event of the first step while waiting for a
later step also restarts the cycle (the worker began a new part).

Without a template (AV-LIV-003) the engine cuts segments at pauses: a pause is
motion energy below motion.idle_energy for at least 0.8 s (TC-LIV-08).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .body import L_WRI, R_WRI, Hold
from .templates import ProcessTemplate

PAUSE_S = 0.8


@dataclass
class LiveStep:
    station: str
    name: str
    start_s: float
    end_s: float
    closed_by: str          # event, timeout, lost, stop, switch, pause


@dataclass
class EngineState:
    open_step: str | None = None
    open_since: float | None = None
    cycles: int = 0
    last_cycle_s: float | None = None
    cycle_times: list = field(default_factory=list)
    events: list = field(default_factory=list)
    closed: list = field(default_factory=list)


class OnlineRuleEngine:
    def __init__(self, template: ProcessTemplate | None, kp_conf_min: float, zone_hold_s: float,
                 idle_energy: float = 0.15, station: str = ""):
        self.t = template
        self.kp_min = kp_conf_min
        self.hold = zone_hold_s
        self.station = station or (template.name if template else "segments")
        self.zone_state: dict = {}
        self.both: dict = {}
        self.k = 0
        self.open: tuple[int, float] | None = None
        self.cycle_start: float | None = None
        self.state = EngineState()
        self.steps: list[LiveStep] = []
        self.idle_energy = idle_energy
        self._seg = None
        self._pause_since = None
        self._prev = None
        self.set_template(template)

    def set_template(self, template: ProcessTemplate | None) -> None:
        """New zones (from the sliders); zone states start again, open steps stay."""
        self.t = template
        zones = template.zones if template else {}
        self.zone_state = {(z, w): Hold(self.hold) for z in zones for w in (L_WRI, R_WRI)}
        self.both = {z: False for z in zones}

    # events ---------------------------------------------------------------------------

    def _inside(self, p: np.ndarray, box, size) -> bool:
        w, h = size
        return box[0] * w <= p[0] < box[2] * w and box[1] * h <= p[1] < box[3] * h

    def _events(self, t: float, kp: np.ndarray, conf: np.ndarray, size) -> list[tuple[str, float]]:
        """(event, time it began) in time order; the time is when the wrist crossed,
        not when the hold ended."""
        ev = []
        for (z, wri), hold in self.zone_state.items():
            seen = conf[wri] >= self.kp_min and np.isfinite(kp[wri]).all()
            raw = bool(seen and self._inside(kp[wri], self.t.zones[z], size)) if seen else hold.value
            hold.update(t, raw)
            if hold.changed:
                ev.append((f"zone_{'enter' if hold.value else 'exit'}:{z}", hold.changed_at))
        for z in self.both:
            hl, hr = self.zone_state[(z, L_WRI)], self.zone_state[(z, R_WRI)]
            now = hl.value and hr.value
            if now and not self.both[z]:
                ev.append((f"both_in:{z}", max(hl.changed_at or t, hr.changed_at or t)
                           if (hl.changed or hr.changed) else t))
            self.both[z] = now
        return sorted(ev, key=lambda e: e[1])

    # steps ----------------------------------------------------------------------------

    def _close(self, t: float, how: str) -> None:
        k, t0 = self.open
        self.steps.append(LiveStep(self.station, self.t.steps[k]["name"], t0, t, how))
        self.state.closed.append(self.steps[-1])
        self.open = None

    def _restart(self) -> None:
        self.k, self.cycle_start = 0, None

    def update(self, t: float, kp: np.ndarray, conf: np.ndarray, size: tuple[int, int]) -> EngineState:
        """One frame: t (s), kp [17, 2] smoothed pixels (NaN = unseen), conf [17], size (w, h)."""
        st = self.state
        st.closed = []
        if self.t is None:
            return self._segments(t, kp, conf, size)
        ev = self._events(t, kp, conf, size)
        st.events = [e for e, _ in ev]
        steps = self.t.steps
        if self.open and t - self.open[1] > steps[self.open[0]]["max_s"]:
            self._close(self.open[1] + steps[self.open[0]]["max_s"], "timeout")
            self._restart()
        for name, te in ev:
            if self.open and name == steps[self.open[0]]["end_event"] \
                    and te - self.open[1] >= steps[self.open[0]]["min_s"]:
                last = self.open[0] == len(steps) - 1
                self._close(te, "event")
                self.k = (self.k + 1) % len(steps)
                if last and self.t.cyclic and self.cycle_start is not None:
                    st.cycles += 1
                    st.last_cycle_s = te - self.cycle_start
                    st.cycle_times.append(st.last_cycle_s)
                    self.cycle_start = None
            if self.open is None and self.k != 0 and name == steps[0]["start_event"] \
                    and name != steps[self.k]["start_event"]:
                self._restart()                       # a new part was picked: start the cycle again
            if self.open is None and name == steps[self.k]["start_event"]:
                self.open = (self.k, te)
                if self.k == 0:
                    self.cycle_start = te
        st.open_step = steps[self.open[0]]["name"] if self.open else None
        st.open_since = self.open[1] if self.open else None
        return st

    def close_all(self, t: float, how: str) -> list[LiveStep]:
        """Close the open step (lost subject, stop, station switch) and restart the cycle."""
        before = len(self.steps)
        if self.t is None:
            if self._seg is not None:
                self.steps.append(LiveStep(self.station, self._seg[0], self._seg[1], t, how))
                self._seg = None
        elif self.open:
            self._close(t, how)
            self._restart()
        self.state.open_step = self.state.open_since = None
        for h in self.zone_state.values():
            h.value, h._since = False, None
        self.both = {z: False for z in self.both}
        return self.steps[before:]

    # fallback without a template: segments between pauses -------------------------------

    def _segments(self, t: float, kp: np.ndarray, conf: np.ndarray, size) -> EngineState:
        st = self.state
        ok = (conf >= self.kp_min) & np.isfinite(kp).all(1)
        energy = None
        if self._prev is not None:
            pt, pkp, pok = self._prev
            both = ok & pok
            dt = t - pt
            ys = kp[ok][:, 1]
            scale = float(ys.max() - ys.min()) if ok.sum() >= 2 else 0.0
            if dt > 0 and both.sum() >= 4 and scale > 0:
                energy = float(np.linalg.norm(kp[both] - pkp[both], axis=1).mean() / dt / (scale / 2))
        self._prev = (t, kp.copy(), ok)
        if energy is None:
            return st
        if energy < self.idle_energy:
            self._pause_since = t if self._pause_since is None else self._pause_since
            if self._seg is not None and t - self._pause_since >= PAUSE_S:
                self.steps.append(LiveStep(self.station, self._seg[0], self._seg[1], self._pause_since, "pause"))
                st.closed.append(self.steps[-1])
                self._seg = None
        else:
            self._pause_since = None
            if self._seg is None:
                n = sum(1 for s in self.steps if s.name.startswith("segment_")) + 1
                self._seg = (f"segment_{n}", t)
        st.open_step = self._seg[0] if self._seg else None
        st.open_since = self._seg[1] if self._seg else None
        return st
