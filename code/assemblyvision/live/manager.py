"""LiveManager and LiveSession (LLD 3.12).

The Gradio stream handler only calls push(): the frame goes into a one frame
slot (a newer frame replaces an older one that was not processed yet, latest
frame wins) and the overlay of the last processed frame comes back at once.
A processing thread takes the newest frame, runs pose and ByteTrack, chooses
the subject (the locked track if seen within live.lost_s, else the largest
box), smooths its keypoints with the One Euro filter and updates the current
station. The page never waits for the GPU, so the delay stays bounded
(NFR-REL-08).

Frame rate: the browser offers up to 25 frames per second (live.stream_every_s)
and the session accepts live.rate_steps[k] of them (RateControl): when the
server delay p95 or the share of dropped frames gets too high it steps down
(25, 20, 15, 10 fps) and later up again. push() skips frames that come faster
than the current rate (paced, not counted as dropped); the page reads the rate
from the panel and throttles its own sends to it (app.js), so the tunnel only
carries frames that are used.

Reload and stream drop: the session lives in this process, not in the page. A
reloaded page finds it and continues when frames arrive again. With no frame
for live.lost_s the camera counts as paused (open steps and lifts end as lost,
no time is added); the session ends by itself only after live.idle_stop_s or
live.max_session_s.

GPU: one session at a time per server. Start takes ~/AssemblyVision/.locks/
gpu<live.gpu>.lock without waiting (AV-LIV-002 if another instance holds it)
and checks the free memory of that GPU (AV-LIV-005). Batch runs use the lock
of their own GPU and are never blocked. The model is kept for
live.model_keep_s after a session.

Files: data/public/live/<session_id>/summary.json at Stop and, only with Save
recording on, recording.mp4 (NFR-SEC-07). Nothing else is written.
"""
from __future__ import annotations

import fcntl
import json
import secrets
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..errors import AvError
from ..storage.files import REPO_ROOT, utc_now
from .draw import draw_overlay
from .filters import OneEuro
from .perception import Detections, LiveTracker
from .recorder import Recorder
from .stations import COMBINED, CombinedStation, Station
from .templates import load_template

NO_FRAME_S = 5.0          # AV-LIV-001 when no frame arrived this long after Start
STATS_S = 5.0             # processed fps over this window


def new_session_id() -> str:
    """'live_' + UTC YYYYMMDD_HHMMSS + '_' + 4 hex digits (25 characters)."""
    return "live_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)


def elapsed_s(t0: float, now: float, ended: float | None = None) -> float:
    """Session time from Start: frozen at Stop, never negative."""
    return max(0.0, (now if ended is None else min(now, ended)) - t0)


class RateControl:
    """Accepted frame rate of a session, one step of rate_steps at a time.

    update() is called with the server delay p95 (ms) and the share of dropped
    frames over the recent window. Down one step when p95 > p95_max_ms or the
    drop share > drop_max; up one step after up_s of good windows (p95 under
    half the limit and drop share under drop_max / 5). Two changes are at
    least hold_s apart, so the rate does not flap."""

    def __init__(self, steps, p95_max_ms: float, drop_max: float, hold_s: float, up_s: float):
        self.steps = [int(x) for x in steps]
        self.p95_max, self.drop_max, self.hold_s, self.up_s = p95_max_ms, drop_max, hold_s, up_s
        self.k = 0
        self.changed_at: float | None = None
        self.good_since: float | None = None

    @property
    def fps(self) -> int:
        return self.steps[self.k]

    def update(self, now: float, p95_ms: float | None, drop_share: float | None) -> int:
        if p95_ms is None and drop_share is None:
            return self.fps
        bad = (p95_ms is not None and p95_ms > self.p95_max) or (drop_share is not None and drop_share > self.drop_max)
        good = (p95_ms is None or p95_ms < self.p95_max / 2) and (drop_share is None or drop_share < self.drop_max / 5)
        held = self.changed_at is None or now - self.changed_at >= self.hold_s
        if bad:
            self.good_since = None
            if held and self.k < len(self.steps) - 1:
                self.k += 1
                self.changed_at = now
        elif good:
            self.good_since = now if self.good_since is None else self.good_since
            if held and self.k > 0 and now - self.good_since >= self.up_s:
                self.k -= 1
                self.changed_at = now
                self.good_since = None
        else:
            self.good_since = None
        return self.fps


RATE_WINDOW_S = 3.0       # delay p95 and drop share over this window
RATE_EVERY_S = 1.0        # the rate is checked this often


def gpu_free_mb(index: int) -> int | None:
    """Free memory of one GPU (PCI bus order, as nvidia-smi counts) or None."""
    try:
        out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        return int(out.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


class Slot:
    """One frame slot: put replaces an unprocessed frame (returns True then)."""

    def __init__(self):
        self._item = None
        self._cv = threading.Condition()

    def put(self, item) -> bool:
        with self._cv:
            replaced = self._item is not None
            self._item = item
            self._cv.notify()
            return replaced

    def take(self, timeout: float):
        with self._cv:
            if self._item is None:
                self._cv.wait(timeout)
            item, self._item = self._item, None
            return item


class LiveSession:
    def __init__(self, sid: str, cfg, backend, template_name: str, borders, folder: Path, save_recording: bool,
                 cycles: bool = False):
        self.id = sid
        self.cfg = cfg
        self.live = cfg["live"]
        self.backend = backend
        self.folder = folder
        self.started_utc = utc_now()
        self.t0 = time.monotonic()
        self.t_end: float | None = None           # set at Stop, so the elapsed time stops counting
        self.slot = Slot()
        self.frames_in = self.processed = self.dropped = self.paced = 0
        live_ = cfg["live"]
        self.rate = RateControl(live_["rate_steps"], live_["rate_p95_ms"], live_["rate_drop_max"],
                                live_["rate_hold_s"], live_["rate_up_s"])
        self.last_accept: float | None = None
        self.recent: deque = deque()              # (time, kind) of accepted frames: 'in', 'drop', 'done:<delay>'
        self._rate_check = 0.0
        self.delays: deque = deque(maxlen=600)
        self.parts: dict = {k: deque(maxlen=600) for k in ("wait", "resize", "pose", "track", "station", "draw")}
        self.done_t: deque = deque(maxlen=600)
        self.notices: deque = deque(maxlen=8)
        self.recorder = None
        self.recording_error = None
        if save_recording:
            folder.mkdir(parents=True, exist_ok=True)
            self.recorder = Recorder(folder / "recording.mp4", self.live["rec_fps"])
        self.tracker = None
        self.locked: int | None = None
        self.locked_seen = 0.0
        self.subject_id: int | None = None
        self.subject_seen: float | None = None
        self.lost_flag = False
        self.paused = False
        self.last_frame_mono: float | None = None
        self.filter = None
        self.stations: list[Station] = []
        self.borders = borders
        self.cycles = cycles
        self.lock = threading.Lock()
        self.latest_overlay: np.ndarray | None = None
        self.latest_boxes = np.zeros((0, 4))
        self.latest_ids = np.zeros(0, np.int64)
        self.status = "running"
        self.stop_reason = ""
        self._stop = threading.Event()
        self.on_auto_stop = None
        self.port = 7860
        self._no_frame_noticed = False
        self.switch_station(template_name, borders, t=0.0)
        self.thread = threading.Thread(target=self._loop, name=f"live-{sid}", daemon=True)

    # control ---------------------------------------------------------------------------

    def now_s(self) -> float:
        return elapsed_s(self.t0, time.monotonic(), self.t_end)

    def notice(self, text: str) -> None:
        self.notices.append((round(self.now_s(), 1), text))

    def switch_station(self, name: str | None, borders=None, t: float | None = None) -> None:
        """Start a new station block; name 'combined' is the one view of the Live page."""
        t = self.now_s() if t is None else t
        if name == COMBINED:
            with self.lock:
                if self.stations:
                    self.stations[-1].close(t, "switch")
                self.stations.append(CombinedStation(self.cfg, len(self.stations) + 1, t, self.cycles, borders))
                self.filter = None
            return
        template = None
        if name:
            try:
                template = load_template(name)
            except AvError as e:
                self.notice(f"{e.code}: {e.text}")
        if template is not None and borders and template.kind == "workbench":
            template = template.with_borders(*borders)
        with self.lock:
            if self.stations:
                self.stations[-1].close(t, "switch")
            self.stations.append(Station(template, self.cfg, len(self.stations) + 1, t))
            self.filter = None

    def set_borders(self, b1: float, b2: float) -> None:
        with self.lock:
            st = self.stations[-1]
            if isinstance(st, CombinedStation):
                if st.cycles_on:
                    st.set_cycles(True, (b1, b2))
                self.borders = (b1, b2)
            elif st.t is not None and st.kind == "workbench":
                st.set_template(st.t.with_borders(b1, b2))
                self.borders = (b1, b2)

    def set_cycles(self, on: bool, borders=None) -> None:
        """Cycle counting of the combined view on or off (zone borders as image x fractions)."""
        with self.lock:
            st = self.stations[-1]
            if isinstance(st, CombinedStation):
                st.set_cycles(on, borders or self.borders)
            self.cycles = on
            if borders:
                self.borders = borders

    def lock_subject(self, x: float, y: float) -> bool:
        """Lock the person whose box contains (x, y) in frame pixels (AV-LIV-009 if none)."""
        with self.lock:
            boxes, ids = self.latest_boxes, self.latest_ids
        hits = [(float((b[2] - b[0]) * (b[3] - b[1])), int(i)) for b, i in zip(boxes, ids)
                if i > 0 and b[0] <= x <= b[2] and b[1] <= y <= b[3]]
        if not hits:
            raise AvError("AV-LIV-009")
        self.locked = min(hits)[1]
        self.locked_seen = self.now_s()
        return True

    def push(self, frame: np.ndarray, t: float):
        """Never blocks: store the frame and return the latest overlay. A frame
        that comes sooner than the current rate allows is skipped (paced)."""
        if self.recorder is not None:
            self.recorder.write(frame, t)
        if self.last_accept is not None and t - self.last_accept < 0.8 / self.rate.fps:
            self.paced += 1
            return self.latest_overlay
        self.last_accept = t
        self.frames_in += 1
        dropped = self.slot.put((frame, t))
        with self.lock:
            self.recent.append((t, "in"))
            if dropped:
                self.dropped += 1
                self.recent.append((t, "drop"))
        return self.latest_overlay

    def check_rate(self, now: float) -> int:
        """Window stats to RateControl (about once per RATE_EVERY_S)."""
        if now - self._rate_check < RATE_EVERY_S:
            return self.rate.fps
        self._rate_check = now
        with self.lock:
            while self.recent and self.recent[0][0] < now - RATE_WINDOW_S:
                self.recent.popleft()
            n_in = sum(1 for _, k in self.recent if k == "in")
            n_drop = sum(1 for _, k in self.recent if k == "drop")
            delays = [d for x, d in zip(self.done_t, self.delays) if x >= now - RATE_WINDOW_S]
        if n_in < 5:
            return self.rate.fps
        p95 = 1000 * float(np.percentile(delays, 95)) if delays else None
        before = self.rate.fps
        fps = self.rate.update(now, p95, n_drop / n_in)
        if fps != before:
            self.notice(f"Frame rate {'down' if fps < before else 'up'} to {fps} fps.")
        return fps

    def stop(self, reason: str = "stopped") -> None:
        self._stop.set()
        if threading.current_thread() is not self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        self.t_end = time.monotonic()
        t = self.now_s()
        with self.lock:
            if self.stations and self.stations[-1].ended_s is None:
                self.stations[-1].close(t, "stop")
        if self.recorder is not None:
            path = self.recorder.close()
            if path is None:
                self.recording_error = self.recorder.error or "no frame was recorded"
                if self.recorder.error:
                    self.notice(f"AV-LIV-007: {AvError('AV-LIV-007', detail=self.recorder.error).text}")
        self.status = "done"
        self.stop_reason = reason

    # processing thread ---------------------------------------------------------------

    def _loop(self) -> None:
        live = self.live
        while not self._stop.is_set():
            item = self.slot.take(timeout=0.5)
            now = time.monotonic()
            if item is None:
                self._idle_checks(now)
                continue
            try:
                self._process(*item)
            except Exception as e:          # a bad frame must not end the session
                self.notice(f"frame skipped: {type(e).__name__}: {str(e)[:120]}")
            if now - self.t0 > live["max_session_s"]:
                self._auto_stop("max_session")

    def _idle_checks(self, now: float) -> None:
        live = self.live
        if self.last_frame_mono is None:
            if now - self.t0 > NO_FRAME_S and not self._no_frame_noticed:
                self._no_frame_noticed = True
                self.notice(AvError("AV-LIV-001", port=self.port).text)
            if now - self.t0 > live["idle_stop_s"]:
                self._auto_stop("no_frames")
            return
        gap = now - self.last_frame_mono
        if gap > live["lost_s"] and not self.paused:
            self.paused = True
            with self.lock:
                self.stations[-1].on_lost(self.last_frame_mono - self.t0)
            self.filter = None
            self.notice("Camera paused: no frames. Press the record button under the camera to continue.")
        if gap > live["idle_stop_s"]:
            self._auto_stop("idle")

    def _auto_stop(self, reason: str) -> None:
        self._stop.set()
        if self.on_auto_stop is not None:
            self.on_auto_stop(self, reason)

    def _process(self, frame: np.ndarray, t: float) -> None:
        live = self.live
        if self.paused:
            self.paused = False
            self.notice("Camera running again.")
        c0 = self.last_frame_mono = time.monotonic()
        ts = t - self.t0
        h, w = frame.shape[:2]
        scale = live["input_px"] / max(h, w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA) \
            if scale < 1 else frame
        c1 = time.monotonic()
        det = self.backend.predict(small)
        if scale < 1:
            det = det.scaled(1 / scale)
        c2 = time.monotonic()
        if self.tracker is None:
            self.tracker = LiveTracker(self.cfg["track"], 1 / live["stream_every_s"], live["lost_s"])
        ids = self.tracker.update(det, (w, h))
        c3 = time.monotonic()
        subj = self._choose(det, ids, ts)
        station = self.stations[-1]
        kp = ok = None
        if subj is None:
            if self.subject_seen is not None and ts - self.subject_seen > live["lost_s"] and not self.lost_flag:
                self.lost_flag = True
                with self.lock:
                    station.on_lost(self.subject_seen)
                self.notice("AV-LIV-004: No person is visible.")
        else:
            if ids[subj] != self.subject_id:
                self.subject_id = int(ids[subj])
                self.filter = None
            if self.filter is None:
                self.filter = OneEuro(live["one_euro_min_cutoff"], live["one_euro_beta"], live["one_euro_d_cutoff"],
                                      gap_s=live["lost_s"])
            self.subject_seen = ts
            self.lost_flag = False
            kconf = det.kconf[subj]
            ok = kconf >= live["kp_conf_min"]
            kp = self.filter(ts, det.kp[subj], ok)
            ok = ok & np.isfinite(kp).all(1)
            with self.lock:
                station.update(ts, kp, kconf, (w, h))
        panel = station.panel(ts)
        c4 = time.monotonic()
        banner, sub, strip = overlay_texts(panel)
        zones = station.draw_zones()
        notice = ""
        if subj is None:
            notice = "No person visible" if not len(det.conf) else "Waiting for the selected person"
        img = draw_overlay(frame, det.boxes, subj, kp, ok, zones, banner, sub, strip, panel["band"], notice, ids)
        done = time.monotonic()
        for k, v in (("wait", c0 - t), ("resize", c1 - c0), ("pose", c2 - c1), ("track", c3 - c2),
                     ("station", c4 - c3), ("draw", done - c4)):
            self.parts[k].append(v)
        with self.lock:
            self.latest_overlay = img
            self.latest_boxes, self.latest_ids = det.boxes, ids
            self.processed += 1
            self.delays.append(done - t)
            self.done_t.append(done)
        self.check_rate(done)
        fps = self.fps()
        if self.processed > 50 and fps is not None and fps < live["warn_fps"] and self.processed % 50 == 0:
            self.notice(AvError("AV-LIV-006", fps=f"{fps:.1f}").text)

    def _choose(self, det: Detections, ids: np.ndarray, ts: float) -> int | None:
        if self.locked is not None:
            hit = np.flatnonzero(ids == self.locked)
            if len(hit):
                self.locked_seen = ts
                return int(hit[0])
            if ts - self.locked_seen <= self.live["lost_s"]:
                return None
            self.locked = None                                   # unlock after lost_s
            with self.lock:
                self.stations[-1].on_lost(self.locked_seen)
            self.filter = None
            self.notice(f"AV-LIV-004: The selected person was not seen for {self.live['lost_s']:g} s, "
                        "so the open step was closed.")
        if not len(det.conf):
            return None
        area = (det.boxes[:, 2] - det.boxes[:, 0]) * (det.boxes[:, 3] - det.boxes[:, 1])
        return int(np.argmax(area))

    # numbers -------------------------------------------------------------------------

    def fps(self) -> float | None:
        with self.lock:
            done = [x for x in self.done_t if x >= time.monotonic() - STATS_S]
        if len(done) < 2:
            return None
        return (len(done) - 1) / max(done[-1] - done[0], 1e-6)

    def delay_ms(self) -> tuple[float | None, float | None]:
        with self.lock:
            d = list(self.delays)
        if not d:
            return None, None
        return round(1000 * float(np.percentile(d, 50))), round(1000 * float(np.percentile(d, 95)))

    def stage_ms(self) -> dict:
        """p50 and p95 per part of a processed frame (ms): wait = time in the slot before processing."""
        out = {}
        for k, d in self.parts.items():
            v = list(d)
            if v:
                out[k] = {"p50": round(1000 * float(np.percentile(v, 50)), 1),
                          "p95": round(1000 * float(np.percentile(v, 95)), 1)}
        return out

    def panel(self) -> dict:
        t = self.now_s()
        with self.lock:
            st = self.stations[-1]
            p = st.panel(t)
            log = st.log_rows()
            steps = st.step_rows() if isinstance(st, CombinedStation) else []
        p50, p95 = self.delay_ms()
        fps = self.fps()
        return {"session_id": self.id, "status": self.status, "elapsed_s": round(t), "station": p, "log": log,
                "steps": steps,
                "fps": None if fps is None else round(fps, 1), "delay_p50_ms": p50, "delay_p95_ms": p95,
                "frames_in": self.frames_in, "processed": self.processed, "dropped": self.dropped,
                "rate_fps": self.rate.fps, "paused": self.paused, "recording": self.recorder is not None,
                "notices": [f"{a:.0f} s: {b}" for a, b in self.notices]}

    def summary(self) -> dict:
        t = self.now_s()
        p50, p95 = self.delay_ms()
        rec = self.folder / "recording.mp4"
        with self.lock:
            stations = [s.summary(t) for s in self.stations]
        return {"session_id": self.id, "started_utc": self.started_utc, "ended_utc": utc_now(),
                "duration_s": round(t, 1), "stop_reason": self.stop_reason, "frames_in": self.frames_in,
                "processed": self.processed, "dropped": self.dropped, "paced": self.paced,
                "rate_fps_end": self.rate.fps,
                "fps_mean": round(self.processed / t, 2) if t > 0 else None, "delay_p50_ms": p50,
                "delay_p95_ms": p95, "stage_ms": self.stage_ms(), "recording": str(rec) if rec.is_file() else None,
                "recording_error": self.recording_error, "stations": stations,
                "notices": [f"{a:.0f} s: {b}" for a, b in self.notices]}


def overlay_texts(p: dict) -> tuple[str, str, str]:
    """Banner, sub line and bottom strip of the overlay for a station panel."""
    trunk = "trunk n/a" if p["trunk_deg"] is None else f"trunk {p['trunk_deg']} deg"
    if p["kind"] == COMBINED:
        ps = p["posture"]
        state = "LIFTING" if p.get("in_lift") else f"BENT {ps['current_s']:.1f} s" if ps["current_s"] > 0 \
            else "UPRIGHT"
        lifts = p.get("lifts", [])
        n_st = sum(1 for x in lifts if x["kind"] == "stoop")
        lift_txt = f"lifts {len(lifts)} (stoop {n_st}, squat {len(lifts) - n_st})" if p.get("lifts_ready") or lifts \
            else "lifts need a side view"
        idle = p.get("idle_time_s")
        sub = f"{lift_txt} | longest bent {ps['longest_s']:.1f} s | idle {'n/a' if idle is None else f'{idle:.0f} s'}"
        if p.get("cycles_on"):
            last = "n/a" if p.get("last_cycle_s") is None else f"{p['last_cycle_s']:.1f} s"
            sub = f"cycles {p.get('cycles', 0)} | last {last} | " + sub
        both = p.get("two_hand_simultaneous")
        knee = "" if p.get("knee_deg") is None else f"  knee {p['knee_deg']} deg"
        return (f"LIVE | {state}", sub, f"{trunk}{knee}  both hands {'n/a' if both is None else f'{100 * both:.0f} %'}")
    if p["kind"] == "workbench":
        step = (p.get("step") or "waiting").upper()
        step_s = f" {p['step_s']:.1f} s" if p.get("step_s") is not None else ""
        last = "n/a" if p.get("last_cycle_s") is None else f"{p['last_cycle_s']:.1f} s"
        mean = "n/a" if p.get("mean_cycle_s") is None else f"{p['mean_cycle_s']:.1f} s"
        both = p.get("two_hand_simultaneous")
        return (f"{p['title']} | {step}{step_s}", f"cycles {p.get('cycles', 0)} | last {last} | mean {mean}",
                f"{trunk}  both hands {'n/a' if both is None else f'{100 * both:.0f} %'}")
    if p["kind"] == "lift":
        lifts = p.get("lifts", [])
        state = "LIFTING" if p.get("in_lift") else f"lifts {len(lifts)}"
        sub = "no lift yet"
        if lifts:
            x = lifts[-1]
            knee = "n/a" if x["knee_deg"] is None else f"{x['knee_deg']:.0f}"
            sub = (f"last: {x['kind'].upper()} trunk {x['peak_trunk_deg']:.0f} knee {knee} "
                   f"B2+B3 {x['b2_s'] + x['b3_s']:.1f} s")
        knee = "n/a" if p.get("knee_deg") is None else f"{p['knee_deg']} deg"
        return f"{p['title']} | {state}", sub, f"{trunk}  knee {knee}"
    ps = p["posture"]
    idle = p.get("idle_time_s")
    return (f"{p['title']} | bent {ps['current_s']:.1f} s",
            f"longest {ps['longest_s']:.1f} s | idle {'n/a' if idle is None else f'{idle:.1f} s'}",
            f"{trunk}  gaze {p.get('gaze_now') or 'n/a'}")


def label_stopped(img: np.ndarray) -> np.ndarray:
    """A copy with a small grey 'Stopped' pill in the top right corner."""
    out = np.ascontiguousarray(img.copy())
    h, w = out.shape[:2]
    u = max(1.0, h / 480)
    (tw, th), _ = cv2.getTextSize("Stopped", cv2.FONT_HERSHEY_DUPLEX, 0.6 * u, 1)
    x1, y1 = w - tw - int(30 * u), int(72 * u)
    cv2.rectangle(out, (x1, y1), (x1 + tw + int(18 * u), y1 + th + int(14 * u)), (110, 110, 110), -1)
    cv2.putText(out, "Stopped", (x1 + int(9 * u), y1 + th + int(7 * u)), cv2.FONT_HERSHEY_DUPLEX, 0.6 * u,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


class LiveManager:
    """One per interface process; at most one session at a time."""

    def __init__(self, cfg, live_root: Path, backend_factory=None, gpu_free=gpu_free_mb,
                 lock_dir: Path = REPO_ROOT / ".locks", port: int = 7860):
        self.cfg = cfg
        self.port = port
        self.live = cfg["live"]
        self.root = Path(live_root)
        self.backend_factory = backend_factory or self._yolo
        self.gpu_free = gpu_free
        self.lock_dir = Path(lock_dir)
        self.session: LiveSession | None = None
        self.last_summary: dict | None = None
        self.cleared = False                       # Clear pressed since the last Stop
        self._backend = None
        self._gpu_lock = None
        self._release_timer: threading.Timer | None = None
        self._mutex = threading.Lock()

    def _yolo(self):
        from .perception import YoloPose
        path = (REPO_ROOT / self.cfg["paths"]["models_dir"] / self.live["pose_model"])
        if not path.is_file():
            raise AvError("AV-POS-001", name=path.name, detail=f"expected file {path}")
        return YoloPose(path, "cuda:0", self.live["half"], self.live["input_px"], self.live["det_conf"])

    def _take_lock(self) -> None:
        self.lock_dir.mkdir(exist_ok=True)
        f = open(self.lock_dir / f"gpu{self.live['gpu']}.lock", "a+")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            f.close()
            raise AvError("AV-LIV-002", detail=f"GPU {self.live['gpu']} is used by another live session or job")
        self._gpu_lock = f

    def _release_lock(self) -> None:
        if self._gpu_lock is not None:
            self._gpu_lock.close()
            self._gpu_lock = None

    def start(self, template: str = COMBINED, save_recording: bool = False, borders=None,
              cycles: bool = False) -> str:
        with self._mutex:
            if self.session is not None and self.session.status == "running":
                raise AvError("AV-LIV-002")
            self._take_lock()
            try:
                if self._release_timer is not None:
                    self._release_timer.cancel()
                if self._backend is None:
                    free = self.gpu_free(self.live["gpu"])
                    if free is not None and free < self.live["min_free_mb"]:
                        raise AvError("AV-LIV-005", free=free, need=self.live["min_free_mb"])
                    self._backend = self.backend_factory()
                sid = new_session_id()
                s = LiveSession(sid, self.cfg, self._backend, template, borders, self.root / sid, save_recording,
                                cycles)
            except BaseException:
                self._release_lock()
                raise
            s.port = self.port
            s.on_auto_stop = lambda sess, reason: threading.Thread(target=self._stop_session,
                                                                   args=(sess, reason), daemon=True).start()
            self.session = s
            self.last_summary = None
            self.cleared = False
            s.thread.start()
            return sid

    def push(self, frame: np.ndarray | None, t: float | None = None):
        s = self.session
        if s is None or s.status != "running" or frame is None:
            return None
        return s.push(frame, time.monotonic() if t is None else t)

    def stop(self) -> dict | None:
        s = self.session
        if s is None or s.status != "running":
            return self.last_summary
        return self._stop_session(s, "stopped")

    def _stop_session(self, s: LiveSession, reason: str) -> dict:
        with self._mutex:
            if s.status != "running":
                return self.last_summary
            s.stop(reason)
            summary = s.summary()
            s.folder.mkdir(parents=True, exist_ok=True)
            tmp = s.folder / "summary.json.tmp"
            tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
            tmp.replace(s.folder / "summary.json")
            self.last_summary = summary
            self._release_lock()
            keep = self.live["model_keep_s"]
            self._release_timer = threading.Timer(keep, self._release_model)
            self._release_timer.daemon = True
            self._release_timer.start()
            return summary

    def _release_model(self) -> None:
        with self._mutex:
            if (self.session is None or self.session.status != "running") and self._backend is not None:
                if hasattr(self._backend, "release"):
                    self._backend.release()
                self._backend = None

    def panel(self) -> dict | None:
        s = self.session
        return None if s is None else s.panel()

    def clear(self) -> bool:
        """Forget the stopped session on the page (panel, overlay, summary); the
        summary.json and a recording stay on disk and Analyse recording still
        works. False while a session runs."""
        s = self.session
        if s is not None and s.status == "running":
            return False
        self.session = None
        self.cleared = True
        return True

    def stopped_overlay(self) -> np.ndarray | None:
        """The last overlay of the stopped session with a small grey Stopped label."""
        s = self.session
        if s is None or s.latest_overlay is None:
            return None
        return label_stopped(s.latest_overlay)

    def to_batch(self, session_id: str, variant: str, run_manager) -> str:
        """Start a batch run on the session recording (GPU of batch pose); AV-LIV-008 without one."""
        import re
        if not re.fullmatch(r"live_\d{8}_\d{6}_[0-9a-f]{4}", session_id or ""):
            raise AvError("AV-LIV-008", detail="unknown session")
        rec = self.root / session_id / "recording.mp4"
        if not rec.is_file():
            raise AvError("AV-LIV-008")
        run_id, _ = run_manager.create_run(rec, "public", variant, overrides={"steps": {"use": False}})
        run_manager.start(run_id)
        summ = self.root / session_id / "summary.json"
        if summ.is_file():
            d = json.loads(summ.read_text())
            d.setdefault("batch_runs", []).append(run_id)
            summ.write_text(json.dumps(d, indent=2, ensure_ascii=False, default=str))
        return run_id
