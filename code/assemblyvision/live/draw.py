"""Overlay of a live frame: skeleton and people boxes on the picture (the
subject in a thick white box with an S label, other people in a thin bright
red box with their track number, so they are easy to tell apart and to click),
workbench zones, and at the bottom exactly the readouts of the batch overlay,
drawn by the same code (output.overlay.draw_bottom): the header with LIVE, the
state and the elapsed time, the mini hand reach map of the last 10 s, the both
hands chip and the trunk readout. The top of the picture stays free.

LiveHud keeps the rolling windows of those readouts. They are time based (the
frame timestamps), so they stay right when the adaptive frame rate changes."""
from __future__ import annotations

import cv2
import numpy as np

from ..analysis.motion import BAND_COLORS, BothHandsWindow
from ..analysis.reach import frame_points, reach_samples
from ..output.hud import MiniReach
from ..output.overlay import draw_bottom

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (0, 5), (0, 6)]
FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, BLACK, LIGHT = (255, 255, 255), (0, 0, 0), (200, 200, 200)
OTHER_RGB = (255, 59, 48)              # #FF3B30, people who are not the subject (frames are RGB)
BAND_RGB = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in BAND_COLORS]   # live frames are RGB


def _text(img, txt, org, scale, color=WHITE, thick=2, outline=True):
    if outline:
        cv2.putText(img, txt, org, FONT, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def _tag(img, txt, x, y, u, bg, fg) -> None:
    """A small filled label on the top left corner of a box."""
    scale = 0.6 * u
    (tw, th), _ = cv2.getTextSize(txt, FONT, scale, 1)
    pad = int(5 * u)
    y0 = max(y - th - 2 * pad, 0)
    cv2.rectangle(img, (x, y0), (x + tw + 2 * pad, y0 + th + 2 * pad), bg, -1)
    cv2.putText(img, txt, (x + pad, y0 + th + pad), FONT, scale, fg, 1, cv2.LINE_AA)


class LiveHud:
    """Rolling 10 s state of the bottom readouts for one live session: the mini
    hand reach map (same body frame and confidence rule as the batch map,
    clean.body_conf_min), the both hands share (motion.BothHandsWindow with the
    live per frame hand activity), and the reference figure, a running mean
    over the session (batch: over the run)."""

    def __init__(self, cfg, window_s: float = 10.0):
        self.mini = MiniReach(None, window_s)
        self.both = BothHandsWindow(window_s)
        self.conf_min = cfg["clean"]["body_conf_min"]
        self.edges = cfg["motion"]["angle_bands"]
        self._sk: dict[str, list] = {}
        self.layout: dict | None = None          # boxes of the last drawn frame (draw_bottom)

    def push(self, t: float, kp: np.ndarray | None, conf: np.ndarray | None,
             active: tuple[bool, bool] | None) -> None:
        """One processed frame at time t (s): the subject's smoothed keypoints and
        their confidence (None without a subject), its hand activity or None."""
        pts = np.full((2, 2), np.nan)
        if kp is not None and conf is not None:
            kp = np.asarray(kp, np.float64)[None]
            meas = ((np.nan_to_num(np.asarray(conf, np.float64)) >= self.conf_min) & np.isfinite(kp[0]).all(1))[None]
            pts = frame_points(kp, meas)[0]
            for name, (sm, n) in reach_samples(kp, meas)[2].items():
                acc = self._sk.setdefault(name, [np.zeros(2), 0])
                acc[0] = acc[0] + sm
                acc[1] += n
            self.mini.skeleton = {k: v[0] / v[1] for k, v in self._sk.items() if v[1]}
        self.mini.push(pts, t)
        self.both.push(t, active)


def live_header(state: str, elapsed_s: float) -> str:
    s = max(0, int(elapsed_s))
    return f"LIVE  {state}  {s // 60}:{s % 60:02d}"


def draw_overlay(frame: np.ndarray, boxes: np.ndarray, subject: int | None, kp: np.ndarray | None,
                 ok: np.ndarray | None, zones: dict | None, header: str, hud: LiveHud | None = None,
                 trunk_deg: float | None = None, ids: np.ndarray | None = None) -> np.ndarray:
    """RGB frame in, RGB overlay out; hud must already hold this frame (LiveHud.push)."""
    img = np.ascontiguousarray(frame.copy())
    h, w = img.shape[:2]
    u = max(1.0, h / 480)                                   # scale for line widths and text
    for name, z in (zones or {}).items():
        x0, x1 = int(z[0] * w), int(z[2] * w)
        cv2.rectangle(img, (x0, int(z[1] * h)), (x1 - 1, int(z[3] * h) - 1), LIGHT, max(1, int(u)))
        _text(img, name.upper(), (x0 + int(8 * u), int(28 * u)), 0.6 * u, LIGHT, 1)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in b)
        if i == subject:
            cv2.rectangle(img, (x1, y1), (x2, y2), BLACK, int(6 * u))
            cv2.rectangle(img, (x1, y1), (x2, y2), WHITE, int(3 * u))
            _tag(img, "S", x1, y1, u, WHITE, BLACK)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), OTHER_RGB, max(2, int(2 * u)))
            if ids is not None and i < len(ids) and ids[i] > 0:
                _tag(img, str(int(ids[i])), x1, y1, u, OTHER_RGB, WHITE)
    if kp is not None and ok is not None:
        for a, b in SKELETON:
            if ok[a] and ok[b]:
                pa, pb = tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b])
                cv2.line(img, pa, pb, BLACK, int(6 * u), cv2.LINE_AA)
                cv2.line(img, pa, pb, WHITE, int(3 * u), cv2.LINE_AA)
        for j in np.flatnonzero(ok):
            cv2.circle(img, tuple(int(v) for v in kp[j]), int(4 * u), WHITE, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(int(v) for v in kp[j]), int(4 * u), BLACK, 1, cv2.LINE_AA)
    if hud is not None:
        hud.layout = draw_bottom(img, header, hud.mini, hud.both.share(), trunk_deg, hud.edges, rgb=True)
    return img
