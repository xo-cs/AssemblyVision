"""Readouts drawn on the batch overlay next to the header (bottom left): the
mini hand reach map of the last 10 s above it and the both hands chip to its
right; the trunk readout sits bottom right (overlay.draw_trunk).

Everything here runs once per video frame, so it is OpenCV and NumPy only: a
48 x 48 count grid updated incrementally (add the new frame, remove the frame
that left the window), cv2.GaussianBlur, and a colour lookup table built once
from matplotlib (inferno reversed, the colormap of the Results chart).
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

MINI_EXTENT = (-1.75, 1.75, -1.25, 2.25)     # x0, x1, y0, y1 in torso lengths: square, mid hip at 0, 0
DOT_BGR = (0xEE, 0xD3, 0x22)                 # cyan #22D3EE
DOT_EDGE = (40, 40, 40)
FIG_BGR = (165, 165, 165)
NA_GREY = (150, 150, 150)
_LUT: np.ndarray | None = None


def inferno_lut() -> np.ndarray:
    """[256, 3] uint8 BGR of matplotlib inferno_r (built once)."""
    global _LUT
    if _LUT is None:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import colormaps
        rgb = colormaps["inferno_r"](np.linspace(0.0, 1.0, 256))[:, :3]
        _LUT = np.ascontiguousarray((rgb[:, ::-1] * 255).round().astype(np.uint8))
    return _LUT


_FADED: np.ndarray | None = None


def faded_lut(fade: float = 0.12) -> np.ndarray:
    """[256, 1, 3] uint8 BGR table for cv2.applyColorMap: inferno_r, with values
    below `fade` (of the maximum) blended into white, so empty is white."""
    global _FADED
    if _FADED is None:
        a = np.clip(np.arange(256) / 255.0 / fade, 0.0, 1.0)[:, None]
        _FADED = np.ascontiguousarray((255 * (1 - a) + inferno_lut() * a).round().astype(np.uint8)[:, None, :])
    return _FADED


def boxes_overlap(a, b, gap: int = 0) -> bool:
    return a[0] < b[2] + gap and b[0] < a[2] + gap and a[1] < b[3] + gap and b[1] < a[3] + gap


# rolling window ------------------------------------------------------------------------

class RollingGrid:
    """Counts of body frame points on an n x n grid over the last window_s seconds
    (frame times t; batch passes frame / fps, live the frame timestamps, so a
    changing live frame rate is fine). Row 0 is the top (highest y). push() adds
    one frame and removes the frames that left the window: O(points) per frame."""

    def __init__(self, window_s: float, n: int = 48, extent=MINI_EXTENT):
        self.window_s, self.n, self.extent = float(window_s), n, extent
        self.grid = np.zeros((n, n), np.float32)
        self.q: deque = deque()
        self.points = 0                                  # points inside the window

    def cells(self, pts: np.ndarray) -> list[tuple[int, int]]:
        x0, x1, y0, y1 = self.extent
        out = []
        for x, y in np.asarray(pts, np.float64).reshape(-1, 2):
            if not (math.isfinite(x) and math.isfinite(y)) or not (x0 <= x < x1 and y0 < y <= y1):
                continue
            out.append((min(self.n - 1, int((y1 - y) / (y1 - y0) * self.n)),
                        min(self.n - 1, int((x - x0) / (x1 - x0) * self.n))))
        return out

    def push(self, pts: np.ndarray, t: float) -> None:
        c = self.cells(pts)
        for r, k in c:
            self.grid[r, k] += 1
        self.q.append((float(t), c))
        self.points += len(c)
        while self.q and self.q[0][0] <= t - self.window_s:
            for r, k in self.q.popleft()[1]:
                self.grid[r, k] -= 1
                self.points -= 1

    def density(self, sigma: float = 1.5) -> np.ndarray | None:
        """Smoothed counts normalised to max 1, None when the window is empty."""
        if self.points <= 0:
            return None
        d = cv2.GaussianBlur(self.grid, (0, 0), sigma)
        m = float(d.max())
        return d / m if m > 0 else None


# mini map --------------------------------------------------------------------------------

def mini_size(oh: int) -> tuple[int, int, int, int]:
    """(panel width, panel height, map side, padding) for a frame of height oh:
    a 140 px map at 480 lines, scaled with the frame."""
    s = max(0.6, oh / 480)
    side, pad, title = round(140 * s), round(6 * s), round(16 * s)
    return side + 2 * pad, title + side + 2 * pad, side, pad


def dot_px(pt, map_box) -> tuple[float, float]:
    """Body frame point (torso lengths) -> pixel inside the map square map_box (x0, y0, x1, y1)."""
    x0, x1, y0, y1 = MINI_EXTENT
    mx0, my0, mx1, my1 = map_box
    return (mx0 + (pt[0] - x0) / (x1 - x0) * (mx1 - mx0), my0 + (y1 - pt[1]) / (y1 - y0) * (my1 - my0))


class MiniReach:
    """The rolling hand reach map of one overlay video or live session.
    push() every frame, in time order, with the frame time in seconds."""

    def __init__(self, skeleton: dict | None = None, window_s: float = 10.0, fade_s: float = 0.5):
        self.grid = RollingGrid(window_s)
        self.skeleton = skeleton or {}
        self.fade_s = fade_s
        self.last: list = [None, None]          # last valid point per wrist
        self.last_t = [-math.inf, -math.inf]    # its time
        self.t = 0.0
        self._mask: dict = {}

    def push(self, pts: np.ndarray, t: float) -> None:
        pts = np.asarray(pts, np.float64).reshape(2, 2)
        self.t = float(t)
        self.grid.push(pts, t)
        for k in (0, 1):
            if np.isfinite(pts[k]).all():
                self.last[k], self.last_t[k] = pts[k].copy(), self.t

    def dot_alpha(self, k: int) -> float:
        """1 while the wrist is seen, fading to 0 over fade_s after it was lost."""
        if self.last[k] is None:
            return 0.0
        return max(0.0, 1.0 - (self.t - self.last_t[k]) / self.fade_s)

    def _panel_mask(self, w: int, h: int, r: int) -> np.ndarray:
        """bool [h, w, 1]: inside the rounded panel (built once per size)."""
        key = (w, h, r)
        if key not in self._mask:
            m = np.zeros((h, w), np.uint8)
            cv2.rectangle(m, (r, 0), (w - 1 - r, h - 1), 255, -1)
            cv2.rectangle(m, (0, r), (w - 1, h - 1 - r), 255, -1)
            for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
                cv2.circle(m, (cx, cy), r, 255, -1)
            self._mask[key] = (m > 0)[..., None]
        return self._mask[key]

    def draw(self, img: np.ndarray, box: tuple[int, int, int, int], opacity: float = 0.88) -> None:
        x0, y0, x1, y1 = box
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        w, h = x1 - x0, y1 - y0
        if w < 20 or h < 20:
            return
        s = max(0.6, H / 480)
        pad, title = round(6 * s), round(16 * s)
        side = min(w - 2 * pad, h - 2 * pad - title)
        panel = np.full((h, w, 3), 255, np.uint8)
        mx0, my0 = pad + (w - 2 * pad - side) // 2, pad + title
        mbox = (mx0, my0, mx0 + side, my0 + side)
        d = self.grid.density()
        if d is not None:
            big = cv2.resize((d * 255).astype(np.uint8), (side, side), interpolation=cv2.INTER_LINEAR)
            panel[my0:my0 + side, mx0:mx0 + side] = cv2.applyColorMap(big, faded_lut())
        cv2.rectangle(panel, (mx0, my0), (mx0 + side - 1, my0 + side - 1), (225, 225, 225), 1)
        self._figure(panel, mbox, s)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(panel, "Hand reach, last 10 s", (pad, pad + title - round(5 * s)), font, 0.38 * s, (60, 60, 60),
                    1, cv2.LINE_AA)
        if d is None:
            (tw, th), _ = cv2.getTextSize("n/a", font, 0.5 * s, 1)
            cv2.putText(panel, "n/a", (mx0 + (side - tw) // 2, my0 + (side + th) // 2), font, 0.5 * s, NA_GREY, 1,
                        cv2.LINE_AA)
        r = max(3, round(4 * s))
        for k in (0, 1):
            al = self.dot_alpha(k)
            if al <= 0:
                continue
            px, py = dot_px(self.last[k], mbox)
            c = tuple(int(255 * (1 - al) + v * al) for v in DOT_BGR)
            e = tuple(int(255 * (1 - al) + v * al) for v in DOT_EDGE)
            cv2.circle(panel, (int(round(px)), int(round(py))), r + 1, e, -1, cv2.LINE_AA)
            cv2.circle(panel, (int(round(px)), int(round(py))), r, c, -1, cv2.LINE_AA)
        roi = img[y0:y1, x0:x1]
        mixed = cv2.addWeighted(roi, 1.0 - opacity, panel, opacity, 0.0)
        np.copyto(roi, mixed, where=self._panel_mask(w, h, round(8 * s)))

    def _figure(self, panel: np.ndarray, mbox, s: float) -> None:
        """Faint grey reference figure: shoulders at y 1, hips at y 0, head (analysis.reach)."""
        sk = self.skeleton
        if "sho_half" not in sk or "hip_half" not in sk:
            return
        t = max(1, round(1.5 * s))

        def p(x, y):
            u, v = dot_px((x, y), mbox)
            return int(round(u)), int(round(v))
        sw, hw = float(sk["sho_half"][0]), float(sk["hip_half"][0])
        cv2.line(panel, p(-sw, 1.0), p(sw, 1.0), FIG_BGR, t, cv2.LINE_AA)
        cv2.line(panel, p(-hw, 0.0), p(hw, 0.0), FIG_BGR, t, cv2.LINE_AA)
        cv2.line(panel, p(0.0, 0.0), p(0.0, 1.0), FIG_BGR, t, cv2.LINE_AA)
        if "head" in sk:
            hd = sk["head"]
            cv2.line(panel, p(0.0, 1.0), p(hd[0], hd[1] - 0.2), FIG_BGR, t, cv2.LINE_AA)
            rr = int(round(0.2 / (MINI_EXTENT[1] - MINI_EXTENT[0]) * (mbox[2] - mbox[0])))
            cv2.circle(panel, p(hd[0], hd[1]), max(2, rr), FIG_BGR, t, cv2.LINE_AA)


# both hands chip -------------------------------------------------------------------------

def both_text(share: float | None) -> tuple[str, bool]:
    """('Both hands 57 %', True), or ('Both hands n/a', False) without a value."""
    if share is None or not math.isfinite(share):
        return "Both hands n/a", False
    return f"Both hands {int(round(100 * share))} %", True


def chip_size(text: str, scale: float, margin: int = 8) -> tuple[int, int]:
    """Width and height of a chip in the style of overlay.draw_header."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    return tw + 2 * margin, th + base + 2 * margin


def draw_chip(img: np.ndarray, text: str, box, scale: float, ok: bool = True, alpha: float = 0.55,
              margin: int = 8) -> None:
    """Text on a semi transparent black box like the header; grey text when not ok."""
    x0, y0, x1, y1 = (int(v) for v in box)
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    (_, _), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.putText(img, text, (x0 + margin, y1 - margin - base), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255) if ok else NA_GREY, 1, cv2.LINE_AA)


# layout ------------------------------------------------------------------------------------

def hud_layout(w: int, h: int, header, chip_wh, trunk_wh, map_wh, margin: int = 8) -> dict:
    """Boxes (x0, y0, x1, y1) of everything at the bottom of a w x h frame, none
    overlapping: header bottom left (given), trunk bottom right (above the header
    when they would meet), mini map directly above the header, chip right of
    the header, or on the line above it next to the map when that row is full."""
    m = margin
    hx0, hy0, hx1, hy1 = header
    tw, th = trunk_wh
    trunk = (w - m - tw, h - m - th, w - m, h - m)
    if boxes_overlap(trunk, header, m):
        trunk = (trunk[0], hy0 - m - th, trunk[2], hy0 - m)
    mw, mh = map_wh
    mp = (m, max(0, hy0 - m - mh), m + mw, hy0 - m)
    cw, ch = chip_wh
    chip = (hx1 + m, hy0, hx1 + m + cw, hy0 + ch)
    if chip[2] > w - m or boxes_overlap(chip, trunk, m):
        chip = (mp[2] + m, hy0 - m - ch, mp[2] + m + cw, hy0 - m)
        if boxes_overlap(chip, trunk, m):
            trunk = (trunk[0], chip[1] - m - th, trunk[2], chip[1] - m)
    return {"header": tuple(header), "trunk": trunk, "map": mp, "chip": chip}
