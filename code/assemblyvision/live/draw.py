"""Overlay of a live frame, black, white and greys: skeleton, people boxes,
workbench zones, a top banner with the station and the current state, a bottom
strip with the timers and a B0 to B3 bar with the current trunk band."""
from __future__ import annotations

import cv2
import numpy as np

SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (0, 5), (0, 6)]
FONT = cv2.FONT_HERSHEY_DUPLEX
WHITE, BLACK, GREY, LIGHT = (255, 255, 255), (0, 0, 0), (128, 128, 128), (200, 200, 200)


def _text(img, txt, org, scale, color=WHITE, thick=2, outline=True):
    if outline:
        cv2.putText(img, txt, org, FONT, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(img, txt, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, boxes: np.ndarray, subject: int | None, kp: np.ndarray | None,
                 ok: np.ndarray | None, zones: dict | None, banner: str, sub: str, strip: str,
                 band: int | None, notice: str = "") -> np.ndarray:
    img = np.ascontiguousarray(frame.copy())
    h, w = img.shape[:2]
    u = max(1.0, h / 480)                                   # scale for line widths and text
    for name, z in (zones or {}).items():
        x0, x1 = int(z[0] * w), int(z[2] * w)
        cv2.rectangle(img, (x0, int(z[1] * h)), (x1 - 1, int(z[3] * h) - 1), LIGHT, max(1, int(u)))
        _text(img, name.upper(), (x0 + int(8 * u), int(h - 70 * u)), 0.6 * u, LIGHT, 1)
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = (int(v) for v in b)
        if i == subject:
            cv2.rectangle(img, (x1, y1), (x2, y2), BLACK, int(5 * u))
            cv2.rectangle(img, (x1, y1), (x2, y2), WHITE, int(2 * u))
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), GREY, max(1, int(u)))
    if kp is not None and ok is not None:
        for a, b in SKELETON:
            if ok[a] and ok[b]:
                pa, pb = tuple(int(v) for v in kp[a]), tuple(int(v) for v in kp[b])
                cv2.line(img, pa, pb, BLACK, int(6 * u), cv2.LINE_AA)
                cv2.line(img, pa, pb, WHITE, int(3 * u), cv2.LINE_AA)
        for j in np.flatnonzero(ok):
            cv2.circle(img, tuple(int(v) for v in kp[j]), int(4 * u), WHITE, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(int(v) for v in kp[j]), int(4 * u), BLACK, 1, cv2.LINE_AA)
    # top banner
    bh = int(62 * u)
    cv2.rectangle(img, (0, 0), (w, bh), BLACK, -1)
    _text(img, banner, (int(10 * u), int(30 * u)), 0.9 * u, WHITE, 2, outline=False)
    _text(img, sub, (int(10 * u), int(54 * u)), 0.6 * u, LIGHT, 1, outline=False)
    # bottom strip with the band bar on the right
    sh = int(44 * u)
    cv2.rectangle(img, (0, h - sh), (w, h), BLACK, -1)
    _text(img, strip, (int(10 * u), h - int(14 * u)), 0.7 * u, WHITE, 2, outline=False)
    cw, ch = int(46 * u), int(28 * u)
    x = w - 4 * cw - int(10 * u)
    for b in range(4):
        x0, y0 = x + b * cw, h - sh + (sh - ch) // 2
        filled = band == b
        cv2.rectangle(img, (x0, y0), (x0 + cw - 4, y0 + ch), WHITE if filled else BLACK, -1)
        cv2.rectangle(img, (x0, y0), (x0 + cw - 4, y0 + ch), WHITE, max(1, int(u)))
        _text(img, f"B{b}", (x0 + int(8 * u), y0 + ch - int(7 * u)), 0.6 * u, BLACK if filled else WHITE, 1,
              outline=False)
    if notice:
        _text(img, notice, (int(10 * u), bh + int(30 * u)), 0.7 * u, WHITE, 2)
    return img
