"""Synthetic subject skeletons with known trunk angles, paths and wrist motion."""
from __future__ import annotations

import numpy as np

from assemblyvision.config import Config
from assemblyvision.kts import empty_series
from assemblyvision.types import Shot

FPS = 30.0


def pose_series(T, angle=0.0, hip=(320.0, 300.0), torso=100.0, wrists=None, sway=None, shot_start=0,
                run_id="run", shot_id="run-s001"):
    """COCO 17 subject. angle: trunk angle (deg) as a number or [T]; hip: mid hip
    (2,) or [T,2]; torso: length or [T]; wrists: [T,2,2] absolute left/right
    wrist or None (hanging below the shoulders); sway: [T] x offset added to
    every keypoint after the wrists are placed."""
    s = empty_series(run_id, shot_id, f"{shot_id}-t01", FPS, 640, 480, shot_start, T)
    a = np.radians(np.broadcast_to(np.asarray(angle, np.float64), (T,)))
    L = np.broadcast_to(np.asarray(torso, np.float64), (T,))
    h = np.broadcast_to(np.asarray(hip, np.float64), (T, 2)).copy()
    u = np.stack([np.sin(a), -np.cos(a)], 1)          # up the trunk
    n = np.stack([np.cos(a), np.sin(a)], 1)           # across the trunk
    sho = h + L[:, None] * u
    kp = np.zeros((T, 17, 2))
    kp[:, 0] = sho + 35 * u                            # nose
    kp[:, 1], kp[:, 2] = sho + 42 * u + 8 * n, sho + 42 * u - 8 * n     # eyes
    kp[:, 3], kp[:, 4] = sho + 30 * u + 15 * n, sho + 30 * u - 15 * n   # ears
    kp[:, 5], kp[:, 6] = sho + 20 * n, sho - 20 * n
    kp[:, 7], kp[:, 8] = kp[:, 5] + [0, 50], kp[:, 6] + [0, 50]
    kp[:, 9], kp[:, 10] = kp[:, 5] + [0, 100], kp[:, 6] + [0, 100]
    if wrists is not None:
        kp[:, 9], kp[:, 10] = wrists[:, 0], wrists[:, 1]
    kp[:, 11], kp[:, 12] = h + 15 * n, h - 15 * n
    kp[:, 13], kp[:, 14] = kp[:, 11] + [0, 90], kp[:, 12] + [0, 90]
    kp[:, 15], kp[:, 16] = kp[:, 11] + [0, 180], kp[:, 12] + [0, 180]
    if sway is not None:
        kp[:, :, 0] += np.asarray(sway)[:, None]
    s.present[:] = True
    s.body_xy[:] = kp
    s.body_conf[:] = 0.9
    s.bbox_xyxy[:] = [200, 100, 440, 480]
    return s


def shot_for(series, shot_type="fixed", scale=None):
    a = int(series.frame_idx[0])
    b = a + len(series.present)
    return Shot(series.shot_id, series.run_id, a, b, a / FPS, b / FPS, shot_type, scale)


def cfg(**motion):
    return Config.load(overrides={"motion": motion} if motion else None)


def t_axis(T):
    return np.arange(T) / FPS


def default_wrists(T, hip=(320.0, 300.0)):
    """Wrist positions of an upright subject at rest, [T,2,2]."""
    h = np.asarray(hip, np.float64)
    return np.broadcast_to(np.array([h + [20, 0], h + [-20, 0]]), (T, 2, 2)).copy()
