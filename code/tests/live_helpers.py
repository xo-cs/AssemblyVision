"""Synthetic live keypoint streams: front view workbench wrists and a side view
body with a given trunk angle and knee angle, at about 10 fps with jitter."""
from __future__ import annotations

import numpy as np

from assemblyvision.config import Config

W, H = 640, 480


def cfg(**live):
    return Config.load(overrides={"live": live} if live else None)


def times(duration_s: float, fps: float = 10.0, jitter: float = 0.02, drop: float = 0.0, seed: int = 0):
    """Irregular frame times: about fps, +-jitter s, a share drop of frames left out."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / fps) + rng.uniform(-jitter, jitter, int(np.ceil(duration_s * fps)))[
        : len(np.arange(0, duration_s, 1 / fps))]
    t = np.clip(np.sort(t), 0, None)
    return t[rng.random(len(t)) >= drop]


def side_body(trunk_deg: float, knee_deg: float = 180.0, x: float = 300.0, floor: float = 440.0,
              leg: float = 90.0, torso: float = 110.0) -> tuple[np.ndarray, np.ndarray]:
    """Side view skeleton facing image right: ankle on the floor, shank vertical,
    thigh at knee_deg to the shank (180 = straight), trunk leaning trunk_deg forward."""
    kp = np.zeros((17, 2))
    ank = np.array([x, floor])
    knee = ank + [0, -leg]
    th = np.radians(knee_deg)
    hip = knee + leg * np.array([-np.sin(th), np.cos(th)])          # 180 deg: straight above the knee
    a = np.radians(trunk_deg)
    sho = hip + torso * np.array([np.sin(a), -np.cos(a)])
    head = sho + 35 * np.array([np.sin(a), -np.cos(a)])
    for l, r, p in ((5, 6, sho), (11, 12, hip), (13, 14, knee), (15, 16, ank)):
        kp[l], kp[r] = p + [-2, 0], p + [2, 0]
    kp[7], kp[8] = sho + [20, 40], sho + [22, 40]
    kp[9], kp[10] = sho + [30, 80], sho + [32, 80]
    kp[0] = head + [12, 0]
    kp[1], kp[2] = head + [6, -5], head + [6, -5]
    kp[3], kp[4] = head + [-6, -2], head + [-6, -2]
    conf = np.full(17, 0.9)
    conf[[1, 3]] = 0.2                                               # far eye and ear hidden in a side view
    return kp, conf


def front_body(lw: tuple[float, float], rw: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Front view upper body at the image centre with the wrists at lw and rw."""
    kp = np.zeros((17, 2))
    kp[0] = [320, 120]
    kp[1], kp[2], kp[3], kp[4] = [312, 112], [328, 112], [300, 118], [340, 118]
    kp[5], kp[6] = [280, 180], [360, 180]
    kp[7], kp[8] = [270, 250], [370, 250]
    kp[9], kp[10] = lw, rw
    kp[11], kp[12] = [295, 300], [345, 300]
    kp[13], kp[14] = [295, 390], [345, 390]
    kp[15], kp[16] = [295, 470], [345, 470]
    return kp, np.full(17, 0.9)


def workbench_script(cycles: int, t0: float = 1.0, pick: float = 1.0, assemble: float = 3.0, place: float = 1.0,
                     gap: float = 1.0):
    """Wrist positions over time for complete cycles; returns (fn(t) -> kp, conf, boundaries).
    Rest: both wrists in the centre. Pick: left wrist to the left zone; assemble: both
    in the centre; place: right wrist to the right zone and back."""
    L, C, R = 100.0, 320.0, 560.0
    bounds = []
    t = t0
    for _ in range(cycles):
        bounds.append({"pick": (t, t + pick), "assemble": (t + pick, t + pick + assemble),
                       "place": (t + pick + assemble, t + pick + assemble + place)})
        t += pick + assemble + place + gap

    def at(tt: float):
        lw, rw = (C - 30, 280.0), (C + 30, 280.0)
        for b in bounds:
            if b["pick"][0] <= tt < b["pick"][1]:
                lw = (L, 280.0)
            elif b["place"][0] <= tt < b["place"][1]:
                rw = (R, 280.0)
        return front_body(lw, rw)
    return at, bounds, t
