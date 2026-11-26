"""Tool pickup and object placement moments (Week 12), EXPERIMENTAL: no ground truth yet.

The simplest honest version, from the subject's wrists and a general object detector (COCO classes, objects.npz of
tools/detect_objects.py). A candidate is a reach that ends: the wrist speed relative to the hip was high, then
stays low for at least STOP_S, with the wrist on or next to a detected portable object. What happens next names it:
- pickup: the hand moves away and the object moves with it;
- placement: the hand moves away and the object stays;
- anything else (the hand stays, the object is not seen again) is not counted.
The time of an event is the reach: from where the wrist speed last rose above the active level to the stop.

Limits, stated wherever the numbers are shown: COCO has almost no workshop tools (no wrench, hammer or drill), so most
real pickups are invisible to it and a detected "object" can be the wrong thing; 2D image distance only; detections at
2 per second; an occluded hand gives no event. Reads and writes no file.
"""
from __future__ import annotations

import numpy as np

# COCO classes a hand can carry (people, vehicles, furniture and animals are left out)
PORTABLE = ("backpack", "umbrella", "handbag", "tie", "suitcase", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "laptop", "mouse", "remote", "keyboard", "cell phone", "book", "scissors", "hair drier", "toothbrush")
STOP_S = 0.3                     # the wrist stays slow this long
REACH_FACTOR = 3.0               # a reach peaks at this many times the active level
REACH_MAX_S = 3.0                # the longest reach looked at
NEAR_TORSO = 0.25                # the wrist is on the object's box grown by this many torso lengths
AFTER_S = 3.0                    # the hand must leave within this time after the stop
LEFT_TORSO = 0.5                 # the hand has moved away by at least this
STAYED_TORSO = 0.25              # an object that moved less than this stayed
MERGE_S = 1.0                    # one event per kind and hand in this time


def _near(box: np.ndarray, p: np.ndarray, pad: float) -> bool:
    return bool(box[0] - pad <= p[0] <= box[2] + pad and box[1] - pad <= p[1] <= box[3] + pad)


def _boxes_at(det: dict, f: int, tol: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """The boxes of the sampled frame nearest to f, within tol frames."""
    fr = det["frames"]
    if not len(fr):
        return None
    i = int(np.searchsorted(fr, f))
    best = min((j for j in (i - 1, i) if 0 <= j < len(fr)), key=lambda j: abs(int(fr[j]) - f))
    if abs(int(fr[best]) - f) > tol:
        return None
    a, b = int(det["offset"][best]), int(det["offset"][best + 1])
    return det["boxes"][a:b], det["cls"][a:b], det["conf"][a:b]


def wrist_events(frames: np.ndarray, wrist: np.ndarray, speed: np.ndarray, torso: float, fps: float, det: dict,
                 names: list[str], active: float, hand: str) -> list[dict]:
    """Events of one wrist. frames [T] run frames, wrist [T, 2] px (NaN when unknown), speed [T] torso lengths/s."""
    T = len(frames)
    hold, after, back = max(1, int(round(STOP_S * fps))), max(1, int(round(AFTER_S * fps))), int(round(REACH_MAX_S * fps))
    tol = max(1, int(round(0.5 * fps)))
    slow = np.isfinite(speed) & (speed < active)
    out: list[dict] = []
    t = 1
    while t < T - hold:
        if not (slow[t:t + hold].all() and np.isfinite(speed[t - 1]) and speed[t - 1] >= active):
            t += 1
            continue
        lo = max(0, t - back)
        s = speed[lo:t]
        if not len(s) or not np.nanmax(s) >= REACH_FACTOR * active:
            t += hold
            continue
        start = t
        while start > lo and np.isfinite(speed[start - 1]) and speed[start - 1] >= active:
            start -= 1
        p0 = wrist[t]
        here = _boxes_at(det, int(frames[t]), tol)
        kind, later, p1 = None, None, None
        if np.isfinite(p0).all():                     # the first detected frame after the hand has left
            for f in det["frames"][(det["frames"] > frames[t]) & (det["frames"] <= frames[t] + after)]:
                row = int(np.searchsorted(frames, f))
                if row < T and frames[row] == f and np.isfinite(wrist[row]).all() \
                        and np.linalg.norm(wrist[row] - p0) >= LEFT_TORSO * torso:
                    later, p1 = _boxes_at(det, int(f), 0), wrist[row]
                    break
        if here is not None and later is not None:
            for box, c, conf in zip(*here):
                if not _near(box, p0, NEAR_TORSO * torso):
                    continue
                same = later[0][later[1] == c]
                if not len(same):
                    continue
                cen = (box[:2] + box[2:]) / 2
                got = (same[:, :2] + same[:, 2:]) / 2
                moved_to = got[int(np.argmin(np.linalg.norm(got - (cen + p1 - p0), axis=1)))]
                stayed_at = got[int(np.argmin(np.linalg.norm(got - cen, axis=1)))]
                if np.linalg.norm(stayed_at - cen) < STAYED_TORSO * torso:
                    kind = "placement"
                elif np.linalg.norm(moved_to - (cen + p1 - p0)) < LEFT_TORSO * torso:
                    kind = "pickup"
                if kind:
                    ev = {"kind": kind, "hand": hand, "frame": int(frames[t]), "t_s": round(float(frames[t]) / fps, 3),
                          "reach_s": round((t - start) / fps, 3), "object": names[int(c)], "conf": round(float(conf), 3)}
                    if not (out and out[-1]["kind"] == kind and ev["t_s"] - out[-1]["t_s"] < MERGE_S):
                        out.append(ev)
                    break
        t += hold
    return out
