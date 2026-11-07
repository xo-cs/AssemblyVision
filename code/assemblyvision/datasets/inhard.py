"""InHARD (Dallel et al., 2020): loader interface and a BVH reader stub.

Access is still pending, so nothing is downloaded and load() refuses. The BVH
reader parses the joint hierarchy and the motion block (17 mocap joints at
120 Hz) and resamples channels to the video frame rate. The skeleton is 3D;
mapping it to COCO 17 image points needs the camera calibration, which is
not known yet (datasets/mapping.INHARD17_TO_COCO17, provisional names).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..errors import AvError
from .base import DatasetLoader, Sample, VideoRef

NAME = "inhard"
MOCAP_HZ = 120.0


@dataclass
class Bvh:
    joints: list[str]                 # joint names in file order (End Sites excluded)
    channels: dict[str, list[str]]    # joint -> channel names, e.g. Xposition, Zrotation
    offsets: dict[str, tuple[float, float, float]]
    parents: dict[str, str | None]
    frame_time: float
    motion: np.ndarray                # float32 [F, total channels]

    @property
    def hz(self) -> float:
        return 1.0 / self.frame_time

    def column(self, joint: str, channel: str) -> int:
        n = 0
        for j in self.joints:
            if j == joint:
                return n + self.channels[j].index(channel)
            n += len(self.channels[j])
        raise KeyError(f"{joint}.{channel}")


def parse_bvh(text: str) -> Bvh:
    """Parse a BVH file (HIERARCHY and MOTION blocks)."""
    lines = [l.strip() for l in text.splitlines()]
    joints, channels, offsets, parents = [], {}, {}, {}
    stack: list[str | None] = []
    current: str | None = None
    i = 0
    while i < len(lines) and lines[i] != "MOTION":
        tok = lines[i].split()
        if tok and tok[0] in ("ROOT", "JOINT"):
            current = tok[1]
            joints.append(current)
            parents[current] = next((s for s in reversed(stack) if s is not None), None)
        elif tok and tok[0] == "End":
            current = None
        elif tok and tok[0] == "{":
            stack.append(current)
        elif tok and tok[0] == "}":
            stack.pop()
        elif tok and tok[0] == "OFFSET" and current is not None and current not in offsets:
            offsets[current] = tuple(float(x) for x in tok[1:4])
        elif tok and tok[0] == "CHANNELS" and current is not None:
            channels[current] = tok[2:2 + int(tok[1])]
        i += 1
    if i >= len(lines):
        raise AvError("AV-STO-003", file="BVH", detail="no MOTION block")
    n_frames = int(lines[i + 1].split(":")[1])
    frame_time = float(lines[i + 2].split(":")[1])
    rows = [list(map(float, l.split())) for l in lines[i + 3:i + 3 + n_frames] if l]
    width = sum(len(channels.get(j, [])) for j in joints)
    motion = np.array(rows, np.float32).reshape(-1, width) if rows else np.zeros((0, width), np.float32)
    if len(motion) != n_frames:
        raise AvError("AV-STO-003", file="BVH", detail=f"{len(motion)} motion rows, header says {n_frames}")
    return Bvh(joints, {j: channels.get(j, []) for j in joints}, offsets, parents, frame_time, motion)


def resample(values: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    """Linear resampling of [F, ...] samples at src_hz to dst_hz (no extrapolation
    past the last sample; no gap filling)."""
    F = len(values)
    if F == 0:
        return values.copy()
    t_src = np.arange(F) / src_hz
    t_dst = np.arange(int(np.floor(t_src[-1] * dst_hz)) + 1) / dst_hz
    flat = values.reshape(F, -1)
    out = np.stack([np.interp(t_dst, t_src, flat[:, k]) for k in range(flat.shape[1])], axis=1)
    return out.reshape(len(t_dst), *values.shape[1:]).astype(values.dtype)


class InHARDLoader(DatasetLoader):
    name = NAME

    def videos(self) -> list[VideoRef]:
        return []

    def load(self, video_id: str, keypoints: str = "ours") -> Sample:
        self._check_source(keypoints)
        raise AvError("AV-DAT-003", name="InHARD", detail="access is still pending; nothing has been downloaded")
