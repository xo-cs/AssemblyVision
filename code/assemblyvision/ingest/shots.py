"""Shot segmentation (LLD 3.3): camera cuts with PySceneDetect, fixed or handheld
from the background motion, floor scale from two marked points."""
from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..errors import AvError
from ..storage.files import RunContext
from ..types import Shot, VideoInfo
from .ingestor import iter_frames, iter_frames_at

_FLOW_WIDTH = 640   # frames wider than this are scaled down for the motion check


def shot_id(run_id: str, idx: int) -> str:
    """<run_id>-sNNN (29 characters, fits shot.shot_id TEXT(29))."""
    return f"{run_id}-s{idx:03d}"


def detect_cuts(video_path: Path, fps: float, threshold: float, min_shot_s: float,
                downscale: int = 0) -> list[int]:
    """Frame numbers where a new shot starts (0 excluded), from ContentDetector.
    downscale > 0 sets PySceneDetect's frame downscale factor (shots.downscale);
    0 keeps its automatic choice."""
    from scenedetect import ContentDetector, SceneManager, open_video
    video = open_video(str(video_path), backend="opencv")
    sm = SceneManager()
    if downscale > 0:
        sm.auto_downscale = False
        sm.downscale = downscale
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=max(1, round(min_shot_s * fps))))
    sm.detect_scenes(video, show_progress=False)
    return [start.frame_num for start, _ in sm.get_scene_list(start_in_scene=True)][1:]


class ShotDetector:
    def __init__(self, cfg: Config, ctx: RunContext):
        self.cfg = cfg
        self.scfg = cfg["shots"]
        self.ctx = ctx

    def detect(self, run_id: str, video_path: Path, info: VideoInfo) -> list[Shot]:
        """Shots of the video with type and motion; AV-SHT-001 falls back to one handheld shot."""
        t0 = time.monotonic()
        fps, n = info.fps, info.frame_count
        method = "pyscenedetect ContentDetector"
        try:
            cuts = detect_cuts(video_path, fps, self.scfg["cut_threshold"], self.scfg["min_shot_s"],
                               self.scfg["downscale"])
        except Exception as e:
            self.ctx.notice(AvError("AV-SHT-001", detail=f"{type(e).__name__}: {e}"), "shots")
            cuts, method = None, "whole video (cut detection failed)"
        t_cuts = time.monotonic() - t0
        bounds = [0] + [c for c in (cuts or []) if 0 < c < n] + [n]
        shots = [Shot(shot_id(run_id, i + 1), run_id, a, b, round(a / fps, 3), round(b / fps, 3), "handheld")
                 for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])) if b > a]
        tm = time.monotonic()
        motion = self.classify_all(video_path, shots, fps) if cuts is not None else {}
        t_motion = time.monotonic() - tm
        for s in shots:
            m = motion.get(s.shot_id)
            s.shot_type = "fixed" if m is not None and m < self.scfg["fixed_motion_max"] else "handheld"
        self.ctx.manifest["shots"] = {
            "count": len(shots), "method": method, "cut_threshold": self.scfg["cut_threshold"],
            "min_shot_s": self.scfg["min_shot_s"], "downscale": self.scfg["downscale"],
            "seconds": round(time.monotonic() - t0, 2),
            "breakdown_s": {"cuts": round(t_cuts, 3), "motion": round(t_motion, 3)},
            "items": [{"shot_id": s.shot_id, "start_frame": s.start_frame, "end_frame": s.end_frame,
                       "start_s": s.start_s, "end_s": s.end_s, "frames": s.end_frame - s.start_frame,
                       "shot_type": s.shot_type, "motion_p90": motion.get(s.shot_id),
                       "scale_m_per_px": s.scale_m_per_px, "ref_frame": None} for s in shots],
        }
        self.ctx.log.info("%d shots (%d fixed)", len(shots), sum(s.shot_type == "fixed" for s in shots))
        return shots

    def pair_starts(self, shot: Shot, fps: float) -> tuple[list[int], int]:
        d = max(1, round(self.scfg["motion_pair_s"] * fps))
        last = shot.end_frame - 1 - d
        if last < shot.start_frame:
            return [], d
        n = min(self.scfg["motion_pairs"], last - shot.start_frame + 1)
        return sorted({int(round(x)) for x in np.linspace(shot.start_frame, last, n)}), d

    def classify_all(self, video_path: Path, shots: list[Shot], fps: float) -> dict[str, float | None]:
        """90th percentile background motion per shot, in one pass over the video.

        For each frame pair 0.5 s apart: Lucas Kanade on up to 200 corners,
        d = median displacement / frame width (LLD 3.3 classify)."""
        starts: dict[int, tuple[str, int]] = {}   # first frame -> (shot_id, second frame)
        for s in shots:
            ps, d = self.pair_starts(s, fps)
            for f in ps:
                starts[f] = (s.shot_id, f + d)
        ends = {e: f for f, (_, e) in starts.items()}
        pending: dict[int, np.ndarray] = {}
        values: dict[str, list[float]] = {s.shot_id: [] for s in shots}
        wanted = set(starts) | set(ends)
        frames = (iter_frames_at(video_path, wanted) if self.scfg["motion_seek"]
                  else iter_frames(video_path, want=wanted.__contains__))
        for idx, frame in frames:
            gray = _gray(frame)
            if idx in ends and ends[idx] in pending:
                f0 = ends[idx]
                d = background_motion(pending.pop(f0), gray)
                if d is not None:
                    values[starts[f0][0]].append(d)
            if idx in starts:
                pending[idx] = gray
        return {sid: (round(float(np.percentile(v, 90)), 6) if v else None) for sid, v in values.items()}

    def classify(self, video_path: Path, shot: Shot, fps: float) -> str:
        m = self.classify_all(video_path, [shot], fps)[shot.shot_id]
        return "fixed" if m is not None and m < self.scfg["fixed_motion_max"] else "handheld"

    @staticmethod
    def scale_from_markings(p1: tuple[float, float], p2: tuple[float, float], metres: float) -> float:
        """Metres per pixel from two marked floor points (AV-SHT-002)."""
        px = math.dist(p1, p2)
        if px <= 0 or metres <= 0:
            raise AvError("AV-SHT-002", detail=f"points {p1} and {p2}, distance {metres} m")
        return metres / px


def _gray(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if g.shape[1] > _FLOW_WIDTH:
        g = cv2.resize(g, (_FLOW_WIDTH, round(g.shape[0] * _FLOW_WIDTH / g.shape[1])), interpolation=cv2.INTER_AREA)
    return g


def background_motion(g0: np.ndarray, g1: np.ndarray) -> float | None:
    """Median corner displacement between two grey frames, as a fraction of the width."""
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=200, qualityLevel=0.01, minDistance=7)
    if p0 is None or len(p0) == 0:
        return 0.0   # no texture at all (blank frame): nothing moves
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None)
    if p1 is None:
        return None
    good = st.reshape(-1) == 1
    if not good.any():
        return None
    disp = np.linalg.norm((p1 - p0).reshape(-1, 2)[good], axis=1)
    return float(np.median(disp)) / g0.shape[1]
