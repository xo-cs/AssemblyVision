"""SegmentationEngine interface (LLD 3.6), the run timeline and the step contract.

Deviation from LLD 3.6: the LLD segments the subject series of one shot, and
its contract keeps every step inside its shot. The QiSens process phases last
1 to 4 minutes and cross many camera cuts, so the engines work on a RunTimeline
(all shots on the clock of the video) and the contract keeps every step inside
the run. segment(series, shot) stays as the per shot interface of the LLD.
"""
from __future__ import annotations

import csv
import io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...errors import AvError
from ...kts import read_kts
from ...storage.files import RunContext, write_atomic
from ...types import KeypointSeries, Shot, Step
from ..cleaning import SignalCleaner, features
from .activity import empty_activity, frame_activity

OTHER = "other"
SOURCE = "engine_a"
_SEGMENT = re.compile(r"segment_[1-9][0-9]*")


@dataclass
class RunTimeline:
    """Per frame evidence of one run, frame f is at f / fps seconds."""
    fps: float
    n_frames: int
    person: np.ndarray                 # bool [N]: some stored track is present
    energy: np.ndarray                 # float32 [N]: subject motion energy, NaN without subject
    cuts: list[int] = field(default_factory=list)   # first frames of shots 2..n
    offset_s: float = 0.0              # added to all times (one shot timelines)
    activity: dict = field(default_factory=dict)    # per frame subject activity channels (activity.py), or empty

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def t(self, frame: int) -> float:
        return round(self.offset_s + frame / self.fps, 3)


def timeline_from_series(series: KeypointSeries, shot: Shot | None, cleaner: SignalCleaner) -> RunTimeline:
    """One shot timeline from one subject series (LLD per shot interface)."""
    f = features(cleaner.clean(series))
    return RunTimeline(float(series.fps), len(series.present), np.asarray(series.present, bool).copy(),
                       f["energy"], [], shot.start_s if shot is not None else 0.0)


def build_timeline(manifest: dict, run_root: Path, cleaner: SignalCleaner,
                   motion_cfg: dict | None = None) -> RunTimeline:
    """Run timeline from the stored shot table and KTS files (no video, no GPU).
    With motion_cfg (seg_a.activity_change on) it also carries the subject
    activity channels, with the thresholds motion.hand_active and motion.idle_energy."""
    cfr = manifest["ingest"]["cfr"]
    fps = float(cfr["fps"])
    items = sorted(manifest["shots"]["items"], key=lambda i: i["start_frame"])
    n = max(int(cfr["frame_count"]), max((i["end_frame"] for i in items), default=0))
    person = np.zeros(n, bool)
    energy = np.full(n, np.nan, np.float32)
    activity = empty_activity(n) if motion_cfg is not None else {}
    for tr in manifest.get("tracking", {}).get("tracks", []):
        series, _ = read_kts(run_root / tr["kts_path"])
        f0 = int(series.frame_idx[0]) if len(series.frame_idx) else 0
        T = min(len(series.present), n - f0)
        person[f0:f0 + T] |= series.present[:T]
        if tr.get("is_subject"):
            cs = cleaner.clean(series)
            e = features(cs)["energy"]
            energy[f0:f0 + T] = e[:T]
            if motion_cfg is not None:
                act = frame_activity(cs, e, motion_cfg["hand_active"], motion_cfg["idle_energy"])
                for k, v in act.items():
                    activity[k][f0:f0 + T] = v[:T]
    cuts = [int(i["start_frame"]) for i in items[1:]]
    return RunTimeline(fps, n, person, energy, cuts, activity=activity)


class SegmentationEngine(ABC):
    name = "?"

    def prepare(self, run_id: str) -> None:
        """Load what the engine needs (weights for Engine B); nothing for Engine A."""

    @abstractmethod
    def segment_run(self, tl: RunTimeline) -> list[Step]:
        """Steps covering the whole timeline, other included."""

    def segment(self, series: KeypointSeries, shot: Shot) -> list[Step]:
        """LLD interface: the subject series of one shot."""
        return self.segment_run(timeline_from_series(series, shot, self.cleaner))


def create_engine(name: str, cfg, ctx: RunContext | None = None) -> SegmentationEngine:
    from .rules import RuleBasedEngine
    if name == "B":   # no MS TCN weights exist yet
        if ctx is not None:
            ctx.notice(AvError("AV-SEG-001"), "segmentation")
        return RuleBasedEngine(cfg, ctx)
    if name != "A":
        raise ValueError(f"unknown engine {name}")
    return RuleBasedEngine(cfg, ctx)


def check_contract(steps: list[Step], tl: RunTimeline, vocabulary: list[str], engine: str = "A",
                   where: str = "run") -> None:
    """AV-SEG-002 unless: sorted, no overlap, inside the timeline, end > start,
    labels from vocabulary, other or segment_N, and no hole (TC-SEG-10)."""
    def bad(detail: str):
        raise AvError("AV-SEG-002", engine=engine, shot=where, detail=detail)
    eps = 0.5 / tl.fps
    t0, t1 = tl.offset_s, tl.offset_s + tl.duration_s
    allowed = set(vocabulary) | {OTHER}
    prev_end = t0
    for k, s in enumerate(steps):
        if not s.end_s > s.start_s:
            bad(f"step {k} ends before it starts")
        if s.start_s < t0 - eps or s.end_s > t1 + eps:
            bad(f"step {k} lies outside {t0:.3f} to {t1:.3f} s")
        if s.start_s < prev_end - eps:
            bad(f"step {k} overlaps the previous step or is out of order")
        if abs(s.start_s - prev_end) > 1.0 / tl.fps + eps:
            bad(f"hole of {s.start_s - prev_end:.3f} s before step {k}")
        if s.label not in allowed and not _SEGMENT.fullmatch(s.label):
            bad(f"label {s.label!r} is not in the vocabulary")
        prev_end = s.end_s
    if steps and abs(prev_end - t1) > 1.0 / tl.fps + eps:
        bad(f"steps end at {prev_end:.3f} s, the run at {t1:.3f} s")


def step_number(label: str, vocabulary: list[str]) -> int:
    if label in vocabulary:
        return vocabulary.index(label) + 1
    if _SEGMENT.fullmatch(label):
        return int(label.split("_")[1])
    return 0


def steps_csv_bytes(steps: list[Step], vocabulary: list[str]) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["step", "label", "start_s", "end_s", "source", "score"])
    for s in steps:
        w.writerow([step_number(s.label, vocabulary), s.label, f"{s.start_s:.3f}", f"{s.end_s:.3f}", s.source,
                    "" if s.score is None else f"{s.score:.3f}"])
    return buf.getvalue().encode("utf-8")


def write_steps_csv(path: Path, steps: list[Step], vocabulary: list[str], ctx: RunContext) -> Path:
    return write_atomic(path, steps_csv_bytes(steps, vocabulary), ctx.paths)
