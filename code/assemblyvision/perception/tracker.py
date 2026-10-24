"""Tracking (LLD 3.5): ByteTrack over the DET v1 rows of one shot, KTS v1 output.

A new tracker is made for every shot, so identity restarts at each cut. Before
tracking, duplicate boxes of one worker (overlapping partial skeletons under
occlusion) are dropped when they overlap strongly with a higher scoring box
whose keypoints they repeat. DET v1 itself is never changed (FR-POSE-08).

Subject: a click on the reference frame picks the track (subject_source
'click'); without a click the track with the most present frames is the
subject (subject_source 'auto') if it covers track.auto_subject_min_coverage
of the shot, else the shot has no subject and gets an AV-POS-005 notice.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..config import Config
from ..errors import AvError
from ..kts import empty_series, write_kts
from ..storage.files import RunContext, write_atomic
from ..types import KeypointSeries, Shot
from .pose import load_det

# assignment codes per DET row in detections/<shot_id>_assign.npz
KEPT_MIN = 1        # >= 1: number NN of the stored track <shot_id>-tNN
SHORT = 0           # tracked, but the track was shorter than track.min_track_s
UNTRACKED = -1      # ByteTrack did not assign it (low score, or the first frame of a track,
                    # which ByteTrack only confirms on its second frame)
SUPPRESSED = -2     # duplicate of a higher scoring box, not given to the tracker


def track_id(shot_id: str, n: int) -> str:
    """<shot_id>-tNN (33 characters, fits worker_track.track_id TEXT(33))."""
    return f"{shot_id}-t{n:02d}"


def _iou_contain(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """IoU and intersection over the smaller box area."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    small = min(area_a, area_b)
    return (inter / union if union > 0 else 0.0), (inter / small if small > 0 else 0.0)


def suppress_duplicates(xyxy: np.ndarray, conf: np.ndarray, kxy: np.ndarray, kconf: np.ndarray,
                        tc: dict) -> np.ndarray:
    """Boolean mask of the rows of one frame that are duplicates.

    Rows are visited by falling score. A row B is a duplicate of a kept row A
    if IoU(A,B) >= dup_iou, or if the smaller box lies >= dup_contain inside
    the other and the skeletons agree: of the keypoints confident in both (at
    least 3), >= dup_kp_agree lie within dup_kp_dist x the smaller diagonal.
    A second worker standing behind someone has different joints and is kept.
    """
    n = len(conf)
    dup = np.zeros(n, bool)
    if n < 2:
        return dup
    kept: list[int] = []
    for j in np.argsort(-conf, kind="stable"):
        for i in kept:
            iou, contain = _iou_contain(xyxy[i], xyxy[j])
            if iou >= tc["dup_iou"]:
                dup[j] = True
                break
            if contain >= tc["dup_contain"]:
                shared = (kconf[i] >= tc["dup_kp_conf"]) & (kconf[j] >= tc["dup_kp_conf"])
                if shared.sum() >= 3:
                    wi, hi = xyxy[i, 2] - xyxy[i, 0], xyxy[i, 3] - xyxy[i, 1]
                    wj, hj = xyxy[j, 2] - xyxy[j, 0], xyxy[j, 3] - xyxy[j, 1]
                    diag = min(np.hypot(wi, hi), np.hypot(wj, hj))
                    d = np.linalg.norm(kxy[i][shared] - kxy[j][shared], axis=1)
                    if np.mean(d <= tc["dup_kp_dist"] * diag) >= tc["dup_kp_agree"]:
                        dup[j] = True
                        break
        if not dup[j]:
            kept.append(j)
    return dup


class Tracker:
    def __init__(self, cfg: Config, ctx: RunContext):
        self.cfg = cfg
        self.tc = cfg["track"]
        self.ctx = ctx

    def _bytetrack(self, fps: float):
        from ultralytics.trackers.byte_tracker import BYTETracker
        args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=self.tc["high_thresh"],
                               track_low_thresh=self.tc["low_thresh"], new_track_thresh=self.tc["new_thresh"],
                               track_buffer=max(1, round(self.tc["buffer_s"] * fps)),
                               match_thresh=self.tc["match_thresh"], fuse_score=True)
        return BYTETracker(args)

    def link(self, det: dict, fps: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Run ByteTrack over every frame of the shot. Returns per DET row the
        local ByteTrack ID (0 = none) and the duplicate mask."""
        from ultralytics.engine.results import Boxes
        bt = self._bytetrack(fps)
        n = det["frame_idx"].shape[0]
        local = np.zeros(n, np.int64)
        dup = np.zeros(n, bool)
        order = np.argsort(det["frame_idx"], kind="stable")
        fidx = det["frame_idx"][order]
        start, count = int(det["frame_start"]), int(det["frame_count"])
        lo = 0
        for f in range(start, start + count):   # every frame, also empty ones, so lost tracks age
            hi = int(np.searchsorted(fidx, f, "right"))
            rows = order[lo:hi]
            lo = hi
            if len(rows) and self.tc["dup_suppress"]:
                d = suppress_duplicates(det["bbox_xyxy"][rows], det["det_conf"][rows],
                                        det["body_xy"][rows], det["body_conf"][rows], self.tc)
                dup[rows[d]] = True
                rows = rows[~d]
            data = np.zeros((len(rows), 6), np.float32)
            if len(rows):
                data[:, :4] = det["bbox_xyxy"][rows]
                data[:, 4] = det["det_conf"][rows]
            out = bt.update(Boxes(data, (height, width)))
            for r in np.asarray(out).reshape(-1, 8):
                local[rows[int(r[7])]] = int(r[4])
        return local, dup

    def track(self, run_id: str, shot: Shot, det_path: Path, ref: dict | None = None,
              click_xy: tuple[float, float] | None = None, fps: float | None = None,
              size: tuple[int, int] | None = None, meta: dict | None = None) -> list[dict]:
        """Track one shot; writes KTS files and the row assignment, returns the
        WorkerTrack records (LLD 4.4 fields). AV-TRK-001, 002, 003, AV-POS-005."""
        t0 = time.monotonic()
        ctx, tc = self.ctx, self.tc
        det = load_det(det_path)
        fps = fps or self.ctx.manifest.get("ingest", {}).get("cfr", {}).get("fps") or 25.0
        width, height = size or (0, 0)
        local, dup = self.link(det, fps, width, height)
        start, count = int(det["frame_start"]), int(det["frame_count"])
        min_frames = round(tc["min_track_s"] * fps)
        buffer = max(1, round(tc["buffer_s"] * fps))

        ids = [i for i in np.unique(local) if i > 0]
        assign = np.where(dup, SUPPRESSED, UNTRACKED).astype(np.int16)
        candidates = []
        for i in ids:
            rows = np.flatnonzero(local == i)
            rows = rows[np.argsort(det["frame_idx"][rows], kind="stable")]
            if len(rows) < min_frames:
                assign[rows] = SHORT
            else:
                candidates.append(rows)
        candidates.sort(key=lambda r: int(det["frame_idx"][r[0]]))
        if len(candidates) > 99:
            ctx.log.warning("shot %s has %d tracks; IDs above t99 exceed TEXT(33)", shot.shot_id, len(candidates))

        series_list: list[KeypointSeries] = []
        records: list[dict] = []
        for n, rows in enumerate(candidates, start=1):
            tid = track_id(shot.shot_id, n)
            s = empty_series(run_id, shot.shot_id, tid, fps, width, height, start, count)
            k = det["frame_idx"][rows] - start
            s.present[k] = True
            s.bbox_xyxy[k] = det["bbox_xyxy"][rows]
            s.body_xy[k] = det["body_xy"][rows]
            s.body_conf[k] = det["body_conf"][rows]
            assign[rows] = n
            gaps = self._long_gaps(s.present, buffer)
            for g0, glen in gaps:
                ctx.notice(AvError("AV-TRK-001", s=f"{glen / fps:.1f}", t=f"{(start + g0) / fps:.1f}",
                                   shot=shot.shot_id), "tracking")
            present = int(s.present.sum())
            series_list.append(s)
            records.append({"track_id": tid, "shot_id": shot.shot_id, "idx": n, "is_subject": False,
                            "subject_source": None, "frames_total": count, "frames_present": present,
                            "coverage": round(present / count, 4) if count else 0.0,
                            "conf_mean": round(float(np.nanmean(s.body_conf[s.present])), 4) if present else None,
                            "gaps_broken": len(gaps), "first_frame": int(start + k.min()),
                            "last_frame": int(start + k.max())})

        subject = None
        if not series_list:
            ctx.notice(AvError("AV-TRK-002", min=f"{tc['min_track_s']:g}", shot=shot.shot_id), "tracking")
        elif click_xy is not None:
            subject = self.pick_subject(series_list, ref, click_xy, shot.shot_id), "click"
        else:
            best = max(range(len(records)), key=lambda i: (records[i]["frames_present"], -i))
            if records[best]["coverage"] >= tc["auto_subject_min_coverage"]:
                subject = series_list[best].track_id, "auto"
            else:
                ctx.notice(AvError("AV-POS-005", shot=shot.shot_id,
                                   detail=f"longest track covers {records[best]['coverage']:.0%} of the shot, "
                                          f"below {tc['auto_subject_min_coverage']:.0%}"), "tracking")

        meta = dict(meta or {})
        meta["tracker"] = "bytetrack"
        for s, rec in zip(series_list, records):
            if subject and s.track_id == subject[0]:
                rec["is_subject"], rec["subject_source"] = True, subject[1]
            path = write_kts(s, ctx.paths, {**meta, "is_subject": rec["is_subject"],
                                            "subject_source": rec["subject_source"]})
            rec["kts_path"] = ctx.rel(path)
        self._save_assign(shot.shot_id, assign, local)

        stats = {"shot_id": shot.shot_id, "tracks_created": len(ids), "tracks_kept": len(records),
                 "short_tracks": len(ids) - len(records), "duplicates_suppressed": int(dup.sum()),
                 "untracked_rows": int((assign == UNTRACKED).sum()), "rows": int(len(assign)),
                 "subject": subject[0] if subject else None,
                 "subject_source": subject[1] if subject else None,
                 "seconds": round(time.monotonic() - t0, 3)}
        ctx.manifest.setdefault("tracking", {}).setdefault("per_shot", []).append(stats)
        ctx.manifest["tracking"].setdefault("tracks", []).extend(records)
        ctx.log.info("shot %s: %d tracks kept, %d short, %d duplicates suppressed, subject %s (%s)",
                     shot.shot_id, stats["tracks_kept"], stats["short_tracks"], stats["duplicates_suppressed"],
                     stats["subject"], stats["subject_source"])
        return records

    @staticmethod
    def _long_gaps(present: np.ndarray, buffer: int) -> list[tuple[int, int]]:
        """(start row, length) of gaps longer than buffer between the first and last present frame."""
        idx = np.flatnonzero(present)
        if len(idx) < 2:
            return []
        steps = np.diff(idx) - 1
        return [(int(idx[i] + 1), int(steps[i])) for i in np.flatnonzero(steps > buffer)]

    @staticmethod
    def pick_subject(series_list: list[KeypointSeries], ref: dict | None, click_xy: tuple[float, float],
                     shot_id: str) -> str:
        """Track whose box at the reference frame contains the click; the one
        with the box centre closest to it (AV-TRK-003)."""
        if ref is None:
            raise AvError("AV-TRK-003", shot=shot_id, detail="no reference frame")
        x, y = click_xy
        best, best_d = None, np.inf
        for s in series_list:
            row = ref["frame_idx"] - int(s.frame_idx[0])
            if not (0 <= row < len(s.present)) or not s.present[row]:
                continue
            x1, y1, x2, y2 = s.bbox_xyxy[row]
            if x1 <= x <= x2 and y1 <= y <= y2:
                d = np.hypot((x1 + x2) / 2 - x, (y1 + y2) / 2 - y)
                if d < best_d:
                    best, best_d = s.track_id, d
        if best is None:
            raise AvError("AV-TRK-003", shot=shot_id)
        return best

    def _save_assign(self, shot_id: str, assign: np.ndarray, local: np.ndarray) -> None:
        buf = io.BytesIO()
        np.savez_compressed(buf, assign=assign, bytetrack_id=local.astype(np.int32))
        write_atomic(self.ctx.paths.assign(shot_id), buf.getvalue(), self.ctx.paths)


def load_assign(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as z:
        return z["assign"]


def summarize(ctx: RunContext) -> dict:
    """Totals of the tracking stage for the manifest and the summary line."""
    t = ctx.manifest.setdefault("tracking", {})
    per = t.get("per_shot", [])
    t["totals"] = {"shots": len(per), "tracks_kept": sum(p["tracks_kept"] for p in per),
                   "short_tracks": sum(p["short_tracks"] for p in per),
                   "duplicates_suppressed": sum(p["duplicates_suppressed"] for p in per),
                   "shots_with_subject": sum(p["subject"] is not None for p in per)}
    return t["totals"]
