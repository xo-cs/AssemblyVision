"""Pose and track overlay videos for visual checking (part of Visualizer, LLD 3.9).

Reads the DET v1 file (not the model), draws boxes and COCO 17 skeletons with
OpenCV and encodes H.264 through ffmpeg (LLD D10). Keypoints below
overlay.kp_conf_min are drawn faded; the stored data is not changed.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..errors import AvError
from ..ingest.ingestor import iter_frames
from ..perception.pose import COCO17_SKELETON, load_det
from ..storage.files import RunContext

_COLORS = ((0, 200, 255), (255, 120, 0), (0, 220, 0), (220, 0, 220), (0, 120, 255),
           (255, 255, 0), (120, 0, 255), (0, 255, 160))
_FADED = (150, 150, 150)


def draw_people(img: np.ndarray, boxes: np.ndarray, conf: np.ndarray, xy: np.ndarray,
                kconf: np.ndarray, kp_min: float, scale: float, fixed_color: tuple | None = None) -> None:
    t = max(1, round(2 * scale))
    r = max(2, round(4 * scale))
    for i in range(len(boxes)):
        color = fixed_color or _COLORS[i % len(_COLORS)]
        x1, y1, x2, y2 = (boxes[i] * scale).astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, t)
        cv2.putText(img, f"{conf[i]:.2f}", (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * max(scale, 0.6), color, t)
        pts = xy[i] * scale
        good = kconf[i] >= kp_min
        for a, b in COCO17_SKELETON:
            if good[a] and good[b] and np.isfinite(pts[[a, b]]).all():
                cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color, t, cv2.LINE_AA)
        for k in range(len(pts)):
            if not np.isfinite(pts[k]).all():
                continue
            p = tuple(pts[k].astype(int))
            if good[k]:
                cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(img, p, r, _FADED, 1, cv2.LINE_AA)


def render_pose_overlay(video: Path, det_path: Path, out_path: Path, fps: float,
                        cfg: Config, ctx: RunContext) -> Path:
    t0 = time.monotonic()
    ocfg = cfg["overlay"]
    out_path = ctx.paths.check_inside(out_path)
    det = load_det(det_path)
    order = np.argsort(det["frame_idx"], kind="stable")
    fidx = det["frame_idx"][order]
    start = int(det["frame_start"])
    end = start + int(det["frame_count"])

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise AvError("AV-ING-001", name=Path(video).name, detail="OpenCV could not open the video")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    scale = min(1.0, ocfg["max_width"] / w)
    ow, oh = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)   # x264 needs even sizes

    tmp = out_path.with_name(out_path.stem + ".tmp.mp4")
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{ow}x{oh}", "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", str(ocfg["crf"]), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0
    try:
        for idx, frame in iter_frames(video, start, end):
            img = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA) if scale < 1 else frame
            if (ow, oh) != (img.shape[1], img.shape[0]):
                img = cv2.resize(img, (ow, oh))
            lo, hi = np.searchsorted(fidx, idx, "left"), np.searchsorted(fidx, idx, "right")
            if hi > lo:
                sel = order[lo:hi]
                draw_people(img, det["bbox_xyxy"][sel], det["det_conf"][sel], det["body_xy"][sel],
                            det["body_conf"][sel], ocfg["kp_conf_min"], scale)
            draw_header(img, f"frame {idx}  people {hi - lo}", scale=0.55 * max(1.0, oh / 480))
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
            n += 1
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        code = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        tmp.unlink(missing_ok=True)
        raise
    if code != 0:
        tmp.unlink(missing_ok=True)
        ctx.log.error("overlay ffmpeg exit %d: %s", code, err.strip())
        raise AvError("AV-STO-004", what="The overlay video", detail=f"ffmpeg exit code {code}")
    os.replace(tmp, out_path)
    ctx.manifest["overlay"] = {"path": ctx.rel(out_path), "frames": n, "width": ow, "height": oh,
                               "seconds": round(time.monotonic() - t0, 2)}
    ctx.log.info("overlay written: %s (%d frames, %dx%d)", ctx.rel(out_path), n, ow, oh)
    return out_path


_SHORT = (0, 140, 255)       # orange: tracked but shorter than track.min_track_s
_UNTRACKED = (160, 160, 160)
_SUPPRESSED = (0, 0, 230)    # red: duplicate skeleton removed before tracking


def _track_color(n: int) -> tuple[int, int, int]:
    return _COLORS[(n - 1) % len(_COLORS)]


def draw_header(img: np.ndarray, text: str, scale: float = 0.55, alpha: float = 0.55, margin: int = 8) -> None:
    """Draw the status text once, bottom left, on a semi transparent black box.
    The top left stays free: the QiSens video carries its own step captions there."""
    font, th = cv2.FONT_HERSHEY_SIMPLEX, 1
    (tw, tht), base = cv2.getTextSize(text, font, scale, th)
    h, w = img.shape[:2]
    x0, y1 = margin, h - margin
    x1, y0 = min(w, x0 + tw + 2 * margin), max(0, y1 - tht - base - 2 * margin)
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    cv2.putText(img, text, (x0 + margin, y1 - margin - base), font, scale, (255, 255, 255), th, cv2.LINE_AA)


def step_at(steps: list | None, t: float) -> str | None:
    """Label of the step covering time t (steps sorted, end exclusive)."""
    for s in steps or ():
        if s.start_s <= t < s.end_s:
            return s.label
    return None


def render_track_overlay(video: Path, shots: list, items: dict, out_path: Path, fps: float,
                         cfg: Config, ctx: RunContext, steps: list | None = None) -> Path:
    """Overlay of all shots: stored tracks in their own colour with label tNN
    (subject marked S), short tracks orange, untracked rows grey, suppressed
    duplicates red. items[shot_id] = {"det": path, "assign": path, "subject": n or None}.
    steps (optional, Step objects) adds the current step label to the header."""
    from ..perception.tracker import SHORT, SUPPRESSED, UNTRACKED, load_assign
    t0 = time.monotonic()
    ocfg = cfg["overlay"]
    out_path = ctx.paths.check_inside(out_path)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise AvError("AV-ING-001", name=Path(video).name, detail="OpenCV could not open the video")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    scale = min(1.0, ocfg["max_width"] / w)
    ow, oh = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)
    kp_min = ocfg["kp_conf_min"]
    t = max(1, round(2 * scale * max(1.0, w / 1280)))

    tmp = out_path.with_name(out_path.stem + ".tmp.mp4")
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{ow}x{oh}", "-r", f"{fps:.6f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", str(ocfg["crf"]), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    shots = sorted(shots, key=lambda s: s.start_frame)
    n_out, k, cur = 0, -1, None
    try:
        for idx, frame in iter_frames(video, shots[0].start_frame, shots[-1].end_frame):
            while k + 1 < len(shots) and idx >= shots[k + 1].start_frame:
                k += 1
                s = shots[k]
                it = items.get(s.shot_id)
                cur = None
                if it and Path(it["det"]).is_file() and Path(it["assign"]).is_file():
                    det, assign = load_det(it["det"]), load_assign(it["assign"])
                    order = np.argsort(det["frame_idx"], kind="stable")
                    cur = (det, assign, order, det["frame_idx"][order], it.get("subject"))
            img = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA) if (ow, oh) != (w, h) else frame
            s = shots[k]
            if cur is not None:
                det, assign, order, fidx, subject = cur
                lo, hi = np.searchsorted(fidx, idx, "left"), np.searchsorted(fidx, idx, "right")
                for r in order[lo:hi]:
                    a = int(assign[r])
                    box = det["bbox_xyxy"][r] * scale
                    x1, y1, x2, y2 = box.astype(int)
                    if a >= 1:
                        color = _track_color(a)
                        draw_people(img, det["bbox_xyxy"][r:r + 1], det["det_conf"][r:r + 1],
                                    det["body_xy"][r:r + 1], det["body_conf"][r:r + 1], kp_min, scale, color)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, t + (1 if a == subject else 0))
                        label = f"t{a:02d}" + (" S" if a == subject else "")
                        cv2.putText(img, label, (x1 + 2, max(y2 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, color, 2, cv2.LINE_AA)
                    else:
                        color = {SHORT: _SHORT, UNTRACKED: _UNTRACKED, SUPPRESSED: _SUPPRESSED}.get(a, _UNTRACKED)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                        if a == SUPPRESSED:
                            cv2.line(img, (x1, y1), (x2, y2), color, 1)
            header = f"shot {k + 1}/{len(shots)} {s.shot_type}  frame {idx}"
            label = step_at(steps, idx / fps)
            if label is not None:
                header += f"  step: {label}"
            draw_header(img, header, scale=0.55 * max(1.0, oh / 480))
            proc.stdin.write(np.ascontiguousarray(img).tobytes())
            n_out += 1
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.stderr.close()
        code = proc.wait()
    except BaseException:
        proc.kill()
        proc.wait()
        tmp.unlink(missing_ok=True)
        raise
    if code != 0:
        tmp.unlink(missing_ok=True)
        ctx.log.error("overlay ffmpeg exit %d: %s", code, err.strip())
        raise AvError("AV-STO-004", what="The overlay video", detail=f"ffmpeg exit code {code}")
    os.replace(tmp, out_path)
    ctx.manifest["overlay"] = {"path": ctx.rel(out_path), "frames": n_out, "width": ow, "height": oh,
                               "legend": "tNN colour = stored track (S = subject), orange = short track, "
                                         "grey = untracked, red crossed = suppressed duplicate",
                               "seconds": round(time.monotonic() - t0, 2)}
    ctx.log.info("overlay written: %s (%d frames, %dx%d)", ctx.rel(out_path), n_out, ow, oh)
    return out_path
