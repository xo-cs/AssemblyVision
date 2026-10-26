"""Pose estimation (LLD 3.4): reference frames, YOLO26 pose on every frame, DET v1 output per shot.

Raw keypoints and confidences are stored unchanged (FR-POSE-08). The caller
pins one GPU with CUDA_VISIBLE_DEVICES, so the model always uses cuda:0.
Weights are loaded only from models/ by explicit path; nothing is downloaded.
"""
from __future__ import annotations

import csv
import io
import json
import time
from pathlib import Path

import numpy as np

from ..config import VARIANTS, Config
from ..errors import AvError
from ..ingest.ingestor import iter_frames
from ..storage.files import REPO_ROOT, RunContext, write_atomic
from ..types import Shot

DET_VERSION = 1
N_BODY = 17
COCO17_NAMES = ("nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle")
COCO17_SKELETON = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
                   (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
                   (1, 3), (2, 4), (3, 5), (4, 6))


def weight_path(variant: str, models_dir: Path) -> Path:
    return models_dir / f"yolo26{variant}-pose.pt"


def save_det(path: Path, ctx: RunContext, rows: dict[str, np.ndarray], frame_start: int,
             frame_count: int, variant: str) -> Path:
    """Write one DET v1 file atomically: one row per detected person per frame."""
    arrays = dict(
        det_version=np.int32(DET_VERSION),
        frame_start=np.int32(frame_start),
        frame_count=np.int32(frame_count),
        pose_variant=np.str_(variant),
        frame_idx=rows["frame_idx"].astype(np.int32),
        bbox_xyxy=rows["bbox_xyxy"].astype(np.float32),
        det_conf=rows["det_conf"].astype(np.float32),
        body_xy=rows["body_xy"].astype(np.float32),
        body_conf=rows["body_conf"].astype(np.float32),
    )
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return write_atomic(path, buf.getvalue(), ctx.paths)


def load_det(path: Path) -> dict[str, np.ndarray]:
    """Read and check a DET v1 file (AV-STO-003)."""
    try:
        with np.load(path, allow_pickle=False) as z:
            d = {k: z[k] for k in z.files}
    except (OSError, ValueError) as e:
        raise AvError("AV-STO-003", file=Path(path).name, detail=str(e)) from e
    if int(d.get("det_version", -1)) != DET_VERSION:
        raise AvError("AV-STO-003", file=Path(path).name, detail="det_version is not 1")
    n = d["frame_idx"].shape[0]
    shapes = {"bbox_xyxy": (n, 4), "det_conf": (n,), "body_xy": (n, N_BODY, 2), "body_conf": (n, N_BODY)}
    for key, shape in shapes.items():
        if key not in d or d[key].shape != shape:
            raise AvError("AV-STO-003", file=Path(path).name, detail=f"{key} shape is not {shape}")
    return d


class PoseEstimator:
    def __init__(self, cfg: Config, ctx: RunContext, device: str = "cuda:0",
                 models_dir: Path | None = None):
        self.cfg = cfg
        self.pcfg = cfg["pose"]
        self.ctx = ctx
        self.device = device
        self.models_dir = models_dir or (REPO_ROOT / cfg["paths"]["models_dir"])
        self.model = None
        self.variant: str | None = None

    def load_models(self, variant: str) -> None:
        """Load YOLO26 pose of one variant from models/ (AV-POS-001, 002)."""
        path = weight_path(variant, self.models_dir)
        name = path.name
        if variant not in VARIANTS or not path.is_file():
            raise AvError("AV-POS-001", name=name, detail=f"expected file {path}")
        import torch                   # imported late: env (offline, GPU pinning) is set by the caller,
        from ultralytics import YOLO   # and CPU only stages (tracking, overlay) never load torch
        self.model = None
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        try:
            model = YOLO(str(path), task="pose")
            model.to(self.device)
        except torch.cuda.OutOfMemoryError as e:
            raise AvError("AV-POS-002", variant=variant, action="Choose a smaller variant.") from e
        except Exception as e:
            raise AvError("AV-POS-001", name=name, detail=repr(e)) from e
        self.model, self.variant = model, variant
        self.ctx.log.info("loaded %s on %s", name, self.device)

    def _infer(self, frame: np.ndarray, conf: float | None = None):
        # Ultralytics 8.4.178 replaced half with quantize (32 = FP32, 16 = FP16).
        return self.model.predict(frame, imgsz=self.pcfg["imgsz"],
                                  conf=self.pcfg["det_conf_min"] if conf is None else conf,
                                  quantize=16 if self.pcfg["half"] else 32, device=self.device,
                                  verbose=False, save=False)[0]

    def reference_frames(self, run_id: str, shots: list[Shot], video: Path | None = None) -> list[dict]:
        """Per shot, the first frame within pose.ref_search_s with a person at
        pose.ref_conf_min: reference/<shot_id>.png with boxes numbered 1..n
        (left to right) and .json with the box list. AV-POS-004 when none."""
        if self.model is None:
            self.load_models(self.pcfg["variant"])
        ctx = self.ctx
        video = video or ctx.paths.cfr
        fps = self._fps(video)
        windows = {s.shot_id: (s.start_frame, min(s.end_frame, s.start_frame + round(self.pcfg["ref_search_s"] * fps)))
                   for s in shots}
        by_frame = {}
        for s in shots:
            a, b = windows[s.shot_id]
            for f in range(a, b):
                by_frame[f] = s.shot_id
        found: dict[str, dict] = {}

        def want(idx: int) -> bool:
            sid = by_frame.get(idx)
            return sid is not None and sid not in found

        last = max((b for _, b in windows.values()), default=0)
        for idx, frame in iter_frames(video, 0, last, want=want):
            sid = by_frame[idx]
            try:
                res = self._infer(frame, conf=self.pcfg["ref_conf_min"])
            except Exception as e:   # one bad frame: try the next one
                ctx.log.warning("AV-POS-003 frame %d (reference): %s: %s", idx, type(e).__name__, e)
                continue
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.detach().cpu().numpy()
            conf = res.boxes.conf.detach().cpu().numpy()
            order = np.argsort(xyxy[:, 0], kind="stable")
            boxes = [{"n": i + 1, "xyxy": [round(float(v), 1) for v in xyxy[k]], "conf": round(float(conf[k]), 3)}
                     for i, k in enumerate(order)]
            ref = {"shot_id": sid, "frame_idx": idx, "t_sec": round(idx / fps, 3), "boxes": boxes}
            self._save_reference(frame, ref)
            found[sid] = ref
        out = []
        for s in shots:
            if s.shot_id in found:
                out.append(found[s.shot_id])
            else:
                ctx.notice(AvError("AV-POS-004", s=f"{self.pcfg['ref_search_s']:g}", shot=s.shot_id), "reference")
        ctx.manifest["reference"] = {"found": len(out), "missing": len(shots) - len(out),
                                     "items": [{"shot_id": r["shot_id"], "frame_idx": r["frame_idx"],
                                                "boxes": len(r["boxes"])} for r in out]}
        return out

    def _save_reference(self, frame: np.ndarray, ref: dict) -> None:
        import cv2
        img = frame.copy()
        t = max(2, img.shape[1] // 400)
        for b in ref["boxes"]:
            x1, y1, x2, y2 = (int(v) for v in b["xyxy"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 255), t)
            label = str(b["n"])
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, t)
            cv2.rectangle(img, (x1, y1), (x1 + w + 8, y1 + h + 8), (0, 220, 255), -1)
            cv2.putText(img, label, (x1 + 4, y1 + h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, (0, 0, 0), t)
        ok, png = cv2.imencode(".png", img)
        paths = self.ctx.paths
        write_atomic(paths.ref_png(ref["shot_id"]), png.tobytes(), paths)
        write_atomic(paths.ref_json(ref["shot_id"]), json.dumps(ref, indent=2).encode("utf-8"), paths)

    @staticmethod
    def _fps(video: Path) -> float:
        import cv2
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
        cap.release()
        return fps if fps and fps > 0 else 25.0

    def estimate_body(self, run_id: str, shot: Shot, video: Path | None = None) -> Path:
        """Pose on every frame of one shot; returns the DET v1 path (LLD 3.4)."""
        return self.estimate_body_shots(run_id, [shot], video)[shot.shot_id]

    def estimate_body_shots(self, run_id: str, shots: list[Shot], video: Path | None = None) -> dict[str, Path]:
        """Pose on every frame of every shot in one pass over the video; one DET
        v1 file per shot (AV-POS-002, 003). Statistics, totals and per shot, go
        to ctx.manifest["pose_body"]."""
        import torch
        if self.model is None:
            self.load_models(self.pcfg["variant"])
        ctx, log = self.ctx, self.ctx.log
        video = video or ctx.paths.cfr
        fallback = self.pcfg["fallback_variant"]
        cuda = self.device.startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        shots = sorted(shots, key=lambda s: s.start_frame)
        first, last = shots[0].start_frame, shots[-1].end_frame
        shot_of = np.full(last, -1, np.int32)
        for k, s in enumerate(shots):
            shot_of[s.start_frame:s.end_frame] = k

        failed: list[tuple[int, str, str]] = []
        fell_back = None
        total = int((shot_of >= 0).sum())
        n_frames = 0
        t_infer = 0.0
        t0 = last_log = time.monotonic()
        every = self.cfg["run"]["progress_every_s"]
        acc = {s.shot_id: {"rows": [[], [], [], [], []], "frames": 0, "failed": 0} for s in shots}
        paths: dict[str, Path] = {}
        current = None

        def flush(k: int) -> None:
            s = shots[k]
            a = acc[s.shot_id]
            fi, bx, dc, bxy, bcf = a["rows"]

            def cat(parts, shape, dtype):
                return np.concatenate(parts).astype(dtype) if parts else np.zeros((0, *shape), dtype)
            rows = {"frame_idx": cat(fi, (), np.int32), "bbox_xyxy": cat(bx, (4,), np.float32),
                    "det_conf": cat(dc, (), np.float32), "body_xy": cat(bxy, (N_BODY, 2), np.float32),
                    "body_conf": cat(bcf, (N_BODY,), np.float32)}
            if a["frames"] != s.end_frame - s.start_frame:
                log.warning("shot %s: decoded %d frames, expected %d", s.shot_id, a["frames"], s.end_frame - s.start_frame)
            paths[s.shot_id] = save_det(ctx.paths.det(s.shot_id), ctx, rows, s.start_frame,
                                        s.end_frame - s.start_frame, self.variant)
            a.update(detections=int(rows["frame_idx"].size),
                     frames_with_person=int(np.unique(rows["frame_idx"]).size))
            a["rows"] = None

        for idx, frame in iter_frames(video, first, last, want=lambda i: shot_of[i] >= 0):
            k = int(shot_of[idx])
            if current is not None and k != current:
                flush(current)
            current = k
            a = acc[shots[k].shot_id]
            a["frames"] += 1
            n_frames += 1
            ti = time.monotonic()
            try:
                try:
                    res = self._infer(frame)
                except torch.cuda.OutOfMemoryError:
                    if not fallback or fallback == self.variant or fell_back:
                        raise AvError("AV-POS-002", variant=self.variant, action="No fallback variant is left.")
                    old = self.variant
                    log.warning("GPU out of memory with variant %s on frame %d, switching to %s", old, idx, fallback)
                    self.load_models(fallback)
                    fell_back = {"from": old, "to": fallback, "frame": idx}
                    ctx.notice(AvError("AV-POS-002", variant=old,
                                       action=f"Variant {fallback} is used from frame {idx} on."), "pose_body")
                    res = self._infer(frame)   # retry once; a second OOM ends the stage
            except torch.cuda.OutOfMemoryError as e:
                raise AvError("AV-POS-002", variant=self.variant, action="The fallback variant also failed.") from e
            except AvError:
                raise
            except Exception as e:
                failed.append((idx, type(e).__name__, str(e)[:200]))
                a["failed"] += 1
                log.warning("AV-POS-003 frame %d: %s: %s", idx, type(e).__name__, e)
                continue
            t_infer += time.monotonic() - ti

            n = 0 if res.boxes is None else len(res.boxes)
            if n and res.keypoints is not None:
                kp = res.keypoints.data.detach().cpu().numpy()          # [n,17,3]: x, y, conf
                fi, bx, dc, bxy, bcf = a["rows"]
                fi.append(np.full(n, idx, np.int32))
                bx.append(res.boxes.xyxy.detach().cpu().numpy())
                dc.append(res.boxes.conf.detach().cpu().numpy())
                bxy.append(kp[:, :, :2])
                bcf.append(kp[:, :, 2] if kp.shape[2] > 2 else np.ones(kp.shape[:2], np.float32))

            now = time.monotonic()
            if now - last_log >= every:
                last_log = now
                rate = n_frames / (now - t0)
                eta = (total - n_frames) / rate if rate > 0 else 0
                log.info("pose %d/%d frames, %.1f fps, eta %.0f s", n_frames, total, rate, eta)
                self.ctx.progress("pose_body", n_frames / max(total, 1))

        wall = time.monotonic() - t0
        if n_frames == 0:
            raise AvError("AV-ING-001", name=Path(video).name, detail="no frames could be decoded")
        if current is not None:
            flush(current)
        for k, s in enumerate(shots):   # shots that got no decoded frame still get a file
            if s.shot_id not in paths:
                flush(k)
        if failed:
            self._write_failed(failed)

        peak_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None
        per_shot = [{"shot_id": s.shot_id, "det_path": ctx.rel(paths[s.shot_id]), "frames": acc[s.shot_id]["frames"],
                     "detections": acc[s.shot_id]["detections"],
                     "frames_with_person": acc[s.shot_id]["frames_with_person"],
                     "failed_frames": acc[s.shot_id]["failed"]} for s in shots]
        dets = sum(p["detections"] for p in per_shot)
        with_person = sum(p["frames_with_person"] for p in per_shot)
        ctx.manifest["pose_body"] = {
            "variant": self.variant,
            "variant_requested": self.pcfg["variant"],
            "fallback": fell_back,
            "imgsz": self.pcfg["imgsz"], "conf": self.pcfg["det_conf_min"], "half": self.pcfg["half"],
            "frames": n_frames,
            "frames_with_person": with_person,
            "detections": dets,
            "failed_frames": len(failed),
            "seconds": round(wall, 2),
            "pose_fps": round(n_frames / wall, 2) if wall > 0 else None,
            "infer_ms_mean": round(1000 * t_infer / max(n_frames - len(failed), 1), 2),
            "gpu_peak_mb": peak_mb,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "per_shot": per_shot,
        }
        log.info("pose done: %d frames in %d shots, %d detections, %d frames with a person, %d failed, "
                 "%.2f fps, peak %s MB", n_frames, len(shots), dets, with_person, len(failed),
                 n_frames / wall if wall > 0 else 0, peak_mb)
        return paths

    def _write_failed(self, failed: list[tuple[int, str, str]]) -> None:
        path = self.ctx.paths.failed_frames
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["stage", "frame_idx", "code", "error_type", "message"])
            for idx, etype, msg in failed:
                w.writerow(["pose_body", idx, "AV-POS-003", etype, msg])
