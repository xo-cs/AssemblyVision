"""Pose estimation (LLD 3.4): YOLO26 pose on every frame of a shot, DET v1 output.

Raw keypoints and confidences are stored unchanged (FR-POSE-08). The caller
pins one GPU with CUDA_VISIBLE_DEVICES, so the model always uses cuda:0.
Weights are loaded only from models/ by explicit path; nothing is downloaded.
"""
from __future__ import annotations

import csv
import io
import time
from pathlib import Path

import numpy as np
import torch

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
        from ultralytics import YOLO   # imported late: env (offline, GPU pinning) is set by the caller
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

    def _infer(self, frame: np.ndarray):
        # Ultralytics 8.4.178 replaced half with quantize (32 = FP32, 16 = FP16).
        return self.model.predict(frame, imgsz=self.pcfg["imgsz"], conf=self.pcfg["det_conf_min"],
                                  quantize=16 if self.pcfg["half"] else 32, device=self.device,
                                  verbose=False, save=False)[0]

    def estimate_body(self, run_id: str, shot: Shot, video: Path | None = None) -> Path:
        """Pose on every frame of the shot; returns the DET v1 path (AV-POS-002, 003).

        Stage statistics (frames, failed frames, fps, peak GPU memory, fallback)
        go to ctx.manifest["pose_body"].
        """
        if self.model is None:
            self.load_models(self.pcfg["variant"])
        ctx, log = self.ctx, self.ctx.log
        video = video or ctx.paths.cfr
        fallback = self.pcfg["fallback_variant"]
        cuda = self.device.startswith("cuda")
        if cuda:
            torch.cuda.reset_peak_memory_stats()

        frame_idx, boxes, det_conf, body_xy, body_conf = [], [], [], [], []
        failed: list[tuple[int, str, str]] = []
        fell_back = None
        total = shot.end_frame - shot.start_frame
        n_frames = 0
        t_infer = 0.0
        t0 = last = time.monotonic()
        every = self.cfg["run"]["progress_every_s"]

        for idx, frame in iter_frames(video, shot.start_frame, shot.end_frame):
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
                log.warning("AV-POS-003 frame %d: %s: %s", idx, type(e).__name__, e)
                continue
            t_infer += time.monotonic() - ti

            n = 0 if res.boxes is None else len(res.boxes)
            if n and res.keypoints is not None:
                kp = res.keypoints.data.detach().cpu().numpy()          # [n,17,3]: x, y, conf
                frame_idx.append(np.full(n, idx, np.int32))
                boxes.append(res.boxes.xyxy.detach().cpu().numpy())
                det_conf.append(res.boxes.conf.detach().cpu().numpy())
                body_xy.append(kp[:, :, :2])
                body_conf.append(kp[:, :, 2] if kp.shape[2] > 2 else np.ones(kp.shape[:2], np.float32))

            now = time.monotonic()
            if now - last >= every:
                last = now
                rate = n_frames / (now - t0)
                eta = (total - n_frames) / rate if rate > 0 else 0
                log.info("pose %d/%d frames, %.1f fps, eta %.0f s", n_frames, total, rate, eta)

        wall = time.monotonic() - t0
        if n_frames == 0:
            raise AvError("AV-ING-001", name=Path(video).name, detail="no frames could be decoded")
        if n_frames != total:
            log.warning("decoded %d frames, the probe reported %d", n_frames, total)

        def cat(parts, shape, dtype):
            return np.concatenate(parts).astype(dtype) if parts else np.zeros((0, *shape), dtype)

        rows = {"frame_idx": cat(frame_idx, (), np.int32), "bbox_xyxy": cat(boxes, (4,), np.float32),
                "det_conf": cat(det_conf, (), np.float32), "body_xy": cat(body_xy, (N_BODY, 2), np.float32),
                "body_conf": cat(body_conf, (N_BODY,), np.float32)}
        if failed:
            self._write_failed(failed)
        path = save_det(ctx.paths.det(shot.shot_id), ctx, rows, shot.start_frame, n_frames, self.variant)

        peak_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None
        frames_with_person = int(np.unique(rows["frame_idx"]).size)
        ctx.manifest["pose_body"] = {
            "det_path": ctx.rel(path),
            "variant": self.variant,
            "variant_requested": self.pcfg["variant"],
            "fallback": fell_back,
            "imgsz": self.pcfg["imgsz"], "conf": self.pcfg["det_conf_min"], "half": self.pcfg["half"],
            "frames": n_frames,
            "frames_with_person": frames_with_person,
            "detections": int(rows["frame_idx"].size),
            "failed_frames": len(failed),
            "seconds": round(wall, 2),
            "pose_fps": round(n_frames / wall, 2) if wall > 0 else None,
            "infer_ms_mean": round(1000 * t_infer / max(n_frames - len(failed), 1), 2),
            "gpu_peak_mb": peak_mb,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        }
        log.info("pose done: %d frames, %d detections, %d frames with a person, %d failed, %.2f fps, peak %s MB",
                 n_frames, rows["frame_idx"].size, frames_with_person, len(failed),
                 n_frames / wall if wall > 0 else 0, peak_mb)
        return path

    def _write_failed(self, failed: list[tuple[int, str, str]]) -> None:
        path = self.ctx.paths.failed_frames
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["stage", "frame_idx", "code", "error_type", "message"])
            for idx, etype, msg in failed:
                w.writerow(["pose_body", idx, "AV-POS-003", etype, msg])
