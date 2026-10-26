"""Pose and tracking of one live frame: YOLO26 s pose (FP16) on the live GPU
and ByteTrack with the batch tracker settings at the live frame rate.

The process must already see only the live GPU (CUDA_VISIBLE_DEVICES set by
the interface before torch is imported), so the model runs on cuda:0.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np


@dataclass
class Detections:
    boxes: np.ndarray      # [N, 4] xyxy pixels of the frame given to predict
    conf: np.ndarray       # [N]
    kp: np.ndarray         # [N, 17, 2]
    kconf: np.ndarray      # [N, 17]

    @classmethod
    def empty(cls) -> "Detections":
        return cls(np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 2)), np.zeros((0, 17)))

    def scaled(self, s: float) -> "Detections":
        return Detections(self.boxes * s, self.conf, self.kp * s, self.kconf)


class YoloPose:
    """YOLO26 pose; predict takes an RGB frame."""

    def __init__(self, model_path: Path, device: str = "cuda:0", half: bool = True, imgsz: int = 640,
                 det_conf: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(str(model_path), task="pose")
        self.model.to(device)
        self.device, self.half, self.imgsz, self.det_conf = device, half, imgsz, det_conf
        self.predict(np.zeros((imgsz * 3 // 4, imgsz, 3), np.uint8))          # warm up

    def predict(self, rgb: np.ndarray) -> Detections:
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        # Ultralytics 8.4.178 replaced half with quantize (16 = FP16), as in perception/pose.py
        res = self.model.predict(bgr, imgsz=self.imgsz, conf=self.det_conf, device=self.device,
                                 quantize=16 if self.half and self.device != "cpu" else 32, verbose=False)[0]
        if res.boxes is None or not len(res.boxes) or res.keypoints is None:
            return Detections.empty()
        k = res.keypoints.data.detach().cpu().numpy().astype(np.float64)
        return Detections(res.boxes.xyxy.detach().cpu().numpy().astype(np.float64),
                          res.boxes.conf.detach().cpu().numpy().astype(np.float64), k[:, :, :2],
                          k[:, :, 2] if k.shape[2] > 2 else np.ones(k.shape[:2]))

    def release(self) -> None:
        import torch
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LiveTracker:
    """ByteTrack over live frames; returns a track ID (0 = none) per detection."""

    def __init__(self, track_cfg: dict, fps: float, lost_s: float):
        from ultralytics.trackers.byte_tracker import BYTETracker
        args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=track_cfg["high_thresh"],
                               track_low_thresh=track_cfg["low_thresh"], new_track_thresh=track_cfg["new_thresh"],
                               track_buffer=max(1, round(lost_s * fps)), match_thresh=track_cfg["match_thresh"],
                               fuse_score=True)
        self.bt = BYTETracker(args)          # track_buffer is in frames at the live rate

    def update(self, det: Detections, size: tuple[int, int]) -> np.ndarray:
        from ultralytics.engine.results import Boxes
        w, h = size
        data = np.zeros((len(det.conf), 6), np.float32)
        if len(det.conf):
            data[:, :4] = det.boxes
            data[:, 4] = det.conf
        ids = np.zeros(len(det.conf), np.int64)
        for r in np.asarray(self.bt.update(Boxes(data, (h, w)))).reshape(-1, 8):
            ids[int(r[7])] = int(r[4])
        return ids
