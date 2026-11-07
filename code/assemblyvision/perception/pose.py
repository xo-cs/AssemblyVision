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
from typing import Callable

import numpy as np

from ..config import VARIANTS, Config
from ..errors import AvError
from ..ingest.ingestor import frame_reader, iter_frames
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
        self.half = False                  # precision in use, set by load_models (resolve_half)
        self.gpu_cc: str | None = None
        self.tf32: dict[str, bool] | None = None   # TF32 flags in use on CUDA, set by load_models (apply_tf32)
        self.pass_refs: list[dict] = []     # reference frames found by the pose pass (with_reference)

    def resolve_half(self, torch) -> bool:
        """pose.half asks for FP16; it is used only on a CUDA GPU with tensor cores
        (compute capability 7.0 or higher, e.g. the Ampere A5000), else FP32."""
        if not self.device.startswith("cuda"):
            return False
        major, minor = torch.cuda.get_device_capability(0)
        self.gpu_cc = f"{major}.{minor}"
        if self.pcfg["half"] and major < 7:
            self.ctx.log.info("pose.half is true but GPU compute capability %s has no tensor cores: FP32",
                              self.gpu_cc)
        return bool(self.pcfg["half"]) and major >= 7

    def apply_tf32(self, torch) -> dict[str, bool] | None:
        """pose.tf32 = false turns TF32 off (strict FP32 convolutions and matmul);
        true leaves the PyTorch defaults. Returns the flags in use (None off CUDA)."""
        if not self.device.startswith("cuda"):
            return None
        if not self.pcfg["tf32"]:
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
        return {"cudnn": bool(torch.backends.cudnn.allow_tf32), "matmul": bool(torch.backends.cuda.matmul.allow_tf32)}

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
        self.half = self.resolve_half(torch)
        self.tf32 = self.apply_tf32(torch)
        self.ctx.log.info("loaded %s on %s (%s, TF32 %s)", name, self.device, "FP16" if self.half else "FP32",
                          self.tf32)

    def _infer(self, frame: np.ndarray, conf: float | None = None):
        # Ultralytics 8.4.178 replaced half with quantize (32 = FP32, 16 = FP16).
        return self.model.predict(frame, imgsz=self.pcfg["imgsz"],
                                  conf=self.pcfg["det_conf_min"] if conf is None else conf,
                                  quantize=16 if self.half else 32, device=self.device,
                                  verbose=False, save=False)[0]

    def _infer_batch(self, frames: list[np.ndarray], conf: float | None = None) -> list:
        """One inference call on several frames (pose.batch). Ultralytics letterboxes
        frames of one size exactly like a single frame (pre_transform auto=same_shapes)."""
        if len(frames) == 1:
            return [self._infer(frames[0]) if conf is None else self._infer(frames[0], conf)]
        return self.model.predict(list(frames), imgsz=self.pcfg["imgsz"],
                                  conf=self.pcfg["det_conf_min"] if conf is None else conf,
                                  quantize=16 if self.half else 32, device=self.device,
                                  verbose=False, save=False)

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
            ref = self._make_ref(sid, idx, fps, res.boxes.xyxy.detach().cpu().numpy(),
                                 res.boxes.conf.detach().cpu().numpy())
            self._save_reference(frame, ref)
            found[sid] = ref
        return self._reference_done(shots, found)

    @staticmethod
    def _make_ref(sid: str, idx: int, fps: float, xyxy: np.ndarray, conf: np.ndarray) -> dict:
        order = np.argsort(xyxy[:, 0], kind="stable")
        boxes = [{"n": i + 1, "xyxy": [round(float(v), 1) for v in xyxy[k]], "conf": round(float(conf[k]), 3)}
                 for i, k in enumerate(order)]
        return {"shot_id": sid, "frame_idx": idx, "t_sec": round(idx / fps, 3), "boxes": boxes}

    def _reference_done(self, shots: list[Shot], found: dict[str, dict]) -> list[dict]:
        """Reference list in shot order, AV-POS-004 for shots without one, manifest section."""
        ctx = self.ctx
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

    def estimate_body_shots(self, run_id: str, shots: list[Shot], video: Path | None = None,
                            with_reference: bool = False,
                            on_shot_done: Callable[[Shot, Path, dict | None], None] | None = None
                            ) -> dict[str, Path]:
        """Pose on every frame of every shot in one pass over the video; one DET
        v1 file per shot (AV-POS-002, 003). Statistics, totals and per shot, go
        to ctx.manifest["pose_body"], with the time breakdown in breakdown_s.

        pose.batch frames go to the model per call (frame order and per frame
        failure accounting are unchanged: a failed batch is retried frame by
        frame; on running out of memory the batch is halved before the fallback
        variant is used). pose.decode picks the frame reader.
        with_reference (pose.reference_in_pass): the reference frames are taken
        from this pass, as reference_frames would find them; the list is in
        self.pass_refs. on_shot_done(shot, det_path, ref) is called as soon as a
        shot's DET file is written (tracking overlap)."""
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

        ref_end: dict[str, int] = {}
        found: dict[str, dict] = {}
        fps = self._fps(video) if with_reference else None
        if with_reference:
            ref_end = {s.shot_id: min(s.end_frame, s.start_frame + round(self.pcfg["ref_search_s"] * fps))
                       for s in shots}

        failed: list[tuple[int, str, str]] = []
        state = {"fell_back": None, "batch": max(1, int(self.pcfg["batch"]))}
        total = int((shot_of >= 0).sum())
        n_frames = 0
        bd = dict.fromkeys(("decode", "infer_call", "preprocess", "inference", "postprocess", "extract",
                            "writing"), 0.0)
        t0 = last_log = time.monotonic()
        every = self.cfg["run"]["progress_every_s"]
        acc = {s.shot_id: {"rows": [[], [], [], [], []], "frames": 0, "failed": 0} for s in shots}
        paths: dict[str, Path] = {}
        current = None

        def flush(k: int) -> None:
            tw = time.monotonic()
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
            bd["writing"] += time.monotonic() - tw
            if on_shot_done is not None:
                on_shot_done(s, paths[s.shot_id], found.get(s.shot_id))

        def run_frames(idxs: list[int], frames: list[np.ndarray]) -> list:
            """Results per frame (None = failed, recorded in failed)."""
            ti = time.monotonic()
            try:
                out = self._infer_batch(frames)
            except torch.cuda.OutOfMemoryError:
                if len(frames) > 1:
                    half = max(1, len(frames) // 2)
                    if state["batch"] > half:
                        log.warning("GPU out of memory with batch %d, batch %d from frame %d on",
                                    len(frames), half, idxs[0])
                        state["batch"] = half
                    return run_frames(idxs[:half], frames[:half]) + run_frames(idxs[half:], frames[half:])
                if not fallback or fallback == self.variant or state["fell_back"]:
                    raise AvError("AV-POS-002", variant=self.variant, action="No fallback variant is left.")
                old = self.variant
                log.warning("GPU out of memory with variant %s on frame %d, switching to %s", old, idxs[0], fallback)
                self.load_models(fallback)
                state["fell_back"] = {"from": old, "to": fallback, "frame": idxs[0]}
                ctx.notice(AvError("AV-POS-002", variant=old,
                                   action=f"Variant {fallback} is used from frame {idxs[0]} on."), "pose_body")
                try:
                    out = self._infer_batch(frames)   # retry once; a second OOM ends the stage
                except torch.cuda.OutOfMemoryError as e:
                    raise AvError("AV-POS-002", variant=self.variant, action="The fallback variant also failed.") from e
            except AvError:
                raise
            except Exception as e:
                if len(frames) > 1:               # find the bad frame(s): one by one
                    return [r for i, f in zip(idxs, frames) for r in run_frames([i], [f])]
                failed.append((idxs[0], type(e).__name__, str(e)[:200]))
                log.warning("AV-POS-003 frame %d: %s: %s", idxs[0], type(e).__name__, e)
                return [None]
            bd["infer_call"] += time.monotonic() - ti
            for r in out:
                sp = getattr(r, "speed", None)
                if isinstance(sp, dict):
                    for key in ("preprocess", "inference", "postprocess"):
                        bd[key] += float(sp.get(key) or 0.0) / 1000.0
            return list(out)

        def consume(idx: int, frame: np.ndarray, res) -> None:
            nonlocal current
            k = int(shot_of[idx])
            if current is not None and k != current:
                flush(current)
            current = k
            s = shots[k]
            a = acc[s.shot_id]
            a["frames"] += 1
            if res is None:
                a["failed"] += 1
                return
            te = time.monotonic()
            n = 0 if res.boxes is None else len(res.boxes)
            if n and res.keypoints is not None:
                kp = res.keypoints.data.detach().cpu().numpy()          # [n,17,3]: x, y, conf
                xyxy = res.boxes.xyxy.detach().cpu().numpy()
                conf = res.boxes.conf.detach().cpu().numpy()
                fi, bx, dc, bxy, bcf = a["rows"]
                fi.append(np.full(n, idx, np.int32))
                bx.append(xyxy)
                dc.append(conf)
                bxy.append(kp[:, :, :2])
                bcf.append(kp[:, :, 2] if kp.shape[2] > 2 else np.ones(kp.shape[:2], np.float32))
                if with_reference and s.shot_id not in found and idx < ref_end[s.shot_id]:
                    keep = conf >= self.pcfg["ref_conf_min"]      # the boxes a conf = ref_conf_min call returns
                    if keep.any():
                        ref = self._make_ref(s.shot_id, idx, fps, xyxy[keep], conf[keep])
                        self._save_reference(frame, ref)
                        found[s.shot_id] = ref
            bd["extract"] += time.monotonic() - te

        reader = frame_reader(self.pcfg["decode"], self.pcfg["prefetch_frames"])
        frames_it = reader(video, first, last, want=lambda i: shot_of[i] >= 0)
        idxs: list[int] = []
        batch: list[np.ndarray] = []
        try:
            while True:
                td = time.monotonic()
                item = next(frames_it, None)
                bd["decode"] += time.monotonic() - td
                if item is not None:
                    idxs.append(item[0])
                    batch.append(item[1])
                    if len(batch) < state["batch"]:
                        continue
                if batch:
                    for i, f, r in zip(idxs, batch, run_frames(idxs, batch)):
                        consume(i, f, r)
                    n_frames += len(batch)
                    idxs, batch = [], []
                if item is None:
                    break
                now = time.monotonic()
                if now - last_log >= every:
                    last_log = now
                    rate = n_frames / (now - t0)
                    eta = (total - n_frames) / rate if rate > 0 else 0
                    log.info("pose %d/%d frames, %.1f fps, eta %.0f s", n_frames, total, rate, eta)
                    self.ctx.progress("pose_body", n_frames / max(total, 1))
        finally:
            close = getattr(frames_it, "close", None)
            if close is not None:
                close()

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
        if with_reference:
            self.pass_refs = self._reference_done(shots, found)

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
            "fallback": state["fell_back"],
            "imgsz": self.pcfg["imgsz"], "conf": self.pcfg["det_conf_min"],
            "half_requested": self.pcfg["half"], "half": self.half, "precision": "FP16" if self.half else "FP32",
            "tf32_requested": self.pcfg["tf32"], "tf32": self.tf32,
            "batch": int(self.pcfg["batch"]), "batch_used": state["batch"], "decode": self.pcfg["decode"],
            "reference_in_pass": bool(with_reference),
            "frames": n_frames,
            "frames_with_person": with_person,
            "detections": dets,
            "failed_frames": len(failed),
            "seconds": round(wall, 2),
            "pose_fps": round(n_frames / wall, 2) if wall > 0 else None,
            "infer_ms_mean": round(1000 * bd["infer_call"] / max(n_frames - len(failed), 1), 2),
            "breakdown_s": {k: round(v, 3) for k, v in bd.items()},
            "gpu_peak_mb": peak_mb,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
            "gpu_cc": self.gpu_cc,
            "gpu_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 2**20) if cuda else None,
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
