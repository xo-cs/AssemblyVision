"""Zero shot phase scores with SigLIP 2 (Week 11): the siglip_scores stage.

Model: google/siglip2-so400m-patch14-384 (Apache 2.0), models/siglip2/so400m-patch14-384,
checksums in models/SHA256SUMS_siglip2.txt, loaded offline. Per phase the text embedding is
the mean of the unit embeddings of its descriptions (prompt ensembling,
analysis/phase_text.py), unit length again. Frames are sampled at fps by ffmpeg (long
side at most max_px, never written to disk). Two views per sample: the full frame, and the
subject's crop (its tracked box, padded by pad on every side, clipped to the frame; none
without a subject). Scores are the cosine similarities [N, K] of each view with each phase.
The image embeddings of both views are stored too, so new descriptions can be scored
without the GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

MODEL_DIR = "siglip2/so400m-patch14-384"
REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"


def load(models_dir: Path, device: str = "cpu", half: bool = False):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoModel, AutoProcessor
    path = models_dir / MODEL_DIR
    model = AutoModel.from_pretrained(path).eval().to(device)
    if half:
        model = model.half()
    return model, AutoProcessor.from_pretrained(path)


def _features(out):
    return out if hasattr(out, "shape") else out.pooler_output


def text_bank(model, proc, phases: list[dict], device: str = "cpu") -> np.ndarray:
    """[K, D] unit phase embeddings: the mean of each phase's description embeddings."""
    import torch
    rows = []
    for p in phases:
        with torch.inference_mode():
            t = proc(text=p["descriptions"], padding="max_length", max_length=64, truncation=True, return_tensors="pt")
            e = _features(model.get_text_features(**{k: v.to(device) for k, v in t.items()})).float()
        e = torch.nn.functional.normalize(e, dim=1).mean(0)
        rows.append(torch.nn.functional.normalize(e, dim=0).cpu().numpy())
    return np.stack(rows).astype(np.float32)


def text_sha(phases: list[dict]) -> str:
    return hashlib.sha256(json.dumps(phases, sort_keys=True).encode()).hexdigest()[:16]


def sample_frames(video: Path, fps: float, max_px: int):
    """Yields (time_s, uint8 RGB [H, W, 3], scale) at fps; scale = output size / video size."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    s = min(1.0, max_px / max(w, h))
    ow, oh = int(round(w * s / 2) * 2), int(round(h * s / 2) * 2)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-vf", f"fps={fps},scale={ow}:{oh}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n, k = ow * oh * 3, 0
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield k / fps, np.frombuffer(buf, np.uint8).reshape(oh, ow, 3), ow / w
            k += 1
    finally:
        proc.stdout.close()
        proc.wait()


def crop(frame: np.ndarray, box: np.ndarray | None, pad: float) -> np.ndarray | None:
    if box is None or not np.isfinite(box).all():
        return None
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    if bw <= 2 or bh <= 2:
        return None
    H, W = frame.shape[:2]
    x0, x1 = int(max(0, x0 - pad * bw)), int(min(W, x1 + pad * bw))
    y0, y1 = int(max(0, y0 - pad * bh)), int(min(H, y1 + pad * bh))
    return frame[y0:y1, x0:x1] if x1 - x0 > 2 and y1 - y0 > 2 else None


def embed_images(model, proc, images: list[np.ndarray], device: str, half: bool = False) -> np.ndarray:
    import torch
    with torch.inference_mode():
        x = proc(images=images, return_tensors="pt")["pixel_values"].to(device)
        e = _features(model.get_image_features(pixel_values=x.half() if half else x)).float()
    return torch.nn.functional.normalize(e, dim=1).cpu().numpy().astype(np.float32)


def subject_boxes(manifest: dict, root: Path) -> dict[int, np.ndarray]:
    """Run frame -> the subject's box (xyxy, run pixels) over every subject track."""
    from ..kts import read_kts
    out: dict[int, np.ndarray] = {}
    for tr in manifest.get("tracking", {}).get("tracks", []):
        if not tr.get("is_subject"):
            continue
        s, _ = read_kts(root / tr["kts_path"])
        for f, ok, b in zip(s.frame_idx, s.present, s.bbox_xyxy):
            if ok:
                out[int(f)] = b
    return out


def score_video(model, proc, bank: np.ndarray, video: Path, boxes: dict[int, np.ndarray], run_fps: float,
                fps: float = 2.0, max_px: int = 1280, pad: float = 0.2, batch: int = 32, device: str = "cpu",
                half: bool = False) -> dict:
    times, full, crops, crop_ok = [], [], [], []
    buf_f, buf_c, idx_c = [], [], []

    def flush():
        if buf_f:
            full.append(embed_images(model, proc, buf_f, device, half))
        if buf_c:
            e = embed_images(model, proc, buf_c, device, half)
            for i, row in zip(idx_c, e):
                crops[i] = row
        buf_f.clear(), buf_c.clear(), idx_c.clear()
    for t, frame, scale in sample_frames(video, fps, max_px):
        times.append(t)
        f = int(round(t * run_fps))
        b = boxes.get(f)
        if b is None:                                   # the nearest subject frame within half a sample
            near = [boxes.get(f + d) for d in range(-int(run_fps / fps / 2), int(run_fps / fps / 2) + 1)]
            b = next((x for x in near if x is not None), None)
        c = crop(frame, None if b is None else np.asarray(b, np.float64) * scale, pad)
        buf_f.append(frame)
        crops.append(None)
        crop_ok.append(c is not None)
        if c is not None:
            buf_c.append(c)
            idx_c.append(len(crops) - 1)
        if len(buf_f) >= batch:
            flush()
    flush()
    full_e = np.concatenate(full) if full else np.zeros((0, bank.shape[1]), np.float32)
    ok = np.asarray(crop_ok, bool)
    crop_e = np.stack([c if c is not None else np.zeros(bank.shape[1], np.float32) for c in crops]) if crops \
        else np.zeros((0, bank.shape[1]), np.float32)
    return {"time_s": np.asarray(times, np.float32), "full_emb": full_e.astype(np.float16),
            "crop_emb": crop_e.astype(np.float16), "crop_ok": ok, "full_cos": full_e @ bank.T,
            "crop_cos": np.where(ok[:, None], crop_e @ bank.T, np.nan).astype(np.float32)}


def save(path: Path, res: dict, meta: dict) -> None:
    tmp = path.with_suffix(".part.npz")
    np.savez_compressed(tmp, **res, meta=json.dumps(meta))
    tmp.replace(path)


def load_scores(path: Path) -> dict:
    z = np.load(path)
    out = {k: z[k] for k in z.files if k != "meta"}
    out["meta"] = json.loads(str(z["meta"]))
    return out


def _on_frames(times: np.ndarray, rows: np.ndarray, n_frames: int, fps: float, span_s: float) -> np.ndarray:
    """Samples at times (each covering span_s from its time) -> per frame rows [n_frames, K],
    NaN where no sample covers the frame."""
    t = np.arange(n_frames) / fps
    out = np.full((n_frames, rows.shape[1]), np.nan, np.float32)
    if len(times) == 0:
        return out
    idx = np.searchsorted(times, t, side="right") - 1
    ok = (idx >= 0) & (t < times[np.clip(idx, 0, len(times) - 1)] + span_s)
    out[ok] = rows[idx[ok]]
    return out


def run_phase_scores(root: Path, n_frames: int, fps: float, seg: dict) -> tuple[dict, list[str]]:
    """{"siglip_full", "siglip_crop", "vlm": [n_frames, K]} from the run's siglip.npz and
    vlm.npz for the sources whose weight in seg (seg_a settings) is above 0; and the
    sources whose file is missing."""
    out, missing = {}, []
    if seg["siglip_weight"] > 0 or seg["siglip_crop_weight"] > 0:
        path = root / "siglip.npz"
        if path.is_file():
            z = load_scores(path)
            span = 1.0 / float(z["meta"]["fps"])
            if seg["siglip_weight"] > 0:
                out["siglip_full"] = _on_frames(z["time_s"], z["full_cos"], n_frames, fps, span)
            if seg["siglip_crop_weight"] > 0:
                out["siglip_crop"] = _on_frames(z["time_s"], z["crop_cos"], n_frames, fps, span)
        else:
            missing.append("siglip.npz")
    if seg["vlm_weight"] > 0:
        path = root / "vlm.npz"
        if path.is_file():
            z = load_scores(path)
            out["vlm"] = _on_frames(z["time_s"], z["probs"], n_frames, fps, float(z["meta"]["window_s"]))
        else:
            missing.append("vlm.npz")
    return out, missing
