"""Visual scene features (Week 11, docs_team/visual_features.md): one DINOv2 ViT S/14
embedding per sampled frame, as Engine B evidence.

The encoder is the pinned local clone of facebookresearch/dinov2 (models/torch_hub/dinov2,
loaded with source="local", no network) with the weights of facebook/dinov2-small
(Apache 2.0, models/dinov2/pytorch_model.bin, checksum in models/SHA256SUMS_dinov2.txt;
the Meta CDN copy could not be fetched: its TLS certificate has expired). The weights are
converted from the transformers layout to the hub layout and loaded strictly.

Frames are sampled at visual.fps by ffmpeg, scaled so the short side is input_px,
centre cropped to input_px square, normalised with the ImageNet mean and std; the
embedding is the CLS token of the last layer norm, unit length. Frames are never written
to disk; the embeddings go to visual.npz in the run folder (or a dataset's derived folder).
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np

HUB_DIR = "torch_hub/dinov2"
WEIGHTS = "dinov2/pytorch_model.bin"
SUMS = "SHA256SUMS_dinov2.txt"
DIM = 384
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def convert_state_dict(sd: dict) -> dict:
    """facebook/dinov2-small (transformers Dinov2Model keys) -> the hub model's keys."""
    import torch
    out = {"cls_token": sd["embeddings.cls_token"], "mask_token": sd["embeddings.mask_token"].reshape(1, -1),
           "pos_embed": sd["embeddings.position_embeddings"],
           "patch_embed.proj.weight": sd["embeddings.patch_embeddings.projection.weight"],
           "patch_embed.proj.bias": sd["embeddings.patch_embeddings.projection.bias"],
           "norm.weight": sd["layernorm.weight"], "norm.bias": sd["layernorm.bias"]}
    n = 1 + max(int(k.split(".")[2]) for k in sd if k.startswith("encoder.layer."))
    for i in range(n):
        a, b = f"encoder.layer.{i}.", f"blocks.{i}."
        for x in ("weight", "bias"):
            out[b + f"norm1.{x}"] = sd[a + f"norm1.{x}"]
            out[b + f"norm2.{x}"] = sd[a + f"norm2.{x}"]
            out[b + f"attn.qkv.{x}"] = torch.cat([sd[a + f"attention.attention.{q}.{x}"]
                                                  for q in ("query", "key", "value")], 0)
            out[b + f"attn.proj.{x}"] = sd[a + f"attention.output.dense.{x}"]
            out[b + f"mlp.fc1.{x}"] = sd[a + f"mlp.fc1.{x}"]
            out[b + f"mlp.fc2.{x}"] = sd[a + f"mlp.fc2.{x}"]
        out[b + "ls1.gamma"] = sd[a + "layer_scale1.lambda1"]
        out[b + "ls2.gamma"] = sd[a + "layer_scale2.lambda1"]
    return out


def check_weights(models_dir: Path) -> Path:
    path = models_dir / WEIGHTS
    want = next(line.split()[0] for line in (models_dir / SUMS).read_text().splitlines()
                if line.strip().endswith(WEIGHTS))
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    if h != want:
        raise RuntimeError(f"{WEIGHTS}: checksum {h} is not the recorded {want}")
    return path


def load_encoder(models_dir: Path, device: str = "cpu", half: bool = False):
    import torch
    model = torch.hub.load(str(models_dir / HUB_DIR), "dinov2_vits14", source="local", pretrained=False)
    sd = torch.load(check_weights(models_dir), map_location="cpu", weights_only=True)
    model.load_state_dict(convert_state_dict(sd), strict=True)
    model = model.eval().to(device)
    return model.half() if half else model


def sample_frames(video: Path, fps: float, px: int):
    """Yields (time_s, uint8 RGB [px, px, 3]) at fps (ffmpeg fps filter, short side px,
    centre crop); nothing is written to disk."""
    vf = (f"fps={fps},scale='if(gt(iw,ih),-2,{px})':'if(gt(iw,ih),{px},-2)':flags=bicubic,"
          f"crop={px}:{px}")
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n, k = px * px * 3, 0
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield k / fps, np.frombuffer(buf, np.uint8).reshape(px, px, 3)
            k += 1
    finally:
        proc.stdout.close()
        proc.wait()
        if proc.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg failed on {video} ({proc.returncode})")


def embed_batch(model, frames: list[np.ndarray], device: str, half: bool = False) -> np.ndarray:
    import torch
    x = (np.stack(frames).astype(np.float32) / 255.0 - MEAN) / STD
    t = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
    with torch.inference_mode():
        out = model(t.half() if half else t).float()
    out = torch.nn.functional.normalize(out, dim=1)
    return out.cpu().numpy()


def video_embeddings(model, video: Path, fps: float = 2.0, px: int = 224, batch: int = 64, device: str = "cpu",
                     half: bool = False) -> dict:
    times, embs, buf = [], [], []
    for t, frame in sample_frames(video, fps, px):
        times.append(t)
        buf.append(frame)
        if len(buf) == batch:
            embs.append(embed_batch(model, buf, device, half))
            buf = []
    if buf:
        embs.append(embed_batch(model, buf, device, half))
    emb = np.concatenate(embs) if embs else np.zeros((0, DIM), np.float32)
    return {"time_s": np.asarray(times, np.float32), "embedding": emb.astype(np.float16)}


def save(path: Path, res: dict, meta: dict) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.npz")
    np.savez_compressed(tmp, time_s=res["time_s"], embedding=res["embedding"], meta=json.dumps(meta))
    tmp.replace(path)


def load(path: Path) -> dict:
    import json
    z = np.load(path)
    return {"time_s": z["time_s"].astype(np.float64), "embedding": z["embedding"].astype(np.float32),
            "meta": json.loads(str(z["meta"]))}


def to_frames(vis: dict, n_frames: int, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """The 2 fps embeddings on the run clock: each frame takes the nearest earlier sample.
    Returns (embedding [n_frames, D], valid [n_frames])."""
    t = np.arange(n_frames) / fps
    ts = vis["time_s"]
    if len(ts) == 0:
        return np.zeros((n_frames, DIM), np.float32), np.zeros(n_frames, bool)
    idx = np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(ts) - 1)
    valid = t <= ts[-1] + 1.0 / max(vis["meta"].get("fps", 2.0), 1e-6)
    return vis["embedding"][idx], valid
