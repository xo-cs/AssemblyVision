"""Engine B (Week 10): a taxonomy free boundary scorer, the probability of an action change per frame.

Model: a small MS TCN (Abu Farha and Gall, CVPR 2019) with [engine_b] stages
stages of layers dilated residual layers each (dilation 1, 2, 4, ...; kernel 3,
not causal), channels wide. Stage 1 sees the 55 body frame features of
datasets/engine_b.py; every later stage refines the sigmoid of the stage before.
One output channel per frame: the boundary logit. The loss is the weighted
binary cross entropy of every stage over the labelled frames (pos_weight
against the class imbalance; 0 = negatives / positives of the training targets).

Time base: model_fps (10 fps). A video is sampled at t = k / model_fps (nearest
frame). Targets: every (action, repetition) change at time c gives
exp(-(t - c)^2 / (2 tol_s^2)), the maximum over the changes ([boundaries] tol_s,
ignore_dont_care), so a frame on a change is 1 and one tol_s away is 0.61.

Evaluation: peaks of the probability (local maxima at or above peak_threshold,
at least peak_min_gap_s apart, the higher first) are matched one to one to the
true change times within a tolerance; precision, recall and F1 per tolerance.

Use in Engine A (rules.py): run_probability() gives the probability on the run
clock from the subject's KTS (YOLO26 x keypoints, same features), pooled over
pool_s, as one more evidence term of the DP.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FORMAT = "assemblyvision engine_b boundary_tcn v1"


# time base and targets --------------------------------------------------------------

def sample_index(n_frames: int, fps: float, model_fps: float) -> np.ndarray:
    """Frame index of every model step t = k / model_fps (nearest frame, inside the video)."""
    if n_frames <= 0:
        return np.zeros(0, np.int64)
    n = int(np.floor((n_frames - 1) / fps * model_fps)) + 1
    return np.minimum(np.round(np.arange(n) * fps / model_fps).astype(np.int64), n_frames - 1)


def change_times(labels: np.ndarray, fps: float, ignore_dont_care: bool, dont_care: int = 0,
                 labelled: np.ndarray | None = None) -> np.ndarray:
    """Times (s) of the (action, repetition) changes of a [T,2] label array. A change at
    frame c lies between frames c - 1 and c, so at (c - 0.5) / fps. With labelled, only
    changes between two labelled frames count."""
    from ...datasets.base import change_frames
    c = np.asarray(change_frames(labels, ignore_dont_care, dont_care), np.int64)
    if labelled is not None and len(c):
        c = c[labelled[c - 1] & labelled[c]]
    return (c - 0.5) / fps


def soft_targets(t: np.ndarray, changes_s: np.ndarray, tol_s: float) -> np.ndarray:
    """Gaussian bumps of sigma tol_s on the change times, the maximum where they overlap."""
    if len(changes_s) == 0:
        return np.zeros(len(t), np.float32)
    d = t[:, None] - changes_s[None, :]
    return np.exp(-0.5 * (d / max(tol_s, 1e-6)) ** 2).max(axis=1).astype(np.float32)


# peaks and boundary F1 -------------------------------------------------------------

def peaks(prob: np.ndarray, threshold: float, min_gap: int) -> np.ndarray:
    """Indices of local maxima >= threshold, greedy from the highest, at least min_gap apart."""
    p = np.asarray(prob, np.float64)
    if len(p) == 0:
        return np.zeros(0, np.int64)
    left = np.r_[-np.inf, p[:-1]]
    right = np.r_[p[1:], -np.inf]
    cand = np.flatnonzero((p >= threshold) & (p >= left) & (p > right))
    keep: list[int] = []
    for i in cand[np.argsort(-p[cand], kind="stable")]:
        if all(abs(int(i) - k) >= max(min_gap, 1) for k in keep):
            keep.append(int(i))
    return np.array(sorted(keep), np.int64)


def match_boundaries(pred_s: np.ndarray, true_s: np.ndarray, tol_s: float) -> int:
    """True positives: one to one matches within tol_s, the closest pairs first."""
    pairs = sorted((abs(p - t), i, j) for i, p in enumerate(pred_s) for j, t in enumerate(true_s)
                   if abs(p - t) <= tol_s + 1e-9)
    used_p, used_t, tp = set(), set(), 0
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            tp += 1
    return tp


def boundary_f1(videos: list[tuple[np.ndarray, np.ndarray]], tols: list[float]) -> dict:
    """videos: (predicted times, true times) per video. Pooled precision, recall, F1 per tolerance."""
    out = {}
    n_pred = sum(len(p) for p, _ in videos)
    n_true = sum(len(t) for _, t in videos)
    for tol in tols:
        tp = sum(match_boundaries(p, t, tol) for p, t in videos)
        pr = tp / n_pred if n_pred else 0.0
        rc = tp / n_true if n_true else 0.0
        out[f"{tol:g}"] = {"tp": tp, "predicted": n_pred, "true": n_true, "precision": round(pr, 4),
                           "recall": round(rc, 4), "f1": round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0}
    return out


def pool(prob: np.ndarray, n: int) -> np.ndarray:
    """Centred moving average over n samples (n <= 1: unchanged); edges average what exists."""
    p = np.asarray(prob, np.float64)
    if n <= 1 or len(p) == 0:
        return p.copy()
    k = np.ones(int(n))
    num = np.convolve(p, k, mode="same")
    den = np.convolve(np.ones(len(p)), k, mode="same")
    return num / den


# the model -------------------------------------------------------------------------------

def build_model(n_in: int, stages: int, layers: int, channels: int, dropout: float):
    import torch
    from torch import nn

    class Residual(nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.conv = nn.Conv1d(channels, channels, 3, padding=d, dilation=d)
            self.out = nn.Conv1d(channels, channels, 1)
            self.drop = nn.Dropout(dropout)

        def forward(self, x, m):
            return (x + self.drop(self.out(torch.relu(self.conv(x))))) * m

    class Stage(nn.Module):
        def __init__(self, n: int):
            super().__init__()
            self.inp = nn.Conv1d(n, channels, 1)
            self.layers = nn.ModuleList([Residual(2 ** i) for i in range(layers)])
            self.head = nn.Conv1d(channels, 1, 1)

        def forward(self, x, m):
            h = self.inp(x) * m
            for layer in self.layers:
                h = layer(h, m)
            return self.head(h) * m

    class BoundaryTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.stages = nn.ModuleList([Stage(n_in)] + [Stage(1) for _ in range(stages - 1)])

        def forward(self, x, m):
            """x [B, F, T], m [B, 1, T] (1 = real frame) -> logits [S, B, T] of every stage."""
            outs, h = [], x
            for k, st in enumerate(self.stages):
                y = st(h if k == 0 else torch.sigmoid(outs[-1]).unsqueeze(1) * m, m)
                outs.append(y.squeeze(1))
            return torch.stack(outs)

    return BoundaryTCN()


def model_settings(cfg) -> dict:
    e = cfg["engine_b"]
    return {k: e[k] for k in ("model_fps", "stages", "layers", "channels", "dropout")}


def save_model(path: Path, model, settings: dict, extra: dict) -> str:
    """Weights plus the settings needed to rebuild the model; returns the sha256 of the file."""
    import torch
    from ...datasets.engine_b import FEATURE_VERSION, N_FEATURES
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"format": FORMAT, "settings": settings, "n_features": N_FEATURES,
                "feature_version": FEATURE_VERSION, "extra": extra,
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, tmp)
    tmp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(path: Path, device: str = "cpu"):
    """(model in eval mode, its settings) from a file written by save_model."""
    import torch
    from ...datasets.engine_b import FEATURE_VERSION
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck.get("format") != FORMAT or ck.get("feature_version") != FEATURE_VERSION:
        raise ValueError(f"{path} is not an Engine B model of feature version {FEATURE_VERSION}")
    s = ck["settings"]
    m = build_model(ck["n_features"], s["stages"], s["layers"], s["channels"], s["dropout"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), s


def predict(model, feats: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Boundary probability [T] of the last stage for features [T, F] at the model fps."""
    import torch
    if len(feats) == 0:
        return np.zeros(0, np.float32)
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(feats.T[None], np.float32)).to(device)
        m = torch.ones(1, 1, x.shape[2], device=device)
        return torch.sigmoid(model(x, m)[-1, 0]).cpu().numpy().astype(np.float32)


# Engine A evidence on a run ---------------------------------------------------------

def run_probability(manifest: dict, run_root: Path, cfg, model_path: Path) -> dict:
    """Boundary probability per run frame from the subject KTS of every shot (CPU).
    Frames without the subject get 0. Returns {"prob": [N] float32, "model": file name,
    "model_sha256", "frames_scored"}; nothing is written."""
    from ...datasets.engine_b import features
    from ...kts import read_kts
    model, s = load_model(model_path)
    cfr = manifest["ingest"]["cfr"]
    fps, n = float(cfr["fps"]), int(cfr["frame_count"])
    image_h = float(cfr["height"])
    prob = np.zeros(n, np.float32)
    scored = 0
    for tr in manifest.get("tracking", {}).get("tracks", []):
        if not tr.get("is_subject"):
            continue
        series, _ = read_kts(run_root / tr["kts_path"])
        f0 = int(series.frame_idx[0]) if len(series.frame_idx) else 0
        T = min(len(series.present), n - f0)
        if T <= 1:
            continue
        xy = np.where(series.present[:T, None, None], series.body_xy[:T], np.nan)
        conf = np.where(series.present[:T, None], np.nan_to_num(series.body_conf[:T]), 0.0)
        idx = sample_index(T, fps, s["model_fps"])
        p = predict(model, features(xy[idx], conf[idx], image_h))
        t_model = idx / fps
        frame_t = np.arange(T) / fps
        prob[f0:f0 + T] = np.interp(frame_t, t_model, p).astype(np.float32) * series.present[:T]
        scored += int(series.present[:T].sum())
    return {"prob": prob, "model": model_path.name, "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "frames_scored": scored}


def describe(path: Path) -> dict:
    """Settings and extra info stored in a model file (no weights)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return json.loads(json.dumps({k: ck[k] for k in ("settings", "extra", "feature_version")}, default=str))
