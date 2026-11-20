"""Pi3 worker of the scene4d stage (Week 11). Runs in envs/scene4d (torch 2.5.1+cu121), never in the av env:

    envs/scene4d/bin/python code/assemblyvision/scene4d/pi3_worker.py JOB.json

Pi3 (Wang et al. 2025, BSD-2-Clause; models/scene4d/pi3) predicts per frame 3D points, a confidence and
the camera pose of a set of frames in one shared frame. Per piece of a shot (from the job) the frames
at about 6 fps are read into memory (nothing is written but the result), in chunks of `chunk` frames
that overlap by `overlap`; each chunk is mapped onto the one before by a similarity transform (scale,
rotation, translation) fitted on the shared frames, so one piece gives one scene. Output per piece,
npz in out_dir:
  frames [S] run frame index; c2w [S,4,4] cameras; size [h, w] of the model input
  bg_pts [M,3], bg_col [M,3] uint8: the static background (outside the subject's box), voxel downsampled
  person_pts [S,P,3] (NaN padded), person_col [S,P,3]: the points inside the subject's box per time
  kp3d [S,17,3]: our 2D keypoints lifted with the predicted points (NaN where not seen)
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
PI3 = ROOT / "third_party" / "Pi3"
VGGT = ROOT / "third_party" / "vggt"
CKPTS = {"pi3": ROOT / "models/scene4d/pi3/model.safetensors", "pi3x": ROOT / "models/scene4d/pi3x/model.safetensors",
         "vggt": ROOT / "models/scene4d/vggt/model.safetensors"}
CKPT = CKPTS["pi3"]


def load_model(kind: str):
    """Pi3 (BSD-2-Clause), Pi3X or VGGT-1B (both CC BY-NC 4.0), on the GPU, eval mode."""
    from safetensors.torch import load_file
    if kind in ("pi3", "pi3x"):
        sys.path.insert(0, str(PI3))
        if kind == "pi3":
            from pi3.models.pi3 import Pi3 as M
        else:
            from pi3.models.pi3x import Pi3X as M
    else:
        sys.path.insert(0, str(VGGT))
        from vggt.models.vggt import VGGT as M
    model = M()
    missing, unexpected = model.load_state_dict(load_file(str(CKPTS[kind])), strict=False)
    if [k for k in missing if "track" not in k] or unexpected:
        raise RuntimeError(f"{kind} weights do not match: {len(missing)} missing, {len(unexpected)} unexpected")
    model.kind = kind
    return model.cuda().eval()


def target_size(w: int, h: int, limit: int = 255000) -> tuple[int, int]:
    """Pi3's own input size rule (pi3.utils.basic.load_images_as_tensor): multiples of 14 under limit pixels."""
    s = math.sqrt(limit / (w * h))
    wt, ht = w * s, h * s
    k, m = round(wt / 14), round(ht / 14)
    while (k * 14) * (m * 14) > limit:
        if k / m > wt / ht:
            k -= 1
        else:
            m -= 1
    return max(1, k) * 14, max(1, m) * 14


def read_sampled(video: str, start: int, idx: list[int], size: tuple[int, int]) -> np.ndarray:
    """RGB frames start + i for i in idx (sorted), resized to size, as [S, h, w, 3] uint8."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    want, out, i = set(idx), [], 0
    while len(out) < len(idx):
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"frame {start + i} could not be read")
        if i in want:
            out.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA))
        i += 1
    cap.release()
    return np.stack(out)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (s, R, t) with dst ~ s R src + t (least squares)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, D, Vt = np.linalg.svd(b.T @ a / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (a ** 2).sum(1).mean())
    return s, R, md - s * R @ ms


def robust_sim3(src, dst, iters: int = 3, keep: float = 0.8):
    """umeyama with the worst (1 - keep) of the pairs dropped, iteratively."""
    sel = np.arange(len(src))
    for _ in range(iters):
        s, R, t = umeyama(src[sel], dst[sel])
        err = np.linalg.norm((s * src @ R.T + t) - dst, axis=1)
        sel = np.argsort(err)[: max(10, int(keep * len(src)))]
    return umeyama(src[sel], dst[sel])


def camera_sim3(c_src, c_dst, p_src, p_dst, m):
    """Similarity transform of a chunk onto the stitched part from the cameras of the shared frames: the
    rotation is the chordal mean of R_dst R_src^T, the scale the median ratio of the points' distances to
    their camera, the shift the mean of the camera positions. (A fit on the points alone let the rotation
    jump by up to 9 degrees at a chunk border on a fixed camera; this keeps the camera path continuous.)"""
    c_src, c_dst = c_src.astype(np.float64), c_dst.astype(np.float64)
    ratios = []
    for i in range(len(c_src)):
        q = m[i]
        if q.sum() >= min(50, q.size // 2):
            ds = np.linalg.norm(p_src[i][q] - c_src[i, :3, 3], axis=-1)
            dd = np.linalg.norm(p_dst[i][q] - c_dst[i, :3, 3], axis=-1)
            ratios.append(np.median(dd / np.maximum(ds, 1e-9)))
    s = float(np.median(ratios)) if ratios else 1.0
    M = sum(c_dst[i, :3, :3] @ c_src[i, :3, :3].T for i in range(len(c_src)))
    U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    t = (c_dst[:, :3, 3] - s * c_src[:, :3, 3] @ R.T).mean(0)
    return s, R, t


def run_model(model, frames: np.ndarray) -> dict:
    """World points [n,h,w,3], a confidence 0..1 [n,h,w] (higher is surer) and cameras c2w [n,4,4]."""
    x = torch.from_numpy(frames).float().permute(0, 3, 1, 2).div(255).cuda()[None]
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        r = model(x)
    if getattr(model, "kind", "pi3") in ("pi3", "pi3x"):
        return {"pts": r["points"][0].float().cpu().numpy(), "conf": torch.sigmoid(r["conf"][0, ..., 0]).float().cpu().numpy(),
                "c2w": r["camera_poses"][0].float().cpu().numpy()}
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    ext, intr = pose_encoding_to_extri_intri(r["pose_enc"].float(), x.shape[-2:])          # w2c [1,n,3,4]
    depth = r["depth"][0].float().cpu().numpy()                                             # [n,h,w,1]
    pts = unproject_depth_map_to_point_map(depth, ext[0].cpu().numpy(), intr[0].cpu().numpy())
    w2c = np.concatenate([ext[0].cpu().numpy(), np.tile([[[0, 0, 0, 1.0]]], (len(depth), 1, 1))], 1)
    conf = r["depth_conf"][0].float().cpu().numpy()                                         # >= 1, higher is surer
    return {"pts": pts.astype(np.float32), "conf": (1 - 1 / np.maximum(conf, 1)).astype(np.float32),
            "c2w": np.linalg.inv(w2c).astype(np.float32)}


def edge_mask(pts: np.ndarray, cam: np.ndarray, rtol: float = 0.03) -> np.ndarray:
    """[h,w] True at depth discontinuities: a 4 neighbour's distance to the camera differs by more than rtol
    (the "flying" points between a foreground edge and the background)."""
    d = np.linalg.norm(pts - cam, axis=-1)
    e = np.zeros(d.shape, bool)
    for a, b in ((d[1:], d[:-1]), (d[:, 1:], d[:, :-1])):
        jump = np.abs(a - b) / np.maximum(np.minimum(a, b), 1e-6) > rtol
        if a.shape[0] != d.shape[0]:
            e[1:] |= jump
            e[:-1] |= jump
        else:
            e[:, 1:] |= jump
            e[:, :-1] |= jump
    return e


def chunked(model, frames: np.ndarray, chunk: int, overlap: int) -> dict:
    """Pi3 on overlapping chunks, each aligned onto the frame of the first."""
    S = len(frames)
    pts = np.zeros(frames.shape[:3] + (3,), np.float32)
    conf = np.zeros(frames.shape[:3], np.float32)
    c2w = np.zeros((S, 4, 4), np.float32)
    starts = [0] if S <= chunk else list(range(0, S - overlap, chunk - overlap))
    done, fits = 0, []
    for a in starts:
        b = min(S, a + chunk)
        r = run_model(model, frames[a:b])
        if a > 0:
            o = done - a                                      # shared frames a .. done-1
            s, R, t = camera_sim3(r["c2w"][:o], c2w[a:done], r["pts"][:o], pts[a:done],
                                  (r["conf"][:o] > 0.5) & (conf[a:done] > 0.5))
            pick = np.arange(o)
            r["pts"] = (s * r["pts"] @ R.T + t).astype(np.float32)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R, t
            cw = r["c2w"].astype(np.float64)
            cw[:, :3, 3] *= s
            r["c2w"] = (T @ cw).astype(np.float32)
            fits.append({"chunk": a, "scale": round(s, 4), "frames": int(len(pick))})
        keep = slice(done - a, b - a)                          # the frames not yet stored
        pts[done:b], conf[done:b], c2w[done:b] = r["pts"][keep], r["conf"][keep], r["c2w"][keep]
        done = b
    return {"pts": pts, "conf": conf, "c2w": c2w, "fits": fits}


def voxel_down(p: np.ndarray, c: np.ndarray, voxel: float, max_n: int, f: np.ndarray | None = None,
               min_frames: int = 1) -> tuple[np.ndarray, np.ndarray, float]:
    """Mean point and median colour per voxel; with f (the frame of each point) a voxel must be seen in
    min_frames frames (one frame's noise does not survive); the voxel grows until at most max_n remain."""
    while True:
        keys, inv, cnt = np.unique(np.floor(p / voxel).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
        inv = inv.ravel()
        keep = np.ones(len(keys), bool)
        if f is not None and min_frames > 1:
            pairs = np.unique(np.stack([inv, f]), axis=1)                    # (voxel, frame) once each
            keep = np.bincount(pairs[0], minlength=len(keys)) >= min_frames
        if keep.sum() <= max_n:
            break
        voxel *= 1.15
    P = np.zeros((len(keys), 3))
    np.add.at(P, inv, p)
    P /= cnt[:, None]
    mid = np.r_[0, np.cumsum(cnt)[:-1]] + cnt // 2                            # median colour per voxel and channel
    C = np.stack([c[np.lexsort((c[:, k], inv)), k][mid] for k in range(3)], 1)
    return P[keep].astype(np.float32), C[keep].astype(np.uint8), voxel


def drop_outliers(p: np.ndarray, c: np.ndarray, k: int = 8, z: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Statistical outlier removal: points whose mean distance to their k nearest neighbours lies more than
    z robust deviations above the median."""
    if len(p) < k + 2:
        return p, c, np.zeros(len(p), np.float32)
    from scipy.spatial import cKDTree
    d = cKDTree(p).query(p, k=k + 1)[0][:, 1:].mean(1)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) * 1.4826 + 1e-12
    keep = d < med + z * mad
    return p[keep], c[keep], d[keep]


def run_piece(model, job: dict, piece: dict) -> dict:
    W, H = int(job["width"]), int(job["height"])
    size = target_size(W, H, job.get("pixel_limit", 255000))
    sx, sy = size[0] / W, size[1] / H
    n, step = len(piece["boxes"]), int(job["step"])
    idx = list(range(0, n, step))
    frames = read_sampled(job["video"], piece["start"], idx, size)
    r = chunked(model, frames, int(job["chunk"]), int(job["overlap"]))
    pts, conf, c2w = r["pts"], r["conf"], r["c2w"]
    S, h, w = conf.shape
    boxes = np.asarray(piece["boxes"], np.float64)[idx]
    person = np.zeros((S, h, w), bool)
    for i, b in enumerate(boxes):
        if np.isfinite(b).all():
            pad = 0.15 * (b[2] - b[0])
            x0, x1 = int(max(0, (b[0] - pad) * sx)), int(min(w, (b[2] + pad) * sx + 1))
            y0, y1 = int(max(0, (b[1] - pad) * sy)), int(min(h, (b[3] + pad) * sy + 1))
            person[i, y0:y1, x0:x1] = True
    # quality: per frame the most confident share of the pixels above an absolute floor, no depth edges
    good = np.zeros_like(conf, bool)
    for i in range(S):
        thr = max(job.get("conf_min", 0.3), float(np.quantile(conf[i], job.get("conf_drop", 0.35))))
        good[i] = (conf[i] >= thr) & ~edge_mask(pts[i], c2w[i, :3, 3], job.get("edge_rtol", 0.03))
    rng = np.random.default_rng(0)
    bg = good & ~person
    bp, bc = pts[bg], frames[bg]
    bf = np.nonzero(bg)[0]
    if len(bp) > 6_000_000:
        sel = rng.choice(len(bp), 6_000_000, replace=False)
        bp, bc, bf = bp[sel], bc[sel], bf[sel]
    depth = float(np.median(np.linalg.norm(bp[::50] - c2w[0, :3, 3], axis=1))) if len(bp) else 1.0
    bg_pts, bg_col, voxel = voxel_down(bp.astype(np.float64), bc.astype(np.float64), depth / job.get("voxel_div", 700),
                                       int(job["max_bg"]), bf, 2 if S >= 4 else 1)
    bg_pts, bg_col, nn = drop_outliers(bg_pts, bg_col)
    med = float(np.median(nn)) if len(nn) else voxel
    bg_size = np.clip(nn * 1.6, 0.8 * med, 4 * med).astype(np.float32)          # point size: about the gap to the neighbours
    if job.get("raw_dir"):                                                    # rule tests only (tmp)
        Path(job["raw_dir"]).mkdir(parents=True, exist_ok=True)
        np.savez(Path(job["raw_dir"]) / f"{piece['name']}.npz", pts=pts.astype(np.float16),
                            conf=conf.astype(np.float16), c2w=c2w, person=person, good=good, imgs=frames)
    P = int(job["person_pts"])
    pp = np.full((S, P, 3), np.nan, np.float16)
    pc = np.zeros((S, P, 3), np.uint8)
    for i in range(S):
        m = person[i] & good[i]
        q, c = pts[i][m], frames[i][m]
        if len(q) > P:
            sel = rng.choice(len(q), P, replace=False)
            q, c = q[sel], c[sel]
        pp[i, :len(q)], pc[i, :len(q)] = q, c
    kp = np.asarray(piece["kp"], np.float64)[idx]
    kc = np.asarray(piece["kpc"], np.float64)[idx]
    kp3d = np.full((S, 17, 3), np.nan, np.float32)
    for i in range(S):
        for j in range(17):
            if kc[i, j] >= 0.3 and np.isfinite(kp[i, j]).all():
                u, v = int(round(kp[i, j, 0] * sx)), int(round(kp[i, j, 1] * sy))
                if 0 <= u < w and 0 <= v < h and good[i, v, u]:
                    kp3d[i, j] = pts[i, v, u]
    return {"frames": (piece["start"] + np.asarray(idx)).astype(np.int32), "c2w": c2w, "size": np.array([h, w], np.int32),
            "bg_pts": bg_pts, "bg_col": bg_col, "bg_size": bg_size, "person_pts": pp, "person_col": pc, "kp3d": kp3d,
            "voxel": np.float32(voxel), "depth": np.float32(depth), "fits": json.dumps(r["fits"])}


def main(job_path: str) -> int:
    job = json.loads(Path(job_path).read_text())
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    model = load_model(job.get("model", "pi3"))
    for piece in job["pieces"]:
        t = time.monotonic()
        if (out / f"{piece['name']}.npz").is_file():
            print(json.dumps({"name": piece["name"], "reused": True}), flush=True)
            continue
        try:
            res = run_piece(model, job, piece)
        except Exception as e:                            # one piece must not stop the others
            print(json.dumps({"name": piece["name"], "failed": f"{type(e).__name__}: {e}"[:300]}), flush=True)
            torch.cuda.empty_cache()
            continue
        tmp = out / f"{piece['name']}.part.npz"
        np.savez_compressed(tmp, **res)
        tmp.replace(out / f"{piece['name']}.npz")
        print(json.dumps({"name": piece["name"], "samples": int(len(res["frames"])), "bg_points": int(len(res["bg_pts"])),
                          "seconds": round(time.monotonic() - t, 1)}), flush=True)
    print(json.dumps({"done": True, "seconds": round(time.monotonic() - t0, 1),
                      "torch_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
