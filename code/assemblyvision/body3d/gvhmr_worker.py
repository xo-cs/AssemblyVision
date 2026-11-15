"""GVHMR worker of the body3d stage (Week 11). Runs in envs/gvhmr (Python 3.10, torch 2.3.0+cu121),
with the GVHMR repository (third_party/GVHMR) as working directory, never in the av env:

    envs/gvhmr/bin/python code/assemblyvision/body3d/gvhmr_worker.py JOB.json

GVHMR (Shen et al., SIGGRAPH Asia 2024; code and weights for non commercial research only) estimates
world grounded human motion from monocular video: the body in a gravity aligned world frame (y up).
Per piece of a shot (a stretch with the subject, from the job), it takes the subject's boxes from our
tracking (no detector of its own), reads the frames of the piece in chunks into memory (nothing is
written but the result), runs ViTPose and the HMR2 image features on the crops, the camera rotation
by SimpleVO (SIFT) for a handheld shot or none for a fixed one, and the GVHMR network. The output per
piece is npz: frames (run frame index), joints_world [L, 17, 3] in metres (COCO 17 order, from
SMPL-X), transl [L, 3]. The world frame starts again at every piece (cuts reset it).
The job file: {"video", "fps", "width", "height", "pieces": [{"name", "start", "boxes" [L, 4] xyxy,
"static"}], "out_dir", "chunk"}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch


def read_frames(video: str, start: int, n: int, scale: float, step: int = 1):
    """Yields (index in the piece, RGB frame scaled by scale) for frames start .. start + n - 1."""
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"frame {start + i} could not be read")
        if i % step == 0:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if scale != 1.0:
                f = cv2.resize(f, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            yield i, f
    cap.release()


def camera_rotation(video: str, start: int, n: int, step: int = 8) -> torch.Tensor:
    """R_w2c [n, 3, 3] of a handheld piece by GVHMR's SimpleVO (SIFT, every step frames, half size)."""
    from hmr4d.utils.preproc.relpose.simple_vo import (CameraParams, Matcher, TwoPairSolver, focal_length_from_mm,
                                                       interpolate_missing_frames, SimpleVO)
    idx, frames = [], []
    for i, f in read_frames(video, start, n, 0.5, step):
        idx.append(i)
        frames.append(f)
    if idx[-1] != n - 1:                                   # the last frame too, as SimpleVO does
        for i, f in read_frames(video, start + n - 1, 1, 0.5):
            idx.append(n - 1)
            frames.append(f)
    frames = np.stack(frames)
    H, W = frames.shape[1:3]
    vo = SimpleVO("", scale=0.5, step=step)
    solver = TwoPairSolver(CameraParams(W, H, focal_length=focal_length_from_mm(W, H, vo.f_mm)), solver="pycolmap")
    T = vo.process_video_T_w2c_list_np(frames, Matcher("sift"), solver)
    T = interpolate_missing_frames(T, np.asarray(idx))
    return torch.from_numpy(np.asarray(T)[:, :3, :3]).float()


def run_piece(model, vitpose, extractor, smplx_coco17, job: dict, piece: dict) -> dict:
    from hmr4d.utils.geo.hmr_cam import estimate_K, get_bbx_xys_from_xyxy
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.net_utils import detach_to_cpu
    from hmr4d.utils.preproc.vitfeat_extractor import get_batch
    boxes = torch.tensor(piece["boxes"], dtype=torch.float32)
    L = len(boxes)
    bbx_xys = get_bbx_xys_from_xyxy(boxes, base_enlarge=1.2).float()
    kp2d, feats = [], []
    chunk = int(job.get("chunk", 128))
    buf = []
    for i, f in read_frames(job["video"], piece["start"], L, 0.5):
        buf.append(f)
        if len(buf) == chunk or i == L - 1:
            a = i + 1 - len(buf)
            sl = bbx_xys[a:i + 1]
            imgs, _ = get_batch(np.stack(buf), sl * 0.5, img_ds=1.0, path_type="np")   # crops 256 px, normalised
            kp2d.append(vitpose.extract(imgs, sl))
            feats.append(extractor.extract_video_features(imgs, sl))
            buf = []
    W, H = int(job["width"]), int(job["height"])
    R_w2c = torch.eye(3).repeat(L, 1, 1) if piece["static"] else camera_rotation(job["video"], piece["start"], L)
    if model is None:                                       # preprocess only (a check without SMPL-X)
        return {"kp2d": torch.cat(kp2d).numpy(), "features": torch.cat(feats).numpy(), "R_w2c": R_w2c.numpy()}
    data = {"length": torch.tensor(L), "bbx_xys": bbx_xys, "kp2d": torch.cat(kp2d), "K_fullimg": estimate_K(W, H).repeat(L, 1, 1),
            "cam_angvel": compute_cam_angvel(R_w2c), "f_imgseq": torch.cat(feats)}
    pred = detach_to_cpu(model.predict(data, static_cam=bool(piece["static"])))
    p = {k: v.cuda() for k, v in pred["smpl_params_global"].items()}
    with torch.no_grad():
        joints = smplx_coco17(**p)                          # [L, 17, 3], y up (gravity aligned)
    return {"joints_world": joints.cpu().numpy().astype(np.float32),
            "transl": pred["smpl_params_global"]["transl"].numpy().astype(np.float32)}


def main(job_path: str) -> int:
    from hydra import compose, initialize_config_module
    import hydra
    from hmr4d.configs import register_store_gvhmr
    from hmr4d.utils.preproc import Extractor, VitPoseExtractor
    from hmr4d.utils.smplx_utils import make_smplx
    job = json.loads(Path(job_path).read_text())
    out = Path(job["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    pre = bool(job.get("preprocess_only"))
    model = smplx_coco17 = None
    if not pre:
        with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
            register_store_gvhmr()
            cfg = compose(config_name="demo", overrides=["video_name=body3d", "static_cam=True"])
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        model.load_pretrained_model(cfg.ckpt_path)
        model = model.eval().cuda()
        smplx_coco17 = make_smplx("supermotion_coco17").cuda()
    vitpose, extractor = VitPoseExtractor(tqdm_leave=False), Extractor(tqdm_leave=False)
    report = []
    for piece in job["pieces"]:
        t = time.monotonic()
        res = run_piece(model, vitpose, extractor, smplx_coco17, job, piece)
        if pre:
            report.append({"name": piece["name"], "frames": len(piece["boxes"]), "seconds": round(time.monotonic() - t, 1),
                           "kp2d": list(res["kp2d"].shape), "features": list(res["features"].shape),
                           "rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(res["R_w2c"][-1]) - 1) / 2,
                                                                                      -1, 1)))), 1)})
            print(json.dumps(report[-1]), flush=True)
            continue
        frames = np.arange(piece["start"], piece["start"] + len(piece["boxes"]), dtype=np.int32)
        tmp = out / f"{piece['name']}.part.npz"
        np.savez_compressed(tmp, frames=frames, **res, static=np.bool_(piece["static"]))
        tmp.replace(out / f"{piece['name']}.npz")
        report.append({"name": piece["name"], "frames": int(len(frames)), "seconds": round(time.monotonic() - t, 1)})
        print(json.dumps(report[-1]), flush=True)
    peak = torch.cuda.max_memory_allocated() / 1e6
    print(json.dumps({"done": len(report), "seconds": round(time.monotonic() - t0, 1), "torch_peak_mb": round(peak)}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
