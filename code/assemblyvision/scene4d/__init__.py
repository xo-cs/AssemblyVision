"""The scene4d stage (Week 11): the scene of each shot part as coloured points, with the person, the camera
path and, when the alignment works, the GVHMR skeleton (rule w11-scene4d-model chose Pi3, BSD-2-Clause).

Pi3 runs in its own env (envs/scene4d) as a subprocess (pi3_worker.py) on the same shot parts as body3d
(the subject's stretches of a shot). This side prepares the job (subject boxes and 2D keypoints from our
tracking), takes no lock of its own (the caller holds the GPU lock), aligns the GVHMR joints of body3d
with the 2D keypoints lifted by Pi3, and writes the player data. Output, in the run folder only:
<run>/scene4d/<part>.npz (pi3_worker), <part>.align.json, <part>.player.bin and .player.json, summary.json.
No frame is written.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from ..storage.files import REPO_ROOT, utc_now

PYTHON = REPO_ROOT / "envs" / "scene4d" / "bin" / "python"
CKPT = REPO_ROOT / "models" / "scene4d" / "pi3" / "model.safetensors"
PI3 = REPO_ROOT / "third_party" / "Pi3"
MODELS = {"pi3": ("Pi3 (yyfz233/Pi3, BSD-2-Clause)", "Hugging Face yyfz233/Pi3", PI3 / "pi3"),
          "pi3x": ("Pi3X (yyfz233/Pi3X, CC BY-NC 4.0)", "Hugging Face yyfz233/Pi3X", PI3 / "pi3"),
          "vggt": ("VGGT-1B (facebook/VGGT-1B, CC BY-NC 4.0)", "Hugging Face facebook/VGGT-1B",
                   REPO_ROOT / "third_party" / "vggt" / "vggt")}


def missing(model: str = "pi3") -> list[str]:
    out = []
    if not PYTHON.is_file():
        out.append("the env envs/scene4d")
    ckpt = REPO_ROOT / "models" / "scene4d" / model / "model.safetensors"
    if not ckpt.is_file():
        out.append(f"models/scene4d/{model}/model.safetensors ({MODELS[model][1]})")
    if not MODELS[model][2].is_dir():
        out.append(f"the code {MODELS[model][2].relative_to(REPO_ROOT)}")
    return out


def subject_keypoints(manifest: dict, root: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Run frame -> (xy [17, 2], conf [17]) of the subject (KTS v1 files)."""
    from ..kts import read_kts
    out = {}
    for tr in manifest.get("tracking", {}).get("tracks", []):
        if not tr.get("is_subject"):
            continue
        s, _ = read_kts(root / tr["kts_path"])
        for f, ok, xy, c in zip(s.frame_idx, s.present, s.body_xy, s.body_conf):
            if ok:
                out[int(f)] = (xy, c)
    return out


def job_pieces(manifest: dict, root: Path, cfg) -> list[dict]:
    from .. import body3d as B
    fps = float(manifest["ingest"]["cfr"]["fps"])
    bc = cfg["body3d"]
    kps = subject_keypoints(manifest, root)
    nan = (np.full((17, 2), np.nan), np.zeros(17))
    out = []
    for p in B.pieces(manifest, root, fps, bc["min_piece_s"], bc["max_gap_s"]):
        fr = range(p["start"], p["start"] + len(p["boxes"]))
        kp = [kps.get(f, nan) for f in fr]
        out.append({**p, "kp": np.nan_to_num(np.stack([k[0] for k in kp]), nan=-1).round(1).tolist(),
                    "kpc": np.stack([k[1] for k in kp]).round(3).tolist()})
    return out


def run(root: Path, manifest: dict, cfg, gpu: int, log=print, only: list[str] | None = None,
        raw_dir: Path | None = None) -> dict:
    """Runs the scene model on every shot part (or the parts named in only; the GPU lock is the caller's),
    then the alignment and the player data. raw_dir (tests of the rule w11-scene4d-quality only) also keeps
    the model's raw points per part there."""
    sc = cfg["scene4d"]
    model = sc.get("model", "pi3")
    need = missing(model)
    if need:
        raise RuntimeError("scene4d needs " + "; ".join(need))
    cfr = manifest["ingest"]["cfr"]
    fps = float(cfr["fps"])
    job = {"video": str(root / cfr["path"]), "fps": fps, "width": cfr["width"], "height": cfr["height"],
           "pieces": job_pieces(manifest, root, cfg), "out_dir": str(root / "scene4d"),
           "step": max(1, int(round(fps / sc["fps"]))), "chunk": sc["chunk"], "overlap": sc["overlap"],
           "max_bg": sc["max_bg_points"], "person_pts": sc["person_points"], "model": model,
           "conf_drop": sc.get("conf_drop", 0.35), "edge_rtol": sc.get("edge_rtol", 0.03),
           "voxel_div": sc.get("voxel_div", 700), "raw_dir": str(raw_dir) if raw_dir else None}
    if only:
        job["pieces"] = [p for p in job["pieces"] if p["name"].rsplit("-", 1)[-1] in only or p["name"] in only]
    out = root / "scene4d"
    out.mkdir(exist_ok=True)
    (root / "tmp").mkdir(exist_ok=True)
    t0 = time.monotonic()
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=root / "tmp", delete=False) as f:
        json.dump(job, f)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), PYTHONNOUSERSITE="1",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", TMPDIR=str(root / "tmp"))
    try:
        p = subprocess.run([str(PYTHON), str(Path(__file__).with_name("pi3_worker.py")), f.name], env=env,
                           capture_output=True, text=True)
    finally:
        Path(f.name).unlink(missing_ok=True)
    (out / "worker.log").write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        last = p.stderr.strip().splitlines()[-1][:300] if p.stderr.strip() else ""
        raise RuntimeError(f"the scene4d worker ({model}) failed (exit {p.returncode}): {last}; see scene4d/worker.log")
    lines = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
    aligned = {}
    for pc in job["pieces"]:
        if (out / f"{pc['name']}.npz").is_file():
            a = align(root, pc["name"])
            aligned[pc["name"]] = a["ok"]
            write_player(root, pc["name"], fps, a)
    summary = {"created_utc": utc_now(), "model": MODELS[model][0], "settings": {k: job[k] for k in ("conf_drop", "edge_rtol", "voxel_div")}, "fps": fps,
               "sample_fps": fps / job["step"], "pieces": [x for x in lines if "name" in x and "failed" not in x],
               "failed": [x for x in lines if "failed" in x], "aligned": aligned, "seconds": round(time.monotonic() - t0, 1),
               "torch_peak_mb": next((x["torch_peak_mb"] for x in lines if "torch_peak_mb" in x), None)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"scene4d: {len(summary['pieces'])} parts in {summary['seconds']} s, {len(summary['failed'])} failed, "
        f"{sum(aligned.values())} with the skeleton aligned")
    return summary


def align(root: Path, name: str) -> dict:
    """Similarity transform GVHMR world -> Pi3 scene from the 17 joints of body3d against our 2D keypoints
    lifted by Pi3 at the same frames (trimmed least squares). It "works" with at least 30 joint pairs and
    a median residual under 15 % of the body height; then the scene's up direction and ground come from
    the transformed GVHMR ankles (gravity is GVHMR's y axis)."""
    from .pi3_worker import robust_sim3
    z = np.load(root / "scene4d" / f"{name}.npz")
    b = root / "body3d" / f"{name}.npz"
    res = {"ok": False, "reason": "no body3d output for this part"}
    if b.is_file():
        g = np.load(b)
        fidx = {int(f): i for i, f in enumerate(g["frames"])}
        rows = [(fidx[int(f)], i) for i, f in enumerate(z["frames"]) if int(f) in fidx]
        src, dst = [], []
        for gi, si in rows:
            for j in range(17):
                q = z["kp3d"][si, j]
                if np.isfinite(q).all():
                    src.append(g["joints_world"][gi, j])
                    dst.append(q)
        if len(src) < 30:
            res = {"ok": False, "reason": f"only {len(src)} joint pairs"}
        else:
            src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
            s, R, t = robust_sim3(src, dst)
            err = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
            J = g["joints_world"].astype(np.float64)
            height = float(np.median(J[:, 0, 1] - J[:, 15:17, 1].min(1)))     # nose above the ankles (GVHMR, m)
            rel = float(np.median(err) / max(s * height, 1e-6))
            ok = rel < 0.15
            ground = float(np.percentile(J[:, 15:17, 1], 5))
            res = {"ok": ok, "reason": "aligned" if ok else f"median residual {rel:.0%} of the body height",
                   "scale": s, "R": R.tolist(), "t": t.tolist(), "pairs": len(src), "residual_rel": round(rel, 3),
                   "up": (R @ np.array([0, 1.0, 0])).tolist(), "ground_point": (s * R @ np.array([0, ground, 0]) + t).tolist()}
    (root / "scene4d" / f"{name}.align.json").write_text(json.dumps(res, indent=1))
    return res


def write_player(root: Path, name: str, fps: float, a: dict) -> None:
    """<part>.player.json (header) and .player.bin (little endian float32 and uint8 blocks) for the
    three.js player: background, person per sample, cameras, trails (the aligned GVHMR joints at every
    frame when the alignment works, else the lifted keypoints), skeleton (aligned GVHMR)."""
    z = np.load(root / "scene4d" / f"{name}.npz")
    blocks, header, off = [], {"name": name, "fps": fps, "frames": z["frames"].tolist(), "size": z["size"].tolist()}, 0

    def add(key, arr, dtype):
        nonlocal off
        a_ = np.ascontiguousarray(arr, dtype=dtype)
        header[key] = {"offset": off, "shape": list(a_.shape), "dtype": "f32" if dtype == np.float32 else "u8"}
        blocks.append(a_.tobytes())
        off += a_.nbytes
        pad = (-off) % 4
        if pad:
            blocks.append(b"\0" * pad)
            off += pad
    add("bg_pts", z["bg_pts"], np.float32)
    add("bg_col", z["bg_col"], np.uint8)
    if "bg_size" in z.files:                     # per point size (neighbour spacing): far, sparse areas get bigger points
        add("bg_size", z["bg_size"], np.float32)
    add("person_pts", np.nan_to_num(z["person_pts"].astype(np.float32), nan=np.nan), np.float32)
    add("person_col", z["person_col"], np.uint8)
    add("c2w", z["c2w"], np.float32)
    header["aligned"] = bool(a.get("ok"))
    header["align_reason"] = a.get("reason")
    trail = z["kp3d"]
    trail_frames = z["frames"]
    if a.get("ok"):
        g = np.load(root / "body3d" / f"{name}.npz")
        J = a["scale"] * g["joints_world"].astype(np.float64) @ np.asarray(a["R"]).T + np.asarray(a["t"])
        add("skeleton", J.astype(np.float32), np.float32)
        header["skeleton_frames"] = g["frames"].tolist()
        trail, trail_frames = J, g["frames"]
        header["up"], header["ground_point"] = a["up"], a["ground_point"]
    add("trail", np.nan_to_num(np.asarray(trail, np.float32), nan=np.nan), np.float32)
    header["trail_frames"] = np.asarray(trail_frames).tolist()
    header["depth"] = float(z["depth"])
    (root / "scene4d" / f"{name}.player.bin").write_bytes(b"".join(blocks))
    (root / "scene4d" / f"{name}.player.json").write_text(json.dumps(header))


def ensure_players(root: Path, fps: float) -> int:
    """Player data for every scene4d part that has none (e.g. a part made by an older version): the
    alignment and write_player, in the run folder. Returns how many were made."""
    d = root / "scene4d"
    n = 0
    for f in sorted(d.glob("*_p*.npz")) if d.is_dir() else []:
        name = f.stem
        if name.endswith(".part") or (d / f"{name}.player.json").is_file():
            continue
        write_player(root, name, fps, align(root, name))
        n += 1
    return n


def load_parts(root: Path) -> list[dict]:
    """The scene4d parts of a run with player data, in frame order: [{"name", "start", "end"}]."""
    d = root / "scene4d"
    out = []
    for f in sorted(d.glob("*.player.json")) if d.is_dir() else []:
        h = json.loads(f.read_text())
        out.append({"name": h["name"], "start": h["frames"][0], "end": h["frames"][-1], "aligned": h["aligned"]})
    return sorted(out, key=lambda x: x["start"])
