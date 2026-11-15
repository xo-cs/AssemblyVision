"""The body3d stage (Week 11): world grounded 3D joints of the subject per shot with GVHMR.

GVHMR runs in its own env (envs/gvhmr) as a subprocess (gvhmr_worker.py); this side prepares the job
from the stored run (shots, the subject's boxes), takes the GPU lock and records the stage. The
output is <run>/body3d/<shot_id>_p<k>.npz per piece of a shot (frames, joints_world [L, 17, 3] in
metres, gravity aligned, y up; transl) and <run>/body3d/summary.json. A QiSens run's files stay
under data/qisens. No frame is written.

Licence: GVHMR code and weights are for non commercial research only (third_party/GVHMR/LICENSE).
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

GVHMR = REPO_ROOT / "third_party" / "GVHMR"
PYTHON = REPO_ROOT / "envs" / "gvhmr" / "bin" / "python"
SMPLX = GVHMR / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
CKPTS = {"gvhmr": "gvhmr/gvhmr_siga24_release.ckpt", "hmr2": "hmr2/epoch=10-step=25000.ckpt",
         "vitpose": "vitpose/vitpose-h-multi-coco.pth"}
COCO17 = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist",
          "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]


def missing() -> list[str]:
    """What the stage still needs (empty when it can run)."""
    out = []
    if not PYTHON.is_file():
        out.append("the env envs/gvhmr")
    out += [f"third_party/GVHMR/inputs/checkpoints/{c}" for c in CKPTS.values()
            if not (GVHMR / "inputs" / "checkpoints" / c).is_file()]
    if not SMPLX.is_file():
        out.append("third_party/GVHMR/inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz (SMPL-X, MPI account)")
    return out


def pieces(manifest: dict, root: Path, fps: float, min_s: float, max_gap_s: float) -> list[dict]:
    """Per shot, the stretches with the subject: gaps up to max_gap_s are filled by linear
    interpolation of the box, longer gaps split the shot; pieces shorter than min_s are left out."""
    from ..analysis.siglip import subject_boxes
    boxes = subject_boxes(manifest, root)
    out = []
    for sh in manifest["shots"]["items"]:
        a, b = int(sh["start_frame"]), int(sh["end_frame"])
        have = [f for f in range(a, b) if f in boxes]
        runs, cur = [], []
        for f in have:
            if cur and f - cur[-1] - 1 > max_gap_s * fps:
                runs.append(cur)
                cur = []
            cur.append(f)
        if cur:
            runs.append(cur)
        for k, r in enumerate(x for x in runs if (x[-1] - x[0] + 1) >= min_s * fps):
            fr = np.arange(r[0], r[-1] + 1)
            known = np.array(r)
            xyxy = np.stack([np.interp(fr, known, [float(boxes[f][c]) for f in r]) for c in range(4)], axis=1)
            out.append({"name": f"{sh['shot_id']}_p{k + 1}", "shot_id": sh["shot_id"], "start": int(fr[0]),
                        "boxes": xyxy.round(2).tolist(), "static": sh.get("shot_type") == "fixed"})
    return out


def run(root: Path, manifest: dict, cfg, gpu: int, log=print) -> dict:
    """Runs GVHMR on every piece of the run (the GPU lock is the caller's); returns the summary."""
    need = missing()
    if need:
        raise RuntimeError("body3d needs " + "; ".join(need))
    bc = cfg["body3d"]
    cfr = manifest["ingest"]["cfr"]
    fps = float(cfr["fps"])
    job = {"video": str(root / cfr["path"]), "fps": fps, "width": cfr["width"], "height": cfr["height"],
           "pieces": pieces(manifest, root, fps, bc["min_piece_s"], bc["max_gap_s"]), "out_dir": str(root / "body3d"),
           "chunk": bc["chunk"]}
    t0 = time.monotonic()
    (root / "body3d").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=root, delete=False) as f:
        json.dump(job, f)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), PYTHONNOUSERSITE="1",
               TMPDIR=str(root / "tmp"), HYDRA_FULL_ERROR="1")
    (root / "tmp").mkdir(exist_ok=True)
    try:
        p = subprocess.run([str(PYTHON), str(Path(__file__).with_name("gvhmr_worker.py")), f.name], cwd=GVHMR, env=env,
                           capture_output=True, text=True)
    finally:
        Path(f.name).unlink(missing_ok=True)
    (root / "body3d" / "worker.log").write_text(p.stdout + "\n" + p.stderr)
    if p.returncode != 0:
        last = p.stderr.strip().splitlines()[-1][:300] if p.stderr.strip() else ""
        raise RuntimeError(f"the GVHMR worker failed (exit {p.returncode}): {last}; see body3d/worker.log")
    lines = [json.loads(x) for x in p.stdout.splitlines() if x.startswith("{")]
    summary = {"created_utc": utc_now(), "model": "GVHMR siga24 release", "joints": COCO17, "fps": fps,
               "pieces": [x for x in lines if "name" in x], "seconds": round(time.monotonic() - t0, 1),
               "torch_peak_mb": next((x["torch_peak_mb"] for x in lines if "torch_peak_mb" in x), None),
               "static": {pc["name"]: pc["static"] for pc in job["pieces"]}}
    (root / "body3d" / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"body3d: {len(summary['pieces'])} pieces in {summary['seconds']} s")
    return summary


def load(root: Path) -> list[dict]:
    """The stored pieces of a run: [{"name", "shot_id", "frames", "joints"}] in frame order."""
    d = root / "body3d"
    out = []
    for f in sorted(d.glob("*_p*.npz")) if d.is_dir() else []:
        z = np.load(f)
        out.append({"name": f.stem, "shot_id": f.stem.rsplit("_p", 1)[0], "frames": z["frames"],
                    "joints": z["joints_world"], "static": bool(z["static"])})
    return sorted(out, key=lambda x: int(x["frames"][0]))
