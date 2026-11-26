"""Objects for the experimental tool pickup and object placement measures (Week 12, analysis/hand_events.py). CPU only.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.detect_objects --run RUN_DIR [--fps 2] [--threads 8]

YOLO26 m (COCO, Ultralytics assets v8.4.0, models/object_det, checksum in SHA256SUMS.txt) on the run's video at --fps,
only the portable classes of hand_events.PORTABLE. Writes objects.npz in the run folder (a QiSens run's stays under
data/qisens): boxes and classes, no image. Prints only counts and the time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # never a GPU

import numpy as np

from ..config import Config
from ..storage.files import REPO_ROOT, run_source, utc_now

WEIGHTS = REPO_ROOT / "models" / "object_det" / "yolo26m.pt"
CONF = 0.25


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.detect_objects")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args(argv)
    import cv2
    import torch
    from ultralytics import YOLO
    from ..analysis.hand_events import PORTABLE
    torch.set_num_threads(a.threads)
    _, root = run_source(a.run, Config.load()["paths"])
    m = json.loads((root / "manifest.json").read_text())
    cfr = m["ingest"]["cfr"]
    model = YOLO(str(WEIGHTS))
    names = [model.names[i] for i in range(len(model.names))]
    keep = [i for i, n in enumerate(names) if n in PORTABLE]
    step = max(1, int(round(float(cfr["fps"]) / a.fps)))
    cap = cv2.VideoCapture(str(root / cfr["path"]))
    frames, offset, boxes, cls, conf = [], [0], [], [], []
    t0, f = time.monotonic(), 0
    while True:
        if not cap.grab():
            break
        if f % step == 0:
            ok, img = cap.retrieve()
            if not ok:
                break
            r = model.predict(img, device="cpu", conf=CONF, classes=keep, verbose=False)[0]
            frames.append(f)
            boxes.append(r.boxes.xyxy.cpu().numpy().astype(np.float32))
            cls.append(r.boxes.cls.cpu().numpy().astype(np.int16))
            conf.append(r.boxes.conf.cpu().numpy().astype(np.float32))
            offset.append(offset[-1] + len(cls[-1]))
        f += 1
    cap.release()
    tmp = root / "objects.part.npz"
    np.savez_compressed(tmp, frames=np.asarray(frames, np.int64), offset=np.asarray(offset, np.int64),
                        boxes=np.concatenate(boxes) if boxes else np.zeros((0, 4), np.float32),
                        cls=np.concatenate(cls) if cls else np.zeros(0, np.int16),
                        conf=np.concatenate(conf) if conf else np.zeros(0, np.float32),
                        meta=json.dumps({"model": "yolo26m (COCO)", "fps": a.fps, "conf": CONF, "names": names, "date_utc": utc_now()}))
    tmp.replace(root / "objects.npz")
    print(f"detect_objects: {len(frames)} frames, {offset[-1]} portable objects, {time.monotonic() - t0:.0f} s on the CPU")
    return 0


if __name__ == "__main__":
    sys.exit(main())
