"""Build a 10 s synthetic test clip from the Ultralytics sample images.

    python -m assemblyvision.tools.make_synthetic_clip [--out data/public/videos/synthetic_people_10s.mp4]

Timeline at 25 fps, 1280x720, H.264 yuv420p (so ingest needs no transcode):
  0.0 to 4.0 s  slow pan and zoom over bus.jpg (four people)
  4.0 to 5.0 s  plain grey frames, no person (TC-POS-02)
  5.0 to 10.0 s slow zoom into zidane.jpg (two people)
Only public data: the output must be under data/public/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..storage.files import REPO_ROOT, source_root

W, H, FPS = 1280, 720, 25


def _crop(img: np.ndarray, cx: float, cy: float, zoom: float) -> np.ndarray:
    """Crop with the output aspect ratio around (cx, cy) in 0..1, zoom >= 1, resize to W x H."""
    ih, iw = img.shape[:2]
    cw = min(iw, ih * W / H) / zoom
    ch = cw * H / W
    x0 = np.clip(cx * iw - cw / 2, 0, iw - cw)
    y0 = np.clip(cy * ih - ch / 2, 0, ih - ch)
    m = np.float32([[W / cw, 0, -x0 * W / cw], [0, H / ch, -y0 * H / ch]])
    return cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def frames():
    import ultralytics
    assets = Path(ultralytics.__file__).parent / "assets"
    bus = cv2.imread(str(assets / "bus.jpg"))
    zidane = cv2.imread(str(assets / "zidane.jpg"))
    for i in range(10 * FPS):
        t = i / FPS
        if t < 4.0:
            a = t / 4.0
            yield _crop(bus, 0.5, 0.35 + 0.3 * a, 1.0 + 0.15 * a)
        elif t < 5.0:
            yield np.full((H, W, 3), 128, np.uint8)
        else:
            a = (t - 5.0) / 5.0
            yield _crop(zidane, 0.5, 0.5, 1.0 + 0.3 * a)


def main(argv: list[str] | None = None) -> int:
    public = source_root("public", Config.load()["paths"])
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=public / "videos" / "synthetic_people_10s.mp4")
    out = p.parse_args(argv).out.resolve()
    if not out.is_relative_to(public):
        print(f"refused: {out} is not under {public}", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames():
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        return 1
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
