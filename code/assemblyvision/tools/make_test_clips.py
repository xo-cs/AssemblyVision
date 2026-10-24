"""Build the public test clips for shot detection and tracking (public data only).

    python -m assemblyvision.tools.make_test_clips

Writes to data/public/videos/:
  shots_tune_480p30.mp4  + .json   threshold tuning clip (ground truth cuts and types)
  shots_check_480p30.mp4 + .json   held out clip, reported once with the tuned value
  occlusion_480p30.mp4   + .json   train platform clip with a sliding opaque pillar

All clips are 854x480 at 30 fps, like the QiSens video. Ground truth: a cut is
the first frame of each segment. The 0.6 s segment is shorter than
shots.min_shot_s, so the cuts around it are listed in dont_care (it may merge).
Shot types are known only for the synthetic segments (a still image is fixed,
a pan over an image is handheld); real segments have type null.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ..config import Config
from ..storage.files import REPO_ROOT, source_root

W, H, FPS = 854, 480, 30
SRC = {
    "c0": "12649153_1920_1080_25fps.mp4", "c1": "12981246_1920_1080_25fps.mp4",
    "c2": "12981262_1920_1080_25fps.mp4", "c3": "12984031_1920_1080_25fps.mp4",
    "c4": "13751344_3840_2160_50fps.mp4", "c5": "13943788_1920_1080_60fps.mp4",
    "c6": "17145555_1920_1080_50fps.mp4", "c7": "19497508-hd_1920_1080_25fps.mp4",
    "c8": "4477607-hd_1920_1080_30fps.mp4",
}
# (source, start s, frames); "still:<img>" and "pan:<img>" are synthetic
TUNE = [("c8", 0.5, 150), ("c1", 1.0, 240), ("c1", 12.0, 120), ("still:bus", 0, 120), ("c5", 2.0, 18),
        ("c5", 8.0, 180), ("c3", 3.0, 210), ("pan:zidane", 0, 120), ("c6", 0.5, 150), ("c0", 1.0, 180)]
CHECK = [("c2", 5.0, 240), ("c7", 2.0, 210), ("c4", 1.0, 180), ("c2", 40.0, 150), ("still:zidane", 0, 120),
         ("c3", 20.0, 180), ("pan:bus", 0, 120), ("c8", 7.0, 150), ("c1", 6.0, 18), ("c5", 12.0, 150)]
ENC = ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-an"]


def _ff(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", *args], check=True)


def _asset(name: str) -> Path:
    import ultralytics
    return Path(ultralytics.__file__).parent / "assets" / f"{name}.jpg"


def _segment(videos: Path, spec: tuple, out: Path) -> str | None:
    src, start, frames = spec
    if src.startswith("still:"):
        img = _asset(src[6:])
        vf = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1"
        _ff(["-loop", "1", "-framerate", str(FPS), "-i", str(img), "-vf", vf, "-frames:v", str(frames), *ENC, str(out)])
        return "fixed"
    if src.startswith("pan:"):
        img = _asset(src[4:])
        # scale so the image is larger than the frame, then slide a W x H window (about 120 px/s)
        vf = (f"scale={2 * W}:{2 * H}:force_original_aspect_ratio=increase,"
              f"crop={W}:{H}:x='40+n*4':y='40+n*2',setsar=1")
        _ff(["-loop", "1", "-framerate", str(FPS), "-i", str(img), "-vf", vf, "-frames:v", str(frames), *ENC, str(out)])
        return "handheld"
    vf = f"fps={FPS},scale={W}:{H},setsar=1"
    _ff(["-ss", str(start), "-i", str(videos / SRC[src]), "-vf", vf, "-frames:v", str(frames), *ENC, str(out)])
    return None


def build_shot_clip(videos: Path, tmp: Path, name: str, specs: list) -> Path:
    parts, segs, pos = [], [], 0
    for k, spec in enumerate(specs):
        part = tmp / f"{name}_{k:02d}.mp4"
        shot_type = _segment(videos, spec, part)
        parts.append(part)
        segs.append({"start_frame": pos, "frames": spec[2], "source": spec[0], "source_start_s": spec[1],
                     "shot_type": shot_type})
        pos += spec[2]
    lst = tmp / f"{name}.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in parts))
    out = videos / f"{name}.mp4"
    _ff(["-f", "concat", "-safe", "0", "-i", str(lst), "-r", str(FPS), *ENC, str(out)])
    min_frames = FPS * 1.0
    dont_care = set()
    for i, s in enumerate(segs):
        if s["frames"] < min_frames:
            dont_care.add(s["start_frame"])
            if i + 1 < len(segs):
                dont_care.add(segs[i + 1]["start_frame"])
    gt = {"video": out.name, "fps": FPS, "width": W, "height": H, "frame_count": pos,
          "cuts": [s["start_frame"] for s in segs[1:]], "dont_care": sorted(dont_care), "segments": segs,
          "note": "cut = first frame of a segment; dont_care cuts touch a segment shorter than 1 s"}
    out.with_suffix(".json").write_text(json.dumps(gt, indent=2))
    return out


def build_occlusion_clip(videos: Path, tmp: Path) -> Path:
    """Train platform (13943788), 18 s, with a grey pillar sliding left to
    right at 25 px/s so workers are half hidden for a few seconds each."""
    base = tmp / "occl_base.mp4"
    _ff(["-i", str(videos / SRC["c5"]), "-vf", f"fps={FPS},scale={W}:{H},setsar=1", "-t", "18", *ENC, str(base)])
    out = videos / "occlusion_480p30.mp4"
    cap = cv2.VideoCapture(str(base))
    proc = subprocess.Popen(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", *ENC, str(out)], stdin=subprocess.PIPE)
    n, x0, speed, width = 0, 20.0, 25.0, 50
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        x = int(x0 + speed * n / FPS)
        cv2.rectangle(frame, (x, 0), (x + width, H), (85, 85, 90), -1)
        cv2.line(frame, (x, 0), (x, H), (60, 60, 60), 2)
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        n += 1
    cap.release()
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    meta = {"video": out.name, "source": SRC["c5"], "fps": FPS, "frames": n,
            "pillar": {"x0_px": x0, "speed_px_s": speed, "width_px": width, "color_bgr": [85, 85, 90]}}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return out


def main(argv: list[str] | None = None) -> int:
    videos = source_root("public", Config.load()["paths"]) / "videos"
    (REPO_ROOT / "tmp").mkdir(exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="testclips_", dir=REPO_ROOT / "tmp"))
    try:
        for name, specs in (("shots_tune_480p30", TUNE), ("shots_check_480p30", CHECK)):
            print("wrote", build_shot_clip(videos, tmp, name, specs).relative_to(REPO_ROOT))
        print("wrote", build_occlusion_clip(videos, tmp).relative_to(REPO_ROOT))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
