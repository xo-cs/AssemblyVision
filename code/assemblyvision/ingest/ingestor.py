"""Video ingestion (LLD 3.3): probe, constant frame rate video, split, frame reading.

Difference from LLD 3.3 agreed for the first pose stage: the video is only
transcoded when needed (variable frame rate, fps above ingest.max_fps, codec not
H.264, pixel format not yuv420p, or an avi container). Otherwise video/cfr.mp4
is a relative symlink to the input, which stays inside the same data area.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from ..config import Config
from ..errors import AvError
from ..storage.files import RunContext
from ..types import SplitSpec, VideoInfo

EXTENSIONS = (".mp4", ".avi")


@dataclass(frozen=True)
class ProbeDetail:
    info: VideoInfo
    codec: str
    pix_fmt: str
    fps_known: bool
    interval_dev_ms: float | None   # largest deviation of a frame interval from the median


def _rate(text: str | None) -> float | None:
    try:
        value = float(Fraction(text))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return value if value > 0 else None


def _run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)


def ffmpeg_version() -> str:
    try:
        return _run(["ffmpeg", "-version"]).stdout.splitlines()[0]
    except (OSError, IndexError):
        return "missing"


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class VideoIngestor:
    def __init__(self, cfg: Config, ctx: RunContext):
        self.cfg = cfg
        self.ctx = ctx
        self.icfg = cfg["ingest"]

    def probe_detail(self, path: Path) -> ProbeDetail:
        """ffprobe the first video stream and its packet timestamps (AV-ING-001)."""
        name = Path(path).name
        if shutil.which("ffprobe") is None:
            raise AvError("AV-ING-001", name=name, detail="ffprobe not found on PATH")
        res = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "stream=codec_name,pix_fmt,width,height,r_frame_rate,avg_frame_rate,nb_frames"
                    ":format=duration", "-of", "json", str(path)])
        try:
            data = json.loads(res.stdout)
            stream = data["streams"][0]
            width, height = int(stream["width"]), int(stream["height"])
        except (ValueError, KeyError, IndexError) as e:
            raise AvError("AV-ING-001", name=name, detail=res.stderr.strip()[:500] or repr(e)) from e

        # Packet timestamps: frame count and the constant frame rate check (TC-ING-02).
        res = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                    "packet=pts_time", "-of", "csv=p=0", str(path)])
        pts = sorted(float(t) for t in (line.strip().rstrip(",") for line in res.stdout.splitlines())
                     if t and t != "N/A")
        dev_ms = None
        if len(pts) >= 3:
            diffs = np.diff(pts)
            med = statistics.median(diffs)
            dev_ms = float(np.max(np.abs(diffs - med)) * 1000.0)

        fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))
        fps_known = fps is not None
        if not fps_known:
            fps = float(self.icfg["default_fps"])
        frame_count = len(pts) or int(stream.get("nb_frames") or 0)
        if frame_count == 0:
            cap = cv2.VideoCapture(str(path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()
        if frame_count <= 0:
            raise AvError("AV-ING-001", name=name, detail="no video frames found")
        info = VideoInfo(str(path), round(fps, 6), width, height, frame_count, round(frame_count / fps, 6))
        return ProbeDetail(info, stream.get("codec_name", "?"), stream.get("pix_fmt", "?"), fps_known, dev_ms)

    def probe(self, path: Path) -> VideoInfo:
        return self.probe_detail(path).info

    def transcode_reasons(self, path: Path, d: ProbeDetail) -> list[str]:
        reasons = []
        if Path(path).suffix.lower() != ".mp4":
            reasons.append("container is not mp4")
        if not d.fps_known:
            reasons.append("fps unknown")
        elif d.info.fps > self.icfg["max_fps"] + 0.01:
            reasons.append(f"fps {d.info.fps:g} above {self.icfg['max_fps']}")
        if d.interval_dev_ms is None:
            reasons.append("frame timestamps unreadable")
        elif d.interval_dev_ms > self.icfg["vfr_tolerance_ms"] + 1e-6:
            reasons.append(f"variable frame rate (interval deviation {d.interval_dev_ms:.2f} ms)")
        if d.codec != "h264":
            reasons.append(f"codec {d.codec}")
        if d.pix_fmt != "yuv420p":
            reasons.append(f"pixel format {d.pix_fmt}")
        return reasons

    def ingest(self, run_id: str, input_path: Path, source_type: str,
               strict_split: bool = True) -> tuple[VideoInfo, SplitSpec | None]:
        """Make video/cfr.mp4 and the split. strict_split=False turns a short
        video (AV-ING-005) into a notice and returns split None."""
        t0 = time.monotonic()
        input_path = Path(input_path)
        ext = input_path.suffix.lower()
        if ext not in EXTENSIONS:
            raise AvError("AV-ING-002", ext=ext or "(none)")
        if not input_path.is_file():
            raise AvError("AV-ING-001", name=input_path.name, detail="file not found")
        paths = self.ctx.paths
        size = input_path.stat().st_size
        free = shutil.disk_usage(paths.root).free
        need = self.icfg["disk_factor"] * size
        if free < need:
            raise AvError("AV-ING-004", free_gb=f"{free / 1e9:.1f}", need_gb=f"{need / 1e9:.1f}")

        src = self.probe_detail(input_path)
        reasons = self.transcode_reasons(input_path, src)
        log = self.ctx.log
        log.info("input %s: %dx%d, %.3f fps, %d frames, %.1f s, %s %s",
                 input_path.name, src.info.width, src.info.height, src.info.fps,
                 src.info.frame_count, src.info.duration_s, src.codec, src.pix_fmt)

        cfr = paths.cfr
        cfr.unlink(missing_ok=True)
        target_fps = None
        if reasons:
            target_fps = min(round(src.info.fps), self.icfg["max_fps"]) if src.fps_known \
                else self.icfg["default_fps"]
            log.info("transcoding to %d fps constant frame rate: %s", target_fps, "; ".join(reasons))
            self._transcode(input_path, cfr, target_fps)
            out = self.probe(cfr)
        else:
            log.info("no transcode needed, video/cfr.mp4 is a symlink to the input")
            os.symlink(os.path.relpath(input_path, cfr.parent), cfr)
            out = VideoInfo(str(cfr), src.info.fps, src.info.width, src.info.height,
                            src.info.frame_count, src.info.duration_s)

        split = None
        try:
            split = self.make_split(out)
        except AvError as e:
            if strict_split or e.code != "AV-ING-005":
                raise
            self.ctx.notice(e, "ingest")

        self.ctx.manifest["ingest"] = {
            "input_path": str(input_path),
            "input_sha256": sha256_file(input_path),
            "input_bytes": size,
            "source_type": source_type,
            "input": {"fps": src.info.fps, "width": src.info.width, "height": src.info.height,
                      "frame_count": src.info.frame_count, "duration_s": src.info.duration_s,
                      "codec": src.codec, "pix_fmt": src.pix_fmt,
                      "interval_dev_ms": src.interval_dev_ms},
            "transcoded": bool(reasons),
            "transcode_reasons": reasons,
            "target_fps": target_fps,
            "cfr": {"path": self.ctx.rel(cfr), "fps": out.fps, "width": out.width,
                    "height": out.height, "frame_count": out.frame_count,
                    "duration_s": out.duration_s},
            "split": None if split is None else {"train_end_s": split.train_end_s,
                                                  "val_end_s": split.val_end_s},
            "ffmpeg_version": ffmpeg_version(),
            "seconds": round(time.monotonic() - t0, 2),
        }
        return out, split

    def _transcode(self, src: Path, dst: Path, fps: int) -> None:
        if shutil.which("ffmpeg") is None:
            raise AvError("AV-ING-003", detail="ffmpeg not found on PATH")
        tmp = dst.with_name(dst.stem + ".tmp.mp4")
        cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src), "-vf", f"fps={fps}",
               "-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", str(self.icfg["crf"]), "-an", str(tmp)]
        self.ctx.log.debug("ffmpeg command: %s", cmd)
        res = _run(cmd)
        if res.returncode != 0:
            tmp.unlink(missing_ok=True)
            self.ctx.log.error("ffmpeg exit %d: %s", res.returncode, res.stderr.strip())
            raise AvError("AV-ING-003", detail=f"ffmpeg exit code {res.returncode}")
        os.replace(tmp, dst)

    def make_split(self, info: VideoInfo) -> SplitSpec:
        """Chronological split by time, never random frames (LLD D3)."""
        d = info.duration_s
        if d < self.icfg["min_duration_s"]:
            raise AvError("AV-ING-005", dur=f"{d:.1f}", min=f"{self.icfg['min_duration_s']:g}")
        train = self.cfg["split"]["train_frac"]
        val = self.cfg["split"]["val_frac"]
        return SplitSpec(d, round(train * d, 3), round((train + val) * d, 3))


def iter_frames(path: Path, start: int = 0, end: int | None = None,
                want: Callable[[int], bool] | None = None, seek: bool = False) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame index, BGR image) one at a time; memory does not grow with length.

    want(idx) selects frames: the others are only grabbed, not converted.
    Frames before start are grabbed too (no seeking), so indices are exact.
    seek=True (overlay chunks) seeks to start instead; if OpenCV does not land
    exactly on start the video is reopened and grabbed as usual.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise AvError("AV-ING-001", name=Path(path).name, detail="OpenCV could not open the video")
    try:
        idx = 0
        if seek and start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) == start:
                idx = start
            else:
                cap.release()
                cap = cv2.VideoCapture(str(path))
        while end is None or idx < end:
            if idx < start or (want is not None and not want(idx)):
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, frame
            idx += 1
    finally:
        cap.release()



_END = object()


def _prefetched(make: Callable[[], Iterator], depth: int, name: str) -> Iterator:
    """Run the iterator make() in a reader thread with a bounded queue of depth
    items; errors are raised in the consumer, and closing the consumer stops the
    reader. OpenCV and pipe reads release the GIL, so decoding overlaps the caller."""
    import queue
    import threading
    q: queue.Queue = queue.Queue(maxsize=depth)
    stop = threading.Event()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def reader() -> None:
        try:
            for item in make():
                if not put(item):
                    return
            put(_END)
        except BaseException as e:      # handed to the consumer, raised there
            put(e)

    t = threading.Thread(target=reader, name=name, daemon=True)
    t.start()
    try:
        while True:
            item = q.get()
            if item is _END:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        t.join(timeout=10)


def iter_frames_prefetch(path: Path, start: int = 0, end: int | None = None,
                         want: Callable[[int], bool] | None = None, depth: int = 64
                         ) -> Iterator[tuple[int, np.ndarray]]:
    """iter_frames in a reader thread (pose.decode = prefetch): the same indices
    and images, decoded while the caller runs inference."""
    return _prefetched(lambda: iter_frames(path, start, end, want), depth, "frame-prefetch")


def _ffmpeg_cuda_frames(path: Path, start: int, end: int | None,
                        want: Callable[[int], bool] | None) -> Iterator[tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise AvError("AV-ING-001", name=Path(path).name, detail="OpenCV could not open the video")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    size = w * h * 3
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-hwaccel", "cuda", "-c:v", "h264_cuvid", "-i", str(path),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=size * 4)
    idx = 0
    try:
        while end is None or idx < end:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            if idx >= start and (want is None or want(idx)):
                yield idx, np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
            idx += 1
    finally:
        proc.kill()
        proc.wait()
        err = proc.stderr.read().decode(errors="replace").strip()
        proc.stderr.close()
        proc.stdout.close()
    if idx == 0:
        raise AvError("AV-ING-001", name=Path(path).name, detail=f"ffmpeg cuda decode gave no frame: {err[:300]}")


def iter_frames_ffmpeg_cuda(path: Path, start: int = 0, end: int | None = None,
                            want: Callable[[int], bool] | None = None, depth: int = 64
                            ) -> Iterator[tuple[int, np.ndarray]]:
    """Frames decoded on the GPU (NVDEC: ffmpeg -hwaccel cuda -c:v h264_cuvid) and
    converted to BGR by ffmpeg, read from a pipe in a reader thread
    (pose.decode = ffmpeg_cuda). Every frame is decoded; want only selects what
    is yielded. Pixels may differ slightly from OpenCV's software decoder, so
    the equivalence check decides whether this reader may be used."""
    return _prefetched(lambda: _ffmpeg_cuda_frames(path, start, end, want), depth, "frame-nvdec")


def frame_reader(kind: str, depth: int = 64) -> Callable[..., Iterator[tuple[int, np.ndarray]]]:
    """The frame iterator of pose.decode: opencv (iter_frames), prefetch or ffmpeg_cuda."""
    if kind == "opencv":
        return iter_frames
    if kind == "prefetch":
        return lambda path, start=0, end=None, want=None: iter_frames_prefetch(path, start, end, want, depth)
    if kind == "ffmpeg_cuda":
        return lambda path, start=0, end=None, want=None: iter_frames_ffmpeg_cuda(path, start, end, want, depth)
    raise ValueError(f"unknown frame reader {kind}")
