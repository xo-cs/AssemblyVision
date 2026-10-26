"""Recording of the raw camera frames (not the overlay) for Analyse recording.

A writer thread pipes frames to ffmpeg (libx264, yuv420p) at a constant
live.rec_fps: at every tick it writes the newest frame. Ticks are written
only while frames arrive (the newest is at most 0.5 s old), so a paused
camera does not leave a frozen stretch. write() never blocks the page. Any
failure stops the recording with AV-LIV-007 and live analysis goes on.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np

STALE_S = 0.5


class Recorder:
    def __init__(self, path: Path, fps: int):
        self.path, self.fps = Path(path), fps
        self.tmp = self.path.with_name(self.path.stem + ".tmp.mp4")
        self._latest: tuple[np.ndarray, float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.error: str | None = None
        self.frames = 0
        self.size: tuple[int, int] | None = None
        self._proc = None
        self._thread = threading.Thread(target=self._run, name="live-recorder", daemon=True)
        self._thread.start()

    def write(self, frame: np.ndarray, t: float) -> None:
        with self._lock:
            self._latest = (frame, t)

    def _open(self, w: int, h: int) -> None:
        w, h = w - w % 2, h - h % 2
        self.size = (w, h)
        cmd = ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.tmp)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def _run(self) -> None:
        period = 1.0 / self.fps
        nxt = time.monotonic()
        while not self._stop.is_set():
            nxt += period
            delay = nxt - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                nxt = time.monotonic()                    # fell behind: do not burst
            with self._lock:
                item = self._latest
            if item is None or time.monotonic() - item[1] > STALE_S:
                continue
            frame = item[0]
            try:
                if self._proc is None:
                    self._open(frame.shape[1], frame.shape[0])
                w, h = self.size
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                self._proc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
                self.frames += 1
            except (OSError, ValueError) as e:
                self.error = str(e)
                return

    def close(self) -> Path | None:
        """Finish the file; returns its path, or None if nothing or nothing valid was written."""
        self._stop.set()
        self._thread.join(timeout=5)
        if self._proc is None:
            return None
        try:
            self._proc.stdin.close()
            err = self._proc.stderr.read().decode(errors="replace")
            code = self._proc.wait(timeout=60)
        except (OSError, subprocess.TimeoutExpired) as e:
            self._proc.kill()
            self.error = self.error or str(e)
            code, err = -1, ""
        if code != 0 or self.error or self.frames == 0:
            self.error = self.error or f"ffmpeg exit {code}: {err.strip()[:200]}"
            self.tmp.unlink(missing_ok=True)
            return None
        self.tmp.replace(self.path)
        return self.path
