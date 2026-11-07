"""Read single members of a large remote zip with HTTP range requests (Week 10).

HA4M ships one zip of 4 to 39 GB per video with every modality inside. The
server answers range requests, so the central directory (a few MB) and the
wanted members are read without downloading the whole zip. zipfile checks the
CRC32 of every member it reads.
"""
from __future__ import annotations

import io
import time
import urllib.error
import urllib.request
import zipfile
from typing import Callable

USER_AGENT = "AssemblyVision/1.0 (research download)"

Opener = Callable[[urllib.request.Request, float], "io.BufferedIOBase"]


def _urlopen(req: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(req, timeout=timeout)


class HttpRangeFile(io.RawIOBase):
    """A read only, seekable file over HTTP range requests.

    Each read is one request (retried with backoff); stats counts requests and
    bytes so a caller can report the transfer."""

    def __init__(self, url: str, opener: Opener = _urlopen, timeout: float = 60.0, retries: int = 5,
                 backoff_s: float = 2.0, sleep: Callable[[float], None] = time.sleep):
        super().__init__()
        self.url, self.opener, self.timeout = url, opener, timeout
        self.retries, self.backoff_s, self.sleep = retries, backoff_s, sleep
        self.pos = 0
        self.stats = {"requests": 0, "bytes": 0}
        head = self._get(0, 0)
        cr = head.headers.get("Content-Range", "")
        if "/" not in cr:
            raise OSError(f"the server ignores range requests for {url}")
        self.size = int(cr.rsplit("/", 1)[1])

    def _get(self, start: int, end: int):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
        for attempt in range(self.retries + 1):
            try:
                resp = self.opener(req, self.timeout)
                if getattr(resp, "status", 206) != 206:
                    raise OSError(f"HTTP {getattr(resp, 'status', '?')} instead of 206 for a range request")
                self.stats["requests"] += 1
                return resp
            except (urllib.error.URLError, OSError, TimeoutError):
                if attempt == self.retries:
                    raise
                self.sleep(self.backoff_s * 2 ** attempt)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.pos = {io.SEEK_SET: offset, io.SEEK_CUR: self.pos + offset, io.SEEK_END: self.size + offset}[whence]
        return self.pos

    def read_range(self, start: int, length: int) -> bytes:
        """Exactly length bytes from start (fewer only at the end of the file)."""
        out = bytearray()
        end = min(start + length, self.size)
        while start + len(out) < end:
            for attempt in range(self.retries + 1):
                try:
                    data = self._get(start + len(out), end - 1).read()
                    break
                except (urllib.error.URLError, OSError, TimeoutError):
                    if attempt == self.retries:
                        raise
                    self.sleep(self.backoff_s * 2 ** attempt)
            if not data:
                raise OSError(f"empty response at byte {start + len(out)} of {self.url}")
            out += data
        self.stats["bytes"] += len(out)
        return bytes(out)

    def readinto(self, b) -> int:
        if self.pos >= self.size or len(b) == 0:
            return 0
        data = self.read_range(self.pos, len(b))
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip(url: str, opener: Opener = _urlopen, buffer_size: int = 1 << 20,
                    **kw) -> tuple[zipfile.ZipFile, HttpRangeFile]:
    """(ZipFile, its HttpRangeFile) for a remote zip; only the central directory is read."""
    raw = HttpRangeFile(url, opener=opener, **kw)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=buffer_size)), raw


def member_span(zf: zipfile.ZipFile, infos: list[zipfile.ZipInfo]) -> tuple[int, int]:
    """(first byte, end byte) of the members in the zip, local headers included: the
    transfer needed to read them in one sequential pass."""
    if not infos:
        return 0, 0
    starts = sorted(i.header_offset for i in zf.infolist())
    lo = min(i.header_offset for i in infos)
    last = max(infos, key=lambda i: i.header_offset)
    nxt = next((s for s in starts if s > last.header_offset), None)
    hi = nxt if nxt is not None else last.header_offset + 30 + len(last.filename) + last.compress_size
    return lo, hi
