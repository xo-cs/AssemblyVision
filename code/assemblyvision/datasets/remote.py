"""Read single members of a large remote zip with HTTP range requests (Week 10).

HA4M ships one zip of 4 to 39 GB per video with every modality inside. The
server answers range requests, so the central directory (a few MB) and the
wanted members are read without downloading the whole zip. zipfile checks the
CRC32 of every member it reads.

Politeness (the server answered HTTP 429 to four parallel streams):
- every request of the process goes through one Pacer, at least
  min_interval_s after the previous one ends;
- HTTP 429 (and 503) waits for Retry-After if the server sends it, else
  backs off exponentially from backoff_start_s up to backoff_max_s; only
  after limited_retries such waits does the request count as failed;
- other network errors get a few short retries.
"""
from __future__ import annotations

import email.utils
import io
import logging
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Callable

USER_AGENT = "AssemblyVision/1.0 (research download)"
LIMITED = (429, 503)

log = logging.getLogger("assemblyvision.fetch_dataset")

Opener = Callable[[urllib.request.Request, float], "io.BufferedIOBase"]


def _urlopen(req: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(req, timeout=timeout)


@dataclass
class Pacer:
    """One request at a time, at least min_interval_s apart (shared by every file)."""
    min_interval_s: float = 3.0
    backoff_start_s: float = 60.0
    backoff_max_s: float = 1800.0
    limited_retries: int = 8
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _last: float = field(default=-1e18, repr=False)
    waits: list = field(default_factory=list, repr=False)      # (status, seconds) of every 429/503 wait

    def run(self, fn):
        """fn() under the lock, after the minimum interval since the previous request."""
        with self._lock:
            gap = self._last + self.min_interval_s - self.clock()
            if gap > 0:
                self.sleep(gap)
            try:
                return fn()
            finally:
                self._last = self.clock()

    def limited_wait(self, attempt: int, retry_after: str | None) -> float:
        """Seconds to wait after the attempt-th 429/503: Retry-After, else exponential."""
        s = parse_retry_after(retry_after)
        if s is None:
            s = self.backoff_start_s * 2 ** attempt
        return float(min(max(s, 1.0), self.backoff_max_s))


def parse_retry_after(value: str | None, now: float | None = None) -> float | None:
    """Retry-After as seconds (a number or an HTTP date), None if absent or unreadable."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError):
        return None
    return max(0.0, when - (time.time() if now is None else now))


DEFAULT_PACER = Pacer()


class HttpRangeFile(io.RawIOBase):
    """A read only, seekable file over HTTP range requests (one request per read, paced).

    stats counts requests, bytes and rate limit waits so a caller can report the transfer."""

    def __init__(self, url: str, opener: Opener = _urlopen, timeout: float = 120.0, retries: int = 4,
                 backoff_s: float = 5.0, pacer: Pacer | None = None):
        super().__init__()
        self.url, self.opener, self.timeout = url, opener, timeout
        self.retries, self.backoff_s = retries, backoff_s
        self.pacer = pacer or DEFAULT_PACER
        self.pos = 0
        self.stats = {"requests": 0, "bytes": 0, "limited_waits": 0, "limited_wait_s": 0.0}
        resp, _ = self._fetch(0, 0)
        cr = resp.headers.get("Content-Range", "")
        if "/" not in cr:
            raise OSError(f"the server ignores range requests for {url}")
        self.size = int(cr.rsplit("/", 1)[1])

    def _fetch(self, start: int, end: int):
        """(response, body) of one range request, with the 429 and error handling."""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT})
        errors = limited = 0
        while True:
            try:
                def one():
                    r = self.opener(req, self.timeout)
                    if getattr(r, "status", 206) != 206:
                        raise OSError(f"HTTP {getattr(r, 'status', '?')} instead of 206 for a range request")
                    return r, r.read()
                resp, body = self.pacer.run(one)
                self.stats["requests"] += 1
                return resp, body
            except urllib.error.HTTPError as e:
                if e.code not in LIMITED:
                    raise
                if limited >= self.pacer.limited_retries:
                    raise
                wait = self.pacer.limited_wait(limited, e.headers.get("Retry-After") if e.headers else None)
                limited += 1
                self.stats["limited_waits"] += 1
                self.stats["limited_wait_s"] += wait
                self.pacer.waits.append((e.code, wait))
                log.warning("HTTP %d from the server: waiting %.0f s (wait %d of %d)", e.code, wait, limited,
                            self.pacer.limited_retries)
                self.pacer.sleep(wait)
            except (urllib.error.URLError, OSError, TimeoutError):
                if errors >= self.retries:
                    raise
                self.pacer.sleep(self.backoff_s * 2 ** errors)
                errors += 1

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
            _, data = self._fetch(start + len(out), end - 1)
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


def read_member(raw: HttpRangeFile, info: zipfile.ZipInfo, extra_margin: int = 1024) -> bytes:
    """One member in one range request (local header + data), CRC32 checked. For members
    far apart, where a buffered sequential read would fetch the bytes between them."""
    import struct
    import zlib
    name_len = len(info.filename.encode("utf-8"))
    head = 30 + name_len + extra_margin
    data = raw.read_range(info.header_offset, head + info.compress_size)
    if data[:4] != b"PK\x03\x04":
        raise zipfile.BadZipFile(f"no local header for {info.filename}")
    n, e = struct.unpack("<HH", data[26:30])
    start = 30 + n + e
    if start + info.compress_size > len(data):            # extra field longer than the margin
        data += raw.read_range(info.header_offset + len(data), start + info.compress_size - len(data))
    payload = data[start:start + info.compress_size]
    if info.compress_type == zipfile.ZIP_DEFLATED:
        payload = zlib.decompress(payload, -15)
    elif info.compress_type != zipfile.ZIP_STORED:
        raise zipfile.BadZipFile(f"compression {info.compress_type} of {info.filename} is not supported")
    if zlib.crc32(payload) & 0xFFFFFFFF != info.CRC:
        raise zipfile.BadZipFile(f"Bad CRC-32 for file {info.filename}")
    return payload
