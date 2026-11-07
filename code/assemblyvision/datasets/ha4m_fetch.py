"""Selective HA4M download from Science Data Bank (tools/fetch_dataset ha4m).

Every video is one zip with all modalities. Only the wanted members are read
through HTTP range requests (datasets/remote.py): Labels.txt and the Skeletons
of every video, and the Color frames of chosen videos, which are encoded to one
H.264 mp4 per video on the fly (the PNGs are not kept). zipfile checks each
member's CRC32; the stored files get sha256 sums. A video counts as done only
when its meta/<video>.json is written, which happens last, so an interrupted
video is simply fetched again.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np

from ..errors import AvError
from ..storage.files import utc_now
from . import ha4m
from .remote import USER_AGENT, member_span, open_remote_zip

log = logging.getLogger("assemblyvision.datasets.ha4m_fetch")
GB = 1e9


def fetch_listing(post: Callable[[str, bytes], bytes] | None = None) -> list[dict]:
    """The V2 file list (one entry per zip: id, name, size, md5) from the Science Data Bank API."""
    def default_post(url: str, body: bytes) -> bytes:
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": USER_AGENT})
        return urllib.request.urlopen(req, timeout=60).read()
    post = post or default_post
    body = json.dumps({"dataSetId": ha4m.DATASET_ID, "version": ha4m.VERSION, "path": f"/{ha4m.VERSION}",
                       "lastIndex": 0, "pageSize": 1000}).encode()
    data = json.loads(post(ha4m.LISTING_URL, body))
    if data.get("code") != 20000:
        raise AvError("AV-DAT-001", name="HA4M listing", detail=str(data.get("messageEn")))
    files = [{"video": f["fileName"].removesuffix(".zip"), "file_id": f["id"], "size": int(f["size"]),
              "md5": f["md5"].lower(), "path": f["path"]} for f in data["data"] if not f.get("dir")]
    if not files:
        raise AvError("AV-DAT-001", name="HA4M listing", detail="empty listing")
    return sorted(files, key=lambda f: f["video"])


def write_source(root: Path, files: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "SOURCE.json"
    old = json.loads(path.read_text()) if path.is_file() else {}
    src = {"name": "HA4M", "doi": ha4m.DOI, "version": ha4m.VERSION, "dataset_id": ha4m.DATASET_ID,
           "url": f"https://doi.org/{ha4m.DOI}", "licence": ha4m.LICENCE, "citation": ha4m.CITATION,
           "listing_utc": utc_now(), "total_bytes": sum(f["size"] for f in files),
           "files": {f["video"]: {k: f[k] for k in ("file_id", "size", "md5")} for f in files},
           "cameras": old.get("cameras", dict(ha4m.CAMERAS)),
           "camera_per_video": old.get("camera_per_video", {})}
    _write_json(path, src)
    return path


def load_source(root: Path) -> dict:
    path = root / "SOURCE.json"
    if not path.is_file():
        raise AvError("AV-DAT-003", name="HA4M", detail="no SOURCE.json yet (run: fetch_dataset ha4m listing)")
    return json.loads(path.read_text())


def _write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def _members(zf: zipfile.ZipFile, video: str, modality: str) -> list[zipfile.ZipInfo]:
    pre = f"{video}/{modality}/"
    return [i for i in zf.infolist() if i.filename.startswith(pre) and not i.is_dir()]


class Budget:
    """Bytes transferred in one command; above the limit only with approval (--yes)."""

    def __init__(self, limit_gb: float, approved: bool):
        self.limit, self.approved, self.used = limit_gb * GB, approved, 0

    def check(self, more: int) -> None:
        if not self.approved and self.used + more > self.limit:
            raise AvError("AV-DAT-002", gb=round((self.used + more) / GB, 1), limit=round(self.limit / GB, 1))

    def add(self, n: int) -> None:
        self.used += n


def _meta_path(root: Path, video: str) -> Path:
    return root / "meta" / f"{video}.json"


def read_meta(root: Path, video: str) -> dict:
    p = _meta_path(root, video)
    return json.loads(p.read_text()) if p.is_file() else {}


def fetch_annotations(root: Path, video: str, entry: dict, budget: Budget, opener=None) -> dict:
    """Labels.txt and Skeletons of one video: raw/<video>_annot.zip, labels/<video>.csv,
    skeleton/<video>.npz and meta/<video>.json (written last)."""
    meta = read_meta(root, video)
    if meta.get("annotations", {}).get("done"):
        return meta
    budget.check(10 * 1024 * 1024)       # central directory plus skeletons, about 10 MB per video
    kw = {"opener": opener} if opener else {}
    zf, raw = open_remote_zip(ha4m.DOWNLOAD_URL.format(file_id=entry["file_id"]), **kw)
    if raw.size != entry["size"]:
        raise AvError("AV-DAT-001", name=f"{video}.zip", detail=f"size {raw.size} instead of {entry['size']}")
    lab = [i for i in zf.infolist() if i.filename == f"{video}/Labels.txt"]
    skel = _members(zf, video, "Skeletons")
    color = _members(zf, video, "Color")
    if not lab:
        raise AvError("AV-DAT-001", name=f"{video}.zip", detail="no Labels.txt")
    spans = [member_span(zf, group) for group in (lab, skel)]   # not contiguous: two spans
    budget.check(sum(hi - lo for lo, hi in spans))
    texts: dict[str, bytes] = {}
    try:
        for info in lab + skel:                      # zipfile checks the CRC32 of each member
            texts[info.filename] = zf.read(info)
    except zipfile.BadZipFile as e:
        raise AvError("AV-DAT-001", name=f"{video}.zip", detail=str(e)) from e
    budget.add(raw.stats["bytes"])
    rows = ha4m.parse_labels(texts[lab[0].filename].decode())
    n_frames = max([int(rows[:, 0].max()) + 1 if len(rows) else 0]
                   + [(ha4m.frame_id(i.filename) or 0) + 1 for i in color])
    sk_files = {ha4m.frame_id(name): t.decode() for name, t in texts.items() if "/Skeletons/" in name}
    sk = ha4m.parse_skeletons({k: v for k, v in sk_files.items() if k is not None}, n_frames)
    cams = {ha4m.member_camera(i.filename) for i in color + skel} - {None}
    stored = {}
    annot = root / "raw" / f"{video}_annot.zip"
    annot.parent.mkdir(parents=True, exist_ok=True)
    tmp = annot.with_name(annot.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for name, data in texts.items():
            out.writestr(name, data)
    os.replace(tmp, annot)
    stored[f"raw/{annot.name}"] = sha256_path(annot)
    stored[f"labels/{video}.csv"] = _write_bytes(root / "labels" / f"{video}.csv", ha4m.labels_csv(rows).encode())
    skp = root / "skeleton" / f"{video}.npz"
    skp.parent.mkdir(parents=True, exist_ok=True)
    tmpz = skp.with_name(skp.stem + ".tmp.npz")
    np.savez_compressed(tmpz, **sk)
    os.replace(tmpz, skp)
    stored[f"skeleton/{skp.name}"] = sha256_path(skp)
    c_lo, c_hi = member_span(zf, color) if color else (0, 0)
    meta.update({"video": video, "file_id": entry["file_id"], "zip_size": entry["size"], "zip_md5": entry["md5"],
                 "camera": sorted(cams)[0] if len(cams) == 1 else sorted(cams),
                 "n_frames": n_frames, "n_color_frames": len(color), "n_skeleton_files": len(skel),
                 "n_label_rows": int(len(rows)),
                 "annotations": {"done": True, "utc": utc_now(), "transfer_bytes": raw.stats["bytes"],
                                 "requests": raw.stats["requests"],
                                 "members": {i.filename: f"{i.CRC:08x}" for i in lab + skel},
                                 "stored_sha256": stored},
                 "color_span": [c_lo, c_hi], "color_span_bytes": c_hi - c_lo})
    root.joinpath("meta").mkdir(exist_ok=True)
    _write_json(_meta_path(root, video), meta)
    return meta


def encoder_cmd(out: Path, fps: float, crf: int) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "image2pipe", "-framerate", f"{fps:g}",
            "-c:v", "png", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-r", f"{fps:g}", "-movflags", "+faststart", str(out)]


def fetch_rgb(root: Path, video: str, entry: dict, budget: Budget, crf: int = 18, opener=None,
              popen=subprocess.Popen, progress: Callable[[str], None] = lambda s: None) -> dict:
    """Color frames of one video -> rgb/<video>.mp4 (mp4 index = FrameID; a missing FrameID
    repeats the frame before it and is listed in frames.json, written last)."""
    meta = read_meta(root, video)
    frames_json = root / "rgb" / f"{video}.frames.json"
    if meta.get("rgb", {}).get("done") and frames_json.is_file():
        return meta
    kw = {"opener": opener} if opener else {}
    zf, raw = open_remote_zip(ha4m.DOWNLOAD_URL.format(file_id=entry["file_id"]), buffer_size=64 << 20, **kw)
    color = sorted(_members(zf, video, "Color"), key=lambda i: ha4m.frame_id(i.filename) or -1)
    color = [i for i in color if ha4m.frame_id(i.filename) is not None]
    if not color:
        raise AvError("AV-DAT-001", name=f"{video}.zip", detail="no Color frames")
    lo, hi = member_span(zf, color)
    budget.check(hi - lo)
    out = root / "rgb" / f"{video}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.mp4")
    proc = popen(encoder_cmd(tmp, ha4m.FPS, crf), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    base = ha4m.frame_id(color[0].filename)         # mp4 index i is FrameID base + i
    ids, filled, prev = [], [], None
    t0 = time.monotonic()
    try:
        for n, info in enumerate(color):
            fid = ha4m.frame_id(info.filename)
            data = zf.read(info)                        # CRC32 checked
            while prev is not None and base + len(ids) < fid:   # gap: repeat the frame before
                proc.stdin.write(prev)
                filled.append(base + len(ids))
                ids.append(base + len(ids))
            if fid < base + len(ids):
                continue                                # duplicate frame id
            proc.stdin.write(data)
            ids.append(fid)
            prev = data
            if n % 300 == 0:
                mb = raw.stats["bytes"] / 1e6
                progress(f"{video}: frame {n + 1}/{len(color)}, {mb:.0f} MB, "
                         f"{mb / max(time.monotonic() - t0, 1e-6):.1f} MB/s")
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise AvError("AV-DAT-001", name=out.name, detail=f"ffmpeg: {err.strip()[-300:]}")
    except BaseException:
        if proc.poll() is None:
            proc.kill()
        tmp.unlink(missing_ok=True)
        raise
    budget.add(raw.stats["bytes"])
    os.replace(tmp, out)
    digest = sha256_path(out)
    info = {"video": video, "n_frames": len(ids), "first_frame_id": base,
            "filled": filled, "fps": ha4m.FPS, "size": list(ha4m.RGB_SIZE),
            "encoder": {"codec": "libx264", "crf": crf, "preset": "medium", "pix_fmt": "yuv420p",
                        "source": "Color PNG members, in FrameID order"},
            "mp4_sha256": digest}
    if base != 0:
        info["note"] = "the first Color frame id is not 0; mp4 index i is FrameID first_frame_id + i"
    meta.setdefault("video", video)
    meta["rgb"] = {"done": True, "utc": utc_now(), "transfer_bytes": raw.stats["bytes"],
                   "requests": raw.stats["requests"], "seconds": round(time.monotonic() - t0, 1),
                   "frames": len(ids), "filled": len(filled), "stored_sha256": {f"rgb/{out.name}": digest}}
    _write_json(frames_json, info)
    _write_json(_meta_path(root, video), meta)
    return meta


def write_sha256sums(root: Path) -> Path:
    """SHA256SUMS of every stored file, from the meta records."""
    lines = {}
    for p in sorted((root / "meta").glob("*.json")):
        m = json.loads(p.read_text())
        for part in ("annotations", "rgb"):
            lines.update(m.get(part, {}).get("stored_sha256", {}))
    path = root / "SHA256SUMS"
    path.write_text("".join(f"{h}  {name}\n" for name, h in sorted(lines.items())))
    return path
