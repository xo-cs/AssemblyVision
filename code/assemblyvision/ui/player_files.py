"""Compact copies of the 3D and 4D player files, for the browser (Week 12).

The 4D binary (scene4d <part>.player.bin, float32 and uint8 blocks) becomes int16 blocks with a per block offset and
scale (the range over 65,534 steps; about 0.3 mm over 20 m; -32768 marks NaN), then gzip; the JSON headers of 3D and 4D
are gzipped as they are. The copies lie in <run>/tmp/ui_player/ (inside the run's data area) and are made again when the
original is newer. The originals are never changed and stay the fallback for a browser without DecompressionStream.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

NAN = -32768


def _stale(out: Path, *src: Path) -> bool:
    return not out.is_file() or any(s.stat().st_mtime > out.stat().st_mtime for s in src)


def quantise(a: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(int16 values, offset, scale) with value = offset + (q + 32767) * scale, NaN as NAN."""
    a = np.asarray(a, np.float32)
    ok = np.isfinite(a)
    if not ok.any():
        return np.full(a.shape, NAN, np.int16), 0.0, 1.0
    lo, hi = float(a[ok].min()), float(a[ok].max())
    scale = (hi - lo) / 65534.0 or 1.0
    q = np.full(a.shape, NAN, np.int16)
    q[ok] = (np.rint((a[ok] - lo) / scale) - 32767).astype(np.int16)
    return q, lo, scale


def compact_4d(json_path: Path, bin_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """(header .json.gz, data .bin.gz) of one 4D part."""
    out_dir.mkdir(parents=True, exist_ok=True)
    jo, bo = paths_4d(json_path, bin_path, out_dir)
    if not (_stale(jo, json_path, bin_path) or _stale(bo, json_path, bin_path)):
        return jo, bo
    head = json.loads(json_path.read_text())
    raw = bin_path.read_bytes()
    blocks, off, new = [], 0, dict(head)
    for k, b in head.items():
        if not isinstance(b, dict) or "dtype" not in b or "offset" not in b:
            continue
        n = int(np.prod(b["shape"]))
        if b["dtype"] == "f32":
            q, lo, scale = quantise(np.frombuffer(raw, np.float32, n, b["offset"]))
            data, entry = q.tobytes(), {"dtype": "q16", "offset": off, "shape": b["shape"], "min": lo, "scale": scale}
        else:
            data, entry = raw[b["offset"]:b["offset"] + n], {"dtype": b["dtype"], "offset": off, "shape": b["shape"]}
        new[k] = entry
        blocks.append(data)
        off += len(data)
        pad = (-off) % 4
        if pad:
            blocks.append(b"\0" * pad)
            off += pad
    tmp = bo.with_suffix(".part")
    tmp.write_bytes(gzip.compress(b"".join(blocks), compresslevel=6))
    tmp.replace(bo)
    tmp = jo.with_suffix(".part")
    tmp.write_bytes(gzip.compress(json.dumps(new).encode(), compresslevel=6))
    tmp.replace(jo)
    return jo, bo


def out_dir(root: Path, mode: str) -> Path:
    """<run>/tmp/ui_player/3d or 4d (a 3D and a 4D part have the same file name)."""
    return Path(root) / "tmp" / "ui_player" / mode


def ready(out: Path, *src: Path) -> Path | None:
    """out if it exists and is not older than its sources (nothing is made here)."""
    return None if _stale(out, *src) else out


def paths_4d(json_path: Path, bin_path: Path, out_dir: Path) -> tuple[Path, Path]:
    return out_dir / (json_path.name + ".gz"), out_dir / (bin_path.stem + ".q16.bin.gz")


def make_all(root: Path) -> int:
    """The compact copies of every 3D and 4D part of a run (made by the app's keep warm loop); returns how many were made."""
    n = 0
    for j in sorted((Path(root) / "scene4d").glob("*.player.json")):
        b = j.with_name(j.name.replace(".player.json", ".player.bin"))
        if b.is_file():
            jo, bo = paths_4d(j, b, out_dir(root, "4d"))
            if _stale(jo, j, b) or _stale(bo, j, b):
                compact_4d(j, b, out_dir(root, "4d"))
                n += 1
    for j in sorted((Path(root) / "body3d").glob("*.player.json")):
        if _stale(out_dir(root, "3d") / (j.name + ".gz"), j):
            compact_json(j, out_dir(root, "3d"))
            n += 1
    return n


def compact_json(json_path: Path, out_dir: Path) -> Path:
    """The gzip copy of a 3D part's JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (json_path.name + ".gz")
    if _stale(out, json_path):
        tmp = out.with_suffix(".part")
        tmp.write_bytes(gzip.compress(json_path.read_bytes(), compresslevel=6))
        tmp.replace(out)
    return out
