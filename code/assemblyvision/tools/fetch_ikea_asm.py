"""IKEA ASM: action annotations and the top view RGB videos (Week 11).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.fetch_ikea_asm [--workers 4] [--limit N]

Public data only (CC BY-NC 4.0, Ben-Shabat et al., WACV 2021). The top view zip on
Google Drive (68 GB, 372 stored AVI members, one per assembly, camera dev3) is read by
HTTP range requests: one request per video (local header and data, CRC32 checked), so
no zip is stored. Videos land in data/public/datasets/ikea_asm/rgb_top/<furniture>/<recording>.avi;
a video counts as done when its file exists with the member's size (resume). Writes
SOURCE.json, meta/rgb_top_index.json and SHA256SUMS; logs to logs/ikea_download.log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..config import Config
from ..datasets.remote import HttpRangeFile, Pacer, open_remote_zip, read_member
from ..storage.files import REPO_ROOT, utc_now
from .fetch_dataset import dataset_root

GDRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
RGB_TOP_ID = "1CFOH-W-6N50AVA_NqHnm06GUsfpcka0L"
ACTIONS_ID = "1SwBNLViktSpk99jhh3sMXVGTMVr6tpju"
SOURCE = {"dataset": "IKEA ASM", "url": "https://ikeaasm.github.io",
          "files": {"rgb_top": GDRIVE.format(RGB_TOP_ID), "action_annotations": GDRIVE.format(ACTIONS_ID)},
          "licence": "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/); code MIT",
          "citation": "Y. Ben-Shabat, X. Yu, F. Saleh, D. Campbell, C. Rodriguez-Opazo, H. Li, S. Gould. The IKEA ASM "
                      "Dataset: Understanding People Assembling Furniture through Actions, Objects and Pose. WACV 2021."}


def video_path(root: Path, member: str) -> Path:
    """ANU_ikea_dataset_video/<furniture>/<recording>/dev3/images/scan_video.avi -> rgb_top/<furniture>/<recording>.avi"""
    parts = member.split("/")
    return root / "rgb_top" / parts[1] / f"{parts[2]}.avi"


def fetch_one(url: str, zf: zipfile.ZipFile, info: zipfile.ZipInfo, root: Path, lg: logging.Logger) -> str:
    out = video_path(root, info.filename)
    if out.is_file() and out.stat().st_size == info.file_size:
        return "kept"
    raw = HttpRangeFile(url, pacer=Pacer(min_interval_s=1.0))
    data = read_member(raw, info)                      # CRC32 checked
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(out)
    lg.info("%s: %.1f MB", out.relative_to(root), len(data) / 1e6)
    return "fetched"


def main(argv: list[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.fetch_ikea_asm")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, help="only the first N videos (a pilot)")
    args = p.parse_args(argv)
    cfg = Config.load()
    root = dataset_root(cfg, "ikea_asm", repo_root)
    (repo_root / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(repo_root / "logs/ikea_download.log"), logging.StreamHandler()])
    lg = logging.getLogger("ikea")
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "SOURCE.json").write_text(json.dumps({**SOURCE, "retrieved_utc": utc_now()}, indent=1) + "\n")
    url = GDRIVE.format(RGB_TOP_ID)
    zf, _ = open_remote_zip(url)
    infos = sorted((i for i in zf.infolist() if i.filename.endswith(".avi")), key=lambda i: i.filename)
    (root / "meta/rgb_top_index.json").write_text(json.dumps(
        [{"member": i.filename, "size": i.file_size, "crc32": i.CRC, "path": str(video_path(root, i.filename)
                                                                                .relative_to(root))} for i in infos],
        indent=0) + "\n")
    todo = infos[:args.limit] if args.limit else infos
    lg.info("%d videos, %.1f GB", len(todo), sum(i.file_size for i in todo) / 1e9)
    failed = []

    def job(info):
        try:
            return fetch_one(url, zf, info, root, lg)
        except Exception as e:                         # one failed video does not stop the others
            lg.error("%s: %s", info.filename, e)
            failed.append(info.filename)
            return "failed"
    with ThreadPoolExecutor(args.workers) as ex:
        res = list(ex.map(job, todo))
    sums = []
    for f in sorted((root / "rgb_top").rglob("*.avi")):
        h = hashlib.sha256()
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        sums.append(f"{h.hexdigest()}  {f.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    lg.info("done: %d fetched, %d kept, %d failed", res.count("fetched"), res.count("kept"), len(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
