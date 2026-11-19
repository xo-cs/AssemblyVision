"""Hand labels linked to a video by its SHA256 (Week 11): every run of that video gets its evaluation.

data/<source>/annotations/gt_links.json: {"links": [{"video_sha256", "video_name", "gt", "created_utc"}]},
gt a CSV path relative to data/<source>/ (step,label,start_s,end_s,source). A run records the SHA256
of its input video at ingest (manifest ingest.input_sha256), so no video is read again to find its
labels. tools/link_gt.py writes the links (for QiSens Khurshid runs it). The file and the labels stay
in their own data area.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..storage.files import utc_now

LINKS = Path("annotations") / "gt_links.json"


def links_path(area: Path) -> Path:
    return area / LINKS


def load(area: Path) -> list[dict]:
    try:
        return json.loads(links_path(area).read_text()).get("links", [])
    except (OSError, ValueError):
        return []


def add(area: Path, video_sha256: str, gt: Path, video_name: str = "") -> dict:
    """Links gt (inside area) to the video; an earlier link of the same video is replaced."""
    gt = Path(gt).resolve()
    rel = gt.relative_to(area.resolve())                 # ValueError outside the area
    links = [x for x in load(area) if x["video_sha256"] != video_sha256]
    rec = {"video_sha256": video_sha256, "video_name": video_name, "gt": str(rel), "created_utc": utc_now()}
    links.append(rec)
    p = links_path(area)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({"links": links}, indent=1))
    tmp.replace(p)
    return rec


def gt_for_run(area: Path, manifest: dict) -> Path | None:
    """The hand label file of the run's video, if linked and present."""
    sha = manifest.get("ingest", {}).get("input_sha256")
    if not sha:
        return None
    for x in load(area):
        if x["video_sha256"] == sha:
            p = area / x["gt"]
            return p if p.is_file() else None
    return None
