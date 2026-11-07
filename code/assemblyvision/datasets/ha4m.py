"""HA4M (Cicirelli et al., Scientific Data 9, 745, 2022): parsing and the loader.

Layout under data/public/datasets/ha4m/ (written by tools/fetch_dataset):
    SOURCE.json                 DOI, version, licence, listing (file id, size, md5 per zip), cameras
    meta/<video>.json           what was fetched: member names and CRC32s, sha256 of the stored files
    raw/<video>_annot.zip       the original Labels.txt and Skeletons/*.txt members, repacked
    labels/<video>.csv          frame,action,repetition
    skeleton/<video>.npz        Azure Kinect joints of the operator (see parse_skeletons)
    rgb/<video>.mp4             Color frames as H.264, mp4 index = FrameID (gaps filled, listed)
    rgb/<video>.frames.json     frame ids, filled gaps, encoder settings
    derived/yolo26x/<video>/    our pose: KTS v1 files, subject.npz, labels.csv, source.json
    SHA256SUMS
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

import numpy as np

from ..errors import AvError
from .base import DatasetLoader, Sample, VideoRef
from .mapping import KINECT32_TO_COCO17

NAME = "ha4m"
DOI = "10.57760/sciencedb.01872"
VERSION = "V2"
DATASET_ID = "c8d743ad2ea549dfa938cea320a38c46"
LISTING_URL = "https://www.scidb.cn/api/gin-sdb-filetree/public/file/childrenFileListByPath"
DOWNLOAD_URL = "https://china.scidb.cn/download?fileId={file_id}"
LICENCE = "CC BY 4.0"
CITATION = ("G. Cicirelli, R. Marani, L. Romeo, M. G. Dominguez, J. Heras, A. G. Perri, T. D'Orazio, "
            "The HA4M dataset: Multi-Modal Monitoring of an assembly task for Human Action recognition in "
            "Manufacturing, Scientific Data 9, 745 (2022)")
FPS = 30.0
RGB_SIZE = (2048, 1536)
N_JOINTS = 32
INVALID_BODY = 0xFFFFFFFF      # K4ABT_INVALID_BODY_ID: rows of no tracked body
# The two laboratories, told apart by the Azure Kinect serial in every member path
# (checked on all 217 zips: subjects 001 to 022 and 023 to 041). Which serial is the
# Italian (STIIMA-CNR, Bari) and which the Spanish lab (Universidad de La Rioja) is
# set from one frame of each (SOURCE.json "cameras").
CAMERAS = {"000124702712": "lab_a", "000099302712": "lab_b"}

VIDEO_RE = re.compile(r"IDU(\d{3})V(\d{3})")
FRAME_RE = re.compile(r"FrameID(\d+)_DeviceTimeStamp(\d+)us\.(png|txt)$")


def video_subject(video_id: str) -> str:
    m = VIDEO_RE.fullmatch(video_id)
    if not m:
        raise ValueError(f"not an HA4M video id: {video_id}")
    return f"S{m.group(1)}"


def frame_id(member: str) -> int | None:
    m = FRAME_RE.search(member)
    return int(m.group(1)) if m else None


def member_camera(member: str) -> str | None:
    """Kinect serial of a member path IDUnnnVmmm/<Modality>/<serial>/<file>."""
    parts = member.split("/")
    return parts[2] if len(parts) >= 4 else None


def parse_labels(text: str) -> np.ndarray:
    """Labels.txt (frame, action 0..12, repetition 0..2 per line) -> int32 [N,3], sorted by frame."""
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3:
            raise AvError("AV-EVL-001", file="Labels.txt", detail=f"line {n} has {len(parts)} columns")
        rows.append([int(p) for p in parts])
    a = np.array(rows, np.int32).reshape(-1, 3)
    a = a[np.argsort(a[:, 0], kind="stable")]
    if len(a) and len(np.unique(a[:, 0])) != len(a):
        raise AvError("AV-EVL-001", file="Labels.txt", detail="a frame is labelled twice")
    return a


def labels_per_frame(rows: np.ndarray, n_frames: int) -> np.ndarray:
    """int32 [T,2] (action, repetition) for frames 0..T-1; unlabelled frames are (-1, 0)."""
    out = np.full((n_frames, 2), (-1, 0), np.int32)
    keep = (rows[:, 0] >= 0) & (rows[:, 0] < n_frames)
    out[rows[keep, 0]] = rows[keep, 1:]
    return out


def parse_skeleton_frame(text: str) -> dict[int, np.ndarray]:
    """One Skeletons/*.txt file -> {body id: float32 [32,12]}: confidence, X, Y, Z (mm),
    Qw, Qx, Qy, Qz, x, y (colour image px), x, y (depth image px)."""
    bodies: dict[int, np.ndarray] = {}
    for line in text.splitlines()[1:]:          # first line is the header
        parts = line.split()
        if len(parts) != 14:
            continue
        body, joint = int(float(parts[0])), int(float(parts[1]))
        if body == INVALID_BODY or not 0 <= joint < N_JOINTS:
            continue
        arr = bodies.setdefault(body, np.full((N_JOINTS, 12), np.nan, np.float32))
        arr[joint] = [float(p) for p in parts[2:]]
    return bodies


def parse_skeletons(files: dict[int, str], n_frames: int) -> dict[str, np.ndarray]:
    """{frame id: file text} -> arrays [T, ...] of the operator. In a frame with several
    bodies the operator is the one with the highest summed joint confidence."""
    xyz = np.full((n_frames, N_JOINTS, 3), np.nan, np.float32)
    xy = np.full((n_frames, N_JOINTS, 2), np.nan, np.float32)
    conf = np.zeros((n_frames, N_JOINTS), np.float32)
    body_id = np.full(n_frames, -1, np.int64)
    n_bodies = np.zeros(n_frames, np.int8)
    for fid, text in files.items():
        if not 0 <= fid < n_frames:
            continue
        bodies = parse_skeleton_frame(text)
        if not bodies:
            continue
        bid, arr = max(bodies.items(), key=lambda kv: np.nansum(kv[1][:, 0]))
        conf[fid], xyz[fid], xy[fid] = np.nan_to_num(arr[:, 0]), arr[:, 1:4], arr[:, 8:10]
        body_id[fid], n_bodies[fid] = bid, len(bodies)
    return {"frame_idx": np.arange(n_frames, dtype=np.int32), "joints_xyz": xyz, "joints_2d_color": xy,
            "confidence": conf, "body_id": body_id, "n_bodies": n_bodies}


def kinect_to_coco(xy: np.ndarray, conf: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kinect [T,32,2] and [T,32] -> COCO 17 [T,17,2] and [T,17]; confidence 0 means missing."""
    idx = np.array(KINECT32_TO_COCO17)
    k_xy, k_conf = xy[:, idx].copy(), conf[:, idx].copy()
    k_xy[k_conf <= 0] = np.nan
    return k_xy.astype(np.float32), k_conf.astype(np.float32)


def labels_csv(rows: np.ndarray) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["frame", "action", "repetition"])
    w.writerows(rows.tolist())
    return buf.getvalue()


def read_labels_csv(path: Path) -> np.ndarray:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = [[int(x["frame"]), int(x["action"]), int(x["repetition"])] for x in r]
    return np.array(rows, np.int32).reshape(-1, 3)


class HA4MLoader(DatasetLoader):
    name = NAME

    def source(self) -> dict:
        p = self.root / "SOURCE.json"
        return json.loads(p.read_text()) if p.is_file() else {}

    def cameras(self) -> dict[str, str]:
        return {**CAMERAS, **self.source().get("cameras", {})}

    def video_location(self, video_id: str) -> str:
        meta = self.root / "meta" / f"{video_id}.json"
        if meta.is_file():
            serial = json.loads(meta.read_text()).get("camera")
            if serial:
                return self.cameras().get(serial, serial)
        listing = self.source().get("camera_per_video", {})
        if video_id in listing:
            return self.cameras().get(listing[video_id], listing[video_id])
        raise AvError("AV-DAT-003", name=f"HA4M {video_id}", detail="its camera (lab) is not known yet")

    def videos(self) -> list[VideoRef]:
        """Videos whose fetch is complete (labels and a known camera; a video still being
        fetched is left out until its meta file is written)."""
        out = []
        for p in sorted((self.root / "labels").glob("IDU*V*.csv")):
            vid = p.stem
            try:
                loc = self.video_location(vid)
            except AvError:
                continue
            out.append(VideoRef(vid, video_subject(vid), loc, FPS, int(read_labels_csv(p)[:, 0].max()) + 1))
        return out

    def _frames(self, video_id: str) -> int:
        rows = read_labels_csv(self.root / "labels" / f"{video_id}.csv")
        n = int(rows[:, 0].max()) + 1 if len(rows) else 0
        rgb = self.root / "rgb" / f"{video_id}.frames.json"
        if rgb.is_file():
            n = max(n, int(json.loads(rgb.read_text())["n_frames"]))
        return n

    def load(self, video_id: str, keypoints: str = "ours") -> Sample:
        self._check_source(keypoints)
        lab_path = self.root / "labels" / f"{video_id}.csv"
        if not lab_path.is_file():
            raise AvError("AV-DAT-003", name=f"HA4M {video_id}", detail="no labels fetched (fetch_dataset annotations)")
        T = self._frames(video_id)
        labels = labels_per_frame(read_labels_csv(lab_path), T)
        ref = VideoRef(video_id, video_subject(video_id), self.video_location(video_id), FPS, T)
        meta: dict = {}
        if keypoints == "dataset":
            with np.load(self.root / "skeleton" / f"{video_id}.npz") as z:
                xy, conf = kinect_to_coco(z["joints_2d_color"][:T], z["confidence"][:T])
            meta["skeleton"] = "azure_kinect_32_to_coco17"
        else:
            sub = self.root / "derived" / "yolo26x" / video_id / "subject.npz"
            if not sub.is_file():
                raise AvError("AV-DAT-003", name=f"HA4M {video_id}",
                              detail="our keypoints are not derived yet (tools/dataset_pose)")
            with np.load(sub) as z:
                xy, conf = z["body_xy"][:T], z["body_conf"][:T]
            meta.update(json.loads((sub.parent / "source.json").read_text()))
        if len(xy) < T:                           # pad the tail (no frames there)
            pad = T - len(xy)
            xy = np.concatenate([xy, np.full((pad, 17, 2), np.nan, np.float32)])
            conf = np.concatenate([conf, np.zeros((pad, 17), np.float32)])
        present = np.isfinite(xy).all(axis=2).any(axis=1)
        return Sample(ref, keypoints, xy.astype(np.float32), np.nan_to_num(conf).astype(np.float32), present,
                      labels, self.boundaries(labels, FPS), meta)
