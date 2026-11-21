"""Hand labels for validation (Week 12, docs_team/validation_measurements.md and train_domain_test.md).

Keypoints (for example the wrists of hand travel) use the existing Label Studio workflow (labelling/). This module
holds the other small labelling tasks on the frames or short stretches of a stored run, and the class of each public
train domain clip. Items are drawn with a fixed seed from the frames where the subject is tracked, so every
annotator labels the same items and no system output decides which. Each annotator has one file:
<data area>/validation/labels/<run>/<task>/<user>.csv, or train_domain/annotations/<user>.csv. A QiSens run's
labels stay under data/qisens. Nothing here changes a run.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from .storage.files import utc_now

SEED = 20261121
CANNOT = "cannot tell"
CLIPS = "clip_class"
TASKS = {
    "trunk_band": {"title": "Waist bending: trunk angle band", "kind": "frame", "n": 100,
                   "ask": "How far is the trunk bent from upright?",
                   "options": ["under 20 degrees", "20 to 45 degrees", "45 to 60 degrees", "over 60 degrees", CANNOT]},
    "gaze": {"title": "Gaze: head direction in the image", "kind": "frame", "n": 100,
             "ask": "Where does the head point, as seen in the image?",
             "options": ["front", "left", "right", "away", CANNOT]},
    "two_hand": {"title": "Two hand cooperation", "kind": "clip", "n": 40, "span_s": 3.0,
                 "ask": "Which hands are doing work in these 3 s?",
                 "options": ["both hands", "one hand", "no hand", CANNOT]},
    "repetition": {"title": "Repetitive work", "kind": "clip", "n": 30, "span_s": 10.0,
                   "ask": "Does the worker repeat the same short motion in these 10 s?",
                   "options": ["repetitive", "not repetitive", CANNOT]},
    "tool_pickup": {"title": "Tool pickup moments (2 minutes)", "kind": "event", "span_s": 120.0,
                    "ask": "Play the video and press Mark at each moment a hand closes on a tool."},
    "object_place": {"title": "Object placement moments (2 minutes)", "kind": "event", "span_s": 120.0,
                     "ask": "Play the video and press Mark at each moment a hand lets go of a placed object."},
}
FIELDS = ["item", "frame", "time_s", "value", "user", "utc"]
CLIP_FIELDS = ["file", "class", "second_class", "change_s", "user", "utc"]


def _rng(run_id: str, task: str) -> np.random.Generator:
    key = hashlib.sha256(f"{SEED}:{run_id}:{task}".encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "little"))


def draw_items(task: str, run_id: str, frames: np.ndarray, fps: float, n_video: int) -> list[dict]:
    """The items of a task: frames, or stretches starting at a tracked frame, or the one event window."""
    t = TASKS[task]
    frames = np.unique(np.asarray(frames, np.int64))
    rng = _rng(run_id, task)
    if t["kind"] == "event":
        span = int(round(t["span_s"] * fps))
        last = max(0, n_video - span)
        ok = frames[frames <= last]
        start = int(rng.choice(ok)) if len(ok) else 0
        return [{"item": 0, "frame": start, "time_s": round(start / fps, 3), "end_s": round(min(n_video, start + span) / fps, 3)}]
    if t["kind"] == "clip":
        span = int(round(t["span_s"] * fps))
        frames = frames[frames <= max(0, n_video - span)]
    if not len(frames):
        return []
    pick = np.sort(rng.choice(frames, min(t["n"], len(frames)), replace=False))
    out = []
    for i, f in enumerate(pick.tolist()):
        d = {"item": i, "frame": f, "time_s": round(f / fps, 3)}
        if t["kind"] == "clip":
            d["end_s"] = round(min(n_video, f + span) / fps, 3)
        out.append(d)
    return out


def check_user(user: str | None) -> str:
    """Labels belong to a logged in account: the name is part of a file name."""
    from .ui.auth import NAME_RE
    if not user or user == "interface" or not NAME_RE.fullmatch(user):
        raise ValueError("log in with your own account to label")
    return user


def label_path(area: Path, run_id: str, task: str, user: str) -> Path:
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}")
    return Path(area) / "validation" / "labels" / run_id / task / f"{check_user(user)}.csv"


def clip_path(folder: Path, user: str) -> Path:
    return Path(folder) / "annotations" / f"{check_user(user)}.csv"


def read(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_item(path: Path, item: dict, value: str, user: str) -> None:
    """One label per item: a new answer replaces the annotator's old one."""
    rows = [r for r in read(path) if int(r["item"]) != int(item["item"])]
    rows.append({"item": item["item"], "frame": item["frame"], "time_s": item["time_s"], "value": value, "user": user, "utc": utc_now()})
    _write(path, sorted(rows, key=lambda r: int(r["item"])), FIELDS)


def add_mark(path: Path, t: float, window: dict, fps: float, user: str) -> None:
    """An event time inside the task's window."""
    if not window["time_s"] <= t <= window["end_s"]:
        raise ValueError(f"the mark must lie between {window['time_s']:.1f} and {window['end_s']:.1f} s")
    rows = read(path)
    rows.append({"item": max([int(r["item"]) for r in rows], default=-1) + 1, "frame": int(round(t * fps)),
                 "time_s": round(t, 3), "value": "mark", "user": user, "utc": utc_now()})
    _write(path, sorted(rows, key=lambda r: float(r["time_s"])), FIELDS)


def remove_last_mark(path: Path) -> None:
    rows = read(path)
    if rows:
        last = max(rows, key=lambda r: int(r["item"]))
        _write(path, [r for r in rows if r is not last], FIELDS)


def save_clip(path: Path, file: str, classes: list[str], first: str, second: str | None, change: float | None,
              duration: float, user: str) -> None:
    second = second or None
    if first not in classes or (second and second not in classes):
        raise ValueError("choose a class from the list")
    if second:
        if second == first:
            raise ValueError("the second class must differ from the first")
        if change is None or not 0 < change < duration:
            raise ValueError(f"a second class needs a change time between 0 and {duration:.1f} s")
    rows = [r for r in read(path) if r["file"] != file]
    rows.append({"file": file, "class": first, "second_class": second or "", "change_s": "" if not second else round(float(change), 2),
                 "user": user, "utc": utc_now()})
    _write(path, sorted(rows, key=lambda r: r["file"]), CLIP_FIELDS)


def subject_frames(manifest: dict, root: Path) -> np.ndarray:
    """The frames where the subject is tracked (stored KTS files)."""
    from .kts import read_kts
    out = []
    for tr in manifest.get("tracking", {}).get("tracks", []):
        if tr.get("is_subject"):
            s, _ = read_kts(Path(root) / tr["kts_path"])
            out.append(np.asarray(s.frame_idx)[np.asarray(s.present, bool)])
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def run_items(task: str, run_id: str, root: Path) -> tuple[list[dict], float]:
    m = json.loads((Path(root) / "manifest.json").read_text())
    cfr = m["ingest"]["cfr"]
    fps = float(cfr["fps"])
    n = int(round(float(cfr["duration_s"]) * fps))
    return draw_items(task, run_id, subject_frames(m, root), fps, n), fps


def read_frame(video: Path, frame: int, max_w: int = 960) -> np.ndarray | None:
    """One RGB frame, at most max_w wide; (nothing is written)."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, f = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    if f.shape[1] > max_w:
        f = cv2.resize(f, (max_w, int(round(f.shape[0] * max_w / f.shape[1]))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
