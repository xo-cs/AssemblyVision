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


# --- measurement ground truth, round 2 (2026-11-28): frames, wrists, events, suggestion ratings ---------------------

MENTORS = ("choi",)               # rate the suggestions; never see the team's labelling tasks
TRUNK = ["B0 (under 20 degrees)", "B1 (20 to 45 degrees)", "B2 (45 to 60 degrees)", "B3 (over 60 degrees)", "cannot see"]
HANDS = ["yes", "no", "cannot see"]
GAZE2 = ["towards the hands or the work", "left (in the image)", "right (in the image)", "up", "away from the camera",
         "cannot see"]
EVENT_KINDS = ["cycle start", "tool pickup", "object placement"]
RATINGS = ["useful", "partly useful", "not useful"]
GT_TASKS = {
    "frames": {"title": "Frames: trunk angle (two clicks), both hands, gaze (100 frames)", "n": 100},
    "wrists": {"title": "Wrists: click left and right (50 frames)", "n": 50},
    "events": {"title": "Events: cycle starts, tool pickups, object placements (2 minute clip)", "span_s": 120.0},
}
# trunk: "clicks" (hip and shoulder midpoints clicked, angle computed) or "cannot see"; the band radio answers of the
# first version (2026-11-28) are dropped (drop_old_bands), since nobody can judge 20, 45 or 60 degrees by eye
FRAME_FIELDS = ["item", "frame", "time_s", "phase", "trunk", "hip_x", "hip_y", "shoulder_x", "shoulder_y", "angle_deg",
                "band", "hands", "gaze", "user", "utc"]
TRUNK_CLICKS, TRUNK_CANNOT = "clicks", "cannot see"
BAND_NAMES = ["B0", "B1", "B2", "B3"]
BAND_TEXT = {"B0": "B0 (under 20 degrees)", "B1": "B1 (20 to 45 degrees)", "B2": "B2 (45 to 60 degrees)",
             "B3": "B3 (over 60 degrees)"}
WRIST_FIELDS = ["item", "frame", "time_s", "phase", "left_x", "left_y", "right_x", "right_y", "user", "utc"]
EVENT_FIELDS = ["item", "time_s", "kind", "user", "utc"]
RATING_FIELDS = ["item", "rule", "phase", "cycle", "rating", "comment", "user", "utc"]


def is_mentor(user: str | None) -> bool:
    return (user or "") in MENTORS


def phase_spans(root: Path, manifest: dict) -> tuple[list[tuple[str, float, float]], str]:
    """The work phases of the run: the linked hand labels if there are any, else the run's steps; (spans, source)."""
    from .evaluation.gt_links import gt_for_run
    from .evaluation.steps_io import read_steps_csv
    gt = gt_for_run(Path(root).parent.parent, manifest)
    steps = read_steps_csv(gt) if gt else read_steps_csv(Path(root) / "steps.csv")
    return [(s.label, s.start_s, s.end_s) for s in steps if s.label != "other"], "hand labels" if gt else "run steps"


def stratified_frames(task: str, run_id: str, frames: np.ndarray, fps: float, spans: list, n: int) -> list[dict]:
    """n frames with a tracked subject, the same number from each phase (the rest from the phases with frames left)."""
    rng = _rng(run_id, task)
    frames = np.unique(np.asarray(frames, np.int64))
    pools = [(name, frames[(frames >= a * fps) & (frames < b * fps)]) for name, a, b in spans]
    pools = [(nm, p) for nm, p in pools if len(p)]
    if not pools:
        return []
    want = {nm: n // len(pools) for nm, _ in pools}
    for nm, _ in pools[: n - sum(want.values())]:
        want[nm] += 1
    picked = []
    for nm, p in pools:
        k = min(want[nm], len(p))
        picked += [(int(f), nm) for f in rng.choice(p, k, replace=False)]
    picked.sort()
    return [{"item": i, "frame": f, "time_s": round(f / fps, 3), "phase": nm} for i, (f, nm) in enumerate(picked)]


def event_window(root: Path, manifest: dict, spans: list, fps: float) -> dict:
    """The 2 minute clip of the event task, chosen by a fixed rule (no person looks at the video to choose it): among the
    phases of at least 2 minutes, the one where the wrists move most (median wrist speed relative to the hip); its middle
    2 minutes. With no phase that long, the longest phase."""
    from .analysis.motion import wrist_speeds
    from .config import Config
    from .output.corrections import _Ctx
    from .tools.run_pose import stored_config
    from .analysis.motion import MotionAnalyzer
    cfg = Config.load(overrides=stored_config(manifest))
    subs = MotionAnalyzer(cfg, _Ctx(manifest, Path(root))).load_subjects(manifest, Path(root))
    span = GT_TASKS["events"]["span_s"]
    long = [s for s in spans if s[2] - s[1] >= span] or [max(spans, key=lambda s: s[2] - s[1])]
    best, score = long[0], -1.0
    for nm, a, b in long:
        v = []
        for s in subs:
            if not s.cs.torso:
                continue
            l, r = wrist_speeds(s.cs.xy, s.cs.valid, s.cs.torso, fps)
            f = np.asarray(s.cs.frame_idx) / fps
            m = (f >= a) & (f < b)
            v += [x for x in np.r_[l[m], r[m]] if np.isfinite(x)]
        med = float(np.median(v)) if v else 0.0
        if med > score:
            best, score = (nm, a, b), med
    nm, a, b = best
    mid = (a + b) / 2
    t0 = max(a, mid - span / 2)
    t1 = min(b, t0 + span)
    return {"item": 0, "phase": nm, "time_s": round(t0, 2), "end_s": round(t1, 2), "frame": int(round(t0 * fps)),
            "rule": "middle 2 minutes of the phase of at least 2 minutes with the highest median wrist speed"}


def gt_items(task: str, run_id: str, root: Path) -> tuple[list[dict], float, str]:
    """(items, fps, phase source) of a round 2 task."""
    m = json.loads((Path(root) / "manifest.json").read_text())
    fps = float(m["ingest"]["cfr"]["fps"])
    spans, src = phase_spans(root, m)
    if task == "events":
        return [event_window(root, m, spans, fps)], fps, src
    return stratified_frames(task, run_id, subject_frames(m, root), fps, spans, GT_TASKS[task]["n"]), fps, src


def gt_path(area: Path, run_id: str, task: str, user: str) -> Path:
    if task not in GT_TASKS and task != "ratings":
        raise ValueError(f"unknown task {task!r}")
    return Path(area) / "validation" / "labels" / run_id / task / f"{check_user(user)}.csv"


def _replace(path: Path, key: str, value, row: dict, fields: list[str]) -> None:
    rows = [r for r in read(path) if str(r[key]) != str(value)]
    rows.append(row)
    _write(path, sorted(rows, key=lambda r: float(r.get("time_s") or r.get("item") or 0)), fields)


def trunk_angle_band(hip, shoulder, edges=(20.0, 45.0, 60.0)) -> tuple[float, str]:
    """(angle in degrees, band) from the clicked hip and shoulder midpoints, exactly as the system computes its trunk angle
    (analysis/motion.trunk_angle: mid shoulder minus mid hip against the upward image vertical, 2D image pixels) and its
    band (motion.band_of with motion.angle_bands)."""
    from .analysis.motion import band_of, trunk_angle
    a = float(trunk_angle(np.asarray(shoulder, np.float64), np.asarray(hip, np.float64)))
    return a, BAND_NAMES[int(band_of(np.array([a]), list(edges))[0])]


def drop_old_bands(rows: list[dict]) -> tuple[list[dict], int]:
    """The rows with every band radio answer of the first version removed (hands and gaze kept); and how many."""
    out, n = [], 0
    for r in rows:
        r = {k: r.get(k, "") for k in FRAME_FIELDS}
        if r["trunk"] not in (TRUNK_CLICKS, TRUNK_CANNOT, ""):
            r.update(trunk="", hip_x="", hip_y="", shoulder_x="", shoulder_y="", angle_deg="", band="")
            n += 1
        out.append(r)
    return out, n


def save_frame(path: Path, item: dict, trunk, hands: str, gaze: str, user: str, edges=(20.0, 45.0, 60.0)) -> None:
    """trunk: TRUNK_CANNOT, or ((hip x, y), (shoulder x, y)) in pixels of the original frame."""
    if hands not in HANDS or gaze not in GAZE2 or trunk is None:
        raise ValueError("answer all three questions (two trunk clicks or cannot see)")
    row = {"item": item["item"], "frame": item["frame"], "time_s": item["time_s"], "phase": item["phase"], "hands": hands,
           "gaze": gaze, "user": user, "utc": utc_now(), "hip_x": "", "hip_y": "", "shoulder_x": "", "shoulder_y": "",
           "angle_deg": "", "band": ""}
    if trunk == TRUNK_CANNOT:
        row["trunk"] = TRUNK_CANNOT
    else:
        (hx, hy), (sx, sy) = trunk
        a, band = trunk_angle_band((hx, hy), (sx, sy), edges)
        row.update(trunk=TRUNK_CLICKS, hip_x=round(float(hx), 1), hip_y=round(float(hy), 1), shoulder_x=round(float(sx), 1),
                   shoulder_y=round(float(sy), 1), angle_deg=round(a, 2), band=band)
    rows, _ = drop_old_bands([r for r in read(path) if str(r["item"]) != str(item["item"])])
    _write(path, sorted(rows + [row], key=lambda r: int(r["item"])), FRAME_FIELDS)


def save_wrists(path: Path, item: dict, left, right, user: str) -> None:
    """left and right: (x, y) in video pixels, or None for not visible."""
    f = lambda p, k: "" if p is None else round(float(p[k]), 1)
    _replace(path, "item", item["item"], {"item": item["item"], "frame": item["frame"], "time_s": item["time_s"],
                                          "phase": item["phase"], "left_x": f(left, 0), "left_y": f(left, 1),
                                          "right_x": f(right, 0), "right_y": f(right, 1), "user": user, "utc": utc_now()},
             WRIST_FIELDS)


def add_event(path: Path, t: float, kind: str, window: dict, user: str) -> None:
    if kind not in EVENT_KINDS:
        raise ValueError("unknown event kind")
    if not window["time_s"] <= t <= window["end_s"]:
        raise ValueError(f"the mark must lie between {window['time_s']:.1f} and {window['end_s']:.1f} s")
    rows = read(path)
    rows.append({"item": max([int(r["item"]) for r in rows], default=-1) + 1, "time_s": round(t, 3), "kind": kind,
                 "user": user, "utc": utc_now()})
    _write(path, sorted(rows, key=lambda r: float(r["time_s"])), EVENT_FIELDS)


def remove_last_event(path: Path) -> None:
    rows = read(path)
    if rows:
        last = max(rows, key=lambda r: int(r["item"]))
        _write(path, [r for r in rows if r is not last], EVENT_FIELDS)


def save_rating(path: Path, i: int, s: dict, rating: str, comment: str, user: str) -> None:
    if rating not in RATINGS:
        raise ValueError("choose a rating")
    _replace(path, "item", i, {"item": i, "rule": s["rule"], "phase": s["phase"], "cycle": s["cycle"], "rating": rating,
                               "comment": (comment or "").strip()[:500], "user": user, "utc": utc_now()}, RATING_FIELDS)
