"""Score the 10 measurements of a run against the hand labels (Week 12, docs_team/validation_measurements.md). CPU.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.score_measurements --run RUN_DIR --gt GT_CSV

The labels are the Labels page files <data area>/validation/labels/<run>/<task>/<user>.csv (each annotator scored
alone). For every labelled item the system's own answer is computed from the run's stored keypoints with the same
code and settings as the metrics (rules below), then compared. The targets are the ones frozen on 2026-11-26.
Prints a table of summary numbers only (no frame, no label value, no time); writes measurements_score.json and .md in
<data area>/validation/. A measurement without labels is reported as such, never as a number.

The system's answer per item:
- trunk band: the band of the trunk angle (motion.subject_trunk, motion.angle_bands) at the labelled frame;
- gaze: motion.gaze_class of the raw face keypoints (conf >= motion.gaze_conf_min) at the frame (front, left, right, away);
- two hand: over the 3 s clip, per frame with both wrist speeds known, both, one or no hand at or above
  motion.hand_active; the most frequent of the three;
- repetitive: over the 10 s clip, a wrist's autocorrelation (motion.acf_period, the repetition metric's lags) peaks at or
  above motion.rep_acf_min; at least motion.cov_min of the frames must be known;
- tool pickup and object placement: the events of output/extras.py inside the labelled window, matched to the marks one
  to one within 1 s (needs objects.npz from tools/detect_objects.py);
- process time: the run's steps against the final phase labels (GT_CSV).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .. import validation_labels as VL
from ..config import Config
from ..storage.files import run_source, utc_now
from . import freeze

CANNOT = VL.CANNOT
BANDS = ["under 20 degrees", "20 to 45 degrees", "45 to 60 degrees", "over 60 degrees"]
GAZE = ["front", "left", "right", "away"]
MATCH_S = 1.0
TARGETS = {
    1: "mean boundary error at most 10 s, each phase duration within 10 %",
    2: "within 20 % over a walk",
    3: "exact band at least 80 %, within one band at least 95 %",
    4: "F1 at least 0.70",
    5: "at least 60 % of the suggestions rated useful",
    6: "wrist within PCK@0.1 for at least 85 %; travel within 20 %",
    7: "events within 1 s: recall and precision at least 0.70",
    8: "events within 1 s: recall and precision at least 0.70",
    9: "agreement at least 75 %",
    10: "agreement at least 70 %",
}
NAMES = {1: "process time", 2: "movement path", 3: "waist bending", 4: "repetitive work", 5: "improvable motions",
         6: "hand travel", 7: "tool pickup time", 8: "object placement time", 9: "two hand cooperation", 10: "gaze direction"}
TASK_OF = {3: "trunk_band", 4: "repetition", 7: "tool_pickup", 8: "object_place", 9: "two_hand", 10: "gaze"}


class Frames:
    """The subject's cleaned and raw keypoints by run frame."""

    def __init__(self, root: Path, manifest: dict, cfg):
        from ..analysis.motion import MotionAnalyzer
        from ..output.corrections import _Ctx
        self.subjects = MotionAnalyzer(cfg, _Ctx(manifest, root)).load_subjects(manifest, root)
        self.where = {}
        for k, s in enumerate(self.subjects):
            for i, f in enumerate(np.asarray(s.cs.frame_idx, np.int64)):
                self.where[int(f)] = (k, i)

    def at(self, f: int):
        return self.where.get(int(f))

    def span(self, a: int, b: int):
        """(subject, row from, row to) of the longest subject stretch in frames [a, b)."""
        best = None
        for k, s in enumerate(self.subjects):
            fr = np.asarray(s.cs.frame_idx, np.int64)
            i, j = int(np.searchsorted(fr, a)), int(np.searchsorted(fr, b))
            if j > i and (best is None or j - i > best[2] - best[1]):
                best = (k, i, j)
        return best


def system_trunk(F: Frames, f: int, m: dict) -> str | None:
    from ..analysis.motion import band_of, subject_trunk
    w = F.at(f)
    if w is None:
        return None
    s = F.subjects[w[0]]
    _, ang = subject_trunk(s.cs)
    a = ang[w[1]]
    return BANDS[int(band_of(np.array([a]), m["angle_bands"])[0])] if np.isfinite(a) else None


def system_gaze(F: Frames, f: int, m: dict) -> str | None:
    from ..analysis.motion import gaze_class
    w = F.at(f)
    if w is None:
        return None
    s = F.subjects[w[0]]
    si = list(np.asarray(s.series.frame_idx, np.int64)).index(f) if f in set(np.asarray(s.series.frame_idx).tolist()) else None
    if si is None or not s.series.present[si]:
        return None
    raw = np.asarray(s.series.body_xy[si:si + 1], np.float64)
    conf = np.nan_to_num(np.asarray(s.series.body_conf[si:si + 1], np.float64))
    valid = (conf >= m["gaze_conf_min"]) & np.isfinite(raw).all(-1)
    c = int(gaze_class(raw, valid, m["gaze_yaw"])[0])
    return GAZE[c] if c >= 0 else None


def system_two_hand(F: Frames, a: int, b: int, fps: float, m: dict) -> str | None:
    from ..analysis.motion import wrist_speeds
    sp = F.span(a, b)
    if sp is None:
        return None
    s = F.subjects[sp[0]]
    if not s.cs.torso:
        return None
    l, r = wrist_speeds(s.cs.xy, s.cs.valid, s.cs.torso, fps)
    l, r = l[sp[1]:sp[2]], r[sp[1]:sp[2]]
    ok = np.isfinite(l) & np.isfinite(r)
    if not ok.any():
        return None
    n = (l[ok] >= m["hand_active"]).astype(int) + (r[ok] >= m["hand_active"]).astype(int)
    return {2: "both hands", 1: "one hand", 0: "no hand"}[Counter(n.tolist()).most_common(1)[0][0]]


def system_repetitive(F: Frames, a: int, b: int, fps: float, m: dict) -> str | None:
    from ..analysis.cleaning import L_HIP, R_HIP
    from ..analysis.motion import L_WRI, R_WRI, acf_period
    sp = F.span(a, b)
    if sp is None:
        return None
    s = F.subjects[sp[0]]
    if not s.cs.torso or sp[2] - sp[1] < 0.8 * (b - a):
        return None
    xy, val = s.cs.xy[sp[1]:sp[2]].astype(np.float64), s.cs.valid[sp[1]:sp[2]]
    hip, hv = (xy[:, L_HIP] + xy[:, R_HIP]) / 2, val[:, L_HIP] & val[:, R_HIP]
    sigs = []
    for w in (L_WRI, R_WRI):
        q = (xy[:, w] - hip) / s.cs.torso
        q[~(val[:, w] & hv)] = np.nan
        sigs.append(q)
    if max(np.isfinite(q).all(1).mean() for q in sigs) < m["cov_min"]:
        return None
    best = max((acf_period(q, fps, m["rep_lag_min_s"], m["rep_lag_max_s"]) for q in sigs), key=lambda r: r[1])
    return "repetitive" if best[0] is not None and best[1] >= m["rep_acf_min"] else "not repetitive"


def agreement(pairs: list[tuple[str, str | None]]) -> dict:
    """Over items the annotator could tell: agreement where the system has an answer, and its coverage."""
    told = [(h, s) for h, s in pairs if h != CANNOT]
    both = [(h, s) for h, s in told if s is not None]
    return {"labels": len(pairs), "told": len(told), "scored": len(both),
            "coverage": round(len(both) / len(told), 3) if told else None,
            "agreement": round(sum(h == s for h, s in both) / len(both), 3) if both else None}


def match_events(marks: list[float], events: list[float]) -> dict:
    """One to one matches within MATCH_S, nearest first."""
    pairs = sorted((abs(m - e), i, j) for i, m in enumerate(marks) for j, e in enumerate(events) if abs(m - e) <= MATCH_S)
    used_m, used_e = set(), set()
    for _, i, j in pairs:
        if i not in used_m and j not in used_e:
            used_m.add(i)
            used_e.add(j)
    tp = len(used_m)
    return {"marks": len(marks), "events": len(events), "matched": tp,
            "recall": round(tp / len(marks), 3) if marks else None, "precision": round(tp / len(events), 3) if events else None}


def verdict(n: int, r: dict) -> str:
    """Whether the frozen target of measurement n is met."""
    def yes(x):
        return "meets the target" if x else "misses the target"
    if n == 1:
        return yes(r["boundary_mae_s"] is not None and r["boundary_mae_s"] <= 10 and r["phases_within_10_pct"] == r["phases"])
    if n == 3:
        return "-" if r["agreement"] is None else yes(r["agreement"] >= 0.80 and r["within_one"] >= 0.95)
    if n == 4:
        return "-" if r.get("f1") is None else yes(r["f1"] >= 0.70)
    if n in (7, 8):
        if r.get("recall") is None:
            return "-"
        if r["recall"] < 0.70 or (r.get("precision") is not None and r["precision"] < 0.70):
            return yes(False)                       # either side below the target is enough, even if the other is undefined
        return "-" if r.get("precision") is None else yes(True)
    if n in (9, 10):
        return "-" if r["agreement"] is None else yes(r["agreement"] >= (0.75 if n == 9 else 0.70))
    return "-"


def score(root: Path, gt_path: Path | None) -> dict:
    from ..evaluation import metrics as M
    from ..evaluation.steps_io import read_steps_csv
    from ..output import extras as X
    from .run_pose import stored_config
    m_all = json.loads((root / "manifest.json").read_text())
    cfg = Config.load(overrides=stored_config(m_all))
    m = cfg["motion"]
    fps = float(m_all["ingest"]["cfr"]["fps"])
    duration = float(m_all["ingest"]["cfr"]["duration_s"])
    area = root.parent.parent
    F = Frames(root, m_all, cfg)
    out: dict = {}
    if gt_path:
        pred, gt = read_steps_csv(root / "steps.csv"), read_steps_csv(gt_path)
        mae, _ = M.boundary_mae(pred, gt, duration)
        dur = lambda steps: {s.label: s.end_s - s.start_s for s in steps if s.label != "other"}
        p, g = dur(pred), dur(gt)
        within = sum(1 for k, v in g.items() if k in p and abs(p[k] - v) <= 0.10 * v)
        out[1] = {"labels": "two annotators (final labels)", "boundary_mae_s": mae, "phases": len(g), "phases_within_10_pct": within,
                  "duration_error_pct": {k: round(100 * (p.get(k, 0) - v) / v, 1) for k, v in g.items()}}
    events = None
    for n, task in TASK_OF.items():
        files = sorted((area / "validation" / "labels" / root.name / task).glob("*.csv"))
        if not files:
            continue
        per_user = {}
        for f in files:
            rows = VL.read(f)
            if n == 3:
                pairs = [(r["value"], system_trunk(F, int(r["frame"]), m)) for r in rows]
                res = agreement(pairs)
                near = [(h, s) for h, s in pairs if h != CANNOT and s is not None]
                res["within_one"] = round(sum(abs(BANDS.index(h) - BANDS.index(s)) <= 1 for h, s in near) / len(near), 3) if near else None
            elif n == 10:
                res = agreement([(r["value"], system_gaze(F, int(r["frame"]), m)) for r in rows])
            elif n == 9:
                span = int(round(VL.TASKS[task]["span_s"] * fps))
                res = agreement([(r["value"], system_two_hand(F, int(r["frame"]), int(r["frame"]) + span, fps, m)) for r in rows])
            elif n == 4:
                span = int(round(VL.TASKS[task]["span_s"] * fps))
                pairs = [(r["value"], system_repetitive(F, int(r["frame"]), int(r["frame"]) + span, fps, m)) for r in rows]
                res = agreement(pairs)
                both = [(h, s) for h, s in pairs if h != CANNOT and s is not None]
                tp = sum(h == s == "repetitive" for h, s in both)
                fp = sum(s == "repetitive" and h != "repetitive" for h, s in both)
                fn = sum(h == "repetitive" and s != "repetitive" for h, s in both)
                res["f1"] = round(2 * tp / (2 * tp + fp + fn), 3) if 2 * tp + fp + fn else None
            else:
                window = VL.run_items(task, root.name, root)[0][0]
                if events is None:
                    det = X.load_objects(root)
                    events = None if det is None else X.hand_event_list(F.subjects, det, fps, m["hand_active"])
                if events is None:
                    per_user[f.stem] = {"labels": len(rows), "note": "no object detection for this run (run tools/detect_objects.py first)"}
                    continue
                kind = "pickup" if n == 7 else "placement"
                sys_t = [e["t_s"] for e in events if e["kind"] == kind and window["time_s"] <= e["t_s"] <= window["end_s"]]
                res = {"labels": len(rows), **match_events([float(r["time_s"]) for r in rows], sys_t)}
            per_user[f.stem] = res
        out[n] = {"annotators": per_user}
    return {"created_utc": utc_now(), "freeze": freeze.record(), "run": root.name, "targets_frozen": "2026-11-26",
            "results": {str(k): v for k, v in out.items()}}


def fmt(n: int, r: dict | None) -> list[tuple[str, str, str]]:
    """(annotator, result text, verdict) rows of one measurement."""
    if r is None:
        why = {2: "no labels yet (needs walks of known length)", 5: "no labels yet (needs an engineer's rating of the suggestions)",
               6: "no labels yet (needs Label Studio wrist keypoints)", 1: "no final phase labels given (--gt)"}
        return [("-", why.get(n, "no labels yet"), "-")]
    if n == 1:
        t = f"boundary MAE {r['boundary_mae_s']} s; {r['phases_within_10_pct']} of {r['phases']} phase durations within 10 %"
        return [(r["labels"], t, verdict(1, r))]
    rows = []
    for user, x in r["annotators"].items():
        if "note" in x:
            rows.append((user, f"{x['labels']} marks; {x['note']}", "-"))
            continue
        if n in (7, 8):
            t = f"{x['marks']} marks, {x['events']} system events, {x['matched']} matched: recall {x['recall']}, precision {x['precision']}"
        else:
            t = f"{x['labels']} labels, {x['told']} not 'cannot tell', {x['scored']} with a system answer: agreement {x['agreement']}"
            if n == 3:
                t += f", within one band {x['within_one']}"
            if n == 4:
                t += f", F1 {x['f1']}"
        rows.append((user, t, verdict(n, x)))
    return rows


def markdown(res: dict) -> str:
    f = res["freeze"]
    L = [f"# Measurement scores of {res['run']}", "", f"Freeze: {f['tag']} ({f['tag_commit'][:7]}), ran on {f['ran_on'][:7]}, "
         f"method unchanged: {f['method_unchanged']}. Targets frozen on {res['targets_frozen']}.", "",
         "| # | measurement | annotator | result | target | verdict |", "| --- | --- | --- | --- | --- | --- |"]
    for n in range(1, 11):
        for user, text, v in fmt(n, res["results"].get(str(n))):
            L.append(f"| {n} | {NAMES[n]} | {user} | {text} | {TARGETS[n]} | {v} |")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.score_measurements")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--gt", type=Path)
    a = p.parse_args(argv)
    _, root = run_source(a.run, Config.load()["paths"])
    res = score(root, a.gt)
    out = root.parent.parent / "validation"
    out.mkdir(exist_ok=True)
    (out / "measurements_score.json").write_text(json.dumps(res, indent=1))
    md = markdown(res)
    (out / "measurements_score.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
