"""Equivalence of two runs of one video (Week 10 speed experiments).

A speed variant must not change the results. compare_runs(reference, candidate)
checks, from the stored files of both run folders:

- pose: per frame the same number of detections; detections are matched per
  frame (greedy on the smallest box difference, so a changed order is not a
  difference); box and keypoint coordinates within 1 px; box and keypoint
  confidences within 0.01. Max and mean differences and the share of frames
  whose detections all lie within 1 px are reported.
- reference frames: the same frame per shot, boxes within 1 px.
- tracking: the same track assignment for every matched detection, and the same
  subject, tracks kept and duplicates suppressed per shot.
- Engine A: the same step labels, boundaries within one frame.
- motion metrics: every row matched by (segment, metric); values within 1 %
  relative or 0.01 absolute; the same NULL or reported decision; identical flags.

The verdict is "equivalent" only if every check holds; every difference is
listed either way. Shot, segment and track IDs carry the run ID, so they are
compared without it (shots by their order).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..perception.pose import load_det

PX_TOL = 1.0
CONF_TOL = 0.01
METRIC_REL = 0.01
METRIC_ABS = 0.01
MAX_LISTED = 20          # differences listed per check (all are counted)


def _manifest(run: Path) -> dict:
    return json.loads((Path(run) / "manifest.json").read_text())


def _by_frame(d: dict) -> dict[int, np.ndarray]:
    order = np.argsort(d["frame_idx"], kind="stable")
    fidx = d["frame_idx"][order]
    out = {}
    for f in np.unique(fidx):
        lo, hi = np.searchsorted(fidx, f, "left"), np.searchsorted(fidx, f, "right")
        out[int(f)] = order[lo:hi]
    return out


def match_rows(a: dict, ra: np.ndarray, b: dict, rb: np.ndarray) -> list[tuple[int, int]]:
    """Pairs (row of a, row of b) of one frame, greedy on the mean absolute box difference."""
    if len(ra) == 0 or len(rb) == 0:
        return []
    cost = np.abs(a["bbox_xyxy"][ra][:, None, :] - b["bbox_xyxy"][rb][None, :, :]).mean(axis=2)
    pairs, used_a, used_b = [], set(), set()
    for flat in np.argsort(cost, axis=None, kind="stable"):
        i, j = divmod(int(flat), cost.shape[1])
        if i in used_a or j in used_b:
            continue
        pairs.append((int(ra[i]), int(rb[j])))
        used_a.add(i)
        used_b.add(j)
        if len(used_a) == min(len(ra), len(rb)):
            break
    return pairs


def compare_det(a: dict, b: dict) -> tuple[dict, dict[int, int]]:
    """Pose statistics of one shot and the row mapping a -> b."""
    fa, fb = _by_frame(a), _by_frame(b)
    frames = sorted(set(fa) | set(fb))
    empty = np.zeros(0, np.int64)
    mapping: dict[int, int] = {}
    count_mismatch, within, bitwise = [], 0, 0
    box_d, kp_d, dconf_d, kconf_d = [], [], [], []
    for f in frames:
        ra, rb = fa.get(f, empty), fb.get(f, empty)
        if len(ra) != len(rb):
            count_mismatch.append({"frame": f, "reference": int(len(ra)), "candidate": int(len(rb))})
        pairs = match_rows(a, ra, b, rb)
        ok = len(ra) == len(rb)
        same = ok
        for i, j in pairs:
            mapping[i] = j
            bd = np.abs(a["bbox_xyxy"][i] - b["bbox_xyxy"][j])
            kd = np.abs(a["body_xy"][i] - b["body_xy"][j])
            box_d.append(bd)
            kp_d.append(kd.reshape(-1))
            dconf_d.append(abs(float(a["det_conf"][i]) - float(b["det_conf"][j])))
            kconf_d.append(np.abs(a["body_conf"][i] - b["body_conf"][j]))
            if bd.max() > PX_TOL or kd.max() > PX_TOL:
                ok = False
            if bd.max() > 0 or kd.max() > 0:
                same = False
        within += ok
        bitwise += same
    cat = (lambda xs: np.concatenate([np.ravel(x) for x in xs]) if xs else np.zeros(0))
    box, kp, kconf = cat(box_d), cat(kp_d), cat(kconf_d)
    dconf = np.asarray(dconf_d, np.float64)

    def mx(x):
        return round(float(x.max()), 4) if x.size else 0.0

    def mean(x):
        return round(float(x.mean()), 5) if x.size else 0.0
    return {"frames": len(frames), "count_mismatch_frames": len(count_mismatch),
            "count_mismatch": count_mismatch[:MAX_LISTED],
            "frames_within_1px": within, "frames_identical": bitwise,
            "box_max_px": mx(box), "box_mean_px": mean(box), "kp_max_px": mx(kp), "kp_mean_px": mean(kp),
            "det_conf_max": mx(dconf), "kp_conf_max": mx(kconf), "matched": len(mapping)}, mapping


def _shots(m: dict) -> list[dict]:
    return sorted(m["shots"]["items"], key=lambda s: s["start_frame"])


def _strip(text: str, run_id: str) -> str:
    return text.replace(run_id, "<run>")


def compare_pose_and_tracking(ref: Path, cand: Path, mr: dict, mc: dict, out: dict) -> None:
    from ..perception.tracker import load_assign
    sr, sc = _shots(mr), _shots(mc)
    if [(s["start_frame"], s["end_frame"]) for s in sr] != [(s["start_frame"], s["end_frame"]) for s in sc]:
        out["differences"].append("shots: the shot boundaries differ")
        out["checks"]["shots"] = False
        return
    out["checks"]["shots"] = True
    det_r = {p["shot_id"]: p["det_path"] for p in mr["pose_body"]["per_shot"]}
    det_c = {p["shot_id"]: p["det_path"] for p in mc["pose_body"]["per_shot"]}
    trk_r = {p["shot_id"]: p for p in mr.get("tracking", {}).get("per_shot", [])}
    trk_c = {p["shot_id"]: p for p in mc.get("tracking", {}).get("per_shot", [])}
    tot = {"frames": 0, "count_mismatch_frames": 0, "frames_within_1px": 0, "frames_identical": 0,
           "box_max_px": 0.0, "kp_max_px": 0.0, "det_conf_max": 0.0, "kp_conf_max": 0.0}
    means, weights = {"box_mean_px": [], "kp_mean_px": []}, []
    assign_diff = 0
    for a, b in zip(sr, sc):
        da, db = load_det(Path(ref) / det_r[a["shot_id"]]), load_det(Path(cand) / det_c[b["shot_id"]])
        st, mapping = compare_det(da, db)
        for k in ("frames", "count_mismatch_frames", "frames_within_1px", "frames_identical"):
            tot[k] += st[k]
        for k in ("box_max_px", "kp_max_px", "det_conf_max", "kp_conf_max"):
            tot[k] = max(tot[k], st[k])
        for k in means:
            means[k].append(st[k])
        weights.append(max(st["matched"], 1))
        for c in st["count_mismatch"]:
            if len(out["differences"]) < MAX_LISTED:
                out["differences"].append(f"pose: frame {c['frame']} has {c['reference']} detections in the "
                                          f"reference, {c['candidate']} in the candidate")
        pa = Path(ref) / "detections" / f"{a['shot_id']}_assign.npz"
        pb = Path(cand) / "detections" / f"{b['shot_id']}_assign.npz"
        if pa.is_file() and pb.is_file():
            xa, xb = load_assign(pa), load_assign(pb)
            bad = sum(int(xa[i]) != int(xb[j]) for i, j in mapping.items())
            bad += len(xa) - len(mapping)        # unmatched rows count as differences
            assign_diff += bad
            if bad:
                out["differences"].append(f"tracking: {bad} detections of shot {a['shot_id'].rsplit('-', 1)[1]} "
                                          "have another track assignment")
        elif pa.is_file() != pb.is_file():
            out["differences"].append(f"tracking: assignment file of shot {a['shot_id']} missing in one run")
            assign_diff += 1
        ta, tb = trk_r.get(a["shot_id"], {}), trk_c.get(b["shot_id"], {})
        for k in ("tracks_kept", "short_tracks", "duplicates_suppressed"):
            if ta.get(k) != tb.get(k):
                out["differences"].append(f"tracking: shot {a['shot_id'].rsplit('-', 1)[1]} {k} "
                                          f"{ta.get(k)} vs {tb.get(k)}")
                assign_diff += 1
        subj_a = _strip(str(ta.get("subject")), mr["run_id"])
        subj_b = _strip(str(tb.get("subject")), mc["run_id"])
        if subj_a != subj_b:
            out["differences"].append(f"tracking: subject {subj_a} vs {subj_b}")
            assign_diff += 1
    w = np.asarray(weights, np.float64)
    for k, v in means.items():
        tot[k] = round(float(np.average(v, weights=w)), 5) if v else 0.0
    tot["share_within_1px"] = round(tot["frames_within_1px"] / tot["frames"], 6) if tot["frames"] else 1.0
    tot["share_identical"] = round(tot["frames_identical"] / tot["frames"], 6) if tot["frames"] else 1.0
    out["pose"] = tot
    out["checks"]["pose_counts"] = tot["count_mismatch_frames"] == 0
    out["checks"]["pose_1px"] = tot["frames_within_1px"] == tot["frames"]
    out["checks"]["pose_conf"] = tot["det_conf_max"] <= CONF_TOL and tot["kp_conf_max"] <= CONF_TOL
    if not out["checks"]["pose_1px"]:
        out["differences"].append(f"pose: {tot['frames'] - tot['frames_within_1px']} frames not within 1 px "
                                  f"(box max {tot['box_max_px']} px, keypoint max {tot['kp_max_px']} px)")
    if not out["checks"]["pose_conf"]:
        out["differences"].append(f"pose: confidence differs by up to {max(tot['det_conf_max'], tot['kp_conf_max'])}")
    out["tracking"] = {"differences": assign_diff}
    out["checks"]["tracking"] = assign_diff == 0


def compare_reference(mr: dict, mc: dict, ref: Path, cand: Path, out: dict) -> None:
    ir = {i["shot_id"].rsplit("-", 1)[1]: i for i in mr.get("reference", {}).get("items", [])}
    ic = {i["shot_id"].rsplit("-", 1)[1]: i for i in mc.get("reference", {}).get("items", [])}
    bad = []
    for k in sorted(set(ir) | set(ic)):
        a, b = ir.get(k), ic.get(k)
        if a is None or b is None or a["frame_idx"] != b["frame_idx"] or a["boxes"] != b["boxes"]:
            bad.append(f"reference: shot {k} {a and (a['frame_idx'], a['boxes'])} vs {b and (b['frame_idx'], b['boxes'])}")
            continue
        ja = json.loads((Path(ref) / "reference" / f"{mr['run_id']}-{k}.json").read_text())
        jb = json.loads((Path(cand) / "reference" / f"{mc['run_id']}-{k}.json").read_text())
        da = np.asarray([x["xyxy"] for x in ja["boxes"]], np.float64)
        db = np.asarray([x["xyxy"] for x in jb["boxes"]], np.float64)
        if da.shape != db.shape or (da.size and np.abs(da - db).max() > PX_TOL):
            bad.append(f"reference: shot {k} boxes differ by more than 1 px")
    out["differences"].extend(bad)
    out["checks"]["reference"] = not bad


def compare_steps(ref: Path, cand: Path, fps: float, out: dict) -> None:
    from .steps_io import read_steps_csv
    pa, pb = Path(ref) / "steps.csv", Path(cand) / "steps.csv"
    if not pa.is_file() or not pb.is_file():
        out["checks"]["steps"] = pa.is_file() == pb.is_file()
        if not out["checks"]["steps"]:
            out["differences"].append("steps: steps.csv missing in one run")
        return
    a, b = read_steps_csv(pa), read_steps_csv(pb)
    bad, worst = [], 0.0
    if [s.label for s in a] != [s.label for s in b]:
        bad.append(f"steps: labels {[s.label for s in a]} vs {[s.label for s in b]}")
    else:
        for x, y in zip(a, b):
            d = max(abs(x.start_s - y.start_s), abs(x.end_s - y.end_s))
            worst = max(worst, d)
            if d > 1.0 / fps + 1e-6:
                bad.append(f"steps: {x.label} {x.start_s:.3f} to {x.end_s:.3f} s vs {y.start_s:.3f} to {y.end_s:.3f} s")
    out["steps"] = {"count": len(a), "boundary_max_diff_s": round(worst, 4)}
    out["differences"].extend(bad)
    out["checks"]["steps"] = not bad


def _metric_rows(path: Path, run_id: str) -> dict[tuple[str, str], dict]:
    with open(path, newline="") as f:
        return {(_strip(r["segment_id"], run_id), r["metric"]): r for r in csv.DictReader(f)}


def _flags(text: str) -> set[str]:
    return {f.strip() for f in text.replace(";", ",").split(",") if f.strip()}


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(METRIC_ABS, METRIC_REL * max(abs(a), abs(b)))


def compare_metrics(ref: Path, cand: Path, mr: dict, mc: dict, out: dict) -> None:
    pa, pb = Path(ref) / "metrics.csv", Path(cand) / "metrics.csv"
    if not pa.is_file() or not pb.is_file():
        out["checks"]["metrics"] = pa.is_file() == pb.is_file()
        if not out["checks"]["metrics"]:
            out["differences"].append("metrics: metrics.csv missing in one run")
        return
    a, b = _metric_rows(pa, mr["run_id"]), _metric_rows(pb, mc["run_id"])
    bad = []
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key), b.get(key)
        name = f"{key[0].replace('<run>-', '')} {key[1]}"
        if x is None or y is None:
            bad.append(f"metrics: {name} only in the {'candidate' if x is None else 'reference'}")
            continue
        if (x["value"] == "") != (y["value"] == ""):
            bad.append(f"metrics: {name} reported in one run, NULL in the other")
        elif x["value"] != "":
            try:
                if not _close(float(x["value"]), float(y["value"])):
                    bad.append(f"metrics: {name} {x['value']} vs {y['value']}")
            except ValueError:
                if x["value"] != y["value"]:
                    bad.append(f"metrics: {name} {x['value']} vs {y['value']}")
        if x.get("coverage", "") != y.get("coverage", "") and not (
                x.get("coverage") and y.get("coverage") and _close(float(x["coverage"]), float(y["coverage"]))):
            bad.append(f"metrics: {name} coverage {x.get('coverage')} vs {y.get('coverage')}")
        if _flags(x["flags"]) != _flags(y["flags"]):
            bad.append(f"metrics: {name} flags {x['flags'] or '-'} vs {y['flags'] or '-'}")
        if x.get("unit") != y.get("unit"):
            bad.append(f"metrics: {name} unit {x.get('unit')} vs {y.get('unit')}")
    out["metrics"] = {"rows": len(a), "differences": len(bad)}
    out["differences"].extend(bad[:MAX_LISTED] + ([f"metrics: {len(bad) - MAX_LISTED} more"] if len(bad) > MAX_LISTED else []))
    out["checks"]["metrics"] = not bad


def compare_runs(reference: Path, candidate: Path) -> dict:
    """Equivalence record of candidate against reference (both run folders)."""
    mr, mc = _manifest(reference), _manifest(candidate)
    out: dict = {"reference_run": mr["run_id"], "candidate_run": mc["run_id"], "checks": {}, "differences": []}
    fps = float(mr["ingest"]["cfr"]["fps"])
    compare_pose_and_tracking(reference, candidate, mr, mc, out)
    if out["checks"].get("shots"):
        compare_reference(mr, mc, reference, candidate, out)
        compare_steps(reference, candidate, fps, out)
        compare_metrics(reference, candidate, mr, mc, out)
    out["verdict"] = "equivalent" if all(out["checks"].values()) else "different"
    return out
