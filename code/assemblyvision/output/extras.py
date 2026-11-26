"""Measures and report content built on a finished run (Week 12). CPU, reads the run's stored files, writes nothing.

- cycle times per phase, per cycle when a phase repeats;
- the movement path in metres on the GVHMR floor;
- tool pickup and object placement (analysis/hand_events.py), EXPERIMENTAL;
- improvement suggestions and improvable motions: a rule list on the measured metrics, EXPERIMENTAL. Every suggestion
  carries its evidence (metric, value, threshold, time ranges). The thresholds are team defaults for flagging what a
  person should look at; they are not ergonomic limits from a standard and have not been validated.

None of this changes steps.csv or metrics.csv of a run.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np

EXPERIMENTAL = "experimental"
BEND_SHARE = 0.25                # share of the measured time over 45 degrees
DEEP_SHARE = 0.10                # share of the measured time over 60 degrees
IDLE_SHARE = 0.25
TWO_HAND_MIN = 0.25              # both hands active / at least one active
REPETITION_SHARE = 0.50
RELATIVE = 1.5                   # against the median of the run's phases (at least 3 phases)
MIN_PATH_M = 5.0                 # a walking suggestion needs at least this path
COVERAGE_MIN = 0.5               # a metric measured on less of the phase gives no suggestion
RANGE_MIN_S = 2.0                # shortest time range shown as evidence
MAX_RANGES = 5


def _num(row: dict | None) -> float | None:
    try:
        v = float(row["value"])
    except (TypeError, ValueError, KeyError):
        return None
    return v if np.isfinite(v) else None


def cycles(steps: list[dict]) -> list[dict]:
    """[{"phase", "cycle", "start_s", "end_s", "duration_s"}]: the cycle number counts the phase's appearances."""
    seen: dict[str, int] = {}
    out = []
    for s in sorted(steps, key=lambda x: x["start_s"]):
        if s["label"] == "other":
            continue
        seen[s["label"]] = seen.get(s["label"], 0) + 1
        out.append({"phase": s["label"], "cycle": seen[s["label"]], "start_s": round(s["start_s"], 2), "end_s": round(s["end_s"], 2),
                    "duration_s": round(s["end_s"] - s["start_s"], 2)})
    return out


def cycle_png(rows: list[dict]) -> np.ndarray | None:
    """Bars of the phase durations in seconds, one bar per cycle of a phase, as an RGB array."""
    if not rows:
        return None
    from ..ui.charts import GREY, GRID, INK, _plt
    plt = _plt()
    phases = list(dict.fromkeys(r["phase"] for r in rows))
    most = max(r["cycle"] for r in rows)
    width = 0.8 / most
    fig, ax = plt.subplots(figsize=(8.6, 3.2), dpi=120)
    shades = [INK, GREY, "#a3a3a3", "#c9c9c9"]
    for c in range(1, most + 1):
        got = [(phases.index(r["phase"]), r["duration_s"]) for r in rows if r["cycle"] == c]
        x = [i - 0.4 + width * (c - 0.5) for i, _ in got]
        y = [d for _, d in got]
        ax.bar(x, y, width * 0.92, color=shades[(c - 1) % len(shades)], label=f"cycle {c}")
        for a, b in zip(x, y):
            ax.text(a, b, f"{b:.0f}", ha="center", va="bottom", fontsize=9, color=GREY)
    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels([p if len(p) <= 16 else p[:15] + "." for p in phases], fontsize=9)
    ax.set_ylabel("duration (s)")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if most > 1:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def stretches(frames: np.ndarray, mask: np.ndarray, fps: float, a: float, b: float) -> list[list[float]]:
    """The longest time ranges inside [a, b) where mask holds, each at least RANGE_MIN_S."""
    t = np.asarray(frames, np.float64) / fps
    keep = (t >= a) & (t < b)
    t, m = t[keep], np.asarray(mask, bool)[keep]
    out, i = [], 0
    while i < len(t):
        if not m[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(t) and m[j + 1] and t[j + 1] - t[j] <= 2.0 / fps:
            j += 1
        if t[j] - t[i] >= RANGE_MIN_S:
            out.append([round(float(t[i]), 1), round(float(t[j]), 1)])
        i = j + 1
    return sorted(sorted(out, key=lambda r: r[0] - r[1])[:MAX_RANGES])


def _subjects(root: Path, manifest: dict):
    from ..analysis.motion import MotionAnalyzer
    from ..config import Config
    from ..tools.run_pose import stored_config
    from .corrections import _Ctx
    cfg = Config.load(overrides=stored_config(manifest))
    return MotionAnalyzer(cfg, _Ctx(manifest, root)).load_subjects(manifest, root), cfg


def load_objects(root: Path) -> dict | None:
    p = Path(root) / "objects.npz"
    if not p.is_file():
        return None
    z = np.load(p)
    return {k: z[k] for k in ("frames", "offset", "boxes", "cls", "conf")} | {"meta": json.loads(str(z["meta"]))}


def hand_event_list(subjects: list, det: dict, fps: float, active: float) -> list[dict]:
    from ..analysis.hand_events import wrist_events
    from ..analysis.motion import L_WRI, R_WRI, wrist_speeds
    out = []
    for s in subjects:
        cs = s.cs
        if not cs.torso:
            continue
        a, b = wrist_speeds(cs.xy, cs.valid, cs.torso, fps)
        frames = np.asarray(cs.frame_idx, np.int64)
        for hand, j, sp in (("left", L_WRI, a), ("right", R_WRI, b)):
            p = np.where(cs.valid[:, j, None], cs.xy[:, j].astype(np.float64), np.nan)
            out += wrist_events(frames, p, sp, float(cs.torso), fps, det, det["meta"]["names"], active, hand)
    return sorted(out, key=lambda e: e["t_s"])


def event_rows(events: list[dict], cyc: list[dict]) -> list[dict]:
    """Per phase and cycle: the count and the mean reach time of the pickups and of the placements."""
    out = []
    for c in cyc:
        for kind, name in (("pickup", "tool_pickup"), ("placement", "object_place")):
            e = [x for x in events if x["kind"] == kind and c["start_s"] <= x["t_s"] < c["end_s"]]
            out.append({**c, "metric": f"{name}_count", "value": len(e), "unit": "count"})
            out.append({**c, "metric": f"{name}_time_s", "value": round(float(np.mean([x["reach_s"] for x in e])), 3) if e else None,
                        "unit": "s"})
    return out


def rule_suggestions(cyc: list[dict], metrics: list[dict], floor: list[dict], ranges: dict) -> list[dict]:
    """The rule list. ranges: {(phase start s, "bend" or "idle"): time ranges}."""
    def row(c, name):
        return next((m for m in metrics if m.get("label") == c["phase"] and m.get("metric") == name
                     and abs(float(m["start_s"]) - c["start_s"]) < 0.51), None)

    def ok(r):
        try:
            return _num(r) is not None and float(r.get("coverage") or 0) >= COVERAGE_MIN
        except ValueError:
            return False

    def add(c, rule, metric, value, unit, threshold, text, tr=None):
        out.append({"rule": rule, "phase": c["phase"], "cycle": c["cycle"], "metric": metric, "value": round(float(value), 3), "unit": unit,
                    "threshold": threshold, "time_ranges": tr or [[c["start_s"], c["end_s"]]], "suggestion": text})
    out: list[dict] = []
    speed: dict[int, float] = {}
    travel: dict[int, float] = {}
    walk: dict[int, tuple[float, float]] = {}
    for i, c in enumerate(cyc):
        bands = [row(c, f"bend_b{k}_s") for k in range(4)]
        if all(ok(b) for b in bands):
            v = [_num(b) for b in bands]
            total = sum(v)
            if total > 0:
                if v[3] / total >= DEEP_SHARE:
                    add(c, "deep bending", "bend_b3_s / measured time", v[3] / total, "share", f">= {DEEP_SHARE}",
                        "Check whether the work can be raised or the part moved, to avoid bending the trunk over 60 degrees.",
                        ranges.get((c["start_s"], "bend")))
                elif (v[2] + v[3]) / total >= BEND_SHARE:
                    add(c, "long bending", "(bend_b2_s + bend_b3_s) / measured time", (v[2] + v[3]) / total, "share", f">= {BEND_SHARE}",
                        "Check the working height: the trunk is bent over 45 degrees for a long share of this phase.",
                        ranges.get((c["start_s"], "bend")))
        r = row(c, "idle_share")
        if ok(r) and _num(r) >= IDLE_SHARE:
            add(c, "long idle", "idle_share", _num(r), "ratio", f">= {IDLE_SHARE}",
                "Look at the waiting in these stretches: missing parts, tools or instructions may be the cause.",
                ranges.get((c["start_s"], "idle")))
        r = row(c, "two_hand_simultaneous")
        if ok(r) and _num(r) < TWO_HAND_MIN:
            add(c, "low two hand use", "two_hand_simultaneous", _num(r), "ratio", f"< {TWO_HAND_MIN}",
                "One hand does most of the work. Check whether a fixture could hold the part so both hands can work.")
        r = row(c, "repetition_share")
        if ok(r) and _num(r) >= REPETITION_SHARE:
            add(c, "repetitive motion", "repetition_share", _num(r), "ratio", f">= {REPETITION_SHARE}",
                "A short motion repeats for much of this phase. Consider a powered tool or rotating the task.")
        minutes = max(c["duration_s"] / 60.0, 1e-6)
        a, b = row(c, "hand_travel_l_torso"), row(c, "hand_travel_r_torso")
        if ok(a) and ok(b):
            travel[i] = (_num(a) + _num(b)) / minutes
        f = floor[i] if i < len(floor) else None
        if f and f.get("path_m") is not None and f.get("coverage", 0) >= COVERAGE_MIN:
            walk[i] = (f["path_m"] / minutes, f["path_m"])
            speed[i] = f["path_m"] / minutes
    if len(cyc) >= 3:
        med = statistics.median(c["duration_s"] for c in cyc)
        for c in cyc:
            if med > 0 and c["duration_s"] >= RELATIVE * med:
                add(c, "long phase", "duration_s", c["duration_s"], "s", f">= {RELATIVE} x the median phase ({med:.0f} s)",
                    "This phase takes much longer than the others. Check it first for waiting or rework.")
    if len(travel) >= 3:
        med = statistics.median(travel.values())
        for i, x in travel.items():
            if med > 0 and x >= RELATIVE * med:
                add(cyc[i], "long hand travel", "hand travel per minute", x, "torso lengths/min", f">= {RELATIVE} x the median phase ({med:.0f})",
                    "The hands travel far in this phase. Check whether parts and tools can be placed closer.")
    if len(speed) >= 3:
        med = statistics.median(speed.values())
        for i, (x, total) in walk.items():
            if med > 0 and x >= RELATIVE * med and total >= MIN_PATH_M:
                add(cyc[i], "long walking", "movement_path_floor per minute", x, "m/min", f">= {RELATIVE} x the median phase ({med:.1f}) and {MIN_PATH_M} m",
                    "The worker walks more in this phase. Check the layout of parts, tools and the work position.")
    return sorted(out, key=lambda s: (s["time_ranges"][0][0], s["rule"]))


def compute(root: Path, steps_csv: Path | None = None, metrics_csv: Path | None = None) -> dict:
    """Everything of this module for one run. A part that cannot be computed is reported in "notes"."""
    from .. import body3d as B
    from ..analysis.motion import subject_trunk
    from ..tools.check_outputs import floor_path
    from . import report as R
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    fps = float(manifest["ingest"]["cfr"]["fps"])
    steps = R.read_steps(steps_csv or root / "steps.csv")
    metrics = R.read_metrics(metrics_csv or root / "metrics.csv") if (metrics_csv or root / "metrics.csv").is_file() else []
    cyc = cycles(steps)
    notes = []
    work = sorted((s for s in steps if s["label"] != "other"), key=lambda s: s["start_s"])
    try:
        floor = floor_path(B.load(root), work, fps)
    except (OSError, ValueError, KeyError) as e:
        floor, _ = [], notes.append(f"movement path: {type(e).__name__}")
    if not floor or all(f["path_m"] is None for f in floor):
        notes.append("movement path in metres: this run has no body3d output")
    ranges: dict = {}
    events: list[dict] = []
    det = load_objects(root)
    try:
        subjects, cfg = _subjects(root, manifest)
        m = cfg["motion"]
        for c in cyc:
            for s in subjects:
                fr, ang = subject_trunk(s.cs)
                with np.errstate(invalid="ignore"):
                    ranges.setdefault((c["start_s"], "bend"), []).extend(stretches(fr, ang >= m["angle_bands"][1], fps, c["start_s"], c["end_s"]))
                    ranges.setdefault((c["start_s"], "idle"), []).extend(
                        stretches(np.asarray(s.cs.frame_idx), np.asarray(s.energy) < m["idle_energy"], fps, c["start_s"], c["end_s"]))
        if det is not None:
            events = hand_event_list(subjects, det, fps, m["hand_active"])
    except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
        notes.append(f"keypoint based evidence not available ({type(e).__name__})")
    if det is None:
        notes.append("tool pickup and object placement: no object detection for this run (tools/detect_objects.py)")
    sug = rule_suggestions(cyc, metrics, floor, {k: v for k, v in ranges.items() if v})
    return {"cycles": cyc, "floor_path": floor, "events": events, "event_rows": event_rows(events, cyc) if det is not None else [],
            "suggestions": sug, "notes": notes}


def metric_rows(extra: dict, run_id: str) -> list[dict]:
    """The added measures as rows of the metrics table (METRIC_COLS)."""
    out = []
    by = {(c["phase"], c["cycle"]): c for c in extra["cycles"]}

    def add(c, metric, value, unit, coverage, flags):
        out.append({"segment_id": f"{run_id}-X-{len(out) + 1:04d}", "step": "", "label": c["phase"], "start_s": c["start_s"], "end_s": c["end_s"],
                    "metric": metric, "value": "" if value is None else value, "unit": unit, "coverage": coverage, "flags": flags})
    for c, f in zip(extra["cycles"], extra["floor_path"]):
        add(c, "movement_path_floor", f["path_m"], "m", f["coverage"], "gvhmr_floor,not_validated")
    for r in extra["event_rows"]:
        add(by[(r["phase"], r["cycle"])], r["metric"], r["value"], r["unit"], "", f"{EXPERIMENTAL},coco_objects")
    for c in extra["cycles"]:
        n = sum(s["phase"] == c["phase"] and s["cycle"] == c["cycle"] for s in extra["suggestions"])
        add(c, "improvable_motions", n, "count", "", f"{EXPERIMENTAL},rule_list")
    return out


def suggestion_lines(extra: dict) -> list[str]:
    head = ["SUGGESTIONS, NOT FINDINGS. Rule based, from the measured metrics below; not validated against ground truth.",
            "The thresholds are team defaults that flag what a person should check, not ergonomic limits from a standard.",
            "A person must look at the video before acting on any of them.", ""]
    if not extra["suggestions"]:
        return head + ["No rule was triggered for this run."]
    out = head
    for i, s in enumerate(extra["suggestions"], 1):
        ranges = ", ".join(f"{a:.0f} to {b:.0f} s" for a, b in s["time_ranges"])
        out += [f"{i}. Suggestion ({s['rule']}), {s['phase']}, cycle {s['cycle']}:", f"   {s['suggestion']}",
                f"   Evidence: {s['metric']} = {s['value']} {s['unit']}; rule threshold {s['threshold']}", f"   Time ranges: {ranges}", ""]
    return out


def measure_lines(extra: dict) -> list[str]:
    out = ["Movement path on the GVHMR floor (mid hip, metres, 5 Hz, inside each shot part; not validated):"]
    out += [f"  {f['phase'][:30]:<30} {'-' if f['path_m'] is None else str(f['path_m']) + ' m':>10}"
            f"   covers {100 * f['coverage']:.0f} % of the phase" for f in extra["floor_path"]] or ["  no body3d output"]
    out += ["", "Tool pickup and object placement (EXPERIMENTAL, no ground truth): reach ending at a detected portable object.",
            "COCO has almost no workshop tools, so most real pickups are not seen and counts are a lower bound at best."]
    if extra["event_rows"]:
        for r in extra["event_rows"]:
            out.append(f"  {r['phase'][:26]:<26} cycle {r['cycle']}  {r['metric']:<20} {'-' if r['value'] is None else r['value']} {r['unit']}")
    else:
        out.append("  not measured for this run")
    return out + ["", *[f"Note: {n}" for n in extra["notes"]]]


def add_pages(pdf, rep: dict, text_page) -> None:
    """The cycle time graph, the added measures and the suggestions, in the report PDF."""
    from matplotlib import pyplot as plt
    extra = rep["extra"]
    img = cycle_png(extra["cycles"])
    if img is not None:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.text(0.07, 0.96, "Cycle times", fontsize=13, weight="bold", va="top")
        fig.text(0.07, 0.93, "Duration of each phase in seconds; one bar per cycle when a phase repeats.", fontsize=9, va="top")
        ax = fig.add_axes([0.05, 0.60, 0.9, 0.30])
        ax.imshow(img)
        ax.axis("off")
        lines = [f"{c['phase'][:30]:<30} cycle {c['cycle']}  {c['start_s']:8.1f} to {c['end_s']:8.1f} s   {c['duration_s']:8.1f} s" for c in extra["cycles"]]
        fig.text(0.07, 0.57, "\n".join(lines[:40]), fontsize=8, family="monospace", va="top")
        pdf.savefig(fig)
        plt.close(fig)
    text_page(pdf, "Added measures", measure_lines(extra))
    text_page(pdf, "Improvement suggestions (rule based, not validated)", suggestion_lines(extra))
