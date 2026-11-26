"""End to end check of the demo outputs of one run, and its movement path in metres (Week 12). CPU.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.check_outputs --run RUN_DIR

Builds each output through the code the app and the report use: process timeline, skeleton overlay video,
movement heatmaps, cycle time graph, the added measures and the PDF report. The run's files are not changed; the test PDF lies in the
run's tmp folder and is removed. Prints only a status per output and the path length per phase, no frame and
no image. Also writes <data area>/validation/outputs_<run>.md.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from ..config import Config
from ..storage.files import run_source, utc_now
from . import freeze

HIPS = (11, 12)                                  # COCO 17: the left and right hip
PATH_HZ = 5.0


def path_length(xz: np.ndarray, every: int) -> float:
    """Length (m) of a floor path sampled every `every` frames, the last frame included."""
    idx = np.unique(np.r_[np.arange(0, len(xz), every), len(xz) - 1])
    return float(np.linalg.norm(np.diff(xz[idx], axis=0), axis=1).sum()) if len(idx) > 1 else 0.0


def floor_path(pieces: list[dict], steps: list[dict], fps: float) -> list[dict]:
    """Mid hip path on the GVHMR floor (x, z in metres, y up) per phase, at PATH_HZ. A shot part has its own world
    frame, so a path is summed inside each part and never across two parts."""
    every = max(1, int(round(fps / PATH_HZ)))
    out = []
    for s in steps:
        if s["label"] == "other":
            continue
        a, b = int(round(s["start_s"] * fps)), int(round(s["end_s"] * fps))
        total, seen, parts = 0.0, 0, 0
        for p in pieces:
            f = np.asarray(p["frames"])
            m = (f >= a) & (f < b)
            if m.sum() < 2:
                continue
            hip = np.asarray(p["joints"])[m][:, HIPS].mean(1)
            total += path_length(hip[:, [0, 2]], every)
            seen += int(m.sum())
            parts += 1
        out.append({"phase": s["label"], "path_m": round(total, 2) if seen else None,
                    "coverage": round(seen / max(b - a, 1), 3), "parts": parts})
    return out


def check(root: Path, source: str, app) -> dict:
    from .. import body3d as B
    from ..output import body3d_view as BV
    from ..output import report as R
    rid = root.name
    m = json.loads((root / "manifest.json").read_text())
    cfr = m["ingest"]["cfr"]
    fps, dur = float(cfr["fps"]), float(cfr["duration_s"])
    r = app.results(rid)
    rows = []
    add = lambda name, status, detail: rows.append({"output": name, "status": status, "detail": detail})
    steps = R.read_steps(root / "steps.csv")
    phases = [s for s in steps if s["label"] != "other"]
    n_vocab = len(m.get("segmentation", {}).get("vocabulary") or [])
    add("process timeline", "ok" if r["timeline"] is not None and phases else "missing",
        f"{len(phases)} of {n_vocab} phases, engine {sorted({s.get('source') for s in phases})}")
    ov = r["overlay"]
    if ov:
        p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_name:format=duration",
                            "-of", "csv=p=0", str(ov)], capture_output=True, text=True)
        bits = p.stdout.split()
        length = float(bits[-1]) if p.returncode == 0 and bits else 0.0
        good = abs(length - dur) <= 1.0 and bits[0].strip(",") == "h264"
        add("skeleton overlay video", "ok" if good else "problem", f"{bits[0].strip(',') if bits else '?'}, {length:.1f} s of {dur:.1f} s")
    else:
        add("skeleton overlay video", "missing", "no overlay file")
    pieces = B.load(root)
    floors = sum(BV.floor_png(p, fps) is not None for p in pieces)
    reach = app.reach_html(rid)
    has_reach = "<img" in (reach or "")
    add("movement heatmap (floor map, GVHMR, metres)", "ok" if floors else "missing", f"{floors} of {len(pieces)} shot parts")
    add("movement heatmap (hand reach, 2D)", "ok" if has_reach else "missing", "shown" if has_reach else "not available")
    from ..output import extras as X
    extra = X.compute(root)
    graph = X.cycle_png(extra["cycles"])
    add("cycle time graph", "ok" if graph is not None else "missing", f"{len(extra['cycles'])} phase cycles, in Results and the PDF")
    placed = sum(f["path_m"] is not None for f in extra["floor_path"])
    add("movement path in metres (GVHMR floor)", "ok" if placed else "missing", f"{placed} of {len(extra['floor_path'])} phases; not validated")
    measured_events = bool(extra["event_rows"])
    add("tool pickup and object placement (experimental)", "ok" if measured_events else "missing",
        f"{sum(e['kind'] == 'pickup' for e in extra['events'])} pickups, {sum(e['kind'] == 'placement' for e in extra['events'])} placements"
        if measured_events else "no object detection for this run (tools/detect_objects.py)")
    (root / "tmp").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pdf", dir=root / "tmp") as f:
        rep = R.build(root)
        R.write_pdf(Path(f.name), rep)
        data = Path(f.name).read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", data))
    measured = sum(x.get("value") not in ("", None) for x in rep["metrics"])
    add("report PDF: measured baseline", "ok" if pages and measured else "problem",
        f"{pages} pages, {len(rep['steps'])} steps, {measured} of {len(rep['metrics'])} metrics with a value")
    n = len(rep["extra"]["suggestions"])
    add("report PDF: labelled improvement suggestions (experimental)", "ok", f"{n} suggestions with evidence" if n else "section present, no rule triggered")
    return {"created_utc": utc_now(), "freeze": freeze.record(), "run": rid, "source": source, "outputs": rows,
            "floor_path": extra["floor_path"], "body3d_parts": len(pieces)}


def markdown(res: dict) -> str:
    f = res["freeze"]
    L = [f"# Demo outputs of {res['run']}", "", f"Freeze: {f['tag']} ({f['tag_commit'][:7]}), ran on {f['ran_on'][:7]}, method unchanged: "
         f"{f['method_unchanged']}.", "", "| output | status | detail |", "| --- | --- | --- |"]
    L += [f"| {x['output']} | {x['status']} | {x['detail']} |" for x in res["outputs"]]
    L += ["", "Movement path per phase (mid hip on the GVHMR floor, metres, 5 Hz, summed inside each shot part; not validated yet):", "",
          "| phase | path m | share of the phase covered | shot parts |", "| --- | --- | --- | --- |"]
    L += [f"| {x['phase']} | {'-' if x['path_m'] is None else x['path_m']} | {100 * x['coverage']:.0f} % | {x['parts']} |" for x in res["floor_path"]]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.check_outputs")
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args(argv)
    from ..ui.app import GradioApp
    cfg = Config.load()
    source, root = run_source(a.run, cfg["paths"])
    res = check(root, source, GradioApp(cfg, [source]))
    out = root.parent.parent / "validation"
    out.mkdir(exist_ok=True)
    md = markdown(res)
    (out / f"outputs_{root.name}.md").write_text(md)
    (out / f"outputs_{root.name}.json").write_text(json.dumps(res, indent=1))
    print(md)
    return 0 if all(x["status"] == "ok" for x in res["outputs"]) else 3


if __name__ == "__main__":
    sys.exit(main())
