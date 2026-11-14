"""Boundary correction of a run (Week 11): move boundaries, recompute the metrics, export
the approved version as a new revision. The original run files are never changed.

- current(root): the boundaries of the latest revision (or of the run's steps.csv).
- apply(steps, boundaries): the steps with each boundary between two neighbouring phases
  moved; the phase order and the coverage stay, every phase keeps at least min_s.
- preview(root, steps): the metrics of corrected steps, computed in memory from the stored
  keypoint files (CPU), nothing written.
- approve(root, steps, by, note): writes <root>/export/rev<N>/ with the corrected steps.csv,
  the recomputed metrics.csv, report.json and report.pdf (output/report.py), and appends an
  entry to <root>/export/revisions.json.
"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

from ..storage.files import utc_now
from . import report as R

MIN_S = 1.0


def latest_steps(root: Path) -> tuple[list[dict], int | None]:
    """(steps of the latest exported revision, its number), or the run's steps and None."""
    n = R.next_revision(root) - 1
    if n >= 0 and (root / "export" / f"rev{n}" / "steps.csv").is_file():
        return R.read_steps(root / "export" / f"rev{n}" / "steps.csv"), n
    return R.read_steps(root / "steps.csv"), None


def boundaries(steps: list[dict]) -> list[dict]:
    """The movable boundaries: between two neighbouring steps that touch."""
    out = []
    for a, b in zip(steps, steps[1:]):
        if abs(a["end_s"] - b["start_s"]) < 1e-6:
            out.append({"after_step": a["step"], "between": f"{a['label']} | {b['label']}", "time_s": a["end_s"]})
    return out


def apply(steps: list[dict], moved: dict[int, float]) -> list[dict]:
    """steps with the boundary after step k at moved[k] seconds. Raises ValueError when a
    step would get shorter than MIN_S or a boundary is not movable."""
    out = [dict(s) for s in steps]
    idx = {s["step"]: i for i, s in enumerate(out)}
    movable = {b["after_step"] for b in boundaries(out)}
    for k, t in sorted(moved.items()):
        if k not in movable:
            raise ValueError(f"no movable boundary after step {k}")
        i = idx[k]
        a, b = out[i], out[i + 1]
        if not (a["start_s"] + MIN_S <= t <= b["end_s"] - MIN_S):
            raise ValueError(f"the boundary after step {k} must stay between {a['start_s'] + MIN_S:.1f} s and "
                             f"{b['end_s'] - MIN_S:.1f} s")
        a["end_s"] = b["start_s"] = round(float(t), 3)
        a["source"] = b["source"] = "corrected"
    return out


class _Ctx:
    """A run context in memory for the motion analyzer (no file is written)."""

    def __init__(self, manifest: dict, root: Path):
        import logging
        self.manifest, self.run_id = manifest, manifest["run_id"]
        self.log = logging.getLogger("assemblyvision.corrections")
        self.notices: list = []

    def notice(self, err, stage: str) -> None:
        self.notices.append({"stage": stage, **err.as_record()})

    def progress(self, *a, **k) -> None:
        pass


def metrics_csv(root: Path, steps: list[dict]) -> tuple[bytes, list]:
    """(metrics.csv bytes of the steps, notices), computed from the stored keypoints."""
    from ..analysis.motion import MotionAnalyzer, metrics_csv_bytes
    from ..config import Config
    from ..types import Step
    manifest = json.loads((root / "manifest.json").read_text())
    from ..tools.run_pose import stored_config
    cfg = Config.load(overrides=stored_config(manifest))
    ctx = _Ctx(manifest, root)
    an = MotionAnalyzer(cfg, ctx)
    subjects = an.load_subjects(manifest, root)
    st = [Step(s["label"], s["start_s"], s["end_s"], s.get("source") or "corrected", s.get("score")) for s in steps]
    results, _ = an.analyse(manifest["run_id"], subjects, st, float(manifest["ingest"]["cfr"]["fps"]))
    return metrics_csv_bytes(results), ctx.notices


def preview(root: Path, steps: list[dict]) -> list[dict]:
    import csv
    data, _ = metrics_csv(root, steps)
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def approve(root: Path, steps: list[dict], by: str, note: str = "") -> Path:
    """Writes the corrected steps as the next revision (never overwriting one)."""
    base = R.next_revision(root) - 1
    if base < 0:                                    # the run as computed is always rev0
        R.export_run(root, 0)
        base = 0
    data, notices = metrics_csv(root, steps)
    with tempfile.TemporaryDirectory(dir=root) as d:
        s_csv, m_csv = Path(d) / "steps.csv", Path(d) / "metrics.csv"
        R.write_csv(s_csv, steps, ["step", "label", "start_s", "end_s", "source", "score"])
        m_csv.write_bytes(data)
        out = R.export_run(root, None, s_csv, m_csv, {"based_on": base, "by": by, "note": note or "boundaries corrected",
                                                      "created_utc": utc_now(), "metric_notices": notices})
    log = root / "export" / "revisions.json"
    entries = json.loads(log.read_text()) if log.is_file() else []
    entries.append({"revision": int(out.name[3:]), "based_on": base, "by": by, "note": note, "created_utc": utc_now(),
                    "boundaries": boundaries(steps)})
    log.write_text(json.dumps(entries, indent=1) + "\n")
    return out
