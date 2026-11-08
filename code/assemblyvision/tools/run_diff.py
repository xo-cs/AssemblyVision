"""Compare two stored runs by their manifests only (Week 10, the QiSens steps drop).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.run_diff RUN_A RUN_B [--gt GT_CSV]

Reads only <run>/manifest.json (and <run>/eval.json for the GT checksum); never
the video, frames, keypoints or steps. Prints summary lines only, safe to share:
  - every configuration value that differs between the two runs (stored configs)
  - pose model, precision, GPU, batch
  - shots: count, whether the cut frames are identical, shot types
  - tracking: per shot, the subject track's coverage and how it was chosen
  - segmentation: phases found, solver, evidence counts
  - GT: whether each run's eval used a GT file with the same sha256 (and, with --gt,
    the same as that file now)
Both runs must lie in the same data area (public or qisens); nothing is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from ..config import Config
from ..errors import AvError
from ..storage.files import run_source


def flat(d: dict, pre: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def summary(m: dict) -> dict:
    pb, sh, tr, sg = m.get("pose_body", {}), m.get("shots", {}), m.get("tracking", {}), m.get("segmentation", {})
    subj = {}
    for t in tr.get("tracks", []):
        if t.get("is_subject"):
            subj[t["shot_id"].rsplit("-", 1)[-1]] = (t.get("coverage"), t.get("subject_source"))
    return {"pose": {k: pb.get(k) for k in ("variant", "precision", "tf32", "batch_used", "imgsz", "conf", "gpu_name")},
            "shots": {"count": sh.get("count"), "cuts": [i.get("start_frame") for i in sh.get("items", [])],
                      "types": [i.get("shot_type") for i in sh.get("items", [])]},
            "subject": subj, "tracks": len(tr.get("tracks", [])),
            "segmentation": {k: sg.get(k) for k in ("phases_found", "phases_expected", "solver_used", "evidence_counts")},
            "active": sg.get("active")}


def gt_sha(run: Path) -> str | None:
    e = run / "eval.json"
    return json.loads(e.read_text()).get("gt_sha256") if e.is_file() else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.run_diff")
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--gt", type=Path, default=None)
    args = p.parse_args(argv)
    try:
        pc = Config.load()["paths"]
        (sa, ra), (sb, rb) = run_source(args.run_a, pc), run_source(args.run_b, pc)
    except AvError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    if sa != sb:
        print("refused: both runs must lie in the same data area", file=sys.stderr)
        return 2
    ma, mb = (json.loads((r / "manifest.json").read_text()) for r in (ra, rb))
    print(f"A {ma['run_id']} ({ma.get('created_utc', '')[:16]})  B {mb['run_id']} ({mb.get('created_utc', '')[:16]})\n")
    fa, fb = flat(ma.get("config", {})), flat(mb.get("config", {}))
    from ..experiments.noop import is_no_op
    both = [k for k in sorted(set(fa) & set(fb)) if fa[k] != fb[k]]
    print("config values that differ (A -> B):")
    for k in both:
        print(f"  {k}: {fa[k]} -> {fb[k]}")
    if not both:
        print("  none")
    for name, only, cfg in (("B", sorted(set(fb) - set(fa)), mb.get("config", {})),
                            ("A", sorted(set(fa) - set(fb)), ma.get("config", {}))):
        if not only:
            continue
        noop = [k for k in only if is_no_op(k, fa.get(k, fb.get(k)), cfg)]
        real = [k for k in only if k not in noop]
        print(f"settings only in {name}: {len(only)}, of them no-op defaults {len(noop)}"
              + ("" if not real else "; NOT no-ops (may act):"))
        for k in real:
            print(f"  {k} = {fa.get(k, fb.get(k))}")
    xa, xb = summary(ma), summary(mb)
    print("\npose:", "A", xa["pose"], "\n      B", xb["pose"])
    print(f"\nshots: A {xa['shots']['count']}, B {xb['shots']['count']}; cut frames "
          f"{'identical' if xa['shots']['cuts'] == xb['shots']['cuts'] else 'DIFFERENT'}; types "
          f"{'identical' if xa['shots']['types'] == xb['shots']['types'] else 'DIFFERENT'}")
    print(f"tracks stored: A {xa['tracks']}, B {xb['tracks']}")
    print("subject per shot (coverage, chosen by):")
    for s in sorted(set(xa["subject"]) | set(xb["subject"])):
        print(f"  {s}: A {xa['subject'].get(s, 'none')}  B {xb['subject'].get(s, 'none')}")
    print("\nsegmentation: A", xa["segmentation"], "\n              B", xb["segmentation"])
    print("active span:  A", xa["active"], "\n              B", xb["active"])
    ga, gb = gt_sha(ra), gt_sha(rb)
    now = hashlib.sha256(args.gt.read_bytes()).hexdigest() if args.gt and args.gt.is_file() else None
    print("\nGT sha256:", "A and B same" if ga and ga == gb else f"A {'set' if ga else 'no eval'}, B {'set' if gb else 'no eval'}"
          + ("" if not (ga and gb) or ga == gb else ", DIFFERENT"),
          *([f"; A vs --gt now: {'same' if ga == now else 'DIFFERENT'}"] if now and ga else []),
          *([f"; B vs --gt now: {'same' if gb == now else 'DIFFERENT'}"] if now and gb else []))
    return 0


if __name__ == "__main__":
    sys.exit(main())
