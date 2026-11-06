"""Equivalence of two runs (evaluation/equivalence.py) on synthetic run folders."""
import json
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.evaluation.equivalence import compare_det, compare_runs
from helpers import fake_root

STEPS = "step,label,start_s,end_s,source,score\n0,other,0.000,2.000,engine_a,\n1,Removal,2.000,10.000,engine_a,1.000\n"
METRICS = [("duration_s", "8", "s", "1.0000", ""), ("back_bending", "12.5", "deg", "0.9000", "body_relative,handheld"),
           ("idle", "", "s", "0.2000", "low_n")]


def det_rows(seed: int = 0, frames: int = 20, people: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    fi = np.repeat(np.arange(frames, dtype=np.int32), people)
    n = fi.size
    x = rng.uniform(0, 600, (n, 1)).astype(np.float32)
    box = np.concatenate([x, x + 10, x + 80, x + 200], axis=1).astype(np.float32)
    return {"det_version": np.int32(1), "frame_start": np.int32(0), "frame_count": np.int32(frames),
            "pose_variant": np.str_("x"), "frame_idx": fi, "bbox_xyxy": box,
            "det_conf": rng.uniform(0.3, 0.9, n).astype(np.float32),
            "body_xy": rng.uniform(0, 400, (n, 17, 2)).astype(np.float32),
            "body_conf": rng.uniform(0, 1, (n, 17)).astype(np.float32)}


def make_run(root: Path, run_id: str, det: dict, assign: np.ndarray | None = None, steps: str = STEPS,
             metrics=METRICS, ref_frame: int = 0) -> Path:
    d = root / run_id
    for sub in ("detections", "reference"):
        (d / sub).mkdir(parents=True)
    sid = f"{run_id}-s001"
    np.savez(d / "detections" / f"{sid}.npz", **det)
    a = assign if assign is not None else np.tile(np.array([1, 2], np.int64), det["frame_idx"].size // 2)
    np.savez(d / "detections" / f"{sid}_assign.npz", assign=a)
    (d / "reference" / f"{sid}.json").write_text(json.dumps(
        {"shot_id": sid, "frame_idx": ref_frame, "boxes": [{"n": 1, "xyxy": [1.0, 2.0, 50.0, 200.0], "conf": 0.8}]}))
    m = {"run_id": run_id, "ingest": {"cfr": {"fps": 25.0}},
         "shots": {"items": [{"shot_id": sid, "start_frame": 0, "end_frame": 20}]},
         "pose_body": {"per_shot": [{"shot_id": sid, "det_path": f"detections/{sid}.npz"}]},
         "reference": {"items": [{"shot_id": sid, "frame_idx": ref_frame, "boxes": 1}]},
         "tracking": {"per_shot": [{"shot_id": sid, "subject": f"{sid}-t01", "tracks_kept": 2, "short_tracks": 0,
                                    "duplicates_suppressed": 0}]}}
    (d / "manifest.json").write_text(json.dumps(m))
    (d / "steps.csv").write_text(steps)
    lines = ["segment_id,step,label,start_s,end_s,metric,value,unit,coverage,flags"]
    lines += [f'{run_id}-A-0001,1,Removal,2.000,10.000,{name},{v},{u},{c},"{f}"' for name, v, u, c, f in metrics]
    (d / "metrics.csv").write_text("\n".join(lines) + "\n")
    return d


class EquivalenceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.root = Path(self._tmp.name)
        self.ref = make_run(self.root, "run_20261106_000000_aaaa", det_rows())

    def tearDown(self):
        self._tmp.cleanup()

    def cand(self, det=None, **kw) -> dict:
        self.n = getattr(self, "n", 0) + 1
        d = make_run(self.root, f"run_20261106_{self.n:06d}_bbbb", det if det is not None else det_rows(), **kw)
        return compare_runs(self.ref, d)

    def test_identical_runs_are_equivalent(self):
        out = self.cand()
        self.assertEqual(out["verdict"], "equivalent", out["differences"])
        self.assertEqual(out["pose"]["share_identical"], 1.0)
        self.assertEqual(out["differences"], [])

    def test_half_pixel_shift_is_within_1px(self):
        det = det_rows()
        det["body_xy"] = det["body_xy"] + 0.5
        out = self.cand(det)
        self.assertEqual(out["verdict"], "equivalent")
        self.assertEqual(out["pose"]["share_within_1px"], 1.0)
        self.assertEqual(out["pose"]["share_identical"], 0.0)
        self.assertAlmostEqual(out["pose"]["kp_max_px"], 0.5, places=4)

    def test_swapped_detection_order_still_matches(self):
        det = det_rows()
        perm = np.arange(det["frame_idx"].size).reshape(-1, 2)[:, ::-1].reshape(-1)    # swap the two people
        det2 = {k: (v[perm] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == perm.size else v)
                for k, v in det.items()}
        assign = np.tile(np.array([2, 1], np.int64), perm.size // 2)      # assignments move with the rows
        out = self.cand(det2, assign=assign)
        self.assertEqual(out["verdict"], "equivalent", out["differences"])

    def test_two_pixel_shift_is_flagged(self):
        det = det_rows()
        det["bbox_xyxy"] = det["bbox_xyxy"] + 2.0
        out = self.cand(det)
        self.assertEqual(out["verdict"], "different")
        self.assertFalse(out["checks"]["pose_1px"])

    def test_count_mismatch_is_flagged(self):
        det = det_rows()
        keep = np.ones(det["frame_idx"].size, bool)
        keep[3] = False
        det2 = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == keep.size else v)
                for k, v in det.items()}
        assign = np.tile(np.array([1, 2], np.int64), 20)[keep]
        out = self.cand(det2, assign=assign)
        self.assertFalse(out["checks"]["pose_counts"])
        self.assertTrue(any("detections in the reference" in d for d in out["differences"]))

    def test_confidence_difference_is_flagged(self):
        det = det_rows()
        det["det_conf"] = det["det_conf"] + 0.02
        self.assertFalse(self.cand(det)["checks"]["pose_conf"])

    def test_tracking_difference_is_flagged(self):
        assign = np.tile(np.array([1, 2], np.int64), 20)
        assign[5] = 3
        out = self.cand(assign=assign)
        self.assertFalse(out["checks"]["tracking"])

    def test_step_boundary_moved_is_flagged(self):
        moved = STEPS.replace("2.000,10.000", "2.200,10.000").replace("0.000,2.000", "0.000,2.200")
        out = self.cand(steps=moved)
        self.assertFalse(out["checks"]["steps"])
        within = STEPS.replace("2.000,10.000", "2.030,10.000").replace("0.000,2.000", "0.000,2.030")
        self.assertTrue(self.cand(steps=within)["checks"]["steps"])     # 0.03 s < one frame at 25 fps

    def test_metric_value_null_and_flags(self):
        def with_row(i, row):
            m = list(METRICS)
            m[i] = row
            return m
        self.assertTrue(self.cand(metrics=with_row(1, ("back_bending", "12.55", "deg", "0.9000",
                                                       "body_relative,handheld")))["checks"]["metrics"])   # 0.4 %
        self.assertFalse(self.cand(metrics=with_row(1, ("back_bending", "12.8", "deg", "0.9000",
                                                        "body_relative,handheld")))["checks"]["metrics"])  # 2.4 %
        self.assertFalse(self.cand(metrics=with_row(2, ("idle", "3", "s", "0.2000", "low_n")))["checks"]["metrics"])
        self.assertFalse(self.cand(metrics=with_row(1, ("back_bending", "12.5", "deg", "0.9000",
                                                        "body_relative")))["checks"]["metrics"])
        self.assertTrue(self.cand(metrics=with_row(1, ("back_bending", "12.5", "deg", "0.9000",
                                                       "handheld,body_relative")))["checks"]["metrics"])

    def test_reference_frame_moved_is_flagged(self):
        self.assertFalse(self.cand(ref_frame=3)["checks"]["reference"])

    def test_compare_det_reports_stats(self):
        st, mapping = compare_det(det_rows(), det_rows())
        self.assertEqual((st["frames"], st["frames_within_1px"], st["matched"]), (20, 20, 40))
        self.assertEqual(len(mapping), 40)


if __name__ == "__main__":
    unittest.main()
