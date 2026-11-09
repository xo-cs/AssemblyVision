"""TC-PSN: Engine A sensitivity to the pose model, built from two stored experiments (Week 10)."""
import json
import unittest
from pathlib import Path

from helpers import fake_root

from assemblyvision.config import Config
from assemblyvision.experiments import registry as reg
from assemblyvision.tools import pose_sensitivity as ps


class PoseSensitivityTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.root = Path(self._tmp.name)
        self.pc = Config.load()["paths"]
        (self.root / "data/public/videos").mkdir(parents=True)
        self.video = self.root / "data/public/videos/clip.mp4"
        self.video.write_bytes(b"x")
        (self.root / "data/public/videos/clip_gt.csv").write_text("start_s,end_s,label\n")

    def tearDown(self):
        self._tmp.cleanup()

    def make(self, exp_id, variant, fa, sets=None):
        exp = reg.create(exp_id, source="public", videos=[str(self.video)], paths_cfg=self.pc, repo_root=self.root,
                         variant=variant, gt="data/public/videos/clip_gt.csv", repeats=1, sets=sets)
        run = self.root / "data/public/runs" / f"run_{exp_id}"
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps({
            "segmentation": {"phases_found": 5, "phases_expected": 5},
            "tracking": {"tracks": [{"is_subject": True, "coverage": 0.91}]}}))
        ev = {"gt_file": "clip_gt.csv", "frame_accuracy": fa, "edit": 100.0, "f1_10": 100.0, "f1_25": 100.0,
              "f1_50": 80.0, "boundary_mae_s": 2.5}
        res = {"id": exp_id, "videos": {exp.meta["videos"][0]: {
            "repeats": [{"run_dir": run.relative_to(self.root).as_posix()}],
            "summary": {"eval": ev, "pose_fps": 30.0, "eval_same_in_repeats": True}}}}
        (exp.dir / "results.json").write_text(json.dumps(res))
        return exp

    def test_rows_and_change(self):
        self.make("t-x", "x", 90.0)
        self.make("t-l", "l", 80.5)
        res = ps.build(["t-x", "t-l"], self.pc, self.root)
        rows = res["clips"]["clip.mp4"]
        self.assertEqual(rows["l"]["change"]["frame_accuracy"], -9.5)
        self.assertEqual(rows["x"]["change"]["frame_accuracy"], 0.0)
        self.assertEqual(rows["l"]["phases"], "5/5")
        md = ps.table_md(res)
        self.assertIn("| YOLO26 l | 80.5 (-9.5) |", md)

    def test_refuses_other_differences(self):
        self.make("t-x", "x", 90.0)
        self.make("t-l", "l", 80.5, sets={"pose": {"batch": 4}})
        with self.assertRaises(reg.ExperimentError):
            ps.build(["t-x", "t-l"], self.pc, self.root)


if __name__ == "__main__":
    unittest.main()
