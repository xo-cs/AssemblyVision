"""tools/run_diff: manifest summaries only, refuses folders that are not runs."""
import unittest

from assemblyvision.tools import run_diff
from helpers import fake_root


class RunDiffTest(unittest.TestCase):
    def test_flat_and_summary(self):
        self.assertEqual(run_diff.flat({"a": {"b": 1, "c": {"d": 2}}, "e": 3}), {"a.b": 1, "a.c.d": 2, "e": 3})
        m = {"pose_body": {"variant": "l", "precision": "FP32", "batch_used": 1},
             "shots": {"count": 2, "items": [{"start_frame": 0, "shot_type": "fixed"},
                                             {"start_frame": 90, "shot_type": "handheld"}]},
             "tracking": {"tracks": [{"shot_id": "r-s001", "is_subject": True, "coverage": 0.9, "subject_source": "auto"},
                                     {"shot_id": "r-s001", "is_subject": False}]},
             "segmentation": {"phases_found": 5, "phases_expected": 5}}
        s = run_diff.summary(m)
        self.assertEqual((s["pose"]["variant"], s["shots"]["cuts"], s["tracks"]), ("l", [0, 90], 2))
        self.assertEqual(s["subject"], {"s001": (0.9, "auto")})
        self.assertEqual(s["segmentation"]["phases_found"], 5)

    def test_refuses_a_folder_that_is_not_a_run(self):
        tmp = fake_root()
        self.addCleanup(tmp.cleanup)
        self.assertEqual(run_diff.main([tmp.name, tmp.name]), 2)


if __name__ == "__main__":
    unittest.main()
