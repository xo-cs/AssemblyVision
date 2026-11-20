"""TC-MIL: a labelled copy of a run with another step engine (Week 11; tools/milestone_run.py). CPU."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from assemblyvision.tools import milestone_run as M


class MilestoneRunTest(unittest.TestCase):
    def test_copy_links_inputs_and_leaves_the_original(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "runs" / "run_20260101_000000_aaaa"
            for sub in ("video", "keypoints", "detections", "body3d", "overlay"):
                (src / sub).mkdir(parents=True)
            (src / "keypoints" / "k.npz").write_bytes(b"k" * 100)
            (src / "video" / "cfr.mp4").symlink_to("../../../videos/clip.mp4")
            (src / "vlm.npz").write_bytes(b"v")
            (src / "steps.csv").write_text("old steps")
            (src / "overlay" / "tracks_overlay.mp4").write_bytes(b"o")
            (src / "manifest.json").write_text(json.dumps({"run_id": src.name, "worker": {"pid": 1}, "eval": {"x": 1}}))
            dst = M.make_copy(src, "Engine B v1", ['seg_a.engine="B v1"'])
            self.assertEqual(dst.parent, src.parent)
            self.assertEqual(os.stat(dst / "keypoints" / "k.npz").st_ino, os.stat(src / "keypoints" / "k.npz").st_ino)
            self.assertTrue((dst / "video" / "cfr.mp4").is_symlink())
            self.assertTrue((dst / "vlm.npz").is_file())
            self.assertFalse((dst / "steps.csv").exists())                   # made again by the rerun
            self.assertFalse((dst / "overlay").exists())
            m = json.loads((dst / "manifest.json").read_text())
            self.assertEqual((m["run_id"], m["milestone"]["label"], m["milestone"]["copied_from"]),
                             (dst.name, "Engine B v1", src.name))
            self.assertNotIn("eval", m)
            self.assertEqual((src / "steps.csv").read_text(), "old steps")


if __name__ == "__main__":
    unittest.main()
