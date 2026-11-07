"""tools/dataset_pose: our keypoints on a dataset video, merged per subject; data/public only."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from assemblyvision.datasets import ha4m
from assemblyvision.kts import empty_series, write_kts
from assemblyvision.tools import dataset_pose
from helpers import fake_root, make_ctx

VID = "IDU001V001"


def track(ctx, shot: str, n: int, start: int, count: int, x: float, subject: bool, present=None):
    s = empty_series(ctx.run_id, shot, f"{shot}-t{n:02d}", 30.0, 320, 240, start, count)
    keep = np.ones(count, bool) if present is None else present
    s.present[:] = keep
    s.body_xy[keep] = x
    s.body_conf[keep] = 0.8
    s.bbox_xyxy[keep] = [x, 0, x + 10, 10]
    return write_kts(s, ctx.paths, {"is_subject": subject})


class DatasetPoseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.repo = Path(self._tmp.name)
        self.cfg, self.ctx = make_ctx(self.repo)
        self.root = self.repo / "data/public/datasets/ha4m"
        for d in ("rgb", "labels"):
            (self.root / d).mkdir(parents=True)
        (self.root / "rgb" / f"{VID}.mp4").write_bytes(b"not decoded in this test")
        (self.root / "rgb" / f"{VID}.frames.json").write_text(json.dumps(
            {"n_frames": 10, "first_frame_id": 0, "mp4_sha256": "ab"}))
        (self.root / "labels" / f"{VID}.csv").write_text(ha4m.labels_csv(np.array([[f, 1, 0] for f in range(10)])))

    def tearDown(self):
        self._tmp.cleanup()

    def test_merge_picks_the_subject_per_shot(self):
        a = track(self.ctx, "s001", 1, 0, 5, 1.0, subject=False)
        b = track(self.ctx, "s001", 2, 0, 5, 2.0, subject=True, present=np.array([1, 1, 0, 1, 1], bool))
        c = track(self.ctx, "s002", 1, 5, 5, 3.0, subject=False)              # no subject: longest track
        sub = dataset_pose.merge_subject([a, b, c], 10)
        x = sub["body_xy"][:5, 0, 0]
        self.assertEqual(x[[0, 1, 3, 4]].tolist(), [2.0] * 4)                 # the is_subject track
        self.assertTrue(np.isnan(sub["body_xy"][2]).all())                    # the subject was missing there
        self.assertEqual(sub["body_xy"][5:, 0, 0].tolist(), [3.0] * 5)
        self.assertEqual(sub["track_id"][0], "s001-t02")
        shifted = dataset_pose.merge_subject([c], 12, first_frame_id=2)       # row = dataset frame id
        self.assertEqual(np.flatnonzero(np.isfinite(shifted["body_xy"][:, 0, 0])).tolist(), [7, 8, 9, 10, 11])

    def fake_run(self, cmd, **kw):
        """Stands in for run_pose: the run folder already holds two tracks."""
        self.cmd = cmd
        track(self.ctx, "s001", 1, 0, 10, 4.0, subject=True)
        self.ctx.paths.manifest.write_text(json.dumps({"run_id": self.ctx.run_id, "config": {"pose": {"batch": 1}},
                                                       "pose_body": {"variant": "x", "precision": "FP32"}}))
        return SimpleNamespace(returncode=0, stdout=f"{self.ctx.run_id} done\n", stderr="")

    def test_derive_writes_the_folder_and_skips_done_videos(self):
        rc = dataset_pose.main(["--dataset", "ha4m", "--videos", VID], repo_root=self.repo, run=self.fake_run)
        self.assertEqual(rc, 0)
        self.assertIn("overlay.enabled=false", self.cmd)
        self.assertEqual(self.cmd[self.cmd.index("--source") + 1], "public")
        out = self.root / "derived/yolo26x" / VID
        src = json.loads((out / "source.json").read_text())
        self.assertEqual((src["run_id"], src["frames"], src["frames_with_subject"], src["tracks"]),
                         (self.ctx.run_id, 10, 10, 1))
        self.assertTrue((out / "kts" / "s001-t01.npz").is_file() and (out / "labels.csv").is_file())
        with np.load(out / "subject.npz") as z:
            self.assertEqual(z["body_xy"].shape, (10, 17, 2))
        self.cmd = None
        self.assertEqual(dataset_pose.main(["--dataset", "ha4m", "--videos", VID], repo_root=self.repo,
                                           run=self.fake_run), 0)
        self.assertIsNone(self.cmd)                                           # already derived: skipped

    def test_video_linked_into_qisens_is_refused(self):
        q = self.repo / "data/qisens/videos"
        q.mkdir(parents=True)
        (q / "x.mp4").write_bytes(b"x")
        link = self.root / "rgb" / f"{VID}.mp4"
        link.unlink()
        link.symlink_to(q / "x.mp4")
        self.cmd = None
        rc = dataset_pose.main(["--dataset", "ha4m", "--videos", VID], repo_root=self.repo, run=self.fake_run)
        self.assertEqual(rc, 2)
        self.assertIsNone(self.cmd)                                           # nothing was run

    def test_missing_rgb_is_an_error_not_a_run(self):
        (self.root / "rgb" / f"{VID}.frames.json").unlink()
        self.cmd = None
        self.assertEqual(dataset_pose.main(["--dataset", "ha4m", "--videos", VID], repo_root=self.repo,
                                           run=self.fake_run), 1)
        self.assertIsNone(self.cmd)


if __name__ == "__main__":
    unittest.main()
