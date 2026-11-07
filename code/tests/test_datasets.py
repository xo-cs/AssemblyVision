"""Dataset loaders (HA4M, InHARD stub), boundary targets and the joint mappings."""
import json
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.config import Config
from assemblyvision.datasets import ha4m, inhard, mapping
from assemblyvision.datasets.base import boundary_targets, change_frames
from assemblyvision.errors import AvError
from helpers import fake_root

HEADER = "BodyID\tJointID\tConfidance\tX\tY\tZ\tQw\tQx\tQy\tQz\tX2DColor\tY2DColor\tX2DDepth\tY2DDepth"


def skeleton_text(bodies: dict[int, float]) -> str:
    """One frame file: per body all 32 joints, joint j at colour pixel (100 + j + shift, 200 + j)."""
    lines = [HEADER]
    for b, shift in bodies.items():
        for j in range(32):
            conf = 0.666667 if j != 3 else 0.0
            lines.append(f"{b}\t{j}\t{conf}\t1\t2\t3\t1\t0\t0\t0\t{100 + j + shift}\t{200 + j}\t10\t20")
    return "\n".join(lines) + "\n"


def write_video(root: Path, vid: str, labels: list[tuple[int, int, int]], camera: str, n: int | None = None) -> None:
    n = n or len(labels)
    rows = np.array(labels, np.int32)
    for sub in ("labels", "skeleton", "meta"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "labels" / f"{vid}.csv").write_text(ha4m.labels_csv(rows))
    sk = ha4m.parse_skeletons({f: skeleton_text({1: float(f)}) for f in range(n)}, n)
    np.savez(root / "skeleton" / f"{vid}.npz", **sk)
    (root / "meta" / f"{vid}.json").write_text(json.dumps({"video": vid, "camera": camera}))


class ParseTest(unittest.TestCase):
    def test_labels(self):
        rows = ha4m.parse_labels("2 1 0\n0 0 0\n1 0 0\n")
        self.assertEqual(rows[:, 0].tolist(), [0, 1, 2])
        per = ha4m.labels_per_frame(rows, 4)
        self.assertEqual(per.tolist(), [[0, 0], [0, 0], [1, 0], [-1, 0]])
        with self.assertRaises(AvError):
            ha4m.parse_labels("0 1\n")
        with self.assertRaises(AvError):
            ha4m.parse_labels("0 1 0\n0 2 0\n")

    def test_skeleton_frame_and_operator_choice(self):
        bodies = ha4m.parse_skeleton_frame(skeleton_text({1: 0.0, 2: 50.0}))
        self.assertEqual(set(bodies), {1, 2})
        self.assertEqual(bodies[1].shape, (32, 12))
        self.assertAlmostEqual(float(bodies[2][5, 8]), 155.0)          # colour x of joint 5
        weak = skeleton_text({1: 0.0}) + "\n".join(
            f"7\t{j}\t0.333333\t1\t2\t3\t1\t0\t0\t0\t9\t9\t1\t1" for j in range(32))
        sk = ha4m.parse_skeletons({0: weak, 2: skeleton_text({4: 0.0})}, 3)
        self.assertEqual(sk["body_id"].tolist(), [1, -1, 4])            # the most confident body
        self.assertEqual(sk["n_bodies"].tolist(), [2, 0, 1])
        self.assertTrue(np.isnan(sk["joints_2d_color"][1]).all())
        only_invalid = skeleton_text({4294967295: 0.0})                    # K4ABT_INVALID_BODY_ID
        self.assertEqual(ha4m.parse_skeleton_frame(only_invalid), {})
        sk = ha4m.parse_skeletons({0: only_invalid + "\n".join(skeleton_text({2: 0.0}).splitlines()[1:])}, 1)
        self.assertEqual(sk["body_id"].tolist(), [2])

    def test_kinect_to_coco(self):
        sk = ha4m.parse_skeletons({0: skeleton_text({1: 0.0})}, 1)
        xy, conf = ha4m.kinect_to_coco(sk["joints_2d_color"], sk["confidence"])
        self.assertEqual(xy.shape, (1, 17, 2))
        self.assertEqual(xy[0, 0].tolist(), [127.0, 227.0])             # nose = Kinect 27
        self.assertEqual(xy[0, 9].tolist(), [107.0, 207.0])             # left wrist = Kinect 7

    def test_video_ids_and_frames(self):
        self.assertEqual(ha4m.video_subject("IDU023V004"), "S023")
        self.assertEqual(ha4m.frame_id("IDU001V001/Color/000124702712/FrameID000017_DeviceTimeStamp45us.png"), 17)
        self.assertEqual(ha4m.member_camera("IDU001V001/Color/000124702712/x.png"), "000124702712")
        with self.assertRaises(ValueError):
            ha4m.video_subject("V001")


class BoundaryTest(unittest.TestCase):
    def test_changes_of_action_and_repetition(self):
        lab = np.array([[0, 0]] * 3 + [[2, 0]] * 3 + [[2, 1]] * 3 + [[0, 0]] * 2 + [[3, 0]] * 2)
        self.assertEqual(change_frames(lab).tolist(), [3, 6, 9, 11])
        self.assertEqual(change_frames(lab, ignore_dont_care=True).tolist(), [6, 11])

    def test_tolerance(self):
        lab = np.array([[1, 0]] * 5 + [[2, 0]] * 5)
        self.assertEqual(np.flatnonzero(boundary_targets(lab, 0)).tolist(), [4, 5])
        self.assertEqual(np.flatnonzero(boundary_targets(lab, 2)).tolist(), [2, 3, 4, 5, 6, 7])
        self.assertFalse(boundary_targets(np.array([[0, 0]] * 4), 1, ignore_dont_care=True).any())

    def test_config_switch(self):
        cfg = Config.load(overrides={"boundaries": {"ignore_dont_care": True}})
        self.assertTrue(cfg["boundaries"]["ignore_dont_care"])
        with self.assertRaises(AvError):
            Config.load(overrides={"boundaries": {"tol_s": -1}})


class HA4MLoaderTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.repo = Path(self._tmp.name)
        self.cfg = Config.load()
        self.root = self.repo / "data/public/datasets/ha4m"
        labels = [(f, 0 if f < 3 else 5, 0) for f in range(8)]
        write_video(self.root, "IDU001V001", labels, "000124702712")
        write_video(self.root, "IDU023V001", labels, "000099302712")

    def tearDown(self):
        self._tmp.cleanup()

    def test_videos_and_dataset_skeleton(self):
        ld = ha4m.HA4MLoader(self.cfg, repo_root=self.repo)
        vids = ld.videos()
        self.assertEqual([(v.video_id, v.subject, v.location, v.n_frames) for v in vids],
                         [("IDU001V001", "S001", "lab_a", 8), ("IDU023V001", "S023", "lab_b", 8)])
        s = ld.load("IDU001V001", keypoints="dataset")
        self.assertEqual((s.keypoints.shape, s.labels.shape, s.boundary.shape), ((8, 17, 2), (8, 2), (8,)))
        self.assertTrue(s.present.all())
        self.assertEqual(np.flatnonzero(s.boundary).tolist(), list(range(0, 8)))   # tol 0.2 s = 6 frames
        self.assertTrue(np.isnan(s.keypoints[:, :, 0]).sum() == 0)

    def test_ours_needs_derived_keypoints(self):
        ld = ha4m.HA4MLoader(self.cfg, repo_root=self.repo)
        with self.assertRaises(AvError) as cm:
            ld.load("IDU001V001")
        self.assertEqual(cm.exception.code, "AV-DAT-003")
        d = self.root / "derived/yolo26x/IDU001V001"
        d.mkdir(parents=True)
        xy = np.full((6, 17, 2), 5.0, np.float32)
        xy[2] = np.nan
        np.savez(d / "subject.npz", body_xy=xy, body_conf=np.ones((6, 17), np.float32))
        (d / "source.json").write_text(json.dumps({"run_id": "run_x"}))
        s = ld.load("IDU001V001")
        self.assertEqual(s.keypoints.shape, (8, 17, 2))                   # padded to the label frames
        self.assertEqual(s.present.tolist(), [True, True, False, True, True, True, False, False])
        self.assertEqual(s.meta["run_id"], "run_x")

    def test_root_outside_data_public_is_refused(self):
        with self.assertRaises(AvError) as cm:
            ha4m.HA4MLoader(self.cfg, root=self.repo / "data/qisens/datasets/ha4m", repo_root=self.repo)
        self.assertEqual(cm.exception.code, "AV-STO-005")


class MappingTest(unittest.TestCase):
    def test_kinect_mapping(self):
        m = mapping.KINECT32_TO_COCO17
        self.assertEqual(len(m), 17)
        self.assertEqual(len(set(m)), 17)
        named = {c: mapping.KINECT32[k] for c, k in zip(mapping.COCO17, m)}
        self.assertEqual(named["left_ear"], "ear_left")
        self.assertEqual(named["right_shoulder"], "shoulder_right")
        self.assertEqual(named["left_ankle"], "ankle_left")
        self.assertEqual(len(mapping.unmapped(mapping.KINECT32, m)), 15)

    def test_report(self):
        text = mapping.mapping_report()
        self.assertIn("Every COCO joint has an equivalent", text)
        self.assertIn("| nose | none |", text)                             # InHARD has no face points
        self.assertIn("pelvis", text)


BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT LeftUpLeg
  {
    OFFSET 10 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 -40 0
    }
  }
}
MOTION
Frames: 5
Frame Time: 0.00833333
0 0 0 0 0 0 0 0 0
1 0 0 0 0 0 0 0 0
2 0 0 0 0 0 0 0 0
3 0 0 0 0 0 0 0 0
4 0 0 0 0 0 0 0 0
"""


class InHARDTest(unittest.TestCase):
    def test_bvh_and_resample(self):
        b = inhard.parse_bvh(BVH)
        self.assertEqual(b.joints, ["Hips", "LeftUpLeg"])
        self.assertEqual(b.parents, {"Hips": None, "LeftUpLeg": "Hips"})
        self.assertEqual(b.motion.shape, (5, 9))
        self.assertAlmostEqual(b.hz, 120.0, places=1)
        self.assertEqual(b.column("LeftUpLeg", "Xrotation"), 7)
        x = b.motion[:, b.column("Hips", "Xposition")]
        r = inhard.resample(x, b.hz, 60.0)
        self.assertEqual(len(r), 2)                                         # 0 and 1/60 s
        np.testing.assert_allclose(r, [0.0, 2.0], atol=1e-3)

    def test_load_refuses_until_access(self):
        tmp = fake_root()
        repo = Path(tmp.name)
        with self.assertRaises(AvError) as cm:
            inhard.InHARDLoader(Config.load(), root=repo / "elsewhere", repo_root=repo)
        self.assertEqual(cm.exception.code, "AV-STO-005")
        ld = inhard.InHARDLoader(Config.load(), repo_root=repo)
        with self.assertRaises(AvError) as cm:
            ld.load("P01_R01")
        self.assertEqual(cm.exception.code, "AV-DAT-003")
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
