"""TC-S4D: the scene4d stage's geometry (Week 11): the similarity fit, chunk alignment, the GVHMR
skeleton placed in the scene. CPU, synthetic data; Pi3 itself is not run here."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from assemblyvision import scene4d as S4
from assemblyvision.scene4d import pi3_worker as W
from test_body3d import walking


def rot(deg, axis=1):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    R = np.eye(3)
    i, j = [k for k in range(3) if k != axis]
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


class Scene4dTest(unittest.TestCase):
    def test_similarity_fit_with_outliers(self):
        rng = np.random.default_rng(0)
        src = rng.normal(size=(400, 3))
        dst = 2.5 * src @ rot(30).T + np.array([1, -2, 0.5])
        dst[:40] += rng.normal(scale=3, size=(40, 3))                       # 10 % outliers
        s, R, t = W.robust_sim3(src, dst)
        self.assertAlmostEqual(s, 2.5, places=3)
        np.testing.assert_allclose(R, rot(30), atol=1e-3)

    def test_chunks_become_one_scene(self):
        S, h, w = 30, 4, 5
        rng = np.random.default_rng(1)
        base = rng.normal(size=(h, w, 3)) + np.array([0, 0, 5.0])           # a static scene seen in every frame
        truth = np.repeat(base[None], S, 0)

        def fake(model, frames):                                            # each chunk in its own frame and scale
            n = len(frames)
            s, R, t = 0.5 + 0.1 * frames[0, 0, 0, 0], rot(10 * frames[0, 0, 0, 0]), np.array([frames[0, 0, 0, 0], 0, 0])
            pts = s * truth[:n] @ R.T + t
            c2w = np.repeat(np.eye(4)[None], n, 0)                        # the camera of the truth frame, mapped too
            c2w[:, :3, :3], c2w[:, :3, 3] = R, t
            return {"pts": pts.astype(np.float32), "conf": np.ones((n, h, w), np.float32), "c2w": c2w.astype(np.float32)}
        frames = np.zeros((S, h, w, 3), np.uint8)
        for k, a in enumerate(range(0, S, 8)):
            frames[a:] = k                                                  # marks which chunk a frame starts in
        with mock.patch.object(W, "run_model", side_effect=lambda m, f: fake(m, f.copy())):
            r = W.chunked(None, frames, chunk=12, overlap=4)
        self.assertEqual(len(r["fits"]), 3)
        spread = np.linalg.norm(r["pts"] - r["pts"][0], axis=-1).max()      # every frame on the first chunk's scene
        self.assertLess(spread, 1e-3)

    def test_camera_stitch_keeps_a_fixed_camera_fixed(self):
        """Two chunks of a fixed camera, the second with points that fit a slightly wrong rotation: the stitch
        follows the cameras of the shared frames, so the camera path has no jump at the border."""
        rng = np.random.default_rng(2)
        n = 6
        cs = np.repeat(np.eye(4)[None], n, 0)
        R0 = rot(25)
        cd = cs.copy()
        cd[:, :3, :3] = R0
        cd[:, :3, 3] = [1.0, 2.0, 0.5]
        ps = rng.normal(size=(n, 8, 8, 3)) + [0, 0, 4]
        pd = 1.5 * ps @ R0.T + cd[0, :3, 3] + rng.normal(scale=0.05, size=ps.shape)
        s, R, t = W.camera_sim3(cs, cd, ps, pd, np.ones((n, 8, 8), bool))
        self.assertAlmostEqual(s, 1.5, delta=0.05)
        self.assertLess(np.abs(R - R0).max(), 1e-9)
        self.assertLess(np.abs(t - cd[0, :3, 3]).max(), 1e-9)

    def test_edges_and_votes(self):
        pts = np.zeros((4, 4, 3))
        pts[..., 2] = 1.0
        pts[:, 2:, 2] = 3.0                                                 # a depth jump between columns 1 and 2
        e = W.edge_mask(pts, np.zeros(3))
        self.assertTrue(e[:, 1:3].all())
        self.assertFalse(e[:, [0, 3]].any())
        p = np.array([[0.01, 0, 0], [0.02, 0, 0], [5.0, 5, 5]])           # one voxel seen in 2 frames, one in 1
        P, C, _ = W.voxel_down(p, np.full((3, 3), 100.0), 0.1, 10, np.array([0, 1, 0]), min_frames=2)
        self.assertEqual(len(P), 1)
        self.assertEqual(C.dtype, np.uint8)

    def test_fusion_rule(self):
        try:
            import scipy  # noqa: F401                                     # in envs/scene4d only
        except ImportError:
            self.skipTest("scipy is in envs/scene4d")
        def vox(ijk, depth, rank, frames):
            ijk = np.array(ijk)
            return {"ijk": ijk, "pts": ijk.astype(float), "col": np.full((len(ijk), 3), 9), "depth": np.array(depth, float),
                    "rank": np.array(rank, float), "frames": np.array(frames)}
        a = vox([[0, 0, 0], [5, 5, 5], [9, 9, 9], [20, 0, 0], [30, 0, 0]], [10, 10, 10, 10, 10], [.5, .9, .2, .9, .9], [3, 3, 3, 3, 1])
        b = vox([[0, 0, 1], [5, 5, 6]], [10.1, 12], [.5, .1], [3, 3])
        P, _, n = W.fuse_voxels(a, b)
        kept = {tuple(x) for x in P.astype(int).tolist()}
        self.assertEqual(kept, {(0, 0, 0), (0, 0, 1), (5, 5, 5), (20, 0, 0)})     # agree; surer of two apart; sure solo
        self.assertEqual((n["agree"], n["apart"], n["solo_kept"]), ([1, 1], [1, 1], [1, 0]))

    def test_skeleton_alignment(self):
        J = walking(60)
        s, R, t = 3.0, rot(40), np.array([0.5, 1.0, 2.0])
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "body3d").mkdir()
            (root / "scene4d").mkdir()
            np.savez(root / "body3d" / "p.npz", frames=np.arange(60), joints_world=J)
            kp3d = (s * J @ R.T + t)[::5].astype(np.float32)
            kp3d[:, 3] = np.nan                                             # an ear never seen
            np.savez(root / "scene4d" / "p.npz", frames=np.arange(0, 60, 5), kp3d=kp3d)
            a = S4.align(root, "p")
            self.assertTrue(a["ok"])
            self.assertAlmostEqual(a["scale"], 3.0, places=3)
            np.testing.assert_allclose(a["up"], R @ [0, 1, 0], atol=1e-3)
            np.savez(root / "scene4d" / "p.npz", frames=np.arange(0, 60, 5),
                     kp3d=np.random.default_rng(0).normal(size=kp3d.shape).astype(np.float32))
            self.assertFalse(S4.align(root, "p")["ok"])                     # unrelated points: not placed


if __name__ == "__main__":
    unittest.main()
