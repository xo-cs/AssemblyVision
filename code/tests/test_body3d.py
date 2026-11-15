"""TC-B3D: the body3d stage pieces, its 3D and 4D views and the Heatmap page (Week 11). CPU, synthetic
joints; GVHMR itself is not run here."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from assemblyvision import body3d as B
from assemblyvision.config import Config
from assemblyvision.output import body3d_view as BV


def walking(n=60, fps=10.0):
    """A synthetic 17 joint skeleton walking along x, 1.7 m tall."""
    t = np.arange(n) / fps
    base = np.array([[0, 1.6, 0], [-.03, 1.65, 0], [.03, 1.65, 0], [-.08, 1.62, 0], [.08, 1.62, 0], [-.2, 1.4, 0],
                     [.2, 1.4, 0], [-.25, 1.1, 0], [.25, 1.1, 0], [-.25, .85, .1], [.25, .85, .1], [-.1, .9, 0],
                     [.1, .9, 0], [-.1, .5, 0], [.1, .5, 0], [-.1, .08, 0], [.1, .08, 0]], np.float32)
    J = np.repeat(base[None], n, 0)
    J[:, :, 0] += 0.5 * t[:, None]
    return J


class Body3dTest(unittest.TestCase):
    def test_pieces_fill_short_gaps_and_split_long_ones(self):
        boxes = {f: np.array([10, 10, 50, 90], float) for f in list(range(0, 40)) + list(range(45, 80)) + list(range(150, 160))}
        manifest = {"shots": {"items": [{"shot_id": "r-s001", "start_frame": 0, "end_frame": 200, "shot_type": "fixed"}]}}
        with mock.patch("assemblyvision.analysis.siglip.subject_boxes", return_value=boxes):
            ps = B.pieces(manifest, Path("."), 10.0, min_s=2.0, max_gap_s=1.0)
        self.assertEqual([(p["name"], p["start"], len(p["boxes"]), p["static"]) for p in ps],
                         [("r-s001_p1", 0, 80, True)])                 # the 5 frame gap is filled, the last 1 s is too short

    def test_views(self):
        J = walking()
        piece = {"name": "r-s001_p1", "shot_id": "r-s001", "frames": np.arange(100, 160), "joints": J, "static": True}
        d = BV.density_3d(piece, 10.0)
        self.assertEqual([t.name for t in d.data], ["ground", "hands (s per voxel)", "feet (s per voxel)"])
        self.assertAlmostEqual(float(d.data[0].z[0]), 0.08, places=5)           # the ground at the feet
        total = sum(float(np.sum(t.marker.color)) for t in d.data if t.name.startswith("feet"))
        self.assertAlmostEqual(total, 2 * 60 / 10.0, places=5)          # both feet, every frame, in seconds
        f = BV.floor_map(piece, 10.0)
        self.assertAlmostEqual(float(np.nansum(f.data[0].z)), 60 / 10.0, places=5)
        z = np.asarray(f.data[0].z, float)
        self.assertFalse((z[np.isfinite(z)] == 0).any())                        # empty cells are blank, never "0 s"
        self.assertIn("y = 0.08", f.layout.title.text)                  # the ground plane at the feet
        s = BV.skeleton_4d(piece, 10.0, anim_fps=5.0)
        self.assertEqual(len(s.frames), 30)
        self.assertEqual(s.data[-1].name, "ground")                             # stays while the frames play
        self.assertEqual(s.layout.sliders[0].steps[0].label, "0:10.0")  # video time of frame 100 at 10 fps

    def test_heatmap_page(self):
        from assemblyvision.ui.app import GradioApp
        from test_report_export import make_run
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = Config.load(overrides={"paths": {"public_root": str(root / "public"), "qisens_root": str(root / "q")}})
            run = root / "public" / "runs" / "run_20260101_000000_abcd"
            run.mkdir(parents=True)
            make_run(run)
            m = json.loads((run / "manifest.json").read_text())
            m["shots"] = {"items": []}
            (run / "manifest.json").write_text(json.dumps(m))
            app = GradioApp(cfg, ["public"], repo_root=root)
            html, f1, f2 = app.heatmap_view(run.name, "3D", None)
            self.assertIn("no 3D body data", html)
            (run / "body3d").mkdir()
            np.savez(run / "body3d" / "run_20260101_000000_abcd-s001_p1.npz", frames=np.arange(60), joints_world=walking(),
                     transl=np.zeros((60, 3), np.float32), static=np.bool_(True))
            self.assertEqual([v for _, v in app.body3d_pieces(run.name)], ["run_20260101_000000_abcd-s001_p1"])
            html, f1, f2 = app.heatmap_view(run.name, "3D", None)
            self.assertIsNotNone(f1)
            self.assertIsNotNone(f2)
            html, f1, f2 = app.heatmap_view(run.name, "4D", "run_20260101_000000_abcd-s001_p1")
            self.assertTrue(len(f1.frames) > 0)
            self.assertIsNone(f2)


if __name__ == "__main__":
    unittest.main()
