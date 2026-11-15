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
        self.assertEqual(f.data[0].hoverinfo, "text")                   # no "%{z:.1f}" template to show literally
        self.assertNotIn("%{", json.dumps(f.to_plotly_json(), default=str))
        self.assertEqual(f.layout.modebar.orientation, "v")
        s = BV.skeleton_4d(piece, 10.0, anim_fps=5.0)
        self.assertEqual(len(s.frames), 30)
        self.assertEqual(s.data[-1].name, "ground")                             # stays while the frames play
        self.assertEqual(s.layout.sliders[0].steps[0].label, "0:10.0")  # video time of frame 100 at 10 fps

    def test_player_data(self):
        J = walking()
        piece = {"name": "r-s001_p1", "frames": np.arange(100, 160), "joints": J}
        d = BV.player_data(piece, 10.0)
        self.assertEqual(len(d["joints"]), 60 * 17 * 3)
        self.assertEqual(len(d["vox_ids"]), 60)
        self.assertEqual(set(d["vox_kind"]), {0, 1})
        ids = np.array(d["vox_ids"])
        self.assertTrue((ids >= 0).all() and ids.max() == len(d["vox_kind"]) - 1)
        self.assertEqual(len(d["cell_ids"]), 60)
        self.assertAlmostEqual(d["ground"], 0.08, places=5)
        json.dumps(d)                                                       # plain JSON for the player

    def test_heatmap_page(self):
        from assemblyvision.ui.app import GradioApp
        from test_report_export import make_run
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = Config.load(overrides={"paths": {"public_root": str(root / "public"), "qisens_root": str(root / "q")}})
            runs = []
            for rid in ("run_20260101_000000_aaaa", "run_20260102_000000_bbbb", "run_20260103_000000_cccc"):
                run = root / "public" / "runs" / rid
                run.mkdir(parents=True)
                make_run(run)                                               # the same video (clip.mp4) three times
                m = json.loads((run / "manifest.json").read_text())
                m.update(run_id=rid, shots={"items": []})
                (run / "manifest.json").write_text(json.dumps(m))
                runs.append(run)
            app = GradioApp(cfg, ["public"], repo_root=root)
            self.assertEqual([r for _, r in app.heat_runs("2D")], ["run_20260103_000000_cccc"])      # duplicates hidden
            self.assertEqual(len(app.heat_runs("2D", show_all=True)), 3)
            self.assertEqual(app.heat_runs("3D"), [])
            self.assertIn("No run has 3D body output", app.heatmap_view(None, "3D"))
            (runs[0] / "body3d").mkdir()
            np.savez(runs[0] / "body3d" / "run_20260101_000000_aaaa-s001_p1.npz", frames=np.arange(60),
                     joints_world=walking(), transl=np.zeros((60, 3), np.float32), static=np.bool_(True))
            self.assertEqual([r for _, r in app.heat_runs("3D")], ["run_20260101_000000_aaaa"])
            self.assertEqual([r for _, r in app.heat_runs("2D")], ["run_20260103_000000_cccc", "run_20260101_000000_aaaa"])
            html = app.heatmap_view("run_20260101_000000_aaaa", "3D")
            self.assertIn("class='av-player'", html)
            self.assertIn("player.js", html)
            self.assertTrue((runs[0] / "body3d" / "run_20260101_000000_aaaa-s001_p1.player.json").is_file())
            self.assertIn("No 3D or 4D output", app.heatmap_view("run_20260101_000000_aaaa", "4D"))

if __name__ == "__main__":
    unittest.main()
