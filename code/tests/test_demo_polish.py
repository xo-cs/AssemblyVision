"""Demo polish: phase names per data area (public runs show Phase 1..N), the
trunk readout of the batch overlay (band colours, never over the header) and
the hand reach heatmap in the body frame (rotation and scale free)."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.analysis.cleaning import SignalCleaner
from assemblyvision.analysis.motion import MotionAnalyzer, subject_trunk
from assemblyvision.analysis.reach import reach_samples, run_reach
from assemblyvision.analysis.segmentation.rules import RuleBasedEngine
from assemblyvision.kts import write_kts
from assemblyvision.output.overlay import band_bgr, draw_header, draw_trunk, trunk_readout
from assemblyvision.phases import PUBLIC_NOTE, display_steps, phase_defs_for, public_label_map
from assemblyvision.storage.files import RunPaths
from motion_helpers import cfg as motion_cfg, pose_series, shot_for
from ui_helpers import make_root, make_run

EDGES = [20.0, 45.0, 60.0]


class PhaseLabelTest(unittest.TestCase):
    def test_vocabulary_per_data_area(self):
        defs = [{"name": "Removal"}, {"name": "Cleaning", "min_s": 5.0}]
        self.assertEqual(phase_defs_for(defs, "qisens"), defs)
        self.assertEqual(phase_defs_for(defs, None), defs)
        self.assertEqual(phase_defs_for(defs, "public"), [{"name": "Phase 1"}, {"name": "Phase 2", "min_s": 5.0}])

        class Ctx:
            source = "public"
        self.assertEqual(RuleBasedEngine(motion_cfg(), Ctx()).vocabulary, [f"Phase {k}" for k in range(1, 6)])
        Ctx.source = "qisens"
        self.assertEqual(RuleBasedEngine(motion_cfg(), Ctx()).vocabulary[0], "Removal")

    def test_public_label_map(self):
        m = public_label_map(["other", "Removal", "other", "Cleaning", "Removal", "segment_3"])
        self.assertEqual(m, {"Removal": "Phase 1", "Cleaning": "Phase 2", "segment_3": "Phase 3"})
        self.assertNotIn("other", m)
        from assemblyvision.types import Step
        steps = [Step("other", 0.0, 2.0, "engine_a"), Step("Removal", 2.0, 9.0, "engine_a", 1.0)]
        self.assertEqual([s.label for s in display_steps(steps, "public")], ["other", "Phase 1"])   # overlay
        self.assertIs(display_steps(steps, "qisens"), steps)

    def test_store_and_page_map_public_runs_only(self):
        from assemblyvision.ui.app import NO_LABELS, GradioApp
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = make_root(root)
            make_run(root, cfg, "run_20261026_010203_aa01", eval_json=False)                  # public
            make_run(root, cfg, "run_20261026_010203_aa02", source="qisens", eval_json=False)
            app = GradioApp(cfg, ["qisens", "public"], root, worker_cmd=["true"])
            pub, qis = "run_20261026_010203_aa01", "run_20261026_010203_aa02"
            self.assertEqual([s["label"] for s in app.store.get_segments(pub)],
                             ["other", "Phase 1", "other", "Phase 2"])
            self.assertEqual([s["label"] for s in app.store.get_segments(qis)],
                             ["other", "Removal", "other", "Cleaning"])
            self.assertEqual({r["label"] for r in app.store.get_metrics(pub)}, {"Phase 1", "Phase 2"})
            self.assertEqual(app.store.get_baseline(qis)[0]["label"], "Removal")
            self.assertIn("Removal", RunPaths(pub, app.store.paths(pub).root).steps_csv.read_text())  # file unchanged
            self.assertIn(PUBLIC_NOTE, app.results_head(pub))
            self.assertNotIn(PUBLIC_NOTE, app.results_head(qis))
            self.assertIn(NO_LABELS, app.eval_html(pub))
            self.assertEqual(app.reach_choices(pub), ["All phases", "Phase 1", "Phase 2"])


class TrunkReadoutTest(unittest.TestCase):
    def test_band_colours(self):
        self.assertEqual(band_bgr(0), (0x5E, 0xC5, 0x22))         # #22C55E green
        self.assertEqual(band_bgr(1), (0x08, 0xB3, 0xEA))         # #EAB308 yellow
        self.assertEqual(band_bgr(2), (0x16, 0x73, 0xF9))         # #F97316 orange
        self.assertEqual(band_bgr(3), (0x26, 0x26, 0xDC))         # #DC2626 red
        for angle, band in ((0, 0), (19.9, 0), (20.0, 1), (34.4, 1), (45.0, 2), (59.9, 2), (60.0, 3), (88, 3)):
            self.assertEqual(trunk_readout(angle, EDGES)[1], band, angle)
        self.assertEqual(trunk_readout(34.4, EDGES)[0], "Trunk 34")
        self.assertEqual(trunk_readout(float("nan"), EDGES), ("Trunk n/a", None))
        self.assertEqual(trunk_readout(None, EDGES), ("Trunk n/a", None))

    def test_box_border_and_segment_in_band_colour(self):
        img = np.zeros((480, 640, 3), np.uint8)
        x0, y0, x1, y1 = draw_trunk(img, 34.0, EDGES)
        self.assertGreater(x0, 320)                                # bottom right
        self.assertGreater(y0, 380)
        self.assertEqual(tuple(img[(y0 + y1) // 2, x0 + 1]), band_bgr(1))       # border
        seg = img[y0:y1, x0 + 4:x1 - 4].reshape(-1, 3)
        self.assertTrue((seg == band_bgr(1)).all(1).sum() > 40)                   # the B1 segment is filled
        self.assertFalse((seg == band_bgr(3)).all(1).any())
        img = np.zeros((480, 640, 3), np.uint8)
        x0, y0, x1, y1 = draw_trunk(img, None, EDGES)
        self.assertEqual(tuple(img[(y0 + y1) // 2, x0 + 1]), (150, 150, 150))  # grey without an angle
        for k in range(4):
            self.assertFalse((img == band_bgr(k)).all(2).any())

    def test_never_covers_the_header(self):
        img = np.zeros((480, 640, 3), np.uint8)
        hb = draw_header(img, "shot 3/12 handheld  frame 12345  step: Adjustment and Inspection")
        tb = draw_trunk(img, 50.0, EDGES, avoid=hb)
        overlap_x = tb[0] < hb[2] and hb[0] < tb[2]
        overlap_y = tb[1] < hb[3] and hb[1] < tb[3]
        self.assertFalse(overlap_x and overlap_y)
        img = np.zeros((480, 640, 3), np.uint8)
        hb = draw_header(img, "shot 1/1 fixed  frame 1")
        self.assertEqual(draw_trunk(img, 50.0, EDGES, avoid=hb)[3], 472)        # short header: stays at the bottom

    def test_angle_follows_the_bending_rule(self):
        s = pose_series(60, angle=np.r_[np.full(30, 10.0), np.full(30, 50.0)])
        s.body_conf[40:43, 11] = 0.1                               # left hip not measured: filled, not counted
        f, ang = subject_trunk(SignalCleaner(motion_cfg()).clean(s))
        self.assertEqual(len(f), 60)
        self.assertAlmostEqual(ang[5], 10.0, delta=0.5)
        self.assertAlmostEqual(ang[55], 50.0, delta=0.5)
        self.assertTrue(np.isnan(ang[40:43]).all())
        self.assertEqual(int(np.isnan(ang).sum()), 3)


def _rotate(xy, theta, scale, shift):
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    return (xy @ R.T) * scale + shift


class ReachTest(unittest.TestCase):
    def _skeleton(self):
        s = pose_series(20, angle=15.0, wrists=np.tile(np.array([[[380.0, 250.0], [300.0, 120.0]]]), (20, 1, 1)))
        return np.asarray(s.body_xy, np.float64), np.ones((20, 17), bool)

    def test_rotation_scale_and_shift_do_not_matter(self):
        xy, meas = self._skeleton()
        ref, used, sk = reach_samples(xy, meas)
        self.assertEqual(used, 20)
        self.assertEqual(ref.shape, (40, 2))
        for theta, scale, shift in ((0.7, 1.0, (0, 0)), (-1.2, 0.4, (55, -30)), (math.pi / 2, 2.5, (900, 400))):
            pts, used2, sk2 = reach_samples(_rotate(xy, theta, scale, np.array(shift)), meas)
            np.testing.assert_allclose(pts, ref, atol=1e-9)
            np.testing.assert_allclose(sk2["head"][0], sk["head"][0], atol=1e-6)
            self.assertEqual(used2, used)

    def test_body_frame_units(self):
        # upright: hips at (320, 300), shoulders 100 px above, right wrist of the picture 50 px right, 100 px up
        xy, meas = self._skeleton()
        s = pose_series(1, angle=0.0, wrists=np.array([[[370.0, 200.0], [320.0, 300.0]]]))
        pts, used, _ = reach_samples(np.asarray(s.body_xy, np.float64), np.ones((1, 17), bool))
        np.testing.assert_allclose(pts, [[0.5, 1.0], [0.0, 0.0]], atol=1e-9)

    def test_skips_frames_below_the_threshold(self):
        xy, meas = self._skeleton()
        meas[:5, 11] = False                       # no hip: frame skipped
        meas[5:10, 9] = False                      # one wrist missing: the other still counts
        meas[10:12, [9, 10]] = False               # no wrist at all
        pts, used, _ = reach_samples(xy, meas)
        self.assertEqual(used, 20 - 5 - 2)
        self.assertEqual(len(pts), 2 * 15 - 5 - 4)

    def test_run_reach_per_phase_and_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = make_root(root)
            p = make_run(root, cfg, "run_20261026_010203_aa01", eval_json=False)
            s = pose_series(600, run_id="run_20261026_010203_aa01", shot_id="run_20261026_010203_aa01-s001")
            s.body_conf[300:600, 9:11] = 0.1          # wrists unseen in the second half
            write_kts(s, p, {"is_subject": True, "subject_source": "auto"})
            m = json.loads(p.manifest.read_text())
            m["tracking"]["tracks"][0].update(is_subject=True, shot_id="run_20261026_010203_aa01-s001")
            m["shots"]["items"][0].update(start_s=0.0, end_s=20.0)
            p.manifest.write_text(json.dumps(m))
            subs = MotionAnalyzer(cfg).load_subjects(m, p.root)
            steps = [{"label": "other", "start_s": 0, "end_s": 2}, {"label": "A", "start_s": 2, "end_s": 10},
                     {"label": "other", "start_s": 10, "end_s": 11}, {"label": "B", "start_s": 11, "end_s": 20}]
            a = run_reach(subs, steps, 30.0, {"A"})
            self.assertEqual((a.frames, a.used), (240, 240))
            b = run_reach(subs, steps, 30.0, {"B"})
            self.assertEqual(b.frames, 270)
            self.assertAlmostEqual(b.coverage, 0.0)
            both = run_reach(subs, steps, 30.0)
            self.assertAlmostEqual(both.coverage, 240 / 510)
            self.assertEqual(set(both.skeleton), {"sho_half", "hip_half", "head"})
            self.assertAlmostEqual(both.skeleton["sho_half"][0], 0.2)       # 20 px of a 100 px torso
            from assemblyvision.ui.app import GradioApp
            app = GradioApp(cfg, ["public"], root, worker_cmd=["true"])
            h = app.reach_html("run_20261026_010203_aa01")
            self.assertIn("Coverage 47 %", h)
            self.assertIn("Coverage is low", h)
            self.assertIn("<img", h)
            self.assertEqual(len(app._reach), 1)
            app.reach_html("run_20261026_010203_aa01")
            self.assertEqual(len(app._reach), 1)                    # cached
            self.assertNotIn("Coverage is low", app.reach_html("run_20261026_010203_aa01", "Phase 1"))


if __name__ == "__main__":
    unittest.main()
