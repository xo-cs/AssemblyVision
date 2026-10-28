"""Overlay readouts: the rolling hand reach grid, the dot position in the mini
map, the both hands share of the last 10 s, the layout (no overlaps at 854x480
and 1920x1080), a render with every readout, and the comparison mode (sync rule
in node, button and raw video URL on the page). CPU only."""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.analysis.cleaning import SignalCleaner
from assemblyvision.analysis.motion import rolling_both_share, wrist_speeds
from assemblyvision.analysis.reach import frame_points
from assemblyvision.output import overlay
from assemblyvision.output.hud import (MINI_EXTENT, MiniReach, RollingGrid, boxes_overlap, both_text, chip_size,
                                       dot_px, hud_layout, mini_size)
from assemblyvision.storage.files import REPO_ROOT
from assemblyvision.types import Shot
from helpers import fake_root, ffmpeg_clip, make_ctx
from motion_helpers import FPS, cfg as motion_cfg, pose_series

UI = Path(__file__).resolve().parents[1] / "assemblyvision" / "ui"


class RollingGridTest(unittest.TestCase):
    def test_add_and_remove(self):
        g = RollingGrid(window=3, n=10, extent=(0.0, 1.0, 0.0, 1.0))
        g.push(np.array([[0.05, 0.95], [0.55, 0.45]]))    # top left cell and a middle cell
        self.assertEqual(g.grid[0, 0], 1)
        self.assertEqual(g.grid[5, 5], 1)
        g.push(np.array([[0.05, 0.95], [np.nan, np.nan]]))  # a missing wrist adds nothing
        g.push(np.full((2, 2), np.nan))                    # an empty frame still takes its place in the window
        self.assertEqual((g.grid[0, 0], g.points), (2, 3))
        g.push(np.array([[2.0, 2.0], [np.nan, np.nan]]))   # outside the extent; frame 1 leaves the window
        self.assertEqual((g.grid[0, 0], g.grid[5, 5], g.points), (1, 0, 1))
        self.assertEqual(g.grid.sum(), 1)
        self.assertIsNotNone(g.density())
        for _ in range(3):
            g.push(np.full((2, 2), np.nan))
        self.assertEqual((g.points, float(g.grid.sum())), (0, 0.0))
        self.assertIsNone(g.density())

    def test_density_is_smoothed_and_normalised(self):
        g = RollingGrid(window=100, n=48)
        for _ in range(10):
            g.push(np.array([[0.0, 0.5], [np.nan, np.nan]]))
        d = g.density()
        self.assertAlmostEqual(float(d.max()), 1.0)
        self.assertGreater(int((d > 0.05).sum()), 4)           # spread by the blur


class DotTest(unittest.TestCase):
    def test_body_frame_point_to_map_pixel(self):
        box = (100, 50, 240, 190)                          # 140 px map
        x0, x1, y0, y1 = MINI_EXTENT
        u, v = dot_px((0.0, 0.0), box)                     # mid hip
        self.assertAlmostEqual(u, 100 + (0 - x0) / (x1 - x0) * 140)
        self.assertAlmostEqual(v, 50 + (y1 - 0) / (y1 - y0) * 140)
        self.assertAlmostEqual(dot_px((x0, y1), box)[0], 100)
        self.assertAlmostEqual(dot_px((x0, y1), box)[1], 50)
        self.assertGreater(dot_px((0.5, 0.0), box)[0], u)  # body right is map right
        self.assertLess(dot_px((0.0, 1.0), box)[1], v)     # up is up

    def test_wrist_of_a_skeleton_lands_where_expected(self):
        # bent forward 30 deg; the right wrist (picture) half a torso to the body's side at shoulder height
        s = pose_series(1, angle=30.0, torso=80.0)
        a = np.radians(30.0)
        u, n = np.array([np.sin(a), -np.cos(a)]), np.array([np.cos(a), np.sin(a)])
        hip = np.array([320.0, 300.0])
        s.body_xy[0, 9] = hip + 80 * u + 0.5 * 80 * n
        pts = frame_points(s.body_xy, np.ones((1, 17), bool))
        np.testing.assert_allclose(pts[0, 0], [0.5, 1.0], atol=1e-6)
        mini = MiniReach(FPS, {})
        mini.push(pts[0])
        self.assertEqual(mini.dot_alpha(0), 1.0)
        img = np.zeros((480, 854, 3), np.uint8)
        mini.draw(img, (8, 100, 8 + 152, 100 + 168))
        px = dot_px(pts[0, 0], (14, 100 + 6 + 16, 14 + 140, 100 + 6 + 16 + 140))
        c = img[int(round(px[1])), int(round(px[0]))].astype(int)
        self.assertTrue(abs(c[0] - 0xEE) < 50 and abs(c[2] - 0x22) < 50, c)   # the cyan dot
        for _ in range(round(0.25 * FPS)):
            mini.push(np.full((2, 2), np.nan))
        self.assertAlmostEqual(mini.dot_alpha(0), 0.5, delta=0.05)             # fading out
        for _ in range(round(0.5 * FPS)):
            mini.push(np.full((2, 2), np.nan))
        self.assertEqual(mini.dot_alpha(0), 0.0)


class BothHandsTest(unittest.TestCase):
    def test_rolling_share(self):
        al = np.array([1, 1, 0, 1, 0, 0, 1, 1], bool)
        ar = np.array([1, 0, 0, 1, 1, 0, 1, 0], bool)
        ok = np.ones(8, bool)
        s = rolling_both_share(al, ar, ok, window=4)
        self.assertAlmostEqual(s[3], 2 / 3)                # frames 0..3: both in 0 and 3, either in 0, 1, 3
        self.assertAlmostEqual(s[7], 1 / 3)                # frames 4..7: both in 6, either in 4, 6, 7
        self.assertAlmostEqual(s[0], 1.0)
        ok[4:7] = False                                    # 1 of 4 frames ok: not enough
        self.assertTrue(np.isnan(rolling_both_share(al, ar, ok, window=4)[7]))
        self.assertTrue(np.isnan(rolling_both_share(np.zeros(4, bool), np.zeros(4, bool), np.ones(4, bool), 4)[3]))
        self.assertEqual(both_text(0.574), ("Both hands 57 %", True))
        self.assertEqual(both_text(float("nan")), ("Both hands n/a", False))

    def test_same_definition_as_the_metric(self):
        T = 300
        t = np.arange(T) / FPS
        hip = np.array([320.0, 300.0])
        w = np.zeros((T, 2, 2))
        w[:, 0] = hip + np.stack([20 + 60 * np.sin(2 * np.pi * 0.5 * t), np.zeros(T)], 1)
        w[:, 1] = hip + np.stack([-20 - 60 * np.sin(2 * np.pi * 0.5 * t) * (t > 5), np.zeros(T)], 1)
        cs = SignalCleaner(motion_cfg()).clean(pose_series(T, wrists=w))
        a, b = wrist_speeds(cs.xy, cs.valid, cs.torso, FPS)
        thr = motion_cfg()["motion"]["hand_active"]
        ok = np.isfinite(a) & np.isfinite(b)
        al, ar = ok & (a >= thr), ok & (b >= thr)
        share = rolling_both_share(al, ar, ok, round(10 * FPS))
        whole = (al & ar).sum() / (al | ar).sum()              # two_hand_simultaneous of the whole clip
        self.assertAlmostEqual(share[-1], whole)
        self.assertLess(share[100], 0.2)                       # first 3 s: the right hand still
        self.assertGreater(whole, 0.3)


class LayoutTest(unittest.TestCase):
    HEADERS = ("shot 1/1 fixed  frame 12  step: Phase 1",
               "shot 17/32 handheld  frame 17891  step: Phase 3",
               "shot 120/131 handheld  frame 17891  step: Adjustment and Inspection",
               "shot 120/131 handheld  frame 178910  step: Adjustment and Inspection and Assembly")   # too wide

    def _check(self, w, h):
        hs = 0.55 * max(1.0, h / 480)
        modes = set()
        for header in self.HEADERS:
            for chip in ("Both hands 100 %", "Both hands n/a"):
                img = np.zeros((h, w, 3), np.uint8)
                hb = overlay.draw_header(img, header, scale=hs)
                lay = hud_layout(w, h, hb, (chip_size(chip, hs)[0], hb[3] - hb[1]), overlay.trunk_size(hs),
                                 mini_size(h)[:2])
                names = list(lay)
                for i, a in enumerate(names):
                    x0, y0, x1, y1 = lay[a]
                    self.assertTrue(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h, (w, h, a, lay[a]))
                    for b in names[i + 1:]:
                        self.assertFalse(boxes_overlap(lay[a], lay[b]), (w, h, header, a, b, lay))
                self.assertEqual(lay["map"][3], hb[1] - 8)          # directly above the header
                self.assertEqual(lay["map"][0], 8)
                self.assertEqual(lay["chip"][3] - lay["chip"][1], hb[3] - hb[1])   # same height as the header
                modes.add("inline" if lay["chip"][1] == hb[1] else "above")
                # drawing into the boxes stays inside them
                MiniReach(FPS, {}).draw(img, lay["map"])
                overlay.draw_trunk(img, 50.0, [20, 45, 60], scale=hs, box=lay["trunk"])
        return modes

    def test_854x480(self):
        self.assertEqual(self._check(854, 480), {"inline", "above"})
        self.assertEqual(mini_size(480)[2], 140)

    def test_1920x1080(self):
        self.assertIn("inline", self._check(1920, 1080))
        self.assertEqual(mini_size(1080)[2], 315)


class RenderTest(unittest.TestCase):
    def test_render_with_every_readout(self):
        tmp = fake_root()
        try:
            cfg, ctx = make_ctx(Path(tmp.name))
            video = ffmpeg_clip(ctx.paths.tmp_dir / "clip.mp4", seconds=2, rate=10, size="854x480")
            shots = [Shot(f"{ctx.run_id}-s001", ctx.run_id, 0, 20, 0.0, 2.0, "handheld")]
            s = pose_series(20, angle=50.0)
            s.frame_idx[:] = np.arange(20)
            from assemblyvision.analysis.motion import MotionAnalyzer
            sub = MotionAnalyzer(cfg).subject(s, shots[0])
            hud = overlay.overlay_series([sub], 21, 10.0, cfg)
            self.assertTrue(np.isfinite(hud["trunk"][:20]).all())
            self.assertTrue(np.isfinite(hud["wrists"][:20]).all())
            self.assertEqual(set(hud["skeleton"]), {"sho_half", "hip_half", "head"})
            overlay.render_track_overlay(video, shots, {}, ctx.paths.tracks_overlay, 10.0, cfg, ctx, hud=hud)
            self.assertEqual(ctx.manifest["overlay"]["frames"], 20)
            self.assertIn("hand reach", ctx.manifest["overlay"]["legend"])
        finally:
            tmp.cleanup()


class CompareTest(unittest.TestCase):
    def test_sync_rule_in_node(self):
        node = shutil.which("node") or str(REPO_ROOT / "tools" / "node" / "bin" / "node")
        if not Path(node).is_file():
            self.skipTest("node is not installed")
        script = (UI / "compare_sync.js").read_text() + """
const assert = require("assert");
assert.strictEqual(avSyncTime(10.0, 10.05, 0, 0.1), null);          // within 0.1 s: leave it
assert.strictEqual(avSyncTime(10.0, 10.2, 0, 0.1), 10.0);           // drifted: seek
assert.strictEqual(avSyncTime(12.5, 0, 2.5, 0.1), 10.0);            // overlay starts 2.5 s into the raw video
assert.strictEqual(avSyncTime(1.0, 0.3, 2.5, 0.1), 0);              // before the overlay: its start
assert.strictEqual(avSyncTime(5, 5.09), null);                      // default tolerance 0.1 s
assert.strictEqual(avClock(125.9), "2:05");
console.log("ok");
"""
        out = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "ok")

    def test_app_js_takes_the_sync_rule(self):
        js = (UI / "app.js").read_text()
        self.assertEqual(js.count("/*COMPARE_SYNC*/"), 1)
        for want in ("requestFullscreen", "fullscreenchange", "Escape", "Raw footage", "AssemblyVision analysis",
                     "#av-overlay video", "data-compare-raw"):
            self.assertIn(want, js)

    def test_button_and_raw_url(self):
        from assemblyvision.ui.app import GradioApp
        from ui_helpers import make_root, make_run
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = make_root(root)
            p = make_run(root, cfg, "run_20261026_010203_aa01", eval_json=False)
            app = GradioApp(cfg, ["public"], root, worker_cmd=["true"])
            rid = "run_20261026_010203_aa01"
            self.assertNotIn("Comparison mode", app.video_head_html(rid))      # no video files yet
            p.cfr.parent.mkdir(parents=True, exist_ok=True)
            p.cfr.write_bytes(b"x")
            p.tracks_overlay.parent.mkdir(parents=True, exist_ok=True)
            p.tracks_overlay.write_bytes(b"x")
            m = json.loads(p.manifest.read_text())
            m["shots"]["items"][0]["start_frame"] = 15
            p.manifest.write_text(json.dumps(m))
            h = app.video_head_html(rid)
            self.assertIn("Comparison mode", h)
            self.assertIn(f"data-compare-raw='gradio_api/file={p.cfr}'", h)
            self.assertIn("data-compare-offset='0.500'", h)                   # 15 frames at 30 fps
            self.assertNotIn("Comparison mode", app.video_head_html(None))
            vals = app.build()                                               # the page still builds
            self.assertIsNotNone(vals)


if __name__ == "__main__":
    unittest.main()
