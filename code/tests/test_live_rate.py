"""Live frame rate and page updates: the adaptive rate steps down and up, push
paces frames to the current rate, the panel is sent only when it changed,
Clear resets the page, and other people are drawn in red (CPU, fake pose)."""
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.live.draw import OTHER_RGB, draw_overlay
from assemblyvision.live.manager import LiveManager, RateControl
from assemblyvision.ui.live_tab import clear_values, panel_html, panel_update, save_label
from live_helpers import cfg
from test_live_manager import FakePose, frame


class RateControlTest(unittest.TestCase):
    def test_steps_down_one_step_per_hold_and_stops_at_the_floor(self):
        r = RateControl([25, 20, 15, 10], p95_max_ms=150, drop_max=0.25, hold_s=3, up_s=10)
        self.assertEqual(r.fps, 25)
        self.assertEqual(r.update(0.0, 300, 0.0), 20)            # delay too high: one step down
        self.assertEqual(r.update(1.0, 300, 0.0), 20)            # held for 3 s
        self.assertEqual(r.update(3.0, 300, 0.0), 15)
        self.assertEqual(r.update(6.0, 40, 0.5), 10)             # too many dropped frames
        self.assertEqual(r.update(9.0, 300, 0.5), 10)            # the floor
        self.assertEqual(r.update(10.0, None, None), 10)         # no data: no change

    def test_steps_up_after_a_good_stretch(self):
        r = RateControl([25, 20, 15, 10], 150, 0.25, 3, 10)
        r.update(0.0, 400, 0.0)
        self.assertEqual(r.fps, 20)
        self.assertEqual(r.update(5.0, 40, 0.0), 20)             # good since 5 s
        self.assertEqual(r.update(14.0, 40, 0.0), 20)
        self.assertEqual(r.update(15.0, 40, 0.0), 25)            # 10 s good: back up
        r.update(20.0, 400, 0.0)
        self.assertEqual(r.update(30.0, 100, 0.0), 20)           # in between: neither up nor down

    def _manager(self, d, pose, **live):
        return LiveManager(cfg(model_keep_s=0.0, **live), Path(d) / "live", backend_factory=lambda: pose,
                           gpu_free=lambda i: 8000, lock_dir=Path(d) / "locks", port=7861)

    def test_push_paces_to_the_rate(self):
        with tempfile.TemporaryDirectory() as d:
            lm = self._manager(d, FakePose(), rate_steps=[10])
            lm.start()
            for i in range(60):                                    # 100 frames per second offered for 0.6 s
                lm.push(frame(i))
                time.sleep(0.01)
            s = lm.session
            self.assertGreater(s.paced, 40)
            self.assertLessEqual(s.frames_in, 9)
            p = lm.panel()
            self.assertEqual(p["rate_fps"], 10)
            self.assertIn("data-live-rate='10'", panel_html(p))
            lm.stop()

    def test_slow_server_steps_the_rate_down(self):
        with tempfile.TemporaryDirectory() as d:
            lm = self._manager(d, FakePose(delay=0.12), rate_hold_s=0.5)
            lm.start()
            t_end = time.monotonic() + 4.0
            i = 0
            while time.monotonic() < t_end:                        # 25 frames per second offered
                lm.push(frame(i))
                i += 1
                time.sleep(0.04)
            p = lm.panel()
            self.assertLess(p["rate_fps"], 25)
            self.assertTrue(any("Frame rate down" in n for n in p["notices"]))
            s = lm.stop()
            self.assertEqual(s["rate_fps_end"], p["rate_fps"])


class PageUpdateTest(unittest.TestCase):
    P = {"session_id": "live_20261026_010203_ab12", "status": "done", "elapsed_s": 65, "fps": None,
         "delay_p50_ms": 18, "delay_p95_ms": 42, "frames_in": 600, "processed": 590, "dropped": 10,
         "rate_fps": 25, "paused": False, "recording": False, "notices": [], "log": [],
         "station": {"kind": "posture", "title": "C. Side work", "trunk_deg": 5, "band": 0,
                     "band_s": [60.0, 5.0, 0.0, 0.0],
                     "posture": {"band": 0, "current_s": 0.0, "longest_s": 4.0, "total_s": 5.0, "stretches": 2},
                     "gaze": {}, "idle_time_s": 3.0}}

    def test_panel_only_when_changed(self):
        panel, log, shown = panel_update(self.P, {})
        self.assertIn("STOPPED", panel)
        self.assertEqual(log, "")
        self.assertEqual(panel_update(self.P, shown)[:2], (None, None))       # nothing changed: nothing sent
        panel2, log2, _ = panel_update(dict(self.P, elapsed_s=66), shown)
        self.assertIsNotNone(panel2)
        self.assertIsNone(log2)
        self.assertIsNone(panel_update(None, {"panel": panel_html(None), "log": ""})[0])

    def test_save_label(self):
        self.assertEqual(save_label(True), "Save recording: on")
        self.assertEqual(save_label(False), "Save recording: off")

    def test_clear_resets_the_page(self):
        with tempfile.TemporaryDirectory() as d:
            lm = LiveManager(cfg(model_keep_s=0.0), Path(d) / "live", backend_factory=FakePose,
                             gpu_free=lambda i: 8000, lock_dir=Path(d) / "locks", port=7861)
            lm.start()
            for i in range(10):
                lm.push(frame(i))
                time.sleep(0.05)
            self.assertEqual(clear_values(lm), {})                 # not while a session runs
            self.assertIsNotNone(lm.panel())
            s = lm.stop()
            last = lm.stopped_overlay()
            self.assertIsNotNone(last)
            self.assertFalse(np.array_equal(last, lm.session.latest_overlay))   # the Stopped label
            vals = clear_values(lm)
            self.assertEqual(vals, {"overlay": None, "panel": panel_html(None), "log": "", "summary": "",
                                    "camera": None})
            self.assertIn("No live session", vals["panel"])
            self.assertIsNone(lm.panel())
            self.assertIsNone(lm.stopped_overlay())
            self.assertTrue(lm.cleared)
            self.assertIs(lm.last_summary, s)                      # Analyse recording still knows the session
            self.assertTrue((Path(d) / "live" / s["session_id"] / "summary.json").is_file())
            lm.start()
            self.assertFalse(lm.cleared)
            lm.stop()


class OverlayColorTest(unittest.TestCase):
    def test_other_people_are_red_with_their_number(self):
        img = np.full((240, 320, 3), 40, np.uint8)
        boxes = np.array([[20.0, 60.0, 120.0, 220.0], [180.0, 60.0, 280.0, 220.0]])
        out = draw_overlay(img, boxes, 0, None, None, None, "LIVE  UPRIGHT  0:05", ids=np.array([1, 7]))
        self.assertTrue((out[150, 180] == OTHER_RGB).all() or (out[150, 181] == OTHER_RGB).all())
        self.assertEqual(OTHER_RGB, (0xFF, 0x3B, 0x30))
        self.assertTrue((out[150, 21] == 255).all())                # the subject box is white


if __name__ == "__main__":
    unittest.main()
