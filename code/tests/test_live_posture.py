"""Side work station: bent posture stretches, idle time and gaze shares (CPU)."""
import unittest

import numpy as np

from assemblyvision.live.posture import GazeShare, PostureTimer
from assemblyvision.live.stations import Station
from assemblyvision.live.templates import load_template
from live_helpers import cfg, side_body

BANDS = [20.0, 45.0, 60.0]


class PostureTest(unittest.TestCase):
    def test_stretches(self):
        p = PostureTimer(load_template("side_work").params, BANDS, gap_s=0.5)
        for t in np.arange(0, 20, 0.1):
            a = 30 if 2 <= t < 8 or 8.3 <= t < 10 or 14 <= t < 16 else 5   # 0.3 s break does not split
            kp, conf = side_body(a)
            st = p.update(float(t), kp, conf >= 0.3)
            if abs(t - 5) < 0.05:
                self.assertAlmostEqual(st["current_s"], 3.0, delta=0.11)
        st = p.state(20.0)
        self.assertAlmostEqual(st["longest_s"], 8.0, delta=0.15)
        self.assertEqual(st["stretches"], 2)
        self.assertAlmostEqual(st["total_s"], 9.7, delta=0.25)
        self.assertEqual(st["current_s"], 0.0)

    def test_gap_ends_a_stretch(self):
        p = PostureTimer(load_template("side_work").params, BANDS, gap_s=0.5)
        for t in list(np.arange(0, 3, 0.1)) + list(np.arange(6, 8, 0.1)):
            kp, conf = side_body(30)
            p.update(float(t), kp, conf >= 0.3)
        st = p.state(8.0)
        self.assertEqual(st["stretches"], 2)
        self.assertLess(st["total_s"], 5.0)


class GazeTest(unittest.TestCase):
    def test_profile_is_side(self):
        g = GazeShare(0.5, 0.3, gap_s=0.5)
        for t in np.arange(0, 3, 0.1):
            kp, conf = side_body(5)
            g.update(float(t), kp, conf)
        self.assertEqual(g.now, "side")
        self.assertEqual(g.shares()["side"], 1.0)

    def test_front_and_away(self):
        g = GazeShare(0.5, 0.3, gap_s=0.5)
        kp, conf = side_body(5)
        front = kp.copy()
        front[1], front[2], front[0] = front[0] + [-8, -5], front[0] + [8, -5], front[0]
        cf = conf.copy()
        cf[[1, 2]] = 0.9
        away = conf.copy()
        away[[0, 1, 2]] = 0.1
        for i, t in enumerate(np.arange(0, 4, 0.1)):
            if t < 2:
                g.update(float(t), front, cf)
            else:
                g.update(float(t), kp, away)
        sh = g.shares()
        self.assertAlmostEqual(sh["front"], 0.5, delta=0.05)
        self.assertAlmostEqual(sh["away"], 0.5, delta=0.05)


class StationTest(unittest.TestCase):
    def test_side_work_panel_and_summary(self):
        st = Station(load_template("side_work"), cfg(), 1, 0.0)
        for t in np.arange(0, 6, 0.1):
            kp, conf = side_body(50 if t > 2 else 5)
            st.update(float(t), kp, conf, (640, 480))
        p = st.panel(6.0)
        self.assertEqual(p["band"], 2)
        self.assertGreater(p["posture"]["current_s"], 3.5)
        self.assertIsNotNone(p["idle_time_s"])                   # the side view torso works
        labels = [r[0] for r in st.summary(6.0)["rows"]]
        self.assertIn("Longest bent stretch (B1 or worse)", labels)

    def test_band_seconds(self):
        st = Station(load_template("floor_pick"), cfg(), 1, 0.0)
        for t in np.arange(0, 4.001, 0.1):
            kp, conf = side_body(50 if t >= 2 else 10)
            st.update(float(t), kp, conf, (640, 480))
        self.assertAlmostEqual(st.band_s[0], 2.0, delta=0.11)
        self.assertAlmostEqual(st.band_s[2], 2.0, delta=0.11)


if __name__ == "__main__":
    unittest.main()
