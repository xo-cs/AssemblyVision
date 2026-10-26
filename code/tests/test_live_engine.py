"""Online Engine A of the workbench station on synthetic wrist streams (CPU)."""
import unittest

import numpy as np

from assemblyvision.live.engine import OnlineRuleEngine
from assemblyvision.live.filters import OneEuro
from assemblyvision.live.templates import load_template
from live_helpers import H, W, front_body, times, workbench_script


def run(engine, at, ts):
    for t in ts:
        kp, conf = at(t)
        engine.update(float(t), kp, conf, (W, H))
    return engine


class WorkbenchTest(unittest.TestCase):
    def test_three_cycles(self):   # TC-LIV-01
        at, bounds, end = workbench_script(3)
        e = run(OnlineRuleEngine(load_template("workbench"), 0.3, 0.2), at, times(end + 1, seed=1))
        st = e.state
        self.assertEqual(st.cycles, 3)
        steps = [s for s in e.steps if s.closed_by == "event"]
        self.assertEqual([s.name for s in steps], ["pick", "assemble", "place"] * 3)
        for s, b in zip(steps, [b[n] for b in bounds for n in ("pick", "assemble", "place")]):
            self.assertAlmostEqual(s.start_s, b[0], delta=0.2)
            self.assertAlmostEqual(s.end_s, b[1], delta=0.2)
        self.assertAlmostEqual(np.mean(st.cycle_times), 5.0, delta=0.2)
        self.assertAlmostEqual(st.last_cycle_s, 5.0, delta=0.2)

    def test_dropped_frames(self):
        at, bounds, end = workbench_script(2)
        e = run(OnlineRuleEngine(load_template("workbench"), 0.3, 0.2), at, times(end + 1, drop=0.15, seed=4))
        self.assertEqual(e.state.cycles, 2)

    def test_short_touch_gives_no_event(self):   # TC-LIV-03
        e = OnlineRuleEngine(load_template("workbench"), 0.3, 0.2)
        events = []
        for t in np.arange(0, 3, 0.05):
            lw = (100.0, 280.0) if 1.0 <= t < 1.15 else (290.0, 280.0)
            kp, conf = front_body(lw, (350.0, 280.0))
            events += e.update(float(t), kp, conf, (W, H)).events
        self.assertNotIn("zone_enter:left", events)
        self.assertIsNone(e.state.open_step)

    def test_timeout(self):   # TC-LIV-04
        e = OnlineRuleEngine(load_template("workbench"), 0.3, 0.2)
        for t in np.arange(0, 25, 0.1):          # the left wrist stays in the left zone: pick never ends
            kp, conf = front_body((100.0, 280.0) if t >= 1 else (290.0, 280.0), (350.0, 280.0))
            e.update(float(t), kp, conf, (W, H))
        self.assertEqual(e.steps[0].name, "pick")
        self.assertEqual(e.steps[0].closed_by, "timeout")
        self.assertAlmostEqual(e.steps[0].end_s - e.steps[0].start_s, 20.0, delta=0.15)
        self.assertEqual(e.state.cycles, 0)

    def test_unseen_wrist_keeps_its_zone(self):
        at, bounds, end = workbench_script(1)
        e = OnlineRuleEngine(load_template("workbench"), 0.3, 0.2)
        for t in times(end + 1, seed=2):
            kp, conf = at(t)
            if 2.5 < t < 3.0:
                conf = conf.copy()
                conf[[9, 10]] = 0.1              # wrists hidden briefly during assembly
            e.update(float(t), kp, conf, (W, H))
        self.assertEqual(e.state.cycles, 1)

    def test_borders_move_the_zones(self):
        """With the borders at 0.1 and 0.9 the scripted wrists (x 100 and 560 of 640)
        never reach the outer zones, so nothing happens; at the default thirds they do."""
        at, _, end = workbench_script(1)
        e = run(OnlineRuleEngine(load_template("workbench").with_borders(0.1, 0.9), 0.3, 0.2), at, times(end + 1))
        self.assertEqual((e.state.cycles, e.steps), (0, []))
        e = run(OnlineRuleEngine(load_template("workbench").with_borders(0.1, 0.9), 0.3, 0.2), at, [])
        e.set_template(load_template("workbench"))
        e = run(e, at, times(end + 1))
        self.assertEqual(e.state.cycles, 1)

    def test_lost_closes_and_restarts(self):
        at, bounds, end = workbench_script(2)
        e = OnlineRuleEngine(load_template("workbench"), 0.3, 0.2)
        for t in times(end + 1, seed=3):
            if bounds[0]["assemble"][0] + 1 < t < bounds[0]["assemble"][0] + 1.1:
                e.close_all(float(t), "lost")
            kp, conf = at(t)
            e.update(float(t), kp, conf, (W, H))
        self.assertIn("lost", [s.closed_by for s in e.steps])
        self.assertEqual(e.state.cycles, 1)       # the interrupted first cycle is not counted


class FallbackTest(unittest.TestCase):
    def test_segments_at_pauses(self):   # TC-LIV-08
        e = OnlineRuleEngine(None, 0.3, 0.2)
        for t in np.arange(0, 12, 0.1):
            moving = (t % 4) < 2.5                # 2.5 s motion, 1.5 s still
            dx = 40 * np.sin(2 * np.pi * t) if moving else 0.0
            kp, conf = front_body((290.0 + dx, 280.0), (350.0 - dx, 280.0))
            kp[:, 0] += dx
            e.update(float(t), kp, conf, (W, H))
        names = [s.name for s in e.steps]
        self.assertEqual(names[:3], ["segment_1", "segment_2", "segment_3"])
        self.assertTrue(all(s.closed_by == "pause" for s in e.steps))


class OneEuroTest(unittest.TestCase):
    def test_still_point_is_smoothed(self):
        rng = np.random.default_rng(0)
        f = OneEuro(1.0, 0.007, 1.0, gap_s=1.0)
        out = []
        for i, t in enumerate(np.arange(0, 5, 0.1)):
            kp = np.full((17, 2), 100.0) + rng.normal(0, 3, (17, 2))
            out.append(f(t, kp, np.ones(17, bool))[0, 0])
        # LLD setting at 10 fps: alpha = 0.39, noise reduced to about 0.49 (raw std 3)
        self.assertLess(np.std(out[10:]), 0.6 * 3)

    def test_follows_a_step(self):
        f = OneEuro(1.0, 0.007, 1.0, gap_s=1.0)
        for t in np.arange(0, 1, 0.1):
            f(t, np.full((17, 2), 100.0), np.ones(17, bool))
        y = [f(t, np.full((17, 2), 200.0), np.ones(17, bool))[0, 0] for t in np.arange(1, 3, 0.1)]
        self.assertGreater(y[-1], 195)

    def test_unseen_is_nan_and_gap_restarts(self):
        f = OneEuro(1.0, 0.007, 1.0, gap_s=1.0)
        f(0.0, np.full((17, 2), 100.0), np.ones(17, bool))
        ok = np.ones(17, bool)
        ok[3] = False
        out = f(0.1, np.full((17, 2), 110.0), ok)
        self.assertTrue(np.isnan(out[3]).all())
        out = f(5.0, np.full((17, 2), 300.0), np.ones(17, bool))     # after a gap: no lag
        self.assertEqual(out[0, 0], 300.0)


if __name__ == "__main__":
    unittest.main()
