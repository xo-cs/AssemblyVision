"""Floor pick station: lifts on synthetic side view bodies, stoop vs squat (CPU)."""
import unittest

import numpy as np

from assemblyvision.analysis.motion import band_of
from assemblyvision.live.lifts import LiftDetector
from assemblyvision.live.templates import load_template
from live_helpers import side_body, times

BANDS = [20.0, 45.0, 60.0]


def profile(t, start, peak, down=1.0, hold=1.0, up=1.0, rest=5.0):
    """0 before start, linear to peak, hold, linear back to 0."""
    x = t - start
    if x < 0 or x > down + hold + up:
        return rest
    if x < down:
        return rest + (peak - rest) * x / down
    if x < down + hold:
        return peak
    return peak + (rest - peak) * (x - down - hold) / up


def script(lifts):
    """lifts: [(start, peak trunk, lowest knee)] -> fn(t) -> (kp, conf, trunk, knee)."""
    def at(t):
        trunk, knee = 5.0, 178.0
        for start, a, k in lifts:
            if start <= t <= start + 3:
                trunk = profile(t, start, a, rest=5.0)
                knee = profile(t, start, k, rest=178.0)
        kp, conf = side_body(trunk, knee)
        return kp, conf, trunk, knee
    return at


def run(lifts, end, seed=0, drop=0.0):
    det = LiftDetector(load_template("floor_pick").params, BANDS, gap_s=0.5)
    at = script(lifts)
    truth = {}
    for t in times(end, seed=seed, drop=drop):
        kp, conf, a, k = at(float(t))
        det.update(float(t), kp, conf, conf >= 0.3)
    fine = np.arange(0, end, 0.001)
    for start, a, k in lifts:
        seg = [(script(lifts)(float(x))[2]) for x in fine if start <= x <= start + 3]
        b = band_of(np.array(seg), BANDS)
        truth[start] = (float((b == 2).sum()) * 0.001, float((b == 3).sum()) * 0.001)
    return det, truth


class LiftTest(unittest.TestCase):
    def test_stoop_and_squat(self):
        lifts = [(2, 70, 175), (8, 30, 90), (14, 75, 170), (20, 35, 85)]
        det, truth = run(lifts, 26)
        self.assertEqual([x.kind for x in det.lifts], ["stoop", "squat", "stoop", "squat"])
        for x, (start, a, k) in zip(det.lifts, lifts):
            self.assertAlmostEqual(x.peak_trunk_deg, a, delta=2)
            self.assertAlmostEqual(x.knee_deg, k, delta=2)
            b2, b3 = truth[start]
            self.assertAlmostEqual(x.b2_s, b2, delta=0.12)       # one frame at about 10 fps
            self.assertAlmostEqual(x.b3_s, b3, delta=0.12)
        self.assertAlmostEqual(det.lifts[1].b2_s + det.lifts[1].b3_s, 0.0, delta=1e-9)   # squat stays below 45
        cmp_ = det.comparison()
        self.assertEqual((cmp_["stoop"]["n"], cmp_["squat"]["n"]), (2, 2))
        self.assertAlmostEqual(cmp_["stoop"]["peak_trunk_deg"], 72.5, delta=2)
        self.assertAlmostEqual(cmp_["squat"]["knee_deg"], 87.5, delta=2)

    def test_lift_times(self):
        det, _ = run([(2, 70, 175)], 7)
        x = det.lifts[0]
        # trunk passes 45 deg at 2 + (45 - 5) / 65 s on the way down and the same before the end of the rise
        self.assertAlmostEqual(x.start_s, 2 + 40 / 65, delta=0.15)
        self.assertAlmostEqual(x.end_s, 5 - 40 / 65, delta=0.15)

    def test_comparison_needs_both_types(self):
        det, _ = run([(2, 70, 175), (8, 72, 172)], 13)
        self.assertEqual(len(det.lifts), 2)
        self.assertIsNone(det.comparison())

    def test_small_bends_are_not_lifts(self):
        det, _ = run([(2, 35, 150), (8, 40, 135)], 13)
        self.assertEqual(det.lifts, [])

    def test_with_dropped_frames(self):
        det, _ = run([(2, 70, 175), (8, 30, 90)], 13, seed=5, drop=0.2)
        self.assertEqual([x.kind for x in det.lifts], ["stoop", "squat"])

    def test_abort_ends_a_lift(self):
        det = LiftDetector(load_template("floor_pick").params, BANDS, gap_s=0.5)
        for t in np.arange(0, 3, 0.1):
            kp, conf = side_body(70 if t > 1 else 5)
            det.update(float(t), kp, conf, conf >= 0.3)
        x = det.abort(3.0)
        self.assertEqual((x.kind, x.end_s), ("stoop", 3.0))


if __name__ == "__main__":
    unittest.main()
