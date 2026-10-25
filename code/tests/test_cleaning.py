"""SignalCleaner and the NumPy Savitzky Golay filter (CPU)."""
import unittest

import numpy as np

from assemblyvision.analysis.cleaning import SignalCleaner, features, savgol, savgol_coeffs
from assemblyvision.config import Config
from assemblyvision.kts import empty_series

FPS = 30.0


def standing_series(T=90, conf=0.9):
    s = empty_series("run", "shot", "shot-t01", FPS, 640, 480, 0, T)
    tmpl = np.array([[.5, .08], [.45, .06], [.55, .06], [.4, .08], [.6, .08], [.3, .25], [.7, .25],
                     [.25, .42], [.75, .42], [.22, .58], [.78, .58], [.38, .58], [.62, .58],
                     [.38, .78], [.62, .78], [.38, .97], [.62, .97]], np.float32)
    s.present[:] = True
    s.body_xy[:] = tmpl * [60, 160] + [100, 100]
    s.body_conf[:] = conf
    s.bbox_xyxy[:] = [100, 100, 160, 260]
    return s


class SavgolTest(unittest.TestCase):
    def test_known_kernel(self):
        # 5 point quadratic Savitzky Golay kernel: [-3, 12, 17, 12, -3] / 35
        np.testing.assert_allclose(savgol_coeffs(5, 2)[2], np.array([-3, 12, 17, 12, -3]) / 35, atol=1e-12)

    def test_quadratic_kept_exactly_also_at_the_ends(self):
        x = np.arange(40, dtype=np.float64)
        y = 0.3 * x ** 2 - 2 * x + 5
        np.testing.assert_allclose(savgol(y, 11, 2), y, atol=1e-8)

    def test_short_run_left_alone(self):
        y = np.array([1.0, 5.0])
        np.testing.assert_array_equal(savgol(y, 11, 2), y)


class CleanerTest(unittest.TestCase):
    def setUp(self):
        self.cleaner = SignalCleaner(Config.load())

    def test_gaps(self):   # TC-CLN-02
        s = standing_series(T=120)
        s.body_xy[:, :, 0] += np.arange(120)[:, None]            # walking right, 1 px per frame
        s.present[20:32] = False                                  # 0.4 s gap
        s.present[60:84] = False                                  # 0.8 s gap
        s.body_xy[~s.present] = np.nan
        cs = self.cleaner.clean(s)
        self.assertTrue(cs.valid[20:32].all())
        self.assertFalse(cs.measured[20:32].any())
        np.testing.assert_allclose(cs.xy[25, 0, 0], s.body_xy[19, 0, 0] + 6, atol=1e-3)
        self.assertFalse(cs.valid[60:84].any())
        self.assertTrue(np.isnan(cs.xy[60:84]).all())

    def test_low_confidence_masked(self):
        s = standing_series()
        s.body_conf[:, 0] = 0.2
        cs = self.cleaner.clean(s)
        self.assertFalse(cs.measured[:, 0].any())
        self.assertTrue(np.isnan(cs.xy[:, 0]).all())

    def test_torso_and_energy(self):
        s = standing_series()
        cs = self.cleaner.clean(s)
        self.assertAlmostEqual(cs.torso, 0.33 * 160, places=3)
        e = features(cs)["energy"]
        self.assertTrue(np.isnan(e[0]))
        self.assertLess(np.nanmax(e), 1e-4)                      # standing still
        s.body_xy[:, :, 0] += (np.arange(90) * 2.0)[:, None]      # 60 px/s = 1.136 torso/s
        e = features(self.cleaner.clean(s))["energy"]
        np.testing.assert_allclose(np.nanmedian(e), 60 / (0.33 * 160), rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
