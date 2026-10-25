"""Streaming motion metrics for Live: past frames only, same results as batch (CPU)."""
import unittest

import numpy as np

from assemblyvision.analysis.motion import MotionAnalyzer
from assemblyvision.analysis.motion_stream import (BendingStream, IdleStream, RepetitionStream, StreamingMotion,
                                                   TwoHandStream)
from assemblyvision.types import Step
from motion_helpers import FPS, cfg, default_wrists, pose_series, shot_for, t_axis


def feed(stream, series, times=None, rows=None):
    times = t_axis(len(series.present)) if times is None else times
    rows = range(len(times)) if rows is None else rows
    out = None
    for i, t in zip(rows, times):
        out = stream.update(float(t), series.body_xy[i], series.body_conf[i])
    return out


def sine_wrists(T, period, both=False):
    w = default_wrists(T)
    x = 30 * np.sin(2 * np.pi * t_axis(T) / period)
    w[:, 0, 0] += x
    if both:
        w[:, 1, 0] += x
    return w


class PastOnlyTest(unittest.TestCase):
    def test_later_frames_do_not_change_earlier_output(self):
        T = int(12 * FPS)
        a = pose_series(T, angle=np.linspace(0, 70, T), wrists=sine_wrists(T, 0.8))
        b = pose_series(T, angle=np.linspace(0, 70, T), wrists=sine_wrists(T, 0.8))
        k = int(8 * FPS)
        b.body_xy[k + 1:] += 50.0                                  # the futures differ after frame k
        sa, sb = StreamingMotion(cfg()), StreamingMotion(cfg())
        t = t_axis(T)
        for i in range(k + 1):
            ra = sa.update(t[i], a.body_xy[i], a.body_conf[i])
            rb = sb.update(t[i], b.body_xy[i], b.body_conf[i])
            self.assertEqual(ra, rb)
        self.assertEqual(sa.summary(), sb.summary())


class StreamVsBatchTest(unittest.TestCase):
    def test_bands_match_batch(self):   # TC-MOT-04, streamed
        T = int(6 * FPS)
        s = pose_series(T, angle=np.where(t_axis(T) < 4.0, 30.0, 50.0))
        st = BendingStream(cfg())
        out = feed(st, s)
        self.assertEqual(out["band"], 2)
        an = MotionAnalyzer(cfg(enabled=["back_bending"]))
        res, _ = an.analyse("run", [an.subject(s, shot_for(s))], [Step("W", 0, T / FPS, "e")], FPS)
        batch = {r.metric_name: r.value for r in res[0][1]}
        live = st.summary()
        # each within one frame of the truth (B1 4 s, B2 2 s): the stream's first frame adds no time,
        # the batch smoothing may move the switch by one frame
        for name, truth in (("bend_b1_s", 4.0), ("bend_b2_s", 2.0)):
            self.assertAlmostEqual(live[name], truth, delta=1 / FPS + 1e-3)
            self.assertAlmostEqual(batch[name], truth, delta=1 / FPS + 1e-6)

    def test_idle_three_seconds(self):   # TC-MOT-10, streamed
        T = int(10 * FPS)
        t = t_axis(T)
        clock = np.where(t < 4, t, np.where(t < 7, 4.0, t - 3))
        s = pose_series(T, sway=20 * np.sin(2 * np.pi * clock))
        st = IdleStream(cfg())
        feed(st, s)
        self.assertAlmostEqual(st.summary()["idle_time_s"], 3.0, delta=0.1)
        self.assertAlmostEqual(st.summary()["idle_share"], 0.3, delta=0.02)

    def test_idle_counts_only_after_min(self):
        T = int(3 * FPS)
        st = IdleStream(cfg())
        s = pose_series(T)                                         # still all the time
        states = [st.update(float(t), s.body_xy[i], s.body_conf[i]) for i, t in enumerate(t_axis(T))]
        first = next(i for i, r in enumerate(states) if r["idle_now"])
        self.assertAlmostEqual(first / FPS, 1.0 + 1 / FPS, delta=1 / FPS + 1e-9)
        self.assertAlmostEqual(states[first]["idle_s"], 1.0, delta=1 / FPS + 1e-9)

    def test_two_hand_same_motion(self):
        T = int(10 * FPS)
        st = TwoHandStream(cfg())
        feed(st, pose_series(T, wrists=sine_wrists(T, 1.0, both=True)))
        r = st.summary()
        self.assertEqual((r["two_hand_simultaneous"], r["two_hand_use_ratio"]), (1.0, 1.0))
        self.assertAlmostEqual(r["two_hand_speed_corr"], 1.0, places=4)


class RepetitionStreamTest(unittest.TestCase):
    def test_period_after_window(self):   # TC-MOT-06, streamed
        T = int(15 * FPS)
        st = RepetitionStream(cfg())
        out = feed(st, pose_series(T, wrists=sine_wrists(T, 1.0)))
        self.assertTrue(out["periodic"])
        self.assertAlmostEqual(out["period_s"], 1.0, delta=0.1)

    def test_irregular_ten_fps_with_drops(self):
        T = int(15 * FPS)
        s = pose_series(T, wrists=sine_wrists(T, 0.8))
        rng = np.random.default_rng(3)
        rows = np.arange(0, T, 3)                                  # about 10 fps ...
        rows = np.unique(np.clip(rows + rng.integers(-1, 2, len(rows)), 0, T - 1))
        rows = rows[rng.random(len(rows)) > 0.1]                   # ... and 10 percent dropped
        st = RepetitionStream(cfg())
        out = feed(st, s, times=rows / FPS, rows=rows)
        self.assertTrue(out["periodic"])
        self.assertAlmostEqual(out["period_s"], 0.8, delta=0.1)

    def test_still_not_periodic(self):
        T = int(12 * FPS)
        st = RepetitionStream(cfg())
        out = feed(st, pose_series(T))
        self.assertFalse(out["periodic"])
        self.assertIsNone(out["period_s"])


class GapTest(unittest.TestCase):
    def test_gap_adds_no_time(self):
        T = int(4 * FPS)
        s = pose_series(T, angle=30.0)
        st = BendingStream(cfg())
        t = t_axis(T)
        t[int(2 * FPS):] += 2.0                                    # 2 s without frames
        feed(st, s, times=t)
        self.assertAlmostEqual(st.summary()["bend_b1_s"], (T - 2) / FPS, delta=1e-3)   # rounded to ms

    def test_missing_keypoints(self):
        st = StreamingMotion(cfg())
        r = st.update(0.0, np.full((17, 2), np.nan), np.zeros(17))
        self.assertIsNone(r["bending"]["angle_deg"])
        self.assertIsNone(r["idle"]["energy"])
        self.assertIsNone(r["two_hand"]["active"])


if __name__ == "__main__":
    unittest.main()
