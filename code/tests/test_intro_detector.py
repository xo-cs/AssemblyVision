"""TC-INT: the motion energy intro and outro detector (seg_a.intro_detect, Week 11)."""
import unittest

import numpy as np

from assemblyvision.analysis.segmentation.base import RunTimeline
from assemblyvision.analysis.segmentation.rules import RuleBasedEngine
from assemblyvision.config import Config

FPS, N = 30.0, 30 * 200                     # 200 s


def cfg(**seg_a):
    return Config.load(overrides={"steps": {"use": True, "step": [{"name": "A"}, {"name": "B"}]},
                                  "seg_a": {"engine": "A", "intro_detect": True, **seg_a}})


def timeline(intro_s=0.0, outro_s=0.0, seed=0) -> RunTimeline:
    rng = np.random.default_rng(seed)
    e = np.abs(rng.normal(0.5, 0.15, N)).astype(np.float32)          # working
    e[:int(intro_s * FPS)] = np.abs(rng.normal(0.02, 0.01, int(intro_s * FPS)))
    if outro_s:
        e[N - int(outro_s * FPS):] = np.abs(rng.normal(0.02, 0.01, int(outro_s * FPS)))
    return RunTimeline(FPS, N, np.ones(N, bool), e, [])


class IntroDetectorTest(unittest.TestCase):
    def test_still_intro_and_outro_become_other(self):
        e = RuleBasedEngine(cfg())
        steps = e.segment_run(timeline(intro_s=20.0, outro_s=15.0))
        rep = e.report["intro"]
        self.assertTrue(rep["found"])
        self.assertAlmostEqual(rep["intro_s"], 20.0, delta=1.5)
        self.assertAlmostEqual(rep["outro_s"], 15.0, delta=1.5)
        self.assertEqual((steps[0].label, steps[-1].label), ("other", "other"))
        self.assertAlmostEqual(steps[1].start_s, 20.0, delta=1.5)

    def test_plain_video_unchanged(self):
        tl = timeline()
        e = RuleBasedEngine(cfg())
        a = e.segment_run(tl)
        self.assertFalse(e.report["intro"]["found"])
        self.assertEqual(a, RuleBasedEngine(cfg(intro_detect=False)).segment_run(timeline()))

    def test_limits(self):
        e = RuleBasedEngine(cfg(intro_min_s=3.0))
        e.segment_run(timeline(intro_s=2.0))                          # shorter than intro_min_s
        self.assertEqual(e.report["intro"]["intro_s"], 0.0)
        e = RuleBasedEngine(cfg(intro_max_share=0.1))
        e.segment_run(timeline(intro_s=40.0))                         # longer than 10 % of 200 s
        self.assertEqual(e.report["intro"]["intro_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
