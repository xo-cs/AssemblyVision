"""TC-EBD: Engine B decoder options (Week 10): change score peaks as candidates,
boundary refinement, per video normalisation and the optional intro and outro."""
import unittest

import numpy as np

from assemblyvision.analysis.segmentation.base import RunTimeline
from assemblyvision.analysis.segmentation.boundary_tcn import normalize_video
from assemblyvision.analysis.segmentation.rules import RuleBasedEngine
from assemblyvision.config import Config

FPS, N, CUT = 30.0, 3600, 1800          # 120 s, the phase change at 60 s


def cfg(**seg_a):
    eb = {"video_norm": seg_a.pop("video_norm", False)}
    over = {"steps": {"use": True, "step": [{"name": "A"}, {"name": "B"}]},
            "seg_a": {"engine": "custom", "embed_weight": 0.4, "embed_window_s": 10.0, **seg_a}, "engine_b": eb}
    return Config.load(overrides=over)


def timeline(still_s: float = 0.0, peak_at: int = CUT + 60) -> RunTimeline:
    """Constant motion (no pause evidence), an embedding that switches at CUT, a boundary
    head peak at peak_at; optionally a still intro (its own embedding) of still_s."""
    rng = np.random.default_rng(0)
    emb = np.zeros((N, 8), np.float32)
    emb[:CUT, 0], emb[CUT:, 1] = 1.0, 1.0
    intro = int(still_s * FPS)
    emb[:intro] = 0.0
    emb[:intro, 2] = 1.0
    emb += rng.normal(0, 0.02, emb.shape).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    prob = np.full(N, 0.05, np.float32)
    prob[peak_at] = 0.9
    tl = RunTimeline(FPS, N, np.ones(N, bool), np.full(N, 0.5, np.float32), [])
    tl.boundary_prob, tl.embedding, tl.embedding_valid = prob, emb, np.ones(N, bool)
    return tl


def bounds(steps):
    return [s.start_s for s in steps if s.label != "other"][1:]


class DecoderTest(unittest.TestCase):
    def test_without_candidates_the_v1_decoder_has_nothing_to_choose(self):
        e = RuleBasedEngine(cfg())
        e.segment_run(timeline())
        self.assertNotEqual(e.report["solver_used"], "dp")

    def test_change_score_peaks_place_the_boundary(self):
        e = RuleBasedEngine(cfg(embed_candidates=True))
        steps = e.segment_run(timeline())
        self.assertEqual(e.report["solver_used"], "dp")
        self.assertAlmostEqual(bounds(steps)[0], CUT / FPS, delta=1.0)
        self.assertGreater(e.report["evidence_counts"]["embed"], 0)

    def test_refinement_moves_to_the_boundary_head_peak(self):
        e = RuleBasedEngine(cfg(embed_candidates=True, refine_window_s=5.0))
        steps = e.segment_run(timeline(peak_at=CUT + 60))
        self.assertAlmostEqual(bounds(steps)[0], (CUT + 60) / FPS, places=2)
        self.assertEqual(len(e.report["refined"]), 1)
        e = RuleBasedEngine(cfg(embed_candidates=True, refine_window_s=1.0))     # peak outside the window
        self.assertAlmostEqual(bounds(e.segment_run(timeline(peak_at=CUT + 60)))[0], CUT / FPS, delta=1.0)

    def test_video_norm(self):
        rng = np.random.default_rng(1)
        e = rng.normal(size=(50, 4)) + np.array([5.0, 0, 0, 0])
        v = np.ones(50, bool)
        v[:5] = False
        z = normalize_video(e, v)
        np.testing.assert_allclose(np.linalg.norm(z[v], axis=1), 1.0, atol=1e-5)
        self.assertTrue((z[~v] == 0).all())
        self.assertLess(abs(float(z[v][:, 0].mean())), 0.3)          # the common offset is gone
        steps = RuleBasedEngine(cfg(embed_candidates=True, video_norm=True)).segment_run(timeline())
        self.assertAlmostEqual(bounds(steps)[0], CUT / FPS, delta=1.0)

    def test_other_ends_finds_a_still_intro_and_no_false_one(self):
        c = cfg(embed_candidates=True, other_ends=True, other_min_strength=0.15)
        steps = RuleBasedEngine(c).segment_run(timeline(still_s=20.0))
        self.assertEqual(steps[0].label, "other")
        self.assertAlmostEqual(steps[0].end_s, 20.0, delta=1.0)
        e = RuleBasedEngine(c)
        steps = e.segment_run(timeline())
        self.assertEqual(e.report["phases_found"], 2)
        self.assertLess(steps[0].end_s - steps[0].start_s if steps[0].label == "other" else 0.0, 1.5)


if __name__ == "__main__":
    unittest.main()
