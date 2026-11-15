"""TC-SRC: the step source records the effective step engine (Week 11, UI bug of 2026-11-15: a run
with engine B showed "Engine a" and "Phases found by Engine A"). CPU only."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from assemblyvision.analysis.segmentation.base import source_code, source_label
from assemblyvision.config import Config
from assemblyvision.types import Step


class _Engine:
    def __init__(self, effective):
        self.report = {"engine": effective}
        self.cleaner = None
        self.vocabulary = ["A", "B"]

    def segment_run(self, tl):
        return [Step("A", 0.0, 5.0, "engine_a"), Step("B", 5.0, 10.0, "engine_a")]


class _Ctx:
    def __init__(self):
        self.notices = []

    def notice(self, err, stage):
        self.notices.append(err.code)


def segment(cfg, effective, missing=()):
    from assemblyvision.tools import run_pose as R
    tl = mock.Mock(n_frames=100, fps=10.0)
    engines = iter(effective)
    with mock.patch("assemblyvision.analysis.segmentation.build_timeline", return_value=tl), \
            mock.patch("assemblyvision.analysis.segmentation.create_engine", side_effect=lambda *a: _Engine(next(engines))), \
            mock.patch("assemblyvision.analysis.siglip.run_phase_scores", return_value=({}, list(missing))), \
            mock.patch.object(R, "boundary_probability", return_value={"prob": None}):
        ctx = _Ctx()
        engine, steps = R.segment_stored({}, Path("."), cfg, ctx)
    return {s.source for s in steps}, engine.report["source"], ctx.notices


class StepSourceTest(unittest.TestCase):
    def test_codes_and_labels(self):
        self.assertEqual(source_code("B zero shot"), "engine_b")
        self.assertEqual(source_code("A"), "engine_a")
        self.assertEqual(source_code("A", "A"), "engine_a")
        self.assertEqual(source_code("A", "B zero shot"), "engine_a:fallback:AV-SEG-001")
        self.assertEqual(source_code("B v1", notice="AV-SEG-006"), "engine_b_v1:fallback:AV-SEG-006")
        self.assertEqual(source_label("engine_b"), "Engine B")
        self.assertEqual(source_label("engine_a"), "Engine A (rules)")                 # old runs keep their label
        self.assertEqual(source_label("engine_b_v1:fallback:AV-SEG-006"), "Engine B v1, fallback from B (AV-SEG-006)")
        self.assertEqual(source_label("engine_a:fallback:AV-SEG-001"), "Engine A (rules), fallback from B (AV-SEG-001)")
        self.assertEqual(source_label("corrected"), "Corrected")

    def test_segmentation_writes_the_effective_engine(self):
        b = Config.load()                                                              # the default, B zero shot
        self.assertEqual(segment(b, ["B zero shot"])[:2], ({"engine_b"}, "engine_b"))
        src, rep, notices = segment(b, ["B zero shot", "B v1"], missing=["vlm.npz"])   # GPU 1 busy: falls back
        self.assertEqual((src, rep, notices), ({"engine_b_v1:fallback:AV-SEG-006"}, "engine_b_v1:fallback:AV-SEG-006",
                                               ["AV-SEG-006"]))
        self.assertEqual(segment(b, ["A"])[0], {"engine_a:fallback:AV-SEG-001"})      # no Engine B weights
        a = Config.load(overrides={"seg_a": {"engine": "A"}})
        self.assertEqual(segment(a, ["A"])[0], {"engine_a"})

    def test_page_and_report_show_it(self):
        from assemblyvision.output import report as R
        from assemblyvision.ui.app import GradioApp
        from test_report_export import make_run
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = Config.load(overrides={"paths": {"public_root": str(root / "public"), "qisens_root": str(root / "q")}})
            app = GradioApp(cfg, ["public"], repo_root=root)
            runs = {}
            for rid, src in (("run_20260101_000000_aaaa", "engine_b"), ("run_20260101_000000_bbbb", "engine_a")):
                run = root / "public" / "runs" / rid
                run.mkdir(parents=True)
                make_run(run)
                with open(run / "steps.csv", "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["step", "label", "start_s", "end_s", "source", "score"])
                    w.writerow([1, "A", 0.0, 10.0, src, 0.5])
                    w.writerow([2, "B", 10.0, 20.0, src, 0.5])
                runs[src] = rid
            html = app.steps_html(runs["engine_b"])
            self.assertIn("Phases found by Engine B,", html)
            self.assertIn(">Engine B<", html)
            self.assertNotIn("Engine a", html)
            self.assertIn("Phases found by Engine A (rules),", app.steps_html(runs["engine_a"]))   # an old run
            self.assertEqual(app.store.get_segments(runs["engine_b"])[0]["source"], "engine_b")  # through SQLite
            rep = R.build(root / "public" / "runs" / runs["engine_b"])
            self.assertEqual(rep["effective_engine"], "Engine B")
            self.assertEqual({s["source"] for s in rep["steps"]}, {"engine_b"})


if __name__ == "__main__":
    unittest.main()
