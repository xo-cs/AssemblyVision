"""Engine A on synthetic timelines and KTS series (CPU)."""
import json
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.analysis.segmentation import RunTimeline, check_contract, create_engine
from assemblyvision.analysis.segmentation.rules import RuleBasedEngine
from assemblyvision.config import Config
from assemblyvision.errors import AvError
from assemblyvision.kts import empty_series, write_kts
from assemblyvision.types import Shot, Step
from helpers import fake_root, make_ctx

FPS = 30.0
PHASES = ["P1", "P2", "P3", "P4", "P5"]


def cfg_with(steps=None, use=True, **seg_a):
    over = {"steps": {"use": use, "step": [{"name": n} for n in PHASES] if steps is None else steps}}
    if seg_a:
        over["seg_a"] = seg_a
    return Config.load(overrides=over)


class TL:
    """Build a timeline in seconds: person everywhere, moving, no cuts."""

    def __init__(self, seconds):
        self.n = int(round(seconds * FPS))
        self.person = np.ones(self.n, bool)
        self.energy = np.ones(self.n, np.float32)
        self.cuts = []

    def f(self, t):
        return int(round(t * FPS))

    def empty(self, a, b):
        self.person[self.f(a):self.f(b)] = False
        self.energy[self.f(a):self.f(b)] = np.nan
        return self

    def still(self, a, b):
        self.energy[self.f(a):self.f(b)] = 0.0
        return self

    def cut(self, *ts):
        self.cuts += [self.f(t) for t in ts]
        return self

    def build(self):
        return RunTimeline(FPS, self.n, self.person.copy(), self.energy.copy(), sorted(self.cuts))


def five_phase_video(gaps=(100, 203, 306, 409), length=512, gap=3.0):
    """Phases of about 100 s separated by 3 s without a person, a cut every 20 s."""
    tl = TL(length)
    for g in gaps:
        tl.empty(g, g + gap)
    tl.cut(*[t for t in range(20, length, 20)])
    return tl


def phases_of(steps):
    return [(s.label, s.start_s, s.end_s) for s in steps if s.label != "other"]


class PhaseModeTest(unittest.TestCase):
    def run_engine(self, tl, **kw):
        eng = RuleBasedEngine(cfg_with(**kw))
        steps = eng.segment_run(tl.build() if isinstance(tl, TL) else tl)
        return eng, steps

    def assert_bounds(self, steps, expected, tol=1 / FPS + 1e-6):
        got = phases_of(steps)
        self.assertEqual([g[0] for g in got], [e[0] for e in expected])
        for (lab, a, b), (_, ea, eb) in zip(got, expected):
            self.assertAlmostEqual(a, ea, delta=tol, msg=lab)
            self.assertAlmostEqual(b, eb, delta=tol, msg=lab)

    def test_five_phases_no_person_gaps(self):
        for solver in ("dp", "forward"):
            eng, steps = self.run_engine(five_phase_video(), phase_solver=solver)
            self.assert_bounds(steps, [("P1", 0, 100), ("P2", 103, 203), ("P3", 206, 306), ("P4", 309, 409),
                                       ("P5", 412, 512)])
            self.assertEqual(eng.report["phases_found"], 5)
            others = [s for s in steps if s.label == "other"]
            self.assertEqual(len(others), 4)                       # the four gaps

    def test_derived_limits_recorded(self):
        eng, _ = self.run_engine(five_phase_video())
        T = eng.report["active"]["T_s"]
        self.assertAlmostEqual(T, 512.0)
        lim = eng.report["limits"][0]
        self.assertEqual((lim["min_s"], lim["max_s"], lim["source"]), (round(0.25 * T / 5, 3), round(2.5 * T / 5, 3),
                                                                        "derived"))

    def test_close_ups_inside_phases(self):
        # 5 s without a person in the middle of phases 2 and 4: dp keeps the true
        # boundaries (lengths closest to T/K), the forward walk ends phase 2 early
        tl = five_phase_video()
        tl.empty(150, 155).empty(355, 360)
        eng, steps = self.run_engine(tl, phase_solver="dp")
        self.assert_bounds(steps, [("P1", 0, 100), ("P2", 103, 203), ("P3", 206, 306), ("P4", 309, 409),
                                   ("P5", 412, 512)])
        _, fw = self.run_engine(tl, phase_solver="forward")
        self.assertAlmostEqual(phases_of(fw)[1][2], 150.0, delta=0.05)

    def test_missing_gap_uses_cut_with_long_pause(self):
        tl = TL(512)
        for g in (100, 306, 409):
            tl.empty(g, g + 3)
        tl.cut(*[t for t in range(20, 512, 20) if t != 200], 203)
        tl.still(198, 206)                                          # 8 s pause around the cut at 203
        eng, steps = self.run_engine(tl)
        self.assert_bounds(steps, [("P1", 0, 100), ("P2", 103, 203), ("P3", 203, 306), ("P4", 309, 409),
                                   ("P5", 412, 512)])
        kinds = {e["kind"] for e in eng.report["evidence"] if e["used"]}
        self.assertEqual(kinds, {"no_person", "cut_pause"})

    def test_leading_and_trailing_no_person_is_other(self):
        tl = five_phase_video(gaps=(110, 213, 316, 419), length=532)
        tl.empty(0, 10).empty(522, 532)
        _, steps = self.run_engine(tl)
        self.assertEqual((steps[0].label, steps[0].start_s, steps[0].end_s), ("other", 0.0, 10.0))
        self.assertEqual((steps[-1].label, steps[-1].end_s), ("other", 532.0))
        self.assertAlmostEqual(phases_of(steps)[0][1], 10.0)

    def test_order_never_backwards_and_deterministic(self):
        tl = five_phase_video()
        _, a = self.run_engine(tl)
        _, b = self.run_engine(tl)
        self.assertEqual(a, b)
        labels = [s.label for s in a if s.label != "other"]
        self.assertEqual(labels, sorted(labels, key=PHASES.index))
        self.assertEqual(len(set(labels)), len(labels))

    def test_force_close_at_max(self):   # TC-SEG-07
        steps = [{"name": "P1", "min_s": 10, "max_s": 60}, {"name": "P2"}]
        for solver in ("dp", "forward"):
            eng, out = self.run_engine(TL(200), steps=steps, phase_solver=solver)
            ph = [s for s in out if s.label != "other"]
            self.assertEqual((ph[0].label, ph[0].end_s, ph[0].score), ("P1", 60.0, 0.0))
            self.assertEqual((ph[1].start_s, ph[1].end_s), (60.0, 200.0))
            if solver == "dp":
                self.assertTrue(eng.report["solver_used"].startswith("forward"))

    def test_steps_toml_limits_override(self):
        steps = [{"name": "P1", "min_s": 120}, {"name": "P2"}]
        tl = TL(300).empty(50, 53).empty(150, 153)
        eng, out = self.run_engine(tl, steps=steps)
        self.assertEqual(eng.report["limits"][0]["source"], "steps.toml")
        self.assertAlmostEqual(phases_of(out)[0][2], 150.0)          # the gap at 50 s is too early

    def test_two_step_vocabulary_boundaries(self):   # TC-SEG-05
        tl = TL(120).empty(58, 61)
        _, out = self.run_engine(tl, steps=[{"name": "A"}, {"name": "B"}])
        self.assert_bounds(out, [("A", 0, 58), ("B", 61, 120)], tol=0.2)

    def test_durations_add_up(self):   # TC-SEG-10
        tl = five_phase_video()
        tl.empty(150, 155)
        built = tl.build()
        _, out = self.run_engine(built)
        self.assertAlmostEqual(sum(s.end_s - s.start_s for s in out), built.duration_s, delta=1 / FPS)

    def test_no_person_at_all(self):
        tl = TL(10)
        tl.person[:] = False
        _, out = self.run_engine(tl)
        self.assertEqual([(s.label, s.start_s, s.end_s) for s in out], [("other", 0.0, 10.0)])


def moving_series(T, still=(), shot_start=0):
    """Standing worker whose joints sway (about 0.8 torso lengths/s), still in the given (a, b) seconds."""
    s = empty_series("run", "shot", "shot-t01", FPS, 640, 480, shot_start, T)
    tmpl = np.array([[.5, .08], [.45, .06], [.55, .06], [.4, .08], [.6, .08], [.3, .25], [.7, .25],
                     [.25, .42], [.75, .42], [.22, .58], [.78, .58], [.38, .58], [.62, .58],
                     [.38, .78], [.62, .78], [.38, .97], [.62, .97]], np.float32)
    t = np.arange(T) / FPS
    amp = np.full(T, 8.0)
    for a, b in still:
        amp[(t >= a) & (t < b)] = 0.0
    phase = np.cumsum(amp * 2 * np.pi * 0.8 / FPS)
    s.present[:] = True
    s.body_xy[:] = tmpl * [60, 160] + [100, 100]
    s.body_xy[:, :, 0] += np.sin(phase)[:, None] * 8.0
    s.body_conf[:] = 0.9
    s.bbox_xyxy[:] = [100, 100, 160, 260]
    return s


class FallbackTest(unittest.TestCase):
    def test_three_still_periods(self):   # TC-SEG-04
        stills = [(10, 13), (25, 28), (40, 43)]
        series = moving_series(int(55 * FPS), still=stills)
        shot = Shot("run-s001", "run", 0, len(series.present), 0.0, 55.0, "fixed")
        eng = create_engine("A", cfg_with(use=False))
        out = eng.segment(series, shot)
        self.assertEqual([s.label for s in out], ["segment_1", "segment_2", "segment_3", "segment_4"])
        for s, (a, b) in zip(out, stills):
            self.assertAlmostEqual(s.end_s, (a + b) / 2, delta=0.2)
        self.assertEqual(eng.report["mode"], "fallback")

    def test_notice_without_vocabulary(self):
        tmp = fake_root()
        try:
            cfg, ctx = make_ctx(Path(tmp.name))
            eng = create_engine("A", cfg_with(steps=[]), ctx)
            eng.segment_run(TL(30).build())
            self.assertIn("AV-SEG-003", [n["code"] for n in ctx.manifest["notices"]])
        finally:
            tmp.cleanup()


class InterfaceTest(unittest.TestCase):
    def test_engine_b_falls_back(self):   # TC-SEG-02
        tmp = fake_root()
        try:
            cfg, ctx = make_ctx(Path(tmp.name))
            eng = create_engine("B", cfg, ctx)
            self.assertIsInstance(eng, RuleBasedEngine)
            self.assertIn("AV-SEG-001", [n["code"] for n in ctx.manifest["notices"]])
        finally:
            tmp.cleanup()

    def test_contract(self):   # TC-SEG-01
        tl = TL(10).build()
        ok = [Step("A", 0, 4, "engine_a"), Step("other", 4, 6, "engine_a"), Step("B", 6, 10, "engine_a")]
        check_contract(ok, tl, ["A", "B"])
        bad_cases = [
            [Step("A", 0, 6, "e"), Step("B", 5, 10, "e")],         # overlap
            [Step("A", 0, 4, "e"), Step("B", 6, 10, "e")],         # hole
            [Step("A", 0, 10.5, "e")],                             # outside
            [Step("Z", 0, 10, "e")],                               # label
            [Step("A", 0, 0, "e"), Step("B", 0, 10, "e")],         # empty step
        ]
        for steps in bad_cases:
            with self.assertRaises(AvError) as cm:
                check_contract(steps, tl, ["A", "B"])
            self.assertEqual(cm.exception.code, "AV-SEG-002")


class StoredRunTest(unittest.TestCase):
    """build_timeline and the segmentation stage from KTS files and a manifest (no video)."""

    def test_stage_from_files(self):
        from assemblyvision.evaluation.steps_eval import evaluate_run
        from assemblyvision.tools.run_pose import stage_segmentation
        tmp = fake_root()
        try:
            _, ctx = make_ctx(Path(tmp.name), source="qisens")      # qisens keeps the steps.toml names
            cfg = cfg_with(steps=[{"name": "A"}, {"name": "B"}])
            rid = ctx.run_id
            # shot 1: worker 0 to 60 s; shot 2: 3 s title card (no track); shot 3: worker 63 to 120 s
            spans = [(0, 1800, True), (1800, 1890, False), (1890, 3600, True)]
            items, tracks = [], []
            for k, (a, b, person) in enumerate(spans, start=1):
                sid = f"{rid}-s{k:03d}"
                items.append({"shot_id": sid, "start_frame": a, "end_frame": b, "start_s": a / FPS,
                              "end_s": b / FPS, "shot_type": "fixed"})
                if person:
                    s = moving_series(b - a, shot_start=a)
                    s.run_id, s.shot_id, s.track_id = rid, sid, f"{sid}-t01"
                    path = write_kts(s, ctx.paths, {"is_subject": True, "subject_source": "auto"})
                    tracks.append({"track_id": s.track_id, "kts_path": ctx.rel(path), "is_subject": True})
            ctx.manifest.update(run_id=rid, ingest={"cfr": {"fps": FPS, "frame_count": 3600, "duration_s": 120.0, "width": 1920, "height": 1080}},
                                shots={"items": items}, tracking={"tracks": tracks}, notices=[])
            steps = stage_segmentation(ctx, cfg)
            self.assertEqual([(s.label, s.start_s, s.end_s) for s in steps],
                             [("A", 0.0, 60.0), ("other", 60.0, 63.0), ("B", 63.0, 120.0)])
            self.assertTrue(ctx.paths.steps_csv.read_text().startswith("step,label,start_s,end_s,source,score\n1,A,"))
            gt = Path(tmp.name) / "gt.csv"
            gt.write_text("step,label,start_s,end_s,source\n1,A,0,62,gt\n2,B,62,120,gt\n")
            rec = evaluate_run(ctx, gt, cfg)
            self.assertEqual(rec["metrics"]["f1_50"], 100.0)
            self.assertEqual(rec["boundaries"][0]["gt_s"], 62.0)
            self.assertEqual(rec["metrics"]["boundary_mae_s"], 1.0)      # 62 vs 63
            self.assertEqual(json.loads(ctx.paths.eval_json.read_text())["gt_file"], "gt.csv")
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
