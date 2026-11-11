"""Activity change evidence of Engine A (seg_a.activity_change), phase order
scoring of public runs and the eval label (CPU, synthetic data only)."""
import contextlib
import io
import json
from pathlib import Path
import unittest

import numpy as np

from assemblyvision.analysis.cleaning import CleanSeries
from assemblyvision.analysis.segmentation import RunTimeline
from assemblyvision.analysis.segmentation.activity import (CHANNELS, FEATURES, activity_change, empty_activity,
                                                           frame_activity, video_stats, window_features)
from assemblyvision.analysis.segmentation.rules import RuleBasedEngine
from assemblyvision.config import Config
from assemblyvision.errors import AvError
from assemblyvision.evaluation.steps_eval import by_phase_order, evaluate_run
from assemblyvision.evaluation.steps_io import read_steps_csv
from assemblyvision.types import Step
from helpers import fake_root, make_ctx
from test_segmentation import FPS, cfg_with, five_phase_video, moving_series, phases_of


def channels(n, **values):
    act = empty_activity(n)
    for k, v in values.items():
        act[k][:] = v
    return act


def activity_video(seconds, phases, rng, noise=0.05):
    """Activity channels with one level per phase [(end_s, wrist, trunk, hand_h, both, idle)]."""
    n = int(round(seconds * FPS))
    act = empty_activity(n)
    lo = 0
    for end_s, wrist, trunk, hand_h, both, idle in phases:
        hi = int(round(end_s * FPS))
        m = hi - lo
        act["wrist"][lo:hi] = np.abs(wrist + noise * rng.standard_normal(m))
        act["trunk"][lo:hi] = trunk + 20 * noise * rng.standard_normal(m)
        act["hand_h"][lo:hi] = hand_h + noise * rng.standard_normal(m)
        act["both"][lo:hi] = rng.random(m) < both
        act["idle"][lo:hi] = rng.random(m) < idle
        lo = hi
    return act


class WindowFeatureTest(unittest.TestCase):
    def test_known_values(self):
        act = channels(4, trunk=30.0, hand_h=-0.5)
        act["wrist"][:] = [1.0, 3.0, 1.0, 3.0]
        act["both"][:] = [0, 1, 0, 1]
        act["idle"][:] = [1, 1, 1, 0]
        f = dict(zip(FEATURES, window_features(act, 0, 4)))
        self.assertAlmostEqual(f["wrist_mean"], 2.0)
        self.assertAlmostEqual(f["wrist_spread"], 1.0)
        self.assertAlmostEqual(f["trunk_mean"], 30.0)
        self.assertAlmostEqual(f["hand_height"], -0.5)
        self.assertAlmostEqual(f["both_share"], 0.5)
        self.assertAlmostEqual(f["idle_share"], 0.75)

    def test_low_coverage_is_nan(self):
        act = channels(100, trunk=10.0)
        act["wrist"][:10] = 1.0                       # 10 % of the window < MIN_COVERAGE
        f = window_features(act, 0, 100)
        self.assertTrue(np.isnan(f[0]) and np.isnan(f[1]))
        self.assertAlmostEqual(f[2], 10.0)
        self.assertTrue(np.isnan(window_features(act, 50, 50)).all())   # empty window

    def test_frame_activity_channels(self):
        T = 20
        xy = np.zeros((T, 17, 2), np.float32)
        xy[:, 5], xy[:, 6] = (90, 0), (110, 0)        # shoulders straight above the hips: trunk 0 deg
        xy[:, 11], xy[:, 12] = (90, 100), (110, 100)  # hips, torso 100 px
        xy[:, 9], xy[:, 10] = (80, 50), (120, 50)     # wrists half a torso above the hips
        xy[:, 9, 0] += np.arange(T) * 10.0            # left wrist moves 10 px/frame = 3 torso/s
        cs = CleanSeries(xy, np.ones((T, 17), bool), np.ones((T, 17), bool), 100.0, FPS, np.arange(T, dtype=np.int32))
        energy = np.r_[np.full(10, 0.0), np.full(10, 1.0)].astype(np.float32)
        act = frame_activity(cs, energy, hand_active=0.3, idle_energy=0.15)
        self.assertEqual(set(act), set(CHANNELS))
        self.assertAlmostEqual(float(np.nanmean(act["hand_h"])), 0.5, places=5)
        self.assertAlmostEqual(float(np.nanmax(act["trunk"])), 0.0, places=5)
        self.assertAlmostEqual(float(act["wrist"][5]), 1.5, places=4)    # mean of 3 and 0
        self.assertEqual(float(np.nanmax(act["both"])), 0.0)              # the right wrist never moves
        self.assertEqual(act["idle"][:10].tolist(), [1.0] * 10)
        self.assertEqual(act["idle"][10:].tolist(), [0.0] * 10)


class NormalisationTest(unittest.TestCase):
    def test_median_and_mad(self):
        n = 600
        act = channels(n, trunk=20.0)                  # trunk never varies
        act["wrist"][:] = np.repeat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 100)
        norm = video_stats(act, 0, n, win=100, hop=100)
        # window means 1..6: median 3.5, MAD 1.5
        self.assertAlmostEqual(norm.center[0], 3.5)
        self.assertAlmostEqual(norm.scale[0], 1.4826 * 1.5)
        self.assertAlmostEqual(norm.center[2], 20.0)
        self.assertTrue(np.isinf(norm.scale[2]))
        z = norm.z(window_features(act, 0, 100))
        self.assertAlmostEqual(z[0], (1.0 - 3.5) / (1.4826 * 1.5))
        self.assertEqual(z[2], 0.0)
        # a feature that never varies cannot make a change; equal windows give 0
        self.assertEqual(activity_change(act, 50, 50, 0, n, 50, norm), 0.0)
        self.assertGreater(activity_change(act, 100, 100, 0, n, 100, norm), 0.3)

    def test_no_activity_gives_zero(self):
        act = empty_activity(300)
        norm = video_stats(act, 0, 300, 60, 30)
        self.assertEqual(activity_change(act, 150, 150, 0, 300, 60, norm), 0.0)


class EngineTest(unittest.TestCase):
    """Three phases of 40, 90 and 50 s; a cut every 20 s plus one at 130 s. All
    cuts are equal without activity, so the solver goes by the T / K tie break
    (60 and 120 s); the activity changes at 40 and 130 s must win when on."""

    def timeline(self):
        rng = np.random.default_rng(7)
        tl = RunTimeline(FPS, int(180 * FPS), np.ones(int(180 * FPS), bool), np.ones(int(180 * FPS), np.float32),
                         sorted(int(t * FPS) for t in (20, 40, 60, 80, 100, 120, 130, 150, 170)))
        tl.activity = activity_video(180, [(40, 0.3, 10, -0.8, 0.1, 0.4),
                                           (130, 1.5, 45, 0.2, 0.7, 0.05),
                                           (180, 0.6, 20, -0.3, 0.3, 0.2)], rng)
        return tl

    def engine(self, **seg_a):
        return RuleBasedEngine(cfg_with(steps=[{"name": n} for n in ("A", "B", "C")], **seg_a))

    def test_change_detected_at_known_times(self):
        eng = self.engine(activity_change=True)
        steps = eng.segment_run(self.timeline())
        bounds = [(a, b) for _, a, b in phases_of(steps)]
        self.assertEqual(bounds, [(0.0, 40.0), (40.0, 130.0), (130.0, 180.0)])
        ev = {e["start_s"]: e["activity_change"] for e in eng.report["evidence"]}
        top2 = sorted(ev, key=ev.get, reverse=True)[:2]
        self.assertEqual(sorted(top2), [40.0, 130.0])
        self.assertLess(max(v for t, v in ev.items() if t not in (40.0, 130.0)), 0.5)
        cut = [e for e in eng.report["evidence"] if e["start_s"] == 40.0][0]
        self.assertAlmostEqual(cut["strength"], 0.2 + 0.4 * cut["activity_change"], places=3)

    def test_off_ignores_activity(self):
        steps = self.engine().segment_run(self.timeline())
        self.assertEqual([(a, b) for _, a, b in phases_of(steps)], [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0)])

    def test_off_switch_gives_the_old_result_exactly(self):
        for with_activity in (False, True):
            tl = five_phase_video().still(150, 160).cut(155).build()
            if with_activity:
                tl.activity = activity_video(512, [(250, 0.3, 10, -0.5, 0.1, 0.3), (512, 1.2, 40, 0.1, 0.6, 0.1)],
                                             np.random.default_rng(1))
            old = RuleBasedEngine(cfg_with())                    # default config: feature off
            new = RuleBasedEngine(cfg_with(activity_change=False, ac_weight=0.9, ac_window_s=3.0))
            s_old, s_new = old.segment_run(tl), new.segment_run(tl)
            self.assertEqual(s_old, s_new)
            self.assertEqual(json.dumps(old.report, sort_keys=True), json.dumps(new.report, sort_keys=True))
            self.assertNotIn("activity_change", json.dumps(new.report))

    def test_no_person_keeps_its_strength(self):
        tl = five_phase_video().build()
        tl.activity = activity_video(512, [(512, 0.5, 15, 0.0, 0.3, 0.2)], np.random.default_rng(2))
        eng = RuleBasedEngine(cfg_with(activity_change=True))
        eng.segment_run(tl)
        for e in eng.report["evidence"]:
            if e["kind"] == "no_person":
                self.assertEqual(e["strength"], 1.0)
                self.assertNotIn("activity_change", e)
            else:
                self.assertIn("activity_change", e)

    def test_config_rules(self):
        cfg = Config.load()
        self.assertIs(cfg["seg_a"]["activity_change"], False)
        self.assertEqual((cfg["seg_a"]["ac_weight"], cfg["seg_a"]["ac_window_s"]), (0.4, 10.0))
        for bad in ({"ac_window_s": 0.0}, {"ac_weight": -0.1}, {"activity_change": 1}):
            with self.assertRaises(AvError) as cm:
                Config.load(overrides={"seg_a": bad})
            self.assertEqual(cm.exception.code, "AV-CFG-002", bad)


class PhaseOrderEvalTest(unittest.TestCase):
    def test_by_phase_order(self):
        names = ["Removal", "Disassembly"]
        steps = [Step("other", 0, 1, "gt"), Step("Disassembly", 1, 2, "gt"), Step("Phase 1", 2, 3, "x")]
        self.assertEqual([s.label for s in by_phase_order(steps, names)], ["other", "Phase 2", "Phase 1"])

    def _run(self, source, label=""):
        tmp = fake_root()
        self.addCleanup(tmp.cleanup)
        cfg, ctx = make_ctx(Path(tmp.name), source)
        if label:
            cfg = Config.load(overrides={"eval": {"label": label}})
        names = [d["name"] for d in cfg["steps"]["step"]]
        pred = "step,label,start_s,end_s,source,score\n" + "".join(
            f"{k + 1},{(f'Phase {k + 1}' if source == 'public' else n)},{k * 10}.000,{k * 10 + 10}.000,engine_a,1.000\n"
            for k, n in enumerate(names))
        ctx.paths.steps_csv.write_text(pred)
        gt = ctx.paths.root / "gt.csv"
        gt.write_text("step,label,start_s,end_s,source\n" + "".join(
            f"{k + 1},{n},{k * 10}.000,{k * 10 + 10}.000,made_up\n" for k, n in enumerate(names)))
        ctx.manifest.update(ingest={"cfr": {"duration_s": 10.0 * len(names)}},
                            segmentation={"engine": "A", "solver_used": "dp",
                                          "vocabulary": [s.label for s in read_steps_csv(ctx.paths.steps_csv)]})
        return evaluate_run(ctx, gt, cfg), ctx

    def test_public_named_gt_scored_by_order(self):
        rec, _ = self._run("public")
        self.assertEqual(rec["metrics"]["frame_accuracy"], 100.0)
        self.assertEqual(rec["metrics"]["f1_50"], 100.0)
        self.assertIn("label_mapping", rec)
        self.assertNotIn("label", rec)

    def test_qisens_names_and_label(self):
        rec, ctx = self._run("qisens", label="after error analysis, not blind")
        self.assertEqual(rec["metrics"]["frame_accuracy"], 100.0)
        self.assertNotIn("label_mapping", rec)
        self.assertEqual(rec["label"], "after error analysis, not blind")
        self.assertEqual(json.loads(ctx.paths.eval_json.read_text())["label"], "after error analysis, not blind")
        self.assertEqual(ctx.manifest["eval"]["label"], "after error analysis, not blind")


class CompareModeTest(unittest.TestCase):
    """run_pose --compare: off and on in memory, nothing in the run folder written."""

    def fake_run(self, root):
        from assemblyvision.kts import write_kts
        _, ctx = make_ctx(root, source="qisens")
        rid = ctx.run_id
        spans = [(0, 1800, True), (1800, 1890, False), (1890, 3600, True)]
        items, tracks = [], []
        for k, (a, b, person) in enumerate(spans, start=1):
            sid = f"{rid}-s{k:03d}"
            items.append({"shot_id": sid, "start_frame": a, "end_frame": b, "start_s": a / FPS, "end_s": b / FPS,
                          "shot_type": "fixed"})
            if person:
                s = moving_series(b - a, shot_start=a)
                s.run_id, s.shot_id, s.track_id = rid, sid, f"{sid}-t01"
                path = write_kts(s, ctx.paths, {"is_subject": True, "subject_source": "auto"})
                tracks.append({"track_id": s.track_id, "kts_path": ctx.rel(path), "is_subject": True})
        ctx.manifest.update(run_id=rid, ingest={"cfr": {"fps": FPS, "frame_count": 3600, "duration_s": 120.0, "width": 1920, "height": 1080}},
                            shots={"items": items}, tracking={"tracks": tracks}, notices=[])
        ctx.write_manifest()
        return ctx

    @staticmethod
    def snapshot(root):
        return {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()}

    def test_off_and_on_without_writing(self):
        from assemblyvision.tools.run_pose import compare_run, format_compare
        tmp = fake_root()
        self.addCleanup(tmp.cleanup)
        ctx = self.fake_run(Path(tmp.name))
        gt = Path(tmp.name) / "data/qisens/gt.csv"
        gt.write_text("step,label,start_s,end_s,source\n1,A,0,62,gt\n2,B,62,120,gt\n")
        cfg = cfg_with(steps=[{"name": "A"}, {"name": "B"}])
        before = self.snapshot(ctx.paths.root)
        rows = compare_run(json.loads(ctx.paths.manifest.read_text()), ctx.paths.root, "qisens", cfg,
                           "seg_a.activity_change", gt)
        self.assertEqual(self.snapshot(ctx.paths.root), before)
        self.assertEqual([r["setting"] for r in rows], ["seg_a.activity_change = false", "seg_a.activity_change = true"])
        self.assertEqual(rows[0]["steps"], [("A", 0.0, 60.0), ("B", 63.0, 120.0)])
        self.assertEqual(rows[0]["metrics"]["boundary_mae_s"], 1.0)
        self.assertIn("frame acc", format_compare(rows))
        for bad in ("seg_a.ac_weight", "seg_a.nope", "nope"):
            with self.assertRaises(AvError) as cm:
                compare_run({}, ctx.paths.root, "qisens", cfg, bad, gt)
            self.assertEqual(cm.exception.code, "AV-CFG-002")

    def test_arguments(self):
        from assemblyvision.tools.run_pose import parse_args
        a = parse_args(["--run", "r", "--compare", "seg_a.activity_change", "--gt", "g.csv"])
        self.assertEqual((a.compare, a.stages), ("seg_a.activity_change", None))
        for argv in (["--run", "r", "--compare", "seg_a.activity_change"],
                     ["--run", "r", "--compare", "k.v", "--gt", "g", "--stages", "eval"],
                     ["--compare", "k.v", "--gt", "g"], ["--run", "r"]):
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    parse_args(argv)


if __name__ == "__main__":
    unittest.main()
