"""Engine B model (boundary_tcn), its use as Engine A evidence, compare values, the fixed
subject split, the noise rule and a CPU training run on a synthetic cache."""
import json
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.analysis.segmentation import RunTimeline
from assemblyvision.analysis.segmentation import boundary_tcn as bt
from assemblyvision.analysis.segmentation.rules import RuleBasedEngine
from assemblyvision.config import Config
from assemblyvision.datasets import engine_b as eb
from assemblyvision.datasets import splits
from assemblyvision.errors import AvError
from assemblyvision.evaluation.equivalence import noise_check, noise_floor
from helpers import fake_root

FPS = 30.0


class TimeBaseTest(unittest.TestCase):
    def test_sample_index_and_targets(self):
        idx = bt.sample_index(31, 30.0, 10)
        self.assertEqual(idx.tolist(), [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30])
        self.assertEqual(len(bt.sample_index(0, 30.0, 10)), 0)
        labels = np.array([[1, 0]] * 10 + [[2, 0]] * 10 + [[-1, 0]] * 5 + [[3, 0]] * 5)
        lab = labels[:, 0] >= 0
        ch = bt.change_times(labels, 10.0, False, 0, lab)
        np.testing.assert_allclose(ch, [0.95])                       # changes next to unlabelled frames dropped
        self.assertEqual(len(bt.change_times(labels, 10.0, False)), 3)
        t = np.arange(0, 2, 0.1)
        y = bt.soft_targets(t, np.array([0.95]), 0.2)
        self.assertAlmostEqual(float(y.max()), np.exp(-0.5 * (0.05 / 0.2) ** 2), places=5)
        self.assertLess(float(y[0]), 1e-4)
        self.assertEqual(bt.soft_targets(t, np.zeros(0), 0.2).sum(), 0)

    def test_peaks_f1_and_pool(self):
        p = np.zeros(50)
        p[[10, 12, 30]] = [0.9, 0.8, 0.6]
        self.assertEqual(bt.peaks(p, 0.5, 5).tolist(), [10, 30])        # 12 is within 5 of the higher 10
        self.assertEqual(bt.peaks(p, 0.7, 1).tolist(), [10, 12])
        r = bt.boundary_f1([(np.array([1.0, 3.1, 7.0]), np.array([1.1, 3.0, 5.0]))], [0.2, 0.5])
        self.assertEqual((r["0.2"]["tp"], r["0.2"]["f1"]), (2, round(2 / 3, 4)))
        self.assertEqual(bt.boundary_f1([(np.zeros(0), np.zeros(0))], [0.2])["0.2"]["f1"], 0.0)
        # one to one: two predictions near one true boundary give one hit
        self.assertEqual(bt.match_boundaries(np.array([1.0, 1.05]), np.array([1.02]), 0.2), 1)
        np.testing.assert_allclose(bt.pool(np.array([0, 0, 3, 0, 0.]), 3), [0, 1, 1, 1, 0])
        np.testing.assert_allclose(bt.pool(np.array([1, 2.]), 1), [1, 2])


class ModelTest(unittest.TestCase):
    def test_forward_save_load_predict(self):
        import torch
        torch.manual_seed(0)
        m = bt.build_model(eb.N_FEATURES, 2, 3, 8, 0.0).eval()
        x = torch.randn(2, eb.N_FEATURES, 40)
        mask = torch.ones(2, 1, 40)
        mask[1, :, 30:] = 0
        out = m(x, mask)
        self.assertEqual(tuple(out.shape), (2, 2, 40))
        self.assertTrue(torch.all(out[:, 1, 30:] == 0))               # padded frames give logit 0
        tmp = fake_root()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "m.pt"
        sha = bt.save_model(path, m, {"model_fps": 10, "stages": 2, "layers": 3, "channels": 8, "dropout": 0.0},
                            {"experiment": "t"})
        self.assertEqual(len(sha), 64)
        m2, s = bt.load_model(path)
        feats = np.random.default_rng(0).normal(size=(40, eb.N_FEATURES)).astype(np.float32)
        np.testing.assert_allclose(bt.predict(m, feats), bt.predict(m2, feats), atol=1e-6)
        self.assertEqual(s["layers"], 3)
        self.assertEqual(bt.describe(path)["extra"]["experiment"], "t")


def two_phase_cfg(**seg_a):
    seg_a.setdefault("engine", "custom")                               # these tests set the Engine B keys
    over = {"steps": {"use": True, "step": [{"name": "A"}, {"name": "B"}]}, "seg_a": seg_a}
    over["engine_b"] = {"pool_s": seg_a.pop("pool_s", 0.0)}          # raw probability unless a test pools
    return Config.load(overrides=over)


def timeline(prob=None):
    n = int(120 * FPS)
    tl = RunTimeline(FPS, n, np.ones(n, bool), np.ones(n, np.float32), [int(40 * FPS), int(80 * FPS)])
    if prob is not None:
        tl.boundary_prob = prob.astype(np.float32)
    return tl


def bounds(steps):
    return [(s.label, s.start_s) for s in steps if s.label != "other"]


class EngineAEvidenceTest(unittest.TestCase):
    def prob(self, spike_s=None, bump=None):
        rng = np.random.default_rng(0)
        p = 0.01 + 0.005 * rng.random(int(120 * FPS))
        if spike_s is not None:
            f = int(spike_s * FPS)
            p[f - 3:f + 3] = 0.95
        if bump is not None:
            a, b = bump
            p[int(a * FPS):int(b * FPS)] = 0.6
        return p

    def test_weight_zero_changes_nothing(self):
        e0 = RuleBasedEngine(two_phase_cfg())
        s0 = e0.segment_run(timeline())
        e1 = RuleBasedEngine(two_phase_cfg(engine_b_weight=0.0))
        s1 = e1.segment_run(timeline(self.prob(spike_s=80)))
        self.assertEqual(s0, s1)
        self.assertEqual(e0.report, e1.report)
        self.assertEqual(bounds(s0), [("A", 0.0), ("B", 40.0)])        # tie: the first cut

    def test_probability_moves_the_boundary(self):
        eng = RuleBasedEngine(two_phase_cfg(engine_b_weight=0.4))
        steps = eng.segment_run(timeline(self.prob(spike_s=80)))
        self.assertEqual(bounds(steps), [("A", 0.0), ("B", 80.0)])
        ev = {e["start_s"]: e for e in eng.report["evidence"]}
        self.assertGreater(ev[80.0]["engine_b"], 0.99)
        self.assertLess(ev[40.0]["engine_b"], 0.5)
        self.assertTrue(eng.report["engine_b"]["on"])
        self.assertIn("engine_b", eng.report["evidence_counts"])
        minutes = eng.report["active"]["T_s"] / 60
        self.assertAlmostEqual(eng.report["engine_b"]["peaks_per_min_raw"], 1 / minutes, places=1)   # the one spike

    def test_pooling_prefers_a_lasting_change_over_a_spike(self):
        p = self.prob(spike_s=80, bump=(30, 50))
        raw = RuleBasedEngine(two_phase_cfg(engine_b_weight=0.4)).segment_run(timeline(p))
        pooled = RuleBasedEngine(two_phase_cfg(engine_b_weight=0.4, pool_s=20.0)).segment_run(timeline(p))
        self.assertEqual(bounds(raw)[1], ("B", 80.0))
        self.assertEqual(bounds(pooled)[1], ("B", 40.0))

    def test_candidates_from_peaks(self):
        p = self.prob(spike_s=60)
        eng = RuleBasedEngine(two_phase_cfg(engine_b_weight=0.4, engine_b_candidates=True))
        steps = eng.segment_run(timeline(p))
        kinds = {e["kind"] for e in eng.report["evidence"]}
        self.assertIn("engine_b", kinds)
        self.assertAlmostEqual(bounds(steps)[1][1], 60.0, delta=0.2)
        off = RuleBasedEngine(two_phase_cfg(engine_b_weight=0.4)).segment_run(timeline(p))
        self.assertIn(bounds(off)[1][1], (40.0, 80.0))                  # without candidates: one of the cuts

    def test_without_probability_the_term_is_off(self):
        eng = RuleBasedEngine(two_phase_cfg(engine_b_weight=0.4))
        steps = eng.segment_run(timeline())
        self.assertEqual(bounds(steps)[1], ("B", 40.0))
        self.assertFalse(eng.report["engine_b"]["on"])


class CompareValuesTest(unittest.TestCase):
    def test_values(self):
        from assemblyvision.tools.run_pose import compare_values
        cfg = Config.load()
        self.assertEqual(compare_values("seg_a.activity_change", cfg), ("seg_a.activity_change", [False, True]))
        self.assertEqual(compare_values("seg_a.engine_b_weight", cfg)[1], [0.0, 0.2, 0.4, 0.8])
        self.assertEqual(compare_values("engine_b.pool_s=0, 10.5,30", cfg), ("engine_b.pool_s", [0, 10.5, 30]))
        for bad in ("seg_a.ac_weight", "seg_a.nope", "engine_b.pool_s=1", "engine_b.pool_s=[", "x"):
            with self.assertRaises(AvError):
                compare_values(bad, cfg)
        with self.assertRaises(AvError):                          # a value outside the range of its key
            Config.load(overrides={"seg_a": {"engine_b_norm": "zscore"}})


class SubjectSplitTest(unittest.TestCase):
    def test_fixed_split(self):
        subj = {f"S{i:03d}": ("lab_a" if i <= 22 else "lab_b") for i in range(1, 42)}
        a = splits.subject_split(subj, 0.3, 42)
        self.assertEqual(a, splits.subject_split(dict(reversed(list(subj.items()))), 0.3, 42))
        self.assertEqual(len(a["train_subjects"]) + len(a["test_subjects"]), 41)
        self.assertFalse(set(a["train_subjects"]) & set(a["test_subjects"]))
        self.assertEqual(sum(s <= "S022" for s in a["test_subjects"]), round(0.3 * 22))
        self.assertEqual(sum(s > "S022" for s in a["test_subjects"]), round(0.3 * 19))
        tmp = fake_root()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "split.json"
        path.write_text(json.dumps({"dataset": "ha4m", **a}))
        self.assertEqual(splits.check_subject_split(path, subj, 0.3, 42)["seed"], 42)
        with self.assertRaises(ValueError):
            splits.check_subject_split(path, subj, 0.3, 7)

    def test_videos_follow_the_fixed_split(self):
        from assemblyvision.datasets.base import VideoRef
        subj = {f"S{i:03d}": ("lab_a" if i <= 3 else "lab_b") for i in range(1, 7)}
        fixed = splits.subject_split(subj, 0.34, 1)
        vids = [VideoRef(f"IDU{i:03d}V001", f"S{i:03d}", subj[f"S{i:03d}"], 30.0, 10) for i in (1, 2, 4, 5)]
        fold = splits.cross_subject(vids, 0.34, 1, fixed)
        self.assertEqual(set(fold["test_subjects"]), set(fixed["test_subjects"]) & {v.subject for v in vids})


class NoiseRuleTest(unittest.TestCase):
    def eq(self, within, tracks, rel, steps=True, nulls=0):
        return {"verdict": "different", "checks": {"shots": True, "steps": steps},
                "pose": {"frames": 1000, "share_within_1px": within}, "tracking": {"differences": tracks},
                "metrics": {"max_rel_delta": rel, "null_changes": nulls, "flag_changes": 0}}

    def test_floor_and_check(self):
        floor = noise_floor([self.eq(0.99, 2, 0.03), self.eq(0.98, 1, 0.001)])
        self.assertEqual(floor, {"not_within_1px_share": 0.02, "track_changes_per_frame": 0.002,
                                 "metric_max_rel_delta": 0.03})
        self.assertTrue(noise_check(self.eq(0.985, 2, 0.02), floor)["within_noise"])
        r = noise_check(self.eq(0.97, 5, 0.08, steps=False, nulls=1), floor)
        self.assertFalse(r["within_noise"])
        self.assertEqual(len(r["failed"]), 5)


def fake_cache(root: Path, n_videos=6, T=600, seed=0):
    """Engine B cache with a boundary every 4 s, marked in the features (so it can be learnt)."""
    rng = np.random.default_rng(seed)
    cache = eb.cache_dir(root / "data/public/datasets/ha4m")
    cache.mkdir(parents=True, exist_ok=True)
    index = {"videos": {}}
    for k in range(n_videos):
        vid, sub = f"IDU{k + 1:03d}V001", f"S{k + 1:03d}"
        lab = np.zeros((T, 2), np.int32)
        lab[:, 0] = (np.arange(T) // 120) % 3 + 1
        feats = rng.normal(0, 0.1, (T, eb.N_FEATURES)).astype(np.float32)
        ch = np.flatnonzero(np.diff(lab[:, 0])) + 1
        for c in ch:
            feats[max(0, c - 3):c + 3, 0] += 2.0
        meta = {"video": vid, "subject": sub, "location": "lab_a" if k % 2 == 0 else "lab_b", "fps": 30.0, "frames": T}
        np.savez(cache / f"{vid}.npz", xy=np.zeros((T, 17, 2), np.float32), kinect_conf=np.zeros((T, 17), np.float32),
                 features=feats, labels=lab, boundary=np.zeros(T, bool), labelled=np.ones(T, bool),
                 median_torso_px=np.float32(200), meta=np.str_(json.dumps(meta)))
        index["videos"][vid] = meta
    (cache / "index.json").write_text(json.dumps(index))
    subj = {m["subject"]: m["location"] for m in index["videos"].values()}
    split = {"dataset": "ha4m", **splits.subject_split(subj, 0.34, 1)}
    (root / "experiments/datasets").mkdir(parents=True, exist_ok=True)
    (root / "experiments/datasets/ha4m_subject_split.json").write_text(json.dumps(split))
    return cache


class TrainToolTest(unittest.TestCase):
    def test_cpu_training_writes_the_experiment(self):
        from assemblyvision.tools import train_engine_b
        tmp = fake_root()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        fake_cache(root)
        code = train_engine_b.main(["--id", "t-engine-b", "--cpu", "--set", "engine_b.epochs=40", "--set",
                                    "engine_b.layers=3", "--set", "engine_b.channels=16", "--set",
                                    "engine_b.augment=false", "--set", "engine_b.dropout=0.0",
                                    "--set", "engine_b.lr=0.01"], repo_root=root)
        self.assertEqual(code, 0)
        res = json.loads((root / "experiments/t-engine-b/results.json").read_text())
        self.assertEqual(res["data"]["test_subjects"], 2)
        self.assertGreater(res["test"]["f1"]["0.5"]["f1"], 0.8)          # the marked boundaries are found
        self.assertTrue((root / "models/engine_b/t-engine-b.pt").is_file())
        self.assertIn("t-engine-b", (root / "experiments/INDEX.md").read_text())
        self.assertIn("layers = 3", (root / "experiments/t-engine-b/config.toml").read_text())

    def test_refused_without_split(self):
        from assemblyvision.tools import train_engine_b
        tmp = fake_root()
        self.addCleanup(tmp.cleanup)
        self.assertEqual(train_engine_b.main(["--id", "t-x", "--cpu"], repo_root=Path(tmp.name)), 2)


if __name__ == "__main__":
    unittest.main()


class StoredRunTest(unittest.TestCase):
    def test_run_probability_and_missing_model(self):
        import torch
        from test_activity_change import CompareModeTest
        from assemblyvision.tools.run_pose import compare_run
        tmp = fake_root()
        self.addCleanup(tmp.cleanup)
        ctx = CompareModeTest.fake_run(None, Path(tmp.name))
        manifest = json.loads(ctx.paths.manifest.read_text())
        manifest["ingest"]["cfr"]["height"] = 1080
        torch.manual_seed(0)
        m = bt.build_model(eb.N_FEATURES, 2, 2, 8, 0.0).eval()
        path = Path(tmp.name) / "m.pt"
        bt.save_model(path, m, {"model_fps": 10, "stages": 2, "layers": 2, "channels": 8, "dropout": 0.0}, {})
        out = bt.run_probability(manifest, ctx.paths.root, Config.load(), path)
        self.assertEqual(out["prob"].shape, (3600,))
        self.assertTrue((out["prob"][1800:1890] == 0).all())           # no subject in the middle shot
        self.assertTrue((out["prob"][:1800] > 0).all())
        self.assertEqual(out["frames_scored"], 3510)
        gt = Path(tmp.name) / "data/qisens/gt.csv"
        gt.write_text("step,label,start_s,end_s,source\n1,A,0,62,gt\n2,B,62,120,gt\n")
        cfg = Config.load(overrides={"steps": {"use": True, "step": [{"name": "A"}, {"name": "B"}]},
                                     "seg_a": {"engine": "custom"}, "engine_b": {"model": "engine_b/does-not-exist.pt"}})
        rows = compare_run(manifest, ctx.paths.root, "qisens", cfg, "seg_a.engine_b_weight=0,0.4", gt)
        self.assertEqual(rows[0]["steps"], rows[1]["steps"])
        self.assertTrue(any(n.startswith("AV-SEG-001") for n in rows[1]["notices"]))


    def test_compare_hands_engine_a_scale_to_the_decoder(self):
        from unittest import mock
        from test_activity_change import CompareModeTest
        from assemblyvision.analysis.segmentation.rules import RuleBasedEngine
        from assemblyvision.tools.run_pose import compare_run
        tmp = fake_root()
        self.addCleanup(tmp.cleanup)
        ctx = CompareModeTest.fake_run(None, Path(tmp.name))
        manifest = json.loads(ctx.paths.manifest.read_text())
        gt = Path(tmp.name) / "data/qisens/gt.csv"
        gt.write_text("step,label,start_s,end_s,source\n1,A,0,62,gt\n2,B,62,120,gt\n")
        cfg = Config.load(overrides={"steps": {"use": True, "step": [{"name": "A"}, {"name": "B"}]},
                                     "seg_a": {"engine": "custom"}})
        seen, strengths, real = [], [], RuleBasedEngine.segment_run

        def spy(engine, tl):
            out = real(engine, tl)
            seen.append(engine.s["engine_a_scale"])
            strengths.append([e["strength"] for e in engine.report["evidence"]])
            return out
        with mock.patch.object(RuleBasedEngine, "segment_run", spy):
            compare_run(manifest, ctx.paths.root, "qisens", cfg, "seg_a.engine_a_scale=0.0,0.5,1.0", gt)
        self.assertEqual(seen, [0.0, 0.5, 1.0])
        self.assertTrue(strengths[2] and all(x == 0 for x in strengths[0]))
        self.assertEqual([round(2 * x, 6) for x in strengths[1]], [round(x, 6) for x in strengths[2]])
        import contextlib
        import io
        err = io.StringIO()
        with contextlib.redirect_stderr(err):               # an Engine B key while Engine B is off: a warning
            compare_run(manifest, ctx.paths.root, "qisens", cfg, 'engine_b.candidates="engine_a","dense"', gt)
        self.assertIn("Engine B is off", err.getvalue())


class CalibrationTest(unittest.TestCase):
    def test_offsets_are_the_plain_mean_of_the_labs(self):
        def video(hip):
            return {"joints": [{"offset_torso": hip if j == 11 else [0.0, 0.0]} for j in range(16)] + [{}]}
        off = eb.calibration_offsets({"videos": {"a": video([0.2, -0.2]), "b": video([0.0, 0.4])}})
        self.assertEqual(off.shape, (17, 2))
        np.testing.assert_allclose(off[11], [0.1, 0.1])
        np.testing.assert_allclose(off[16], [0.0, 0.0])                  # not measured -> no offset

    def test_offset_is_applied_in_the_body_frame(self):
        from test_engine_b_data import upright
        xy = upright(3)
        conf = np.full((3, 17), 0.67)
        conf[2, 11] = 0.0                                                # no torso -> unchanged
        rot = xy.copy()
        rot[1] = (xy[1] - [500, 800]) @ np.array([[0, 1], [-1, 0]]) + [500, 800]   # person turned 90 degrees
        off = np.zeros((17, 2))
        off[11] = [0.1, -0.2]
        out = eb.calibrate(rot, conf, off)
        L = 200.0
        np.testing.assert_allclose(out[0, 11] - rot[0, 11], [0.1 * L, 0.2 * L], atol=1e-6)   # ex right, ey up (image y down)
        hip, ex, ey, _ = eb.body_axes(rot[1:2])
        np.testing.assert_allclose(out[1, 11] - rot[1, 11], (0.1 * ex[0] - 0.2 * ey[0]) * L, atol=1e-6)
        np.testing.assert_allclose(out[2], rot[2])
        np.testing.assert_allclose(np.delete(out[:2], 11, axis=1), np.delete(rot[:2], 11, axis=1))


class AboveFloorTest(unittest.TestCase):
    def test_metric_changes_keep_their_absolute_values(self):
        from assemblyvision.evaluation.equivalence import change_text
        floor = {"not_within_1px_share": 0.02, "track_changes_per_frame": 0.002, "metric_max_rel_delta": 0.03}
        eq = NoiseRuleTest().eq(0.99, 1, 0.074)
        eq["metrics"]["changes"] = [{"metric": "A-0010 idle_share", "reference": 0.31, "candidate": 0.333,
                                     "abs_delta": 0.023, "rel_delta": 0.0691},
                                    {"metric": "A-0010 path_m", "reference": 10.0, "candidate": 10.1,
                                     "abs_delta": 0.1, "rel_delta": 0.0099}]
        r = noise_check(eq, floor)
        self.assertEqual([c["metric"] for c in r["metrics_above_floor"]], ["A-0010 idle_share"])
        self.assertEqual(change_text(r["metrics_above_floor"][0]), "A-0010 idle_share 0.31 to 0.333 (+0.023, 6.9 %)")


class FoldsTest(unittest.TestCase):
    def test_cross_location_folds_and_calibration(self):
        from assemblyvision.tools import train_engine_b
        tmp = fake_root()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        fake_cache(root)
        vids = [f"IDU{k + 1:03d}V001" for k in range(6)]
        a, b = vids[0::2], vids[1::2]
        (root / "experiments/datasets/ha4m_splits.json").write_text(json.dumps({"folds": [
            {"name": "cross_location_test_lab_a", "train_videos": b, "test_videos": a},
            {"name": "cross_location_test_lab_b", "train_videos": a, "test_videos": b}]}))
        args = ["--cpu", "--set", "engine_b.epochs=2", "--set", "engine_b.layers=2", "--set", "engine_b.channels=8"]
        self.assertEqual(train_engine_b.main(["--id", "t-f", "--folds", "cross_location_test_lab_a", *args],
                                             repo_root=root), 2)                      # cross subject must come first
        self.assertEqual(train_engine_b.main(["--id", "t-f", "--set", "engine_b.hip_calibration=true", *args],
                                             repo_root=root), 2)                      # no domain gap experiment
        (root / "experiments/w10-ha4m-domain-gap").mkdir(parents=True)
        (root / "experiments/w10-ha4m-domain-gap/results.json").write_text(json.dumps(
            {"id": "w10-ha4m-domain-gap", "title": "gap", "index": {"cells": [], "verdict": "measurement"},
             "videos": {"x": {"joints": [{"offset_torso": [0.1, 0.0]}] * 17}}}))
        code = train_engine_b.main(["--id", "t-f", "--folds", "cross_subject,cross_location_test_lab_b",
                                    "--set", "engine_b.hip_calibration=true", *args], repo_root=root)
        self.assertEqual(code, 0)
        res = json.loads((root / "experiments/t-f/results.json").read_text())
        self.assertEqual(list(res["folds"]), ["cross_subject", "cross_location_test_lab_b"])
        f = res["folds"]["cross_location_test_lab_b"]
        self.assertEqual((f["data"]["train_locations"], list(f["test"]["per_location"])), (["lab_a"], ["lab_b"]))
        self.assertEqual(res["calibration_offsets_torso"][0], [0.1, 0.0])
        self.assertTrue((root / "models/engine_b/t-f.cross_location_test_lab_b.pt").is_file())
        self.assertIn("| cross_location_test_lab_b | lab_b |", (root / "experiments/t-f/table.md").read_text())


class GridToolTest(unittest.TestCase):
    def test_only_public_runs(self):
        from assemblyvision.tools import engine_b_grid
        tmp = fake_root()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        elsewhere = root / "runs" / "run_20260101_000000_abcd"
        self.assertEqual(engine_b_grid.main(["--id", "t-g", "--model", "x.pt", "--run", str(elsewhere),
                                             "--gt", "gt.csv"], repo_root=root), 2)
        self.assertEqual(engine_b_grid.main(["--id", "t-g", "--model", "x.pt", "--run", str(elsewhere), "--gt", "a.csv",
                                             "--gt", "b.csv"], repo_root=root), 2)            # one GT per run
        self.assertFalse((root / "experiments/t-g").exists())


class EmbeddingTest(unittest.TestCase):
    def test_change_score_peaks_at_the_phase_change(self):
        T, n = 600, 100
        emb = np.zeros((T, 4))
        emb[:300, 0], emb[300:, 1] = 1.0, 1.0
        valid = np.ones(T, bool)
        sc = bt.change_score(emb, valid, n)
        self.assertEqual(int(sc.argmax()), 300)
        self.assertAlmostEqual(float(sc.max()), np.sqrt(2), places=6)
        self.assertEqual(float(sc[100]), 0.0)                             # far from the change
        valid[:300] = False                                               # no subject before 300: no score there
        self.assertEqual(float(bt.change_score(emb, valid, n)[300]), 0.0)

    def test_multitask_model_heads_and_file(self):
        import torch
        tmp = fake_root()
        self.addCleanup(tmp.cleanup)
        cfg = Config.load(overrides={"engine_b": {"layers": 2, "channels": 8, "action_weight": 1.0}})
        m = bt.build_model(eb.N_FEATURES, 2, 2, 8, 0.0, 12)
        x, mask = torch.zeros(3, eb.N_FEATURES, 50), torch.ones(3, 1, 50)
        b, a, h = m(x, mask, all_heads=True)
        self.assertEqual((tuple(b.shape), tuple(a.shape), tuple(h.shape)), ((2, 3, 50), (2, 3, 12, 50), (3, 8, 50)))
        self.assertEqual(tuple(m(x, mask).shape), (2, 3, 50))
        path = Path(tmp.name) / "v1.pt"
        bt.save_model(path, m, bt.model_settings(cfg), {})
        m2, s = bt.load_model(path)
        self.assertEqual(s["n_actions"], 12)
        out = bt.predict_all(m2, np.zeros((40, eb.N_FEATURES), np.float32))
        self.assertEqual((out["actions"].shape, out["embedding"].shape), ((40, 12), (40, 8)))
        np.testing.assert_allclose(np.linalg.norm(out["embedding"], axis=1), 1.0, atol=1e-5)
        self.assertNotIn("n_actions", bt.model_settings(Config.load()))  # v0 settings unchanged

    def test_embed_term_moves_the_boundary_and_weight_zero_changes_nothing(self):
        def tl_with_embedding():
            tl = timeline()
            emb = np.zeros((tl.n_frames, 4), np.float32)
            emb[:int(80 * FPS), 0], emb[int(80 * FPS):, 1] = 1.0, 1.0       # the phase changes at 80 s
            tl.embedding, tl.embedding_valid = emb, np.ones(tl.n_frames, bool)
            return tl
        base = RuleBasedEngine(two_phase_cfg())
        s0 = base.segment_run(timeline())
        off = RuleBasedEngine(two_phase_cfg(embed_weight=0.0))
        self.assertEqual(off.segment_run(tl_with_embedding()), s0)
        self.assertEqual(off.report, base.report)
        on = RuleBasedEngine(two_phase_cfg(embed_weight=0.4, embed_window_s=20.0))
        self.assertEqual(bounds(on.segment_run(tl_with_embedding())), [("A", 0.0), ("B", 80.0)])
        self.assertTrue(on.report["embed_change"]["on"])
        ev = {e["start_s"]: e for e in on.report["evidence"]}
        self.assertEqual(ev[80.0]["embed_change"], 1.0)


class StartOfWorkTest(unittest.TestCase):
    def tl(self, idle_s):
        tl = timeline()
        tl.energy = np.full(tl.n_frames, 0.5, np.float32)
        tl.energy[:int(idle_s * FPS)] = 0.02                              # present but still: an intro
        return tl

    def test_intro_becomes_other(self):
        eng = RuleBasedEngine(two_phase_cfg(start_detect=True))
        steps = eng.segment_run(self.tl(20))
        self.assertEqual(steps[0].label, "other")
        self.assertEqual(steps[1].start_s, 20.0)                         # the first moving frame
        self.assertTrue(eng.report["start_of_work"]["found"])
        self.assertEqual(eng.report["start_of_work"]["intro_s"], 20.0)

    def test_off_and_no_intro_change_nothing(self):
        base = RuleBasedEngine(two_phase_cfg()).segment_run(self.tl(20))
        self.assertEqual(base[0].start_s, 0.0)
        self.assertNotEqual(base[0].label, "other")
        on = RuleBasedEngine(two_phase_cfg(start_detect=True))
        no_intro = self.tl(0)
        self.assertEqual(on.segment_run(no_intro), RuleBasedEngine(two_phase_cfg()).segment_run(self.tl(0)))
        self.assertEqual(on.report["start_of_work"]["intro_s"], 0.0)

    def test_short_still_start_is_not_an_intro(self):
        eng = RuleBasedEngine(two_phase_cfg(start_detect=True))
        steps = eng.segment_run(self.tl(6))                               # 6 s < start_min_intro_s 10
        self.assertFalse(eng.report["start_of_work"]["found"])
        self.assertEqual(eng.report["start_of_work"]["candidate_s"], 6.0)
        self.assertEqual(steps[0].start_s, 0.0)

    def test_intro_longer_than_the_cap_is_not_cut(self):
        eng = RuleBasedEngine(two_phase_cfg(start_detect=True, start_max_share=0.1))
        steps = eng.segment_run(self.tl(30))                              # 30 s idle > 0.1 x 120 s
        self.assertFalse(eng.report["start_of_work"]["found"])
        self.assertEqual(steps[0].start_s, 0.0)


class MultitaskTrainTest(unittest.TestCase):
    def test_cpu_training_with_the_action_head(self):
        from assemblyvision.tools import train_engine_b
        tmp = fake_root()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        fake_cache(root)
        code = train_engine_b.main(["--id", "t-v1", "--cpu", "--set", "engine_b.epochs=3", "--set", "engine_b.layers=2",
                                    "--set", "engine_b.channels=8", "--set", "engine_b.action_weight=1.0"], repo_root=root)
        self.assertEqual(code, 0)
        res = json.loads((root / "experiments/t-v1/results.json").read_text())
        self.assertIn("action_frame_accuracy", res["test"])
        self.assertIn("Action head", (root / "experiments/t-v1/table.md").read_text())
        self.assertEqual(bt.load_model(root / "models/engine_b/t-v1.pt")[1]["n_actions"], 12)
