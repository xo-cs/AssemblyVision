"""Experiment registry (experiments/registry.py) in a fake repo root, with a stub
runner instead of run_pose (no GPU, no model)."""
import json
import re
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace

from assemblyvision.config import Config
from assemblyvision.errors import AvError
from assemblyvision.experiments import registry as reg
from assemblyvision.experiments.provenance import git_state, model_checksum, sha256_file
from helpers import fake_root


class Stub:
    """Plays run_pose: makes a run folder with a manifest and prints the summary line."""
    made = 0

    def __init__(self, root: Path, walls=(10.0, 12.0, 11.0)):
        self.root, self.walls, self.calls = root, list(walls), []

    def __call__(self, cmd, cwd):
        self.calls.append(cmd)
        k = len(self.calls)
        Stub.made += 1
        source = cmd[cmd.index("--source") + 1]
        run_id = f"run_20261106_{Stub.made:06d}_ab{k:02d}"
        d = self.root / "data" / source / "runs" / run_id
        d.mkdir(parents=True)
        stages = {"ingest": {"seconds": 0.5}, "shots": {"seconds": 2.0}, "pose_body": {"seconds": 5.0 + k},
                  "tracking": {"seconds": 1.0}, "overlay": {"seconds": 3.0}}
        m = {"run_id": run_id, "status": "done", "stages": stages,
             "pose_body": {"frames": 600, "pose_fps": 600 / (5.0 + k), "batch": 1, "batch_used": 1,
                           "precision": "FP32", "decode": "opencv", "gpu_peak_mb": 900.0 + k,
                           "breakdown_s": {"decode": 1.0, "inference": 3.0, "writing": 0.1}},
             "overlay": {"encoder": "libx264", "workers": 1, "frames": 600},
             "profile": {"run": {"samples": 10, "gpu_util_mean": 40.0 + k, "gpu_util_max": 90, "gpu_mem_peak_mb": 2000 + k},
                         "stages": {"pose_body": {"gpu_util_mean": 70.0, "gpu_util_max": 95, "gpu_mem_peak_mb": 2000}},
                         "breakdown": {}}}
        (d / "manifest.json").write_text(json.dumps(m))
        return 0, f"{run_id} done: variant x, 600 frames\n", "", self.walls[(k - 1) % len(self.walls)]


def fake_equivalence(a, b):
    return {"verdict": "equivalent", "checks": {"pose_1px": True}, "differences": [],
            "pose": {"share_within_1px": 1.0, "share_identical": 1.0, "box_max_px": 0.0, "kp_max_px": 0.0}}


def fake_provenance(exp):
    return {"git": {"commit": "abc123", "dirty": []}, "gpu": {"name": "Fake GPU", "driver": "1.0"},
            "model": {"sha256": "0" * 64}, "versions": {"torch": "x"}, "data_area": exp.source, "date_utc": "now"}


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.root = Path(self._tmp.name)
        self.pc = Config.load()["paths"]
        for s in ("public", "qisens"):
            (self.root / "data" / s / "videos").mkdir(parents=True)
        self.video = self.root / "data/public/videos/clip.mp4"
        self.video.write_bytes(b"not a real video")
        self.qvideo = self.root / "data/qisens/videos/q.mp4"
        self.qvideo.write_bytes(b"x")

    def tearDown(self):
        self._tmp.cleanup()

    def new(self, exp_id="w10-ref", source="public", video=None, **kw):
        return reg.create(exp_id, source=source, videos=[str(video or self.video)], paths_cfg=self.pc,
                          repo_root=self.root, **kw)

    def run_exp(self, exp, **kw):
        return reg.run(exp, self.pc, repo_root=self.root, runner=kw.pop("runner", Stub(self.root)),
                       gpu_check=kw.pop("gpu_check", lambda g: []), provenance=fake_provenance,
                       equivalence=fake_equivalence, log=lambda s: None, **kw)

    def test_new_stores_every_setting(self):
        exp = self.new(sets={"pose": {"batch": 8}}, title="batch 8", repeats=2)
        data = tomllib.loads((exp.dir / "config.toml").read_text())
        meta = data.pop("experiment")
        self.assertEqual((meta["id"], meta["source"], meta["repeats"]), ("w10-ref", "public", 2))
        self.assertEqual(meta["videos"], ["data/public/videos/clip.mp4"])      # relative to the repo root
        default = Config.load().snapshot()
        for section, values in default.items():
            self.assertEqual(set(values), set(data[section]), section)
        self.assertEqual(data["pose"]["batch"], 8)
        self.assertEqual(data["steps"]["step"], default["steps"]["step"])   # the vocabulary is stored too
        self.assertEqual(reg.load("w10-ref", "public", self.pc, self.root).settings["pose"]["batch"], 8)

    def test_from_copies_settings_then_sets(self):
        base = self.new(sets={"pose": {"batch": 16}, "overlay": {"workers": 4}})
        exp = self.new("w10-fp16", base=base, sets={"pose": {"half": True}}, reference="w10-ref")
        self.assertEqual((exp.settings["pose"]["batch"], exp.settings["pose"]["half"],
                          exp.settings["overlay"]["workers"]), (16, True, 4))
        self.assertEqual(exp.meta["from"], "w10-ref")

    def strip(self, exp, *keys):
        text = (exp.dir / "config.toml").read_text()
        for k in keys:
            text, n = re.subn(rf"\n{k} = [^\n]*\n", "\n", text, count=1)
            self.assertEqual(n, 1, k)
        (exp.dir / "config.toml").write_text(text)

    NEW_IN_172375A = ("embed_weight", "embed_window_s", "start_detect", "start_sustain_s", "start_active_share",
                      "start_max_share", "start_min_intro_s", "action_weight")

    def test_from_fills_settings_added_later_when_they_are_no_ops(self):
        old = self.new()
        self.strip(old, "hip_calibration", "calibration_from")              # stored before these keys existed
        with self.assertRaises(reg.ExperimentError):
            reg.load("w10-ref", "public", self.pc, self.root)                # a plain load stays strict
        base = reg.load("w10-ref", "public", self.pc, self.root, fill_new=True)
        self.assertEqual(base.meta["filled_at_load"]["keys"], ["engine_b.hip_calibration", "engine_b.calibration_from"])
        exp = self.new("w10-new", base=base)
        stored = reg.load("w10-new", "public", self.pc, self.root)
        self.assertFalse(stored.settings["engine_b"]["hip_calibration"])
        self.assertEqual(stored.meta["filled_from_defaults"], ["engine_b.hip_calibration", "engine_b.calibration_from"])
        self.assertIn("filled_at_commit", stored.meta)
        self.assertIn("filled_from_defaults", (exp.dir / "config.toml").read_text())

    def test_run_and_recheck_fill_in_memory_and_record_it(self):
        exp = self.new()
        self.strip(exp, *self.NEW_IN_172375A)                                 # stored before 172375a
        before = (exp.dir / "config.toml").read_text()
        loaded = reg.load("w10-ref", "public", self.pc, self.root, fill_new=True)
        self.assertEqual(len(loaded.meta["filled_at_load"]["keys"]), 8)
        res = self.run_exp(loaded)
        rec = res["filled_from_defaults"]["at_run"]
        self.assertIn("seg_a.start_detect", rec["keys"])
        self.assertIn("commit", rec)
        self.assertEqual(json.loads((exp.dir / "results.json").read_text())["filled_from_defaults"]["at_run"], rec)
        self.assertEqual((exp.dir / "config.toml").read_text(), before)      # the stored config is unchanged
        res = reg.recheck(loaded, self.pc, repo_root=self.root, equivalence=fake_equivalence)
        self.assertEqual(res["filled_from_defaults"]["at_run"]["keys"], rec["keys"])

    def test_from_refuses_a_missing_setting_that_is_not_a_no_op(self):
        old = self.new()
        self.strip(old, "batch")                                             # pose.batch: not a known no-op
        with self.assertRaises(reg.ExperimentError) as cm:
            reg.load("w10-ref", "public", self.pc, self.root, fill_new=True)
        self.assertIn("pose.batch = 8", str(cm.exception))
        self.assertIn("not a no-op", str(cm.exception))
        old2 = self.new("w10-eb", sets={"seg_a": {"engine_b_weight": 0.4}})
        self.strip(old2, "pool_s")                                           # a no-op only while the weight is 0
        with self.assertRaises(reg.ExperimentError) as cm:
            reg.load("w10-eb", "public", self.pc, self.root, fill_new=True)
        self.assertIn("engine_b.pool_s = 30.0", str(cm.exception))

    def test_bad_values_and_ids_are_refused(self):
        with self.assertRaises(AvError):
            self.new(sets={"pose": {"batch": 0}})
        with self.assertRaises(AvError):
            self.new(sets={"pose": {"no_such_key": 1}})
        with self.assertRaises(reg.ExperimentError):
            self.new("Bad ID")
        self.new()
        with self.assertRaises(reg.ExperimentError):     # no silent overwrite
            self.new()

    def test_incomplete_config_is_refused(self):
        exp = self.new()
        text = (exp.dir / "config.toml").read_text()
        (exp.dir / "config.toml").write_text(re.sub(r"\nbatch = \d+\n", "\n", text, count=1))
        with self.assertRaises(reg.ExperimentError) as cm:
            reg.load("w10-ref", "public", self.pc, self.root)
        self.assertIn("pose.batch", str(cm.exception))

    def test_video_outside_its_area_is_refused(self):
        with self.assertRaises(AvError) as cm:
            self.new(video=self.qvideo)
        self.assertEqual(cm.exception.code, "AV-RUN-001")
        with self.assertRaises(AvError):
            self.new("q", source="qisens", video=self.video)

    def test_run_records_repeats_medians_and_index(self):
        exp = self.new(repeats=3, title="reference")
        stub = Stub(self.root)
        res = self.run_exp(exp, runner=stub)
        self.assertEqual(len(stub.calls), 3)
        cmd = stub.calls[0]
        self.assertIn("profile.enabled=true", cmd)
        self.assertEqual(cmd[cmd.index("--gpu") + 1], "0")
        layer = Path(cmd[cmd.index("--config") + 1])
        self.assertTrue(layer.is_relative_to(self.root / "data/public/tmp"))
        self.assertFalse(layer.exists())                        # removed after the run
        v = res["videos"]["data/public/videos/clip.mp4"]
        s = v["summary"]
        self.assertEqual(s["wall_s"]["median"], 11.0)
        self.assertEqual(s["stages_s"]["pose_body"], 7.0)      # 6, 7, 8
        self.assertEqual(s["gpu_mem_peak_mb"], 2003)
        self.assertEqual(s["torch_peak_mb"], 903.0)
        self.assertEqual(s["gpu_util_mean"], 42.0)
        self.assertEqual(v["repeat_consistency"], ["equivalent", "equivalent"])
        self.assertNotIn("equivalence", v)                     # no reference
        self.assertEqual(res["provenance"]["git"]["commit"], "abc123")
        table = (exp.dir / "table.md").read_text()
        for word in ("pose_body", "full run", "mean GPU util", "peak GPU memory", "pose only fps"):
            self.assertIn(word, table)
        index = (self.root / "experiments/INDEX.md").read_text()
        self.assertIn("[w10-ref](w10-ref/table.md)", index)
        self.assertIn("MB", index)

    def test_reference_equivalence_and_compare_table(self):
        self.run_exp(self.new(repeats=1))
        exp = self.new("w10-b8", sets={"pose": {"batch": 8}}, reference="w10-ref", repeats=1)
        res = self.run_exp(exp, runner=Stub(self.root, walls=(8.0,)))
        self.assertEqual(res["videos"]["data/public/videos/clip.mp4"]["equivalence"]["verdict"], "equivalent")
        text = reg.compare_table([reg.load_results(i, "public", self.pc, self.root) for i in ("w10-ref", "w10-b8")])
        for col in ("GPU util %", "peak MB", "torch MB", "pose fps", "verdict"):
            self.assertIn(col, text)
        self.assertIn("| w10-b8 |", text)
        self.assertIn("equivalent |", (self.root / "experiments/INDEX.md").read_text())

    def test_recheck_compares_with_the_rerun_reference(self):
        self.run_exp(self.new(repeats=1))
        exp = self.new("w10-b8", sets={"pose": {"batch": 8}}, reference="w10-ref", repeats=1)
        before = self.run_exp(exp)
        v = "data/public/videos/clip.mp4"
        self.run_exp(reg.load("w10-ref", "public", self.pc, self.root))          # the reference is rerun
        new_ref = reg.load_results("w10-ref", "public", self.pc, self.root)["videos"][v]["repeats"][0]["run_dir"]
        seen = []
        res = reg.recheck(exp, self.pc, repo_root=self.root,
                          equivalence=lambda a, b: seen.append((a, b)) or fake_equivalence(a, b))
        self.assertEqual(seen, [(self.root / new_ref, self.root / before["videos"][v]["repeats"][0]["run_dir"])])
        self.assertIn("rechecked_utc", res)
        self.assertEqual(res["videos"][v]["repeats"], before["videos"][v]["repeats"])   # timings unchanged
        self.assertEqual(reg.load_results("w10-b8", "public", self.pc, self.root)["rechecked_utc"],
                         res["rechecked_utc"])

    def test_missing_reference_results_stop_the_run(self):
        exp = self.new("w10-b8", reference="w10-ref", repeats=1)
        with self.assertRaises(reg.ExperimentError):
            self.run_exp(exp)

    def test_foreign_gpu_process_stops_before_any_run(self):
        exp = self.new(repeats=2)
        stub = Stub(self.root)
        with self.assertRaises(reg.ExperimentError) as cm:
            self.run_exp(exp, runner=stub, gpu_check=lambda g: [{"pid": 7, "name": "SRBMiner", "used_mb": 3000}])
        self.assertIn("SRBMiner", str(cm.exception))
        self.assertEqual(stub.calls, [])
        self.assertFalse((exp.dir / "results.json").exists())

    def test_failed_run_pose_is_an_error(self):
        exp = self.new(repeats=1)
        with self.assertRaises(reg.ExperimentError):
            self.run_exp(exp, runner=lambda cmd, cwd: (1, "", "AV-POS-001: missing weights", 0.1))

    def test_qisens_experiment_stays_under_data_qisens(self):
        base = self.new(sets={"pose": {"batch": 16}})
        exp = self.new("w10-qisens-best", source="qisens", video=self.qvideo, base=base, repeats=2)
        self.assertTrue(exp.dir.is_relative_to(self.root / "data/qisens/experiments"))
        stub = Stub(self.root)
        self.run_exp(exp, runner=stub)
        for cmd in stub.calls:
            self.assertEqual(cmd[cmd.index("--source") + 1], "qisens")
            self.assertTrue(Path(cmd[cmd.index("--config") + 1]).is_relative_to(self.root / "data/qisens"))
        q = self.root / "data/qisens/experiments/w10-qisens-best"
        self.assertTrue((q / "results.json").is_file() and (q / "table.md").is_file())
        public = self.root / "experiments"
        self.assertEqual(sorted(p.name for p in public.iterdir()), ["w10-ref"])   # only the public base
        self.assertFalse((public / "INDEX.md").exists() and "qisens" in (public / "INDEX.md").read_text())
        with self.assertRaises(reg.ExperimentError):                    # not visible from the public area
            reg.load("w10-qisens-best", "public", self.pc, self.root)

    def test_public_experiment_never_touches_data_qisens(self):
        before = sorted(str(p) for p in (self.root / "data/qisens").rglob("*"))
        self.run_exp(self.new(repeats=1))
        self.assertEqual(sorted(str(p) for p in (self.root / "data/qisens").rglob("*")), before)


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_model_checksum_against_sums_file(self):
        w = self.root / "yolo26x-pose.pt"
        w.write_bytes(b"weights")
        digest = sha256_file(w)
        (self.root / "SHA256SUMS_yolo26_pose.txt").write_text(f"{digest}  yolo26x-pose.pt\n{'1' * 64}  other.pt\n")
        self.assertEqual(model_checksum(w)["matches_sums"], True)
        (self.root / "SHA256SUMS_yolo26_pose.txt").write_text(f"{'2' * 64}  yolo26x-pose.pt\n")
        self.assertEqual(model_checksum(w)["matches_sums"], False)

    def test_git_state(self):
        def run(cmd, **kw):
            out = {"HEAD": "deadbeef", "--abbrev-ref": "feat/w10-speed"}
            if "status" in cmd:
                return SimpleNamespace(returncode=0, stdout=" M code/x.py\n M experiments/INDEX.md\n")
            return SimpleNamespace(returncode=0, stdout=out["--abbrev-ref"] if "--abbrev-ref" in cmd else out["HEAD"])
        st = git_state(self.root, run=run)
        self.assertEqual(st, {"commit": "deadbeef", "branch": "feat/w10-speed", "dirty": ["code/x.py"],
                              "outputs_changed": ["experiments/INDEX.md"]})


if __name__ == "__main__":
    unittest.main()
