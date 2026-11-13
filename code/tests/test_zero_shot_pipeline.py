"""TC-ZS: the zero shot stages in a full run and the "B zero shot" step engine (Week 11).

CPU only, no model and no server: the prompt kept as frozen, the stage choice, the
fallback to "B v1" without the score files, the busy VLM GPU and the stored VLM file.
"""
import fcntl
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from assemblyvision.analysis import phase_text as pt
from assemblyvision.analysis import vlm as V
from assemblyvision.analysis import zero_shot as Z
from assemblyvision.config import Config
from assemblyvision.errors import AvError


class _Ctx:
    def __init__(self):
        self.notices = []

    def notice(self, err, stage):
        self.notices.append((err.code, stage, err.detail))


class ZeroShotPipelineTest(unittest.TestCase):
    def test_prompt_of_2_s_is_unchanged(self):
        ph = pt.phases("qisens")
        text = V.prompt(ph, Z.PROCESS["qisens"])
        self.assertTrue(text.startswith("These 5 frames come from a video of a worker maintaining a railway vehicle "
                                        "(bogie maintenance) in a workshop, in time order over about two seconds."))
        self.assertEqual(text, V.prompt(ph, Z.PROCESS["qisens"], 2.0))
        self.assertIn("about four seconds", V.prompt(ph, Z.PROCESS["qisens"], 4.0))

    def test_phase_sets(self):
        self.assertEqual(len(Z.phase_set("qisens")[0]), 5)
        phases, process, label = Z.phase_set("ikea_asm:Lack_Coffee_Table")
        self.assertEqual((len(phases), label), (3, "Lack_Coffee_Table"))
        self.assertIn("Lack coffee table", process)
        with self.assertRaises(ValueError):
            Z.phase_set("ikea_asm:Unknown")
        with self.assertRaises(AvError):
            Config.load(overrides={"zero_shot": {"phases": "other"}})

    def test_stage_choice(self):
        from assemblyvision.tools.run_pose import zero_shot_needs
        self.assertEqual(zero_shot_needs(Config.load()), (True, True))
        self.assertEqual(zero_shot_needs(Config.load(overrides={"seg_a": {"engine": "B v1"}})), (False, False))
        self.assertEqual(zero_shot_needs(Config.load(overrides={"zero_shot": {"in_pipeline": False}})), (False, False))
        c = Config.load(overrides={"seg_a": {"engine": "custom", "embed_weight": 0.2, "vlm_weight": 1.0}})
        self.assertEqual(zero_shot_needs(c), (False, True))

    def test_fallback_config_is_b_v1(self):
        from assemblyvision.tools.run_pose import fallback_config
        c = fallback_config(Config.load(overrides={"ingest": {"crf": 20}}))
        b = Config.load(overrides={"seg_a": {"engine": "B v1"}, "ingest": {"crf": 20}})
        self.assertEqual(c.snapshot(), b.snapshot())

    def test_missing_files_fall_back_to_b_v1(self):
        from assemblyvision.tools import run_pose as R
        seen = []

        def fake(manifest, root, cfg, ctx, bprob=None):
            seen.append(cfg["seg_a"]["engine"])
            return None, []
        tl = mock.Mock(n_frames=10, fps=10.0)
        ctx = _Ctx()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch("assemblyvision.analysis.segmentation.build_timeline", return_value=tl), \
                mock.patch("assemblyvision.analysis.segmentation.create_engine"):
            orig = R.segment_stored
            with mock.patch.object(R, "segment_stored", side_effect=lambda *a, **k: fake(*a, **k)):
                orig({}, Path(d), Config.load(), ctx)
        self.assertEqual(seen, ["B v1"])
        self.assertEqual([n[0] for n in ctx.notices], ["AV-SEG-006"])
        self.assertIn("siglip.npz", ctx.notices[0][2])
        self.assertIn("vlm.npz", ctx.notices[0][2])

    def test_busy_vlm_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            locks = Path(d) / ".locks"
            locks.mkdir()
            with mock.patch.object(Z, "REPO_ROOT", Path(d)):
                held = open(locks / "gpu7.lock", "a+")
                fcntl.flock(held, fcntl.LOCK_EX)
                zs = dict(Config.load()["zero_shot"], vlm_gpu=7)
                job = Z.VlmJob(Path(d), Path(d) / "v.mp4", pt.phases("qisens"), "p", zs, Path(d) / "log").start()
                r = job.result()
                held.close()
                self.assertEqual(r["status"], "busy")
                again = Z.try_lock(7)
                self.assertIsNotNone(again)              # free again
                again.close()

    def test_vlm_file_written_in_order(self):
        phases = pt.phases("ikea_asm", "Lack_Coffee_Table")
        names = [p["name"] for p in phases]
        wins = [(2.0 * i, [np.zeros((4, 4, 3), np.uint8)]) for i in range(9)]

        def ask(port, text, frames, px):
            return json.dumps({"probabilities": {names[0]: 1.0}, "reason": "r"})
        zs = dict(Config.load()["zero_shot"], vlm_slots=3)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(V, "windows", return_value=iter(wins)), \
                mock.patch.object(V, "ask", side_effect=ask):
            out = Path(d) / "vlm_w2f4.npz"
            rec = Z.vlm_score_video(Path(d) / "v.mp4", out, phases, "p", zs, 1, threading.Event())
            z = np.load(out)
            self.assertEqual(rec, {"windows": 9, "unparsed": 0})
            self.assertEqual(list(z["time_s"]), [2.0 * i for i in range(9)])
            self.assertEqual(json.loads(str(z["meta"]))["slots"], 3)
            self.assertFalse((Path(d) / "vlm_w2f4.part.npz").exists())


if __name__ == "__main__":
    unittest.main()
