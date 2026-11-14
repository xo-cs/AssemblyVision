"""TC-FAIL: the failure suite (Week 11, TA review). Every failure gives a clear record (a notice
or an error with its code in the manifest, or a refused call) and never a silent result.

CPU only, synthetic media. Failures covered here, and where the rest are tested:

| failure | code | test |
| --- | --- | --- |
| corrupt media, full run | AV-ING-001 | test_corrupt_video_fails_the_run_with_a_record |
| decode failure in ffmpeg | AV-ING-003 | test_conversion_failure |
| decode failure of a frame | AV-POS-003 | test_pose_det.test_failed_frame_logged_and_run_continues |
| GPU out of memory | AV-POS-002 | test_gpu_out_of_memory (and test_pose_det for the fallback) |
| no person in the video | AV-SEG-007 | test_no_person |
| low coverage | AV-MOT-001 | test_low_coverage_is_not_reported_as_a_value |
| VLM failure | AV-SEG-006 | test_vlm_server_failure_falls_back_with_a_notice |
| VLM stopped midway | (no file) | test_stopped_vlm_writes_no_scores |
| interrupted run | AV-RUN-004 | test_runstore.test_lost_worker_gives_av_run_004, test_interrupted_run |
"""
import json
import os
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from assemblyvision.analysis import phase_text as pt
from assemblyvision.analysis import zero_shot as Z
from assemblyvision.config import Config
from assemblyvision.errors import AvError
from helpers import fake_root, ffmpeg_clip, make_ctx

CODE = Path(__file__).resolve().parent.parent


class FailureSuiteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_corrupt_video_fails_the_run_with_a_record(self):
        videos = self.root / "public" / "videos"
        videos.mkdir(parents=True)
        good = ffmpeg_clip(self.root / "good.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p")
        bad = videos / "corrupt.mp4"
        bad.write_bytes(good.read_bytes()[:2000])                      # a truncated file
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        p = subprocess.run([sys.executable, "-m", "assemblyvision.tools.run_pose", "--source", "public", "--video",
                            str(bad), "--variant", "n", "--gpu", "-1", "--set", f'paths.public_root="{self.root / "public"}"',
                            "--set", f'paths.qisens_root="{self.root / "qisens"}"'],
                           cwd=CODE, capture_output=True, text=True, env=env, timeout=300)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("AV-ING-001", p.stderr + p.stdout)
        run = next((self.root / "public" / "runs").iterdir())
        m = json.loads((run / "manifest.json").read_text())
        self.assertEqual(m["status"], "failed")
        self.assertEqual(m["error"]["code"], "AV-ING-001")
        self.assertEqual(m["stages"]["ingest"]["status"], "failed")
        self.assertFalse((run / "steps.csv").exists())                 # no partial result

    def test_conversion_failure(self):
        from assemblyvision.ingest import ingestor
        cfg, ctx = make_ctx(self.root)
        src = ffmpeg_clip(self.root / "v.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p")
        failed = subprocess.CompletedProcess([], 1, "", "Invalid data found when processing input")
        with mock.patch.object(ingestor, "_run", return_value=failed):
            with self.assertRaises(AvError) as cm:
                ingestor.VideoIngestor(cfg, ctx)._transcode(src, ctx.paths.cfr, 25)
        self.assertEqual(cm.exception.code, "AV-ING-003")
        self.assertFalse(ctx.paths.cfr.exists())

    def test_gpu_out_of_memory(self):
        import torch
        from assemblyvision.perception.pose import PoseEstimator
        cfg, ctx = make_ctx(self.root)
        pe = PoseEstimator(cfg, ctx, device="cpu")
        with mock.patch("ultralytics.YOLO", side_effect=torch.cuda.OutOfMemoryError("CUDA out of memory")), \
                mock.patch.object(Path, "is_file", return_value=True):
            with self.assertRaises(AvError) as cm:
                pe.load_models("x")
        self.assertEqual(cm.exception.code, "AV-POS-002")

    def test_no_person(self):
        from assemblyvision.analysis.segmentation.base import RunTimeline
        from assemblyvision.analysis.segmentation.rules import RuleBasedEngine
        cfg, ctx = make_ctx(self.root)
        cfg = Config.load(overrides={"seg_a": {"engine": "A"}})
        tl = RunTimeline(10.0, 300, np.zeros(300, bool), np.full(300, np.nan, np.float32), [])
        steps = RuleBasedEngine(cfg, ctx).segment_run(tl)
        self.assertEqual([s.label for s in steps], ["other"])
        self.assertEqual([n["code"] for n in ctx.manifest["notices"]], ["AV-SEG-007"])

    def test_low_coverage_is_not_reported_as_a_value(self):
        from test_motion import FPS, analyse, cfg, one_step, pose_series
        T = int(10 * FPS)
        s = pose_series(T)
        s.present[int(2 * FPS):] = False                                # keypoints on 20 percent of the step
        s.body_xy[int(2 * FPS):] = np.nan
        per, an, _ = analyse([s], one_step(T), cfg(enabled=["back_bending"]))
        r = per[0]["bend_b0_s"]
        self.assertIsNone(r.value)                                      # no value from too few frames
        self.assertIn("low_coverage", r.flags)
        self.assertEqual([e.code for e in an.notices], ["AV-MOT-001"])

    def test_vlm_server_failure_falls_back_with_a_notice(self):
        from assemblyvision.analysis import vlm as V
        from assemblyvision.tools import run_pose as R
        cfg, ctx = make_ctx(self.root)
        ctx.manifest["stages"] = {"vlm_scores": {"status": "pending"}}
        ctx.manifest["ingest"] = {"cfr": {"path": "video/cfr.mp4"}}
        locks = self.root / ".locks"
        with mock.patch.object(Z, "REPO_ROOT", self.root), \
                mock.patch.object(V, "start_server", side_effect=RuntimeError("llama-server exited (1)")):
            job = R.start_vlm(ctx, Config.load(overrides={"zero_shot": {"vlm_gpu": 7}}))
            R.finish_vlm(ctx, job)
            again = Z.try_lock(7)
            self.assertIsNotNone(again)                                 # the GPU lock was given back
            again.close()
        rec = ctx.manifest["stages"]["vlm_scores"]
        self.assertEqual(rec["status"], "skipped")
        self.assertIn("llama-server exited", rec["detail"])
        self.assertEqual(ctx.manifest["notices"][-1]["code"], "AV-SEG-006")
        self.assertTrue(locks.is_dir())

    def test_stopped_vlm_writes_no_scores(self):
        phases = pt.phases("ikea_asm", "Lack_Coffee_Table")
        stop = threading.Event()
        stop.set()
        wins = [(0.0, [np.zeros((4, 4, 3), np.uint8)])]
        out = self.root / "vlm.npz"
        with mock.patch("assemblyvision.analysis.vlm.windows", return_value=iter(wins)):
            with self.assertRaises(RuntimeError):
                Z.vlm_score_video(self.root / "v.mp4", out, phases, "p", Config.load()["zero_shot"], 1, stop)
        self.assertFalse(out.exists())

    def test_interrupted_run(self):
        """A worker that dies mid run: the run shows failed with AV-RUN-004, not running forever."""
        from assemblyvision.storage.runstore import RunStore
        cfg = Config.load(overrides={"paths": {"public_root": str(self.root / "public"),
                                               "qisens_root": str(self.root / "qisens")}})
        run = self.root / "public" / "runs" / "run_20260101_000000_abcd"
        run.mkdir(parents=True)
        p = subprocess.Popen([sys.executable, "-c", "pass"])
        p.wait()                                                        # a pid that is gone
        (run / "manifest.json").write_text(json.dumps({"run_id": run.name, "status": "running", "stages": {
            "pose_body": {"status": "running"}}, "worker": {"pid": p.pid}}))
        st = RunStore(["public"], cfg).status(run.name)
        self.assertEqual(st.status, "failed")
        self.assertIn("AV-RUN-004", st.error)


if __name__ == "__main__":
    unittest.main()
