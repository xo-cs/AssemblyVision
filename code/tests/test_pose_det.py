"""PoseEstimator.estimate_body with a fake model on CPU (no GPU, no weights needed)."""
import csv
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from assemblyvision.errors import AvError
from assemblyvision.output.overlay import render_pose_overlay
from assemblyvision.perception.pose import PoseEstimator, load_det
from assemblyvision.types import Shot
from helpers import fake_root, ffmpeg_clip, make_ctx


def fake_result(n: int):
    boxes = torch.tensor([[10.0 + 50 * i, 20, 60 + 50 * i, 200] for i in range(n)]).reshape(n, 4)
    kp = torch.rand(n, 17, 3) * torch.tensor([300.0, 230.0, 1.0])
    return SimpleNamespace(boxes=SimpleNamespace(xyxy=boxes, conf=torch.full((n,), 0.8), __len__=None),
                           keypoints=SimpleNamespace(data=kp))


class _Boxes(SimpleNamespace):
    def __len__(self):
        return self.xyxy.shape[0]


def result(n: int):
    r = fake_result(n)
    r.boxes = _Boxes(xyxy=r.boxes.xyxy, conf=r.boxes.conf)
    return r


class PoseDetTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.root = Path(self._tmp.name)
        self.cfg, self.ctx = make_ctx(self.root)
        videos = self.ctx.paths.root.parent.parent / "videos"
        videos.mkdir()
        self.video = ffmpeg_clip(videos / "c.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", seconds=1)
        self.shot = Shot("s01", self.ctx.run_id, 0, 25, 0.0, 1.0, "handheld")
        self.pose = PoseEstimator(self.cfg, self.ctx, device="cpu", models_dir=self.root / "models")
        self.pose.model, self.pose.variant = object(), "m"
        self.calls = 0

    def tearDown(self):
        self._tmp.cleanup()

    def people_by_frame(self, frame):
        # frames 0-4: two people, 5-9: nobody, 10-24: one person
        i = self.calls
        self.calls += 1
        return result(2 if i < 5 else 0 if i < 10 else 1)

    def test_det_v1_rows_and_empty_frames(self):
        self.pose._infer = self.people_by_frame
        path = self.pose.estimate_body(self.ctx.run_id, self.shot, video=self.video)
        d = load_det(path)
        self.assertEqual(int(d["frame_count"]), 25)
        self.assertEqual(d["frame_idx"].shape, (25,))          # 5*2 + 15*1 rows
        self.assertEqual(d["body_xy"].shape, (25, 17, 2))
        self.assertEqual(d["body_xy"].dtype, np.float32)
        self.assertFalse(np.isin(np.arange(5, 10), d["frame_idx"]).any())   # TC-POS-02
        self.assertTrue(((d["body_conf"] >= 0) & (d["body_conf"] <= 1)).all())
        m = self.ctx.manifest["pose_body"]
        self.assertEqual((m["frames"], m["frames_with_person"], m["failed_frames"]), (25, 20, 0))
        self.assertIsNotNone(m["pose_fps"])

    def test_failed_frame_logged_and_run_continues(self):
        def infer(frame):   # TC-POS-04
            self.calls += 1
            if self.calls == 4:
                raise RuntimeError("boom")
            return result(1)
        self.pose._infer = infer
        path = self.pose.estimate_body(self.ctx.run_id, self.shot, video=self.video)
        d = load_det(path)
        self.assertEqual(d["frame_idx"].shape, (24,))
        self.assertNotIn(3, d["frame_idx"])
        with open(self.ctx.paths.failed_frames) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual((rows[0]["frame_idx"], rows[0]["code"]), ("3", "AV-POS-003"))

    def test_oom_switches_to_fallback(self):
        def infer(frame):   # TC-POS-05
            self.calls += 1
            if self.calls == 3:
                raise torch.cuda.OutOfMemoryError("fake")
            return result(1)
        loaded = []

        def load(v):
            loaded.append(v)
            self.pose.variant = v
        self.pose._infer = infer
        self.pose.load_models = load
        self.pose.estimate_body(self.ctx.run_id, self.shot, video=self.video)
        m = self.ctx.manifest["pose_body"]
        self.assertEqual(loaded, ["s"])
        self.assertEqual(m["fallback"], {"from": "m", "to": "s", "frame": 2})
        self.assertEqual(m["variant"], "s")
        self.assertEqual(self.ctx.manifest["notices"][0]["code"], "AV-POS-002")

    def test_second_oom_fails_stage(self):
        def infer(frame):
            raise torch.cuda.OutOfMemoryError("fake")
        self.pose._infer = infer
        self.pose.load_models = lambda v: setattr(self.pose, "variant", v)
        with self.assertRaises(AvError) as cm:
            self.pose.estimate_body(self.ctx.run_id, self.shot, video=self.video)
        self.assertEqual(cm.exception.code, "AV-POS-002")

    def test_missing_weights(self):
        with self.assertRaises(AvError) as cm:
            self.pose.load_models("l")
        self.assertEqual(cm.exception.code, "AV-POS-001")

    def test_overlay_from_det(self):
        self.pose._infer = self.people_by_frame
        det = self.pose.estimate_body(self.ctx.run_id, self.shot, video=self.video)
        out = render_pose_overlay(self.video, det, self.ctx.paths.pose_overlay, 25.0, self.cfg, self.ctx)
        self.assertTrue(out.is_file())
        self.assertEqual(self.ctx.manifest["overlay"]["frames"], 25)


class PrecisionTest(unittest.TestCase):
    """pose.half: FP16 only on a CUDA GPU with tensor cores (compute capability >= 7.0)."""

    def setUp(self):
        self._tmp = fake_root()
        self.cfg, self.ctx = make_ctx(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def resolve(self, device, cc, half=True):
        pose = PoseEstimator(self.cfg, self.ctx, device=device, models_dir=Path(self._tmp.name))
        pose.pcfg = {**self.cfg["pose"], "half": half}
        fake = SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda i: cc))
        return pose.resolve_half(fake), pose.gpu_cc

    def test_resolve_half(self):
        self.assertIs(self.cfg["pose"]["half"], False)                       # default FP32
        self.assertEqual(self.resolve("cuda:0", (8, 6)), (True, "8.6"))        # Ampere A5000, A4000
        self.assertEqual(self.resolve("cuda:0", (6, 1)), (False, "6.1"))       # Pascal GTX 1080 Ti
        self.assertEqual(self.resolve("cuda:0", (8, 6), half=False), (False, "8.6"))
        self.assertEqual(self.resolve("cpu", (8, 6)), (False, None))


if __name__ == "__main__":
    unittest.main()
