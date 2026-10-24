import os
import subprocess
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.errors import AvError
from assemblyvision.ingest.ingestor import VideoIngestor, iter_frames
from assemblyvision.types import VideoInfo
from helpers import fake_root, ffmpeg_clip, make_ctx


class IngestTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.root = Path(self._tmp.name)
        self.cfg, self.ctx = make_ctx(self.root)
        self.ing = VideoIngestor(self.cfg, self.ctx)
        self.videos = self.ctx.paths.root.parent.parent / "videos"
        self.videos.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def test_cfr_h264_is_symlinked(self):
        src = ffmpeg_clip(self.videos / "cfr.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p")
        info, split = self.ing.ingest(self.ctx.run_id, src, "open", strict_split=False)
        cfr = self.ctx.paths.cfr
        self.assertTrue(cfr.is_symlink())
        self.assertFalse(os.path.isabs(os.readlink(cfr)))
        self.assertEqual(cfr.resolve(), src.resolve())
        self.assertEqual((info.fps, info.width, info.height, info.frame_count), (25.0, 320, 240, 75))
        self.assertIsNone(split)   # 3 s < 60 s gives a notice, not a failure
        self.assertEqual(self.ctx.manifest["notices"][0]["code"], "AV-ING-005")
        self.assertFalse(self.ctx.manifest["ingest"]["transcoded"])

    def test_short_video_strict_split_fails(self):
        src = ffmpeg_clip(self.videos / "cfr.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p")
        with self.assertRaises(AvError) as cm:
            self.ing.ingest(self.ctx.run_id, src, "open")
        self.assertEqual(cm.exception.code, "AV-ING-005")

    def test_vfr_is_transcoded_to_equal_intervals(self):
        # TC-ING-02: drop frames irregularly to get a variable frame rate.
        src = ffmpeg_clip(self.videos / "vfr.mp4", "-vf", "select='not(between(n,10,20))*not(eq(mod(n,7),3))'",
                          "-vsync", "vfr", "-c:v", "libx264", "-pix_fmt", "yuv420p")
        info, _ = self.ing.ingest(self.ctx.run_id, src, "open", strict_split=False)
        m = self.ctx.manifest["ingest"]
        self.assertTrue(m["transcoded"])
        self.assertTrue(any("variable" in r for r in m["transcode_reasons"]))
        self.assertFalse(self.ctx.paths.cfr.is_symlink())
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "packet=pts_time", "-of", "csv=p=0", str(self.ctx.paths.cfr)],
                             capture_output=True, text=True).stdout.split()
        d = np.diff(sorted(float(t.rstrip(",")) for t in out))
        self.assertLess(np.max(np.abs(d - 1 / info.fps)), 0.001)

    def test_high_fps_capped(self):
        src = ffmpeg_clip(self.videos / "fast.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", rate=60)
        info, _ = self.ing.ingest(self.ctx.run_id, src, "open", strict_split=False)
        self.assertEqual(info.fps, 30.0)
        self.assertEqual(info.frame_count, 90)
        self.assertEqual(self.ctx.manifest["ingest"]["target_fps"], 30)

    def test_avi_accepted_mov_refused(self):
        avi = ffmpeg_clip(self.videos / "a.avi", "-c:v", "mpeg4")
        info, _ = self.ing.ingest(self.ctx.run_id, avi, "open", strict_split=False)
        self.assertEqual(info.frame_count, 75)
        with self.assertRaises(AvError) as cm:
            self.ing.ingest(self.ctx.run_id, self.videos / "a.mov", "open")
        self.assertEqual(cm.exception.code, "AV-ING-002")

    def test_corrupt_file(self):
        bad = self.videos / "bad.mp4"
        bad.write_bytes(b"not a video" * 100)
        with self.assertRaises(AvError) as cm:
            self.ing.ingest(self.ctx.run_id, bad, "open")
        self.assertEqual(cm.exception.code, "AV-ING-001")

    def test_split(self):
        s = self.ing.make_split(VideoInfo("x", 30.0, 1, 1, 18000, 600.0))
        self.assertEqual((s.train_end_s, s.val_end_s), (360.0, 420.0))

    def test_iter_frames_range(self):
        src = ffmpeg_clip(self.videos / "cfr.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p")
        idx = [i for i, _ in iter_frames(src, 10, 20)]
        self.assertEqual(idx, list(range(10, 20)))


if __name__ == "__main__":
    unittest.main()
