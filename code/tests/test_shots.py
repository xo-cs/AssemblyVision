import subprocess
import unittest
from pathlib import Path

from assemblyvision.errors import AvError
from assemblyvision.ingest.ingestor import VideoIngestor
from assemblyvision.ingest.shots import ShotDetector
from helpers import fake_root, make_ctx

PAN = "testsrc2=size=1280x720:rate=30,crop=320:240:x='40+t*120':y=200"
STILL = "smptehdbars=size=320x240:rate=30"
MANDEL = "mandelbrot=size=320x240:rate=30:end_scale=0.3"


def concat(out: Path, sources: list[str], seconds: float = 2.0) -> Path:
    args = ["ffmpeg", "-nostdin", "-y", "-v", "error"]
    for src in sources:
        args += ["-f", "lavfi", "-t", str(seconds), "-i", src]
    n = len(sources)
    chain = "".join(f"[{i}:v]setsar=1,format=yuv420p[v{i}];" for i in range(n))
    chain += "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[out]"
    subprocess.run(args + ["-filter_complex", chain, "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                           "-r", "30", str(out)], check=True)
    return out


class ShotTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.cfg, self.ctx = make_ctx(Path(self._tmp.name))
        self.dir = self.ctx.paths.root.parent.parent / "videos"
        self.dir.mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def detect(self, video):
        info, _ = VideoIngestor(self.cfg, self.ctx).ingest(self.ctx.run_id, video, "open", strict_split=False)
        return ShotDetector(self.cfg, self.ctx).detect(self.ctx.run_id, self.ctx.paths.cfr, info)

    def test_two_cuts_three_shots_with_types(self):   # TC-SHT-01, TC-SHT-02
        video = concat(self.dir / "cuts.mp4", [STILL, PAN, MANDEL])
        shots = self.detect(video)
        self.assertEqual(len(shots), 3)
        for got, want in zip([s.start_frame for s in shots[1:]], [60, 120]):
            self.assertLessEqual(abs(got - want), 1)
        self.assertEqual(shots[0].shot_type, "fixed")
        self.assertEqual(shots[1].shot_type, "handheld")
        self.assertEqual(shots[0].shot_id, f"{self.ctx.run_id}-s001")
        self.assertEqual(len(shots[0].shot_id), 29)
        m = self.ctx.manifest["shots"]
        self.assertEqual(m["count"], 3)
        self.assertEqual(shots[-1].end_frame, 180)

    def test_short_segment_merges(self):
        video = concat(self.dir / "short.mp4", [STILL, PAN], seconds=2.0)
        shots = self.detect(video)
        self.assertTrue(all(s.end_s - s.start_s >= 1.0 for s in shots))

    def test_failure_falls_back_to_one_shot(self):
        video = concat(self.dir / "one.mp4", [STILL])
        info, _ = VideoIngestor(self.cfg, self.ctx).ingest(self.ctx.run_id, video, "open", strict_split=False)
        import assemblyvision.ingest.shots as mod
        orig = mod.detect_cuts
        mod.detect_cuts = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            shots = ShotDetector(self.cfg, self.ctx).detect(self.ctx.run_id, self.ctx.paths.cfr, info)
        finally:
            mod.detect_cuts = orig
        self.assertEqual([(s.start_frame, s.end_frame, s.shot_type) for s in shots], [(0, 60, "handheld")])
        self.assertIn("AV-SHT-001", [n["code"] for n in self.ctx.manifest["notices"]])

    def test_scale_from_markings(self):   # TC-SHT-05
        self.assertAlmostEqual(ShotDetector.scale_from_markings((0, 0), (200, 0), 2.0), 0.01)
        with self.assertRaises(AvError) as cm:
            ShotDetector.scale_from_markings((5, 5), (5, 5), 2.0)
        self.assertEqual(cm.exception.code, "AV-SHT-002")


if __name__ == "__main__":
    unittest.main()
