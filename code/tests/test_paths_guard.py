import os
import unittest
from pathlib import Path

from assemblyvision.config import Config
from assemblyvision.errors import AvError
from assemblyvision.storage.files import RunPaths, check_video_source, new_run_id, write_atomic
from helpers import fake_root, make_ctx


class PathGuardTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.root = Path(self._tmp.name)
        self.cfg = Config.load()
        self.pc = self.cfg["paths"]
        self.q = self.root / self.pc["qisens_root"]
        self.p = self.root / self.pc["public_root"]
        (self.q / "videos").mkdir(parents=True)
        (self.p / "videos").mkdir(parents=True)
        self.qv = self.q / "videos" / "a.mp4"
        self.pv = self.p / "videos" / "b.mp4"
        self.qv.write_bytes(b"x")
        self.pv.write_bytes(b"x")

    def tearDown(self):
        self._tmp.cleanup()

    def check(self, video, source):
        return check_video_source(video, source, self.pc, repo_root=self.root)

    def test_matching_source_accepted(self):
        self.assertEqual(self.check(self.qv, "qisens"), self.qv.resolve())
        self.assertEqual(self.check(self.pv, "public"), self.pv.resolve())

    def test_mismatch_refused(self):
        for video, source in ((self.qv, "public"), (self.pv, "qisens"),
                              (self.root / "elsewhere.mp4", "public"),
                              (self.root / "elsewhere.mp4", "qisens")):
            with self.assertRaises(AvError) as cm:
                self.check(video, source)
            self.assertEqual(cm.exception.code, "AV-RUN-001")

    def test_symlink_into_other_area_refused(self):
        link = self.p / "videos" / "link.mp4"
        os.symlink(self.qv, link)
        with self.assertRaises(AvError):
            self.check(link, "public")
        dotdot = self.p / "videos" / ".." / ".." / Path(self.pc["qisens_root"]).name / "videos" / "a.mp4"
        with self.assertRaises(AvError):
            self.check(dotdot, "public")

    def test_run_folder_under_source(self):
        for source, area in (("qisens", self.q), ("public", self.p)):
            rp = RunPaths.for_source(source, "run_x", self.pc, repo_root=self.root)
            self.assertTrue(rp.root.is_relative_to(area.resolve()))
            for p in (rp.cfr, rp.det("s"), rp.pose_overlay, rp.run_log, rp.tmp_dir, rp.manifest):
                self.assertTrue(p.is_relative_to(area.resolve() / "runs" / "run_x"))

    def test_run_id(self):
        ids = {new_run_id() for _ in range(1000)}
        self.assertEqual(len(ids), 1000)
        self.assertTrue(all(len(i) == 24 and i.startswith("run_") for i in ids))

    def test_write_atomic_guard(self):
        _, ctx = make_ctx(self.root)
        out = write_atomic(ctx.paths.root / "a.bin", b"abc", ctx.paths)
        self.assertEqual(out.read_bytes(), b"abc")
        self.assertFalse((ctx.paths.root / "a.bin.tmp").exists())
        for bad in (self.root / "outside.bin", ctx.paths.root / ".." / "x.bin"):
            with self.assertRaises(AvError) as cm:
                write_atomic(bad, b"abc", ctx.paths)
            self.assertEqual(cm.exception.code, "AV-STO-005")

    def test_write_atomic_keeps_old_file_on_failure(self):
        _, ctx = make_ctx(self.root)
        target = ctx.paths.root / "f.bin"
        write_atomic(target, b"old", ctx.paths)

        def broken(f):
            f.write(b"half")
            raise OSError("disk full")
        with self.assertRaises(AvError) as cm:
            write_atomic(target, broken, ctx.paths)
        self.assertEqual(cm.exception.code, "AV-STO-004")
        self.assertEqual(target.read_bytes(), b"old")
        self.assertFalse((ctx.paths.root / "f.bin.tmp").exists())


if __name__ == "__main__":
    unittest.main()
