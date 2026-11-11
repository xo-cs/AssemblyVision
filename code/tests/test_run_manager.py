"""RunManager starts a worker process and returns at once; run_pose --run-id (CPU, fake worker)."""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from assemblyvision.errors import AvError
from assemblyvision.runs.manager import RunManager
from assemblyvision.storage.runstore import RunStore
from ui_helpers import make_root

FAKE = [sys.executable, str(Path(__file__).with_name("fake_worker.py"))]


def wait_finished(store, run_id, timeout=20.0):
    t0, seen = time.monotonic(), []
    while time.monotonic() - t0 < timeout:
        st = store.status(run_id)
        seen.append(st.progress)
        if st.status in ("done", "partial", "failed"):
            return st, seen
        time.sleep(0.02)
    raise AssertionError("worker did not finish")


class RunManagerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = make_root(self.root)
        self.video = self.root / "data/public/videos/clip.mp4"
        self.video.write_bytes(b"x")
        self.store = RunStore(["public"], self.cfg, self.root)
        self.rm = RunManager(self.store, self.cfg, worker_cmd=FAKE)
        os.environ["AV_TEST_ROOT"] = str(self.root)

    def tearDown(self):
        os.environ.pop("AV_TEST_ROOT", None)
        os.environ.pop("AV_TEST_STEP_S", None)
        self._tmp.cleanup()

    def test_create_and_start(self):   # TC-APP-01
        run_id, notices = self.rm.create_run(self.video, "public", "s")
        self.assertEqual(notices, [])
        m = self.store.get_run(run_id)
        self.assertEqual((m["status"], m["variant"], m["engine"], m["source"]), ("queued", "s", "A+B v1", "public"))
        t0 = time.monotonic()
        pid = self.rm.start(run_id)
        self.assertLess(time.monotonic() - t0, 1.0)              # returns at once (NFR-PERF-04)
        self.assertEqual(int(self.store.paths(run_id).worker_pid.read_text()), pid)
        st, seen = wait_finished(self.store, run_id)
        self.rm.reap()
        self.assertEqual(st.status, "done")
        self.assertEqual(seen, sorted(seen))                      # TC-APP-07
        self.assertEqual(seen[-1], 1.0)
        argv = self.store.get_run(run_id)["argv"]
        self.assertEqual(argv[argv.index("--run-id") + 1], run_id)
        self.assertIn('seg_a.engine="A+B v1"', argv)                    # the step engine reaches the worker
        self.assertEqual(argv[argv.index("--gpu") + 1], str(self.cfg["gpu"]["pose"]))
        self.assertTrue(self.store.paths(run_id).worker_out.is_file())

    def test_step_engine_choice(self):
        run_id, notices = self.rm.create_run(self.video, "public", "s", "A")
        self.assertEqual((notices, self.store.get_run(run_id)["engine"]), ([], "A"))
        self.assertEqual(self.store.get_run(run_id)["overrides"]["seg_a"]["engine"], "A")
        run_id, notices = self.rm.create_run(self.video, "public", "s", "B only")
        self.assertEqual([n.code for n in notices], ["AV-APP-006"])
        self.assertEqual(self.store.get_run(run_id)["engine"], "A+B v1")

    def test_second_start_refused(self):   # AV-RUN-007
        os.environ["AV_TEST_STEP_S"] = "0.3"
        run_id, _ = self.rm.create_run(self.video, "public", "m")
        self.rm.start(run_id)
        with self.assertRaises(AvError) as cm:
            self.rm.start(run_id)
        self.assertEqual(cm.exception.code, "AV-RUN-007")
        self.assertTrue(self.store.status(run_id).active or self.store.status(run_id).status == "done")
        wait_finished(self.store, run_id)
        self.rm.reap()

    def test_unsupported_variant_and_wrong_area(self):
        run_id, notices = self.rm.create_run(self.video, "public", "xl")
        self.assertEqual([n.code for n in notices], ["AV-APP-006"])
        self.assertEqual(self.store.get_run(run_id)["variant"], self.cfg["pose"]["variant"])
        q = self.root / "data/qisens/videos/q.mp4"
        q.write_bytes(b"x")
        for video, source in ((q, "public"), (q, "qisens")):
            with self.assertRaises(AvError) as cm:
                self.rm.create_run(video, source, "m")
            self.assertEqual(cm.exception.code, "AV-RUN-001")
        self.assertEqual(list((self.root / "data/qisens/runs").iterdir()), [])


class RunPoseRunIdTest(unittest.TestCase):
    def test_bad_run_id_is_an_argument_error(self):
        from assemblyvision.tools.run_pose import parse_args
        with self.assertRaises(SystemExit):
            parse_args(["--source", "public", "--video", "x.mp4", "--variant", "n", "--gpu", "-1",
                        "--run-id", "../run"])
        with self.assertRaises(SystemExit):
            parse_args(["--run", "x", "--stages", "metrics", "--run-id", "run_20261026_010203_ab12"])

    def test_run_id_without_queued_folder_is_refused(self):
        """No folder is created and nothing heavy is imported."""
        from assemblyvision.storage.files import REPO_ROOT
        video = REPO_ROOT / "data/public/videos/process_480p30.mp4"
        if not video.is_file():
            self.skipTest("public clip not present")
        rid = "run_19990101_000000_0000"
        code = (f"import sys; from assemblyvision.tools.run_pose import main; "
                f"r = main(['--source', 'public', '--video', {str(video)!r}, '--variant', 'n', '--gpu', '-1', "
                f"'--run-id', {rid!r}]); print('torch' in sys.modules); sys.exit(r)")
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(res.returncode, 1, res.stderr)
        self.assertIn("no queued run folder", res.stderr)
        self.assertEqual(res.stdout.strip(), "False")
        self.assertFalse((REPO_ROOT / "data/public/runs" / rid).exists())


if __name__ == "__main__":
    unittest.main()
