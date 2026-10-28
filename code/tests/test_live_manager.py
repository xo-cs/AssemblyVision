"""LiveManager with a fake pose model (CPU): one frame slot, subject choice,
GPU lock and memory checks, recording and Analyse recording."""
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.errors import AvError
from assemblyvision.live.manager import LiveManager
from assemblyvision.live.perception import Detections
from live_helpers import cfg, side_body

FAKE_WORKER = [sys.executable, str(Path(__file__).with_name("fake_worker.py"))]


def frame(i: int, h: int = 240, w: int = 320) -> np.ndarray:
    f = np.full((h, w, 3), 40, np.uint8)
    f[0, 0, 0], f[0, 0, 1] = i % 256, i // 256
    return f


class FakePose:
    """People per frame from a function of the frame index (pixel 0,0); optional delay."""

    def __init__(self, people=None, delay=0.0):
        self.people = people or (lambda i: [(150.0, 1.0)])     # [(x of the body, size factor)]
        self.delay = delay
        self.seen: list[int] = []

    def predict(self, rgb):
        i = int(rgb[0, 0, 0]) + 256 * int(rgb[0, 0, 1])
        self.seen.append(i)
        if self.delay:
            time.sleep(self.delay)
        boxes, conf, kps, kconf = [], [], [], []
        for x, s in self.people(i):
            kp, c = side_body(5.0, x=x, floor=230.0, leg=40 * s, torso=50 * s)
            kps.append(kp)
            kconf.append(c)
            boxes.append([kp[:, 0].min() - 10, kp[:, 1].min() - 10, kp[:, 0].max() + 10, kp[:, 1].max() + 5])
            conf.append(0.9)
        if not boxes:
            return Detections.empty()
        return Detections(np.array(boxes), np.array(conf), np.array(kps), np.array(kconf))


class ManagerTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = cfg(lost_s=0.5, idle_stop_s=30.0, model_keep_s=0.0)
        self.pose = FakePose()
        self.lm = LiveManager(self.cfg, self.root / "live", backend_factory=lambda: self.pose,
                              gpu_free=lambda i: 8000, lock_dir=self.root / "locks", port=7861)

    def tearDown(self):
        self.lm.stop()
        self._tmp.cleanup()

    def feed(self, n, fps=20.0, start=0):
        for i in range(start, start + n):
            self.lm.push(frame(i))
            time.sleep(1 / fps)

    def wait_processed(self, k, timeout=10):
        t0 = time.monotonic()
        while self.lm.session.processed < k and time.monotonic() - t0 < timeout:
            time.sleep(0.01)

    def test_push_never_waits_and_latest_wins(self):   # TC-LIV-05
        self.pose.delay = 0.05
        self.lm.start("side_work")
        self.lm.session.rate.steps = [1000]                        # no pacing: this test is about the slot
        took = []
        for i in range(50):
            t0 = time.perf_counter()
            self.lm.push(frame(i))
            took.append(time.perf_counter() - t0)
            time.sleep(0.002)
        time.sleep(0.3)
        s = self.lm.session
        self.assertLess(max(took), 0.005)
        self.assertEqual(self.pose.seen[-1], 49)                  # the newest frame was processed
        self.assertGreater(s.dropped, 30)
        self.assertEqual(s.frames_in, 50)
        self.assertEqual(s.processed + s.dropped, 50)
        self.assertIsNotNone(self.lm.panel()["delay_p95_ms"])

    def test_largest_person_click_lock_and_lost(self):   # TC-LIV-06
        self.pose.people = lambda i: [(80.0, 1.3), (240.0, 1.0)] if i < 40 else [(80.0, 1.3)]
        self.lm.start("side_work")
        self.feed(10)
        self.wait_processed(8)
        s = self.lm.session
        big = s.subject_id
        boxes, ids = s.latest_boxes, s.latest_ids
        small = int(np.argmin((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        self.assertNotEqual(ids[small], big)
        x, y = (boxes[small, 0] + boxes[small, 2]) / 2, (boxes[small, 1] + boxes[small, 3]) / 2
        self.assertTrue(s.lock_subject(x, y))
        self.feed(10, start=10)
        self.wait_processed(18)
        self.assertEqual(s.subject_id, ids[small])
        with self.assertRaises(AvError) as cm:
            s.lock_subject(5, 5)
        self.assertEqual(cm.exception.code, "AV-LIV-009")
        self.feed(30, start=40)                                    # the locked person is gone for 1.5 s
        notes = " ".join(self.lm.panel()["notices"])
        self.assertIn("AV-LIV-004", notes)
        self.assertEqual(s.subject_id, big)                        # unlocked after lost_s: largest again

    def test_second_start_and_gpu_checks(self):   # TC-LIV-07
        self.lm.start("side_work")
        with self.assertRaises(AvError) as cm:
            self.lm.start("side_work")
        self.assertEqual(cm.exception.code, "AV-LIV-002")
        self.lm.stop()
        f = open(self.root / "locks" / "gpu1.lock", "a+")          # another instance holds GPU 1
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with self.assertRaises(AvError) as cm:
            self.lm.start("side_work")
        self.assertEqual(cm.exception.code, "AV-LIV-002")
        f.close()
        low = LiveManager(self.cfg, self.root / "live", backend_factory=lambda: self.pose,
                          gpu_free=lambda i: 100, lock_dir=self.root / "locks")
        with self.assertRaises(AvError) as cm:
            low.start("side_work")
        self.assertEqual(cm.exception.code, "AV-LIV-005")
        self.lm.start("side_work")                                 # the lock was given back
        self.lm.stop()

    def test_no_recording_writes_only_the_summary(self):   # TC-LIV-13
        sid = self.lm.start("floor_pick")
        self.feed(20)
        summary = self.lm.stop()
        self.assertEqual(sorted(os.listdir(self.root / "live" / sid)), ["summary.json"])
        self.assertIsNone(summary["recording"])
        self.assertEqual(summary["stations"][0]["title"], "B. Floor pick")
        self.assertEqual(json.loads((self.root / "live" / sid / "summary.json").read_text())["session_id"], sid)

    def test_recording_at_ten_fps_and_analyse(self):   # TC-LIV-09, TC-LIV-10
        sid = self.lm.start("workbench", save_recording=True)
        self.feed(60, fps=25)                                       # 2.4 s of frames
        summary = self.lm.stop()
        rec = Path(summary["recording"])
        self.assertTrue(rec.is_file())
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_frames",
                                "-show_entries", "stream=r_frame_rate,nb_read_frames", "-of", "json", str(rec)],
                               capture_output=True, text=True)
        st = json.loads(probe.stdout)["streams"][0]
        self.assertEqual(st["r_frame_rate"], "10/1")
        self.assertAlmostEqual(int(st["nb_read_frames"]), 24, delta=4)
        # Analyse recording: a batch run on the recording with pause segments
        from assemblyvision.runs.manager import RunManager
        from assemblyvision.storage.runstore import RunStore
        from ui_helpers import make_root
        repo = self.root / "repo"
        c = make_root(repo)
        live_root = repo / "data/public/live"
        live_root.mkdir(parents=True)
        os.replace(self.root / "live" / sid, live_root / sid)
        store = RunStore(["public"], c, repo)
        rm = RunManager(store, c, worker_cmd=FAKE_WORKER)
        lm2 = LiveManager(self.cfg, live_root, backend_factory=lambda: self.pose, gpu_free=lambda i: 8000,
                          lock_dir=self.root / "locks")
        os.environ["AV_TEST_ROOT"] = str(repo)
        try:
            run_id = lm2.to_batch(sid, "s", rm)
            t0 = time.monotonic()
            while store.status(run_id).status != "done" and time.monotonic() - t0 < 20:
                time.sleep(0.05)
            rm.reap()
        finally:
            os.environ.pop("AV_TEST_ROOT", None)
        argv = store.get_run(run_id)["argv"]
        self.assertIn("steps.use=false", argv)
        self.assertEqual(store.get_run(run_id)["video"], str((live_root / sid / "recording.mp4").resolve()))
        self.assertIn(run_id, json.loads((live_root / sid / "summary.json").read_text())["batch_runs"])

    def test_analyse_without_recording(self):   # TC-LIV-10
        sid = self.lm.start("side_work")
        self.feed(5)
        self.lm.stop()
        with self.assertRaises(AvError) as cm:
            self.lm.to_batch(sid, "s", None)
        self.assertEqual(cm.exception.code, "AV-LIV-008")

    def test_stations_and_pause(self):
        self.lm.start("workbench")
        self.feed(10)
        self.lm.session.switch_station("floor_pick")
        self.feed(10, start=10)
        time.sleep(0.8)                                             # camera paused (lost_s 0.5)
        self.assertTrue(self.lm.panel()["paused"])
        self.feed(5, start=20)
        self.wait_processed(22)
        self.assertFalse(self.lm.panel()["paused"])
        summary = self.lm.stop()
        self.assertEqual([s["title"] for s in summary["stations"]], ["A. Workbench", "B. Floor pick"])

    def test_bad_template_falls_back(self):   # AV-LIV-003
        self.lm.start("no_such_station")
        self.feed(5)
        self.wait_processed(3)
        p = self.lm.panel()
        self.assertIn("AV-LIV-003", " ".join(p["notices"]))
        self.assertEqual(p["station"]["title"], "Segments (no template)")


if __name__ == "__main__":
    unittest.main()
