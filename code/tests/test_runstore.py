"""RunStore: reads run folders of the allowed data areas only (CPU, temporary data root)."""
import json
import os
import tempfile
import unittest
from pathlib import Path

from assemblyvision.errors import AvError
from assemblyvision.storage.runstore import RunStore
from ui_helpers import make_root, make_run

RID = "run_20261026_010203_ab12"
QID = "run_20261026_010203_cd34"


class RunStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = make_root(self.root)
        self.p = make_run(self.root, self.cfg, RID)
        self.store = RunStore(["public"], self.cfg, self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_reads_results(self):
        self.assertEqual([r["run_id"] for r in self.store.list_runs()], [RID])
        segs = self.store.get_segments(RID)
        self.assertEqual([(s["seg"], s["step"], s["label"]) for s in segs],
                         [(1, 0, "other"), (2, 1, "Removal"), (3, 0, "other"), (4, 2, "Cleaning")])
        rows = self.store.get_metrics(RID)
        self.assertEqual(len(rows), 12)
        null = [r for r in rows if r["metric"] == "two_hand_simultaneous"][0]
        self.assertIsNone(null["value"])
        self.assertEqual(null["flags"], "low_coverage")
        self.assertEqual(self.store.get_eval(RID)["metrics"]["f1_50"], 50.0)
        self.assertEqual(self.store.get_baseline(RID)[0]["label"], "Removal")
        ref = self.store.reference_items(RID)[0]
        self.assertEqual((ref["subject_box"], ref["box_frame"]), ([10, 20, 60, 110], 5))

    def test_missing_files(self):
        make_run(self.root, self.cfg, "run_20261026_010203_0000", with_results=False)
        self.assertEqual(self.store.get_segments("run_20261026_010203_0000"), [])
        self.assertEqual(self.store.get_metrics("run_20261026_010203_0000"), [])
        self.assertIsNone(self.store.get_eval("run_20261026_010203_0000"))
        self.assertIsNone(self.store.overlay_path(RID))

    def test_bad_run_ids(self):   # AV-APP-003
        for bad in ("../" + RID, RID + "/..", "run_x", "", None, QID):
            with self.assertRaises(AvError) as cm:
                self.store.get_run(bad)
            self.assertEqual(cm.exception.code, "AV-APP-003")

    def test_closed_source_is_never_touched(self):
        """With sources=public, a QiSens area that cannot even be listed causes no error."""
        make_run(self.root, self.cfg, QID, source="qisens")
        q = self.root / "data" / "qisens"
        (q / "videos" / "x.mp4").write_bytes(b"0")
        os.chmod(q, 0)
        try:
            self.assertEqual([r["run_id"] for r in self.store.list_runs()], [RID])
            self.assertTrue(all(lab.startswith("public/") for lab, _ in self.store.list_videos()))
            with self.assertRaises(AvError):
                self.store.get_run(QID)
            self.assertIsNone(self.store.active_run())
            self.assertNotIn("qisens", self.store.roots)
        finally:
            os.chmod(q, 0o755)
        both = RunStore(["qisens", "public"], self.cfg, self.root)
        self.assertEqual({r["run_id"] for r in both.list_runs()}, {RID, QID})

    def test_upload(self):   # TC-APP-02
        src = self.root / "up"
        src.mkdir()
        names = []
        for i, name in enumerate(["clip.mp4", "clip.mp4", "Clip Two.AVI"]):
            f = src / f"t{i}"
            f.write_bytes(b"x")
            names.append(self.store.save_upload(f, name, "public").name)
        self.assertEqual(names, ["clip.mp4", "clip_1.mp4", "Clip_Two.avi"])
        self.assertTrue((self.root / "data/public/input/clip_1.mp4").is_file())
        f = src / "m"
        f.write_bytes(b"x")
        with self.assertRaises(AvError) as cm:
            self.store.save_upload(f, "clip.mov", "public")
        self.assertEqual(cm.exception.code, "AV-APP-001")
        with self.assertRaises(AvError):
            self.store.save_upload(f, "clip.mp4", "qisens")       # source not open
        self.assertFalse((self.root / "data/qisens/input").exists())

    def test_progress_never_goes_down(self):   # TC-APP-07
        m = json.loads(self.p.manifest.read_text())
        m["status"] = "running"
        m.pop("summary", None)
        names = ["ingest", "shots", "reference", "pose_body", "tracking", "segmentation", "metrics", "overlay"]
        seen = []
        for k, name in enumerate(names):
            for frac in (0.0, 0.5, 0.9):
                m["stages"] = {n: {"status": "done"} for n in names[:k]}
                m["stages"][name] = {"status": "running", "progress": frac}
                m["stages"].update({n: {"status": "pending"} for n in names[k + 1:]})
                self.p.manifest.write_text(json.dumps(m))
                st = self.store.status(RID)
                self.assertEqual(st.stage, name)
                seen.append(st.progress)
        m["status"] = "done"
        m["stages"] = {n: {"status": "done"} for n in names}
        self.p.manifest.write_text(json.dumps(m))
        seen.append(self.store.status(RID).progress)
        self.assertEqual(seen, sorted(seen))
        self.assertEqual(seen[-1], 1.0)
        self.assertLess(seen[-2], 1.0)

    def test_lost_worker_gives_av_run_004(self):
        m = json.loads(self.p.manifest.read_text())
        m["status"] = "running"
        m["stages"]["pose_body"] = {"status": "running", "progress": 0.3}
        self.p.manifest.write_text(json.dumps(m))
        self.p.worker_pid.write_text("999999999")                # no such process
        st = self.store.status(RID)
        self.assertEqual(st.status, "failed")
        self.assertFalse(st.active)
        self.assertIn("AV-RUN-004", st.error)
        self.assertEqual(json.loads(self.p.manifest.read_text())["stages"]["pose_body"]["status"], "failed")

    def test_pid_of_another_program_is_not_the_worker(self):
        m = json.loads(self.p.manifest.read_text())
        m["status"] = "running"
        self.p.manifest.write_text(json.dumps(m))
        self.p.worker_pid.write_text(str(os.getpid()))          # this test process, without the run ID
        self.assertEqual(self.store.status(RID).status, "failed")


if __name__ == "__main__":
    unittest.main()
