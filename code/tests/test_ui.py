"""GradioApp: bind rules, page build and the Run and Results handlers on stored
runs (CPU, temporary data root, no server started, fake worker)."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from assemblyvision.errors import AvError
from assemblyvision.ui.app import ALL_STEPS, GradioApp, check_bind, parse_sources, prepare_env
from ui_helpers import make_root, make_run

RID = "run_20261026_010203_ab12"
FAKE = [sys.executable, str(Path(__file__).with_name("fake_worker.py"))]
_ENV_KEYS = ("GRADIO_TEMP_DIR", "GRADIO_ANALYTICS_ENABLED", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
             "MPLCONFIGDIR", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "TMPDIR", "AV_TEST_ROOT")


class BindTest(unittest.TestCase):
    def test_refuses_share_and_public_hosts(self):   # TC-APP-16
        check_bind("127.0.0.1", False)
        check_bind("localhost", False)
        check_bind("10.0.0.5", False)
        for host, share in (("127.0.0.1", True), ("0.0.0.0", False), ("8.8.8.8", False), ("example.com", False)):
            with self.assertRaises(AvError) as cm:
                check_bind(host, share)
            self.assertEqual(cm.exception.code, "AV-APP-005")

    def test_launch_refuses_before_starting(self):
        app = GradioApp.__new__(GradioApp)
        with self.assertRaises(AvError):
            app.launch(share=True)
        with self.assertRaises(AvError):
            app.launch(host="0.0.0.0")

    def test_sources_argument(self):
        self.assertEqual(parse_sources("public"), ["public"])
        self.assertEqual(parse_sources("qisens,public"), ["qisens", "public"])
        for bad in ("", "other", "public,../x"):
            with self.assertRaises(Exception):
                parse_sources(bad)


class AppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._env = {k: os.environ.get(k) for k in _ENV_KEYS}
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.cfg = make_root(cls.root)
        cls.tmpdir = prepare_env(["public"], cls.cfg, cls.root)

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        tempfile.tempdir = None
        cls._tmp.cleanup()

    def setUp(self):
        for d in (self.root / "data/public/runs").iterdir():
            import shutil
            shutil.rmtree(d)
        self.app = GradioApp(self.cfg, ["public"], self.root, worker_cmd=FAKE)

    def test_env_stays_in_the_data_area(self):
        self.assertEqual(self.tmpdir, self.root / "data/public/tmp/gradio")
        self.assertEqual(os.environ["GRADIO_TEMP_DIR"], str(self.tmpdir))
        self.assertEqual(os.environ["GRADIO_ANALYTICS_ENABLED"], "False")
        q = Path(tempfile.mkdtemp())
        self.assertTrue(q.is_relative_to(self.tmpdir))
        q.rmdir()

    def test_build(self):
        demo = self.app.build()
        labels = {c.label for c in demo.blocks.values() if getattr(c, "label", None)}
        for want in ("Run", "Video", "Overlay video", "Process timeline (Engine A)", "Step"):
            self.assertIn(want, labels)

    def test_stored_run_fills_results_quickly(self):   # TC-APP-10
        make_run(self.root, self.cfg, RID)
        t0 = time.monotonic()
        with mock.patch("subprocess.Popen", side_effect=AssertionError("a process was started")):
            r = self.app.results(RID)
            html_, active = self.app.status_html(RID)
            gal = self.app.subject_gallery(RID)
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertFalse(active)
        self.assertIn("Finished", html_)
        self.assertEqual([row[1] for row in r["steps"]], ["Phase 1", "Phase 2"])   # public run: display names
        self.assertEqual(r["timeline"].ndim, 3)
        self.assertEqual(r["bending"].ndim, 3)
        self.assertIn("F1@50", r["eval"])
        self.assertEqual(r["steps_choices"], [ALL_STEPS, ("1 Phase 1", "2"), ("2 Phase 2", "4")])
        self.assertEqual(len(gal), 1)
        self.assertIn("subject t01", gal[0][1])
        # metrics of one step, NULL shown as n/a with its flag
        rows = self.app.metrics_table(RID, "4")
        self.assertTrue(all(row[0] == "2 Phase 2" for row in rows))
        null = [row for row in rows if row[1] == "two_hand_simultaneous"][0]
        self.assertEqual((null[2], null[3], null[4]), ("n/a", "30 %", "low_coverage"))
        self.assertEqual(self.app.bending_rows(RID)[0], ("1 Phase 1", [5.0, 2.0, 1.0, 0.0]))

    def test_partial_run(self):   # TC-APP-09
        stages = {s: {"status": "done"} for s in ("ingest", "shots", "reference", "pose_body", "tracking")}
        stages["segmentation"] = {"status": "failed", "error": {"code": "AV-SEG-002", "text": "bad steps"}}
        stages["metrics"] = {"status": "skipped"}
        stages["overlay"] = {"status": "done"}
        make_run(self.root, self.cfg, RID, status="partial", stages=stages, with_results=False)
        r = self.app.results(RID)
        self.assertIn("not available. AV-SEG-002 bad steps", r["note"])
        self.assertIn("Motion metrics**: not available (skipped)", r["note"])
        self.assertIsNone(r["timeline"])
        self.assertIsNone(r["bending"])
        self.assertEqual(r["steps"], [])
        self.assertIn("No evaluation", r["eval"])
        html_, _ = self.app.status_html(RID)
        self.assertIn("AV-SEG-002", html_)

    def test_unknown_run(self):   # AV-APP-003
        html_, active = self.app.status_html("run_20991231_000000_ffff")
        self.assertIn("AV-APP-003", html_)
        self.assertFalse(active)

    def test_reload_picks_a_run_from_storage(self):   # FR-APP-08
        self.assertIsNone(self.app.initial_run(None))
        make_run(self.root, self.cfg, "run_20261026_010203_0001")
        make_run(self.root, self.cfg, RID)
        self.assertEqual(self.app.initial_run(None), RID)                       # newest
        self.assertEqual(self.app.initial_run("run_20261026_010203_0001"), "run_20261026_010203_0001")
        self.assertEqual(self.app.initial_run("../etc"), RID)

    def test_start(self):   # TC-APP-01, AV-APP-002
        self.assertEqual(self.app.start(None, "m")[0], None)
        self.assertIn("AV-APP-002", self.app.start(None, "m")[1])
        video = self.root / "data/public/videos/clip.mp4"
        video.write_bytes(b"x")
        os.environ["AV_TEST_ROOT"] = str(self.root)
        run_id, msg = self.app.start(str(video), "s")
        self.assertIsNotNone(run_id, msg)
        self.assertEqual(self.app.initial_run(None), run_id)
        t0 = time.monotonic()
        while self.app.status_html(run_id)[1] or self.app.store.status(run_id).status == "queued":
            self.assertLess(time.monotonic() - t0, 20)
            time.sleep(0.05)
        self.app.runs.reap()
        self.assertEqual(self.app.store.status(run_id).status, "done")
        outside = self.root / "elsewhere.mp4"
        outside.write_bytes(b"x")
        self.assertIn("AV-RUN-001", self.app.start(str(outside), "m")[1])

    def test_upload(self):
        f = self.root / "upload.mp4"
        f.write_bytes(b"x")
        dest, msg = self.app.upload(str(f), "public")
        self.assertEqual(Path(dest).parent, self.root / "data/public/input")
        self.assertIn(dest, [p for _, p in self.app.video_choices()])
        g = self.root / "bad.mov"
        g.write_bytes(b"x")
        self.assertEqual(self.app.upload(str(g), "public")[0], None)


if __name__ == "__main__":
    unittest.main()


class LiveTabTest(unittest.TestCase):
    def test_port_range(self):
        check_bind("127.0.0.1", False, 7861)
        check_bind("127.0.0.1", False, 7869)
        for port in (7859, 7870, 80):
            with self.assertRaises(AvError):
                check_bind("127.0.0.1", False, port)
        from assemblyvision.ui.app import main
        with self.assertRaises(SystemExit):
            main(["--port", "7870"])

    def test_overlay_is_sent_from_memory(self):
        import gradio as gr
        import numpy as np
        from assemblyvision.ui.live_tab import live_image
        before = set(Path(os.environ.get("GRADIO_TEMP_DIR", tempfile.gettempdir())).rglob("*")) \
            if os.environ.get("GRADIO_TEMP_DIR") else set()
        img = live_image(gr, interactive=False, type="numpy")
        out = img.postprocess(np.zeros((48, 64, 3), np.uint8))
        self.assertTrue(out.url.startswith("data:image/jpeg;base64,"))
        self.assertFalse(hasattr(out, "path") and getattr(out, "path", None))
        if os.environ.get("GRADIO_TEMP_DIR"):
            self.assertEqual(set(Path(os.environ["GRADIO_TEMP_DIR"]).rglob("*")), before)

    def test_panels_render(self):
        from assemblyvision.ui.live_tab import log_html, panel_html, station_choices, station_info, summary_html
        self.assertEqual([c[1] for c in station_choices()], ["workbench", "floor_pick", "side_work"])
        self.assertIn("FRONT", station_info("workbench"))
        self.assertIn("SIDE", station_info("floor_pick"))
        self.assertIn("Start live", panel_html(None))
        base = {"session_id": "live_20261026_010203_ab12", "status": "running", "elapsed_s": 12, "fps": 9.8,
                "delay_p50_ms": 60, "delay_p95_ms": 110, "frames_in": 120, "processed": 100, "dropped": 20,
                "paused": False, "recording": True, "notices": ["1 s: x"]}
        common = {"trunk_deg": 50, "band": 2, "band_s": [3.0, 2.0, 1.0, 0.0]}
        lift = {"n": 1, "kind": "stoop", "start_s": 1.0, "end_s": 3.0, "peak_trunk_deg": 70.0, "knee_deg": 172.0,
                "b2_s": 0.5, "b3_s": 1.0, "duration_s": 2.0}
        cmp_ = {k: {"n": 1, "peak_trunk_deg": a, "knee_deg": b, "b2_b3_s": c, "duration_s": 2.0}
                for k, a, b, c in (("stoop", 70.0, 172.0, 1.5), ("squat", 30.0, 88.0, 0.0))}
        stations = [
            {"kind": "workbench", "title": "A. Workbench", "step": "assemble", "step_s": 2.0, "cycles": 2,
             "last_cycle_s": 5.1, "mean_cycle_s": 5.0, "two_hand_simultaneous": 0.6, "two_hand_use_ratio": 0.8},
            {"kind": "lift", "title": "B. Floor pick", "lifts": [lift], "in_lift": False, "knee_deg": 170,
             "comparison": cmp_},
            {"kind": "posture", "title": "C. Side work", "posture": {"band": 1, "current_s": 4.0, "longest_s": 9.0,
             "total_s": 20.0, "stretches": 3}, "gaze_now": "side", "gaze": {"front": 0.2, "side": 0.7, "away": 0.1},
             "idle_time_s": 3.0, "idle_share": 0.1}]
        for st in stations:
            p = dict(base, station={**common, **st}, log=[lift] if st["kind"] == "lift" else [])
            h = panel_html(p)
            self.assertIn("9.8 fps", h)
            self.assertIn("REC", h)
            if st["kind"] == "lift":
                self.assertIn("STOOP", h)
                self.assertIn("SQUAT", h)
                self.assertIn("88°", h)
                self.assertIn("Peak trunk", log_html(p))
        s = {"session_id": base["session_id"], "duration_s": 60.0, "processed": 590, "fps_mean": 9.8,
             "delay_p50_ms": 60, "delay_p95_ms": 110, "recording": None, "recording_error": None,
             "stations": [{"number": 1, "title": "B. Floor pick", "rows": [("Stoop lifts", "1")]}]}
        self.assertIn("Station 1: B. Floor pick", summary_html(s))
