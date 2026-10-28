"""Pure helpers of the redesigned page: notice grouping, readable metric names,
the neutral QiSens video name, round time ticks, the GPU state from lock files
and the live elapsed time (CPU, no browser, temporary data root only)."""
import os
import tempfile
import unittest
from pathlib import Path

from assemblyvision.live.manager import elapsed_s
from assemblyvision.ui import views as v
from assemblyvision.ui.charts import time_ticks

RUN = "run_20261026_123831_f667"


def _notice(code, text, stage="reference"):
    return {"stage": stage, "code": code, "effect": "Notice", "text": text}


class NoticeGroupTest(unittest.TestCase):
    def setUp(self):
        self.notes = [_notice("AV-POS-004", f"No worker was found in the first 5 s of shot {RUN}-s{i:03d}. "
                                            "Skip this shot or choose another.") for i in (1, 5, 7, 8)]
        self.notes += [_notice("AV-TRK-002", f"No worker could be tracked for at least 1 s in shot {RUN}-s001.",
                               "tracking"),
                       _notice("AV-MOT-001", "path of step Cleaning is not reported: only 25 percent of frames "
                                             "had the keypoints it needs.", "metrics"),
                       _notice("AV-MOT-001", "two_hand of step Cleaning is not reported: only 37 percent of "
                                             "frames had the keypoints it needs.", "metrics")]

    def test_groups_by_code_in_order(self):
        g = v.group_notices(self.notes)
        self.assertEqual([x["code"] for x in g], ["AV-POS-004", "AV-TRK-002", "AV-MOT-001"])
        self.assertEqual([x["count"] for x in g], [4, 1, 2])
        self.assertEqual(g[0]["shots"], ["s001", "s005", "s007", "s008"])
        self.assertEqual(g[0]["title"], "no worker was found in the first 5 s")
        self.assertEqual(g[1]["title"], "no worker could be tracked for at least 1 s")
        self.assertEqual(g[2]["title"], v.NOTICE_TITLES["AV-MOT-001"])      # texts differ: the short title
        self.assertEqual(g[2]["shots"], [])

    def test_summary_lines(self):
        g = v.group_notices(self.notes)
        self.assertEqual(v.notice_line(g[0]),
                         "AV-POS-004, no worker was found in the first 5 s, 4 shots: s001, s005, s007...")
        self.assertEqual(v.notice_line(g[1]), "AV-TRK-002, no worker could be tracked for at least 1 s, 1 shot: s001")
        self.assertTrue(v.notice_line(g[2]).endswith(", 2 notices"))

    def test_html_is_collapsed_and_complete(self):
        h = v.notices_html(self.notes)
        self.assertEqual(h.count("<details"), 3)
        self.assertNotIn("<details open", h)
        self.assertEqual(h.count("<li>"), len(self.notes))        # the full list on expand
        self.assertNotIn(RUN, h)                                   # shot IDs shortened to s001
        self.assertIn("No notices", v.notices_html([]))

    def test_unknown_code_keeps_its_first_sentence(self):
        g = v.group_notices([{"code": "AV-XYZ-001", "text": "Something odd. More detail here."}])
        self.assertEqual(g[0]["title"], "something odd")


class MetricNameTest(unittest.TestCase):
    def test_band_names_follow_the_bands(self):
        self.assertEqual(v.metric_label("bend_b1_s", [20, 45, 60]), "Time in B1 (20 to 45°)")
        self.assertEqual(v.metric_label("bend_b0_s", [20, 45, 60]), "Time in B0 (under 20°)")
        self.assertEqual(v.metric_label("bend_b3_s", [15, 40, 70]), "Time in B3 (70° or more)")

    def test_known_gaze_and_fallback(self):
        self.assertEqual(v.metric_label("two_hand_simultaneous"), "Both hands active together")
        self.assertEqual(v.metric_label("gaze_front_share"), "Gaze front share")
        self.assertEqual(v.metric_label("some_new_metric"), "Some new metric")

    def test_values(self):
        self.assertEqual(v.metric_value(0.41, "ratio"), "41 %")
        self.assertEqual(v.metric_value(52.4, "deg"), "52°")
        self.assertEqual(v.metric_value(8.0, "s"), "8 s")
        self.assertEqual(v.metric_value(3.2, "torso"), "3.2 torso lengths")
        self.assertEqual(v.metric_value(None, "s"), "n/a")

    def test_flags_become_pills(self):
        h = v.flag_pills("view_dependent,low_coverage")
        self.assertEqual(h.count("<span"), 2)
        self.assertIn("view dependent", h)
        self.assertEqual(v.flag_pills(""), "")


class QisensAliasTest(unittest.TestCase):
    def test_alias(self):
        self.assertEqual(v.qisens_alias(594.0), "QiSens video (9.9 min)")
        self.assertEqual(v.qisens_alias(42.0), "QiSens video (42 s)")
        self.assertEqual(v.qisens_alias(None), "QiSens video")
        self.assertEqual(v.video_label("qisens", "secret_name.mp4", 594.0), "QiSens video (9.9 min)")
        self.assertEqual(v.video_label("public", "process_480p30.mp4", 326.0), "process_480p30.mp4")

    def test_repeated_names_are_numbered(self):
        self.assertEqual(v.unique_labels(["QiSens video (9.9 min)", "QiSens video (9.9 min)", "a.mp4"]),
                         ["QiSens video (9.9 min)", "QiSens video 2 (9.9 min)", "a.mp4"])

    def test_app_never_shows_the_file_name(self):
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from ui_helpers import make_root, make_run
        from assemblyvision.ui.app import GradioApp
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = make_root(root)
            secret = "社名_組立_secret.mp4"
            vid = root / "data/qisens/videos" / secret
            vid.write_bytes(b"x")
            p = make_run(root, cfg, "run_20261026_010203_ab12", source="qisens")
            import json
            m = json.loads(p.manifest.read_text())
            m["video"] = str(vid)
            p.manifest.write_text(json.dumps(m))
            app = GradioApp(cfg, ["qisens", "public"], root, worker_cmd=["true"])
            shown = " ".join(lab for lab, _ in app.video_choices() + app.run_choices())
            shown += app.runs_html() + app.results_head("run_20261026_010203_ab12")
            shown += app.status_html("run_20261026_010203_ab12")[0]
            self.assertNotIn("secret", shown)
            self.assertNotIn("社名", shown)
            self.assertIn("QiSens video (20 s)", shown)      # make_run stores 20 s


class TicksTest(unittest.TestCase):
    def test_round_ticks(self):
        self.assertEqual(time_ticks(326)[:3], [0.0, 30.0, 60.0])
        self.assertEqual(time_ticks(594)[:3], [0.0, 60.0, 120.0])
        self.assertTrue(all(t % 60 == 0 for t in time_ticks(1800)))
        self.assertLessEqual(len(time_ticks(7200)), 12)
        self.assertEqual(time_ticks(20)[:3], [0.0, 5.0, 10.0])


class GpuStateTest(unittest.TestCase):
    def test_reads_proc_locks(self):
        with tempfile.TemporaryDirectory() as d:
            f0, f1 = Path(d) / "gpu0.lock", Path(d) / "gpu1.lock"
            f0.touch()
            f1.touch()
            st = f1.stat()
            line = f"1: FLOCK  ADVISORY  WRITE 1234 {os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino} 0 EOF"
            self.assertEqual(v.gpu_busy(Path(d), proc_locks=line), {0: False, 1: True})
            self.assertEqual(v.gpu_busy(Path(d), proc_locks=""), {0: False, 1: False})
            self.assertIn("GPU0 free, GPU1 busy", v.server_pill({0: False, 1: True}))

    def test_real_lock_is_seen(self):
        import fcntl
        with tempfile.TemporaryDirectory() as d:
            with open(Path(d) / "gpu0.lock", "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertTrue(v.gpu_busy(Path(d))[0])
            self.assertFalse(v.gpu_busy(Path(d))[0])


class LiveElapsedTest(unittest.TestCase):
    def test_counts_from_start_and_stops_at_stop(self):
        self.assertEqual(elapsed_s(100.0, 112.5), 12.5)
        self.assertEqual(elapsed_s(100.0, 66000.0, ended=165.0), 65.0)     # hours after Stop: still 65 s
        self.assertEqual(elapsed_s(100.0, 90.0), 0.0)

    def test_session_elapsed_is_frozen_after_stop(self):
        import time
        from live_helpers import cfg
        from test_live_manager import FakePose, frame
        from assemblyvision.live.manager import LiveManager
        from assemblyvision.ui.live_tab import panel_html, summary_html
        with tempfile.TemporaryDirectory() as d:
            lm = LiveManager(cfg(model_keep_s=0.0), Path(d) / "live", backend_factory=FakePose,
                             gpu_free=lambda i: 8000, lock_dir=Path(d) / "locks", port=7861)
            lm.start("workbench")
            for i in range(10):
                lm.push(frame(i))
                time.sleep(0.05)
            running = panel_html(lm.panel())
            self.assertIn("av-pill-live", running)
            self.assertNotIn("STOPPED", running)
            s = lm.stop()
            t_stop = lm.panel()["elapsed_s"]
            time.sleep(1.2)
            p = lm.panel()
            self.assertEqual(p["elapsed_s"], t_stop)
            self.assertLessEqual(abs(p["elapsed_s"] - s["duration_s"]), 1)
            self.assertIn("STOPPED", panel_html(p))
            self.assertIn("Session summary", summary_html(s))

    def test_stopped_panel_says_stopped(self):
        from assemblyvision.ui.live_tab import panel_html
        p = {"session_id": "live_20261026_010203_ab12", "status": "done", "elapsed_s": 65, "fps": None,
             "delay_p50_ms": 60, "delay_p95_ms": 110, "frames_in": 600, "processed": 590, "dropped": 10,
             "paused": False, "recording": False, "notices": [], "log": [],
             "station": {"kind": "workbench", "title": "A. Workbench", "step": None, "step_s": None, "cycles": 0,
                         "trunk_deg": None, "band": 0, "band_s": [0.0] * 4}}
        h = panel_html(p)
        self.assertIn("STOPPED", h)
        self.assertNotIn("WAITING", h)
        self.assertIn("elapsed 1:05", h)
        self.assertNotIn("live", h.split("elapsed")[1].split("delay")[0].replace("live_", ""))


if __name__ == "__main__":
    unittest.main()
