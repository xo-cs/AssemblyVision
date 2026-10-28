"""Combined live view (the one Live page mode): every measure on every frame,
lifts only with the knees in view, cycles only when switched on, and the
summary after Stop (CPU, synthetic keypoints and a fake pose model)."""
import tempfile
import time
import unittest
from pathlib import Path

from assemblyvision.live.manager import LiveManager, overlay_texts
from assemblyvision.live.stations import COMBINED, CombinedStation
from assemblyvision.ui.live_tab import log_html, panel_html, summary_html
from live_helpers import H, W, cfg, times, workbench_script
from test_live_lifts import script

LIFTS = [(2, 70, 175), (8, 30, 90)]                 # one stoop, one squat


def feed(st, at, ts, hide_knees=False):
    for t in ts:
        kp, conf = at(float(t))[:2]
        if hide_knees:
            conf = conf.copy()
            conf[13:17] = 0.1                         # front view at a bench: legs hidden
        st.update(float(t), kp, conf, (W, H))
    return st


class CombinedStationTest(unittest.TestCase):
    def test_side_view_gives_every_measure_and_lifts(self):
        st = feed(CombinedStation(cfg(), 1, 0.0), script(LIFTS), times(14, seed=3))
        p = st.panel(14.0)
        self.assertEqual(p["kind"], COMBINED)
        self.assertEqual([x["kind"] for x in p["lifts"]], ["stoop", "squat"])
        self.assertTrue(p["lifts_ready"])
        self.assertIsNotNone(p["comparison"])
        self.assertGreater(p["posture"]["total_s"], 1.0)              # the stoop is bent time
        self.assertGreater(p["posture"]["longest_s"], 1.0)
        self.assertGreater(sum(p["band_s"]), 12.0)
        self.assertIn("idle_time_s", p)
        self.assertIn("two_hand_simultaneous", p)
        self.assertFalse(p["cycles_on"])
        self.assertNotIn("cycles", p)
        rows = dict(st.summary(14.0)["rows"])
        for key in ("Time analysed", "Trunk bands B0 / B1 / B2 / B3", "Time bent (B1 or worse)",
                    "Longest bent stretch", "Idle time", "Both hands active together", "Hand use ratio",
                    "Stoop lifts", "Squat lifts", "Stoop: mean peak trunk / knee / B2+B3"):
            self.assertIn(key, rows)
        self.assertEqual((rows["Stoop lifts"], rows["Squat lifts"]), ("1", "1"))
        self.assertNotIn("Cycles", rows)

    def test_lifts_need_the_knees(self):
        st = feed(CombinedStation(cfg(), 1, 0.0), script(LIFTS), times(14, seed=3), hide_knees=True)
        p = st.panel(14.0)
        self.assertEqual(p["lifts"], [])
        self.assertGreaterEqual(st.lifts.rejected, 1)                   # the stoop bend was not a lift
        self.assertFalse(p["lifts_ready"])
        self.assertGreater(p["posture"]["total_s"], 1.0)                # but it is still bent time
        rows = dict(st.summary(14.0)["rows"])
        self.assertIn("side view", rows["Lifts"])
        self.assertNotIn("Stoop lifts", rows)
        p_full = {"session_id": "live_20261026_010203_ab12", "status": "running", "elapsed_s": 14, "fps": 9.9,
                  "delay_p50_ms": 50, "delay_p95_ms": 90, "frames_in": 140, "processed": 140, "dropped": 0,
                  "paused": False, "recording": False, "notices": [], "log": [], "steps": [], "station": p}
        h = panel_html(p_full)
        self.assertIn("Lifts need a side view with the whole body.", h)
        self.assertNotIn("Cycles", h)

    def test_cycles_only_when_switched_on(self):
        at, _, end = workbench_script(2)
        st = CombinedStation(cfg(), 1, 0.0)
        feed(st, lambda t: at(t), times(end + 1, seed=1))
        self.assertIsNone(st.engine)
        st = CombinedStation(cfg(), 1, 0.0, cycles=True, borders=(1 / 3, 2 / 3))
        feed(st, lambda t: at(t), times(end + 1, seed=1))
        p = st.panel(end + 1)
        self.assertTrue(p["cycles_on"])
        self.assertEqual(p["cycles"], 2)
        self.assertIsNotNone(st.draw_zones())
        st.set_cycles(False)
        self.assertIsNone(st.draw_zones())
        p = st.panel(end + 1)
        self.assertEqual(p["cycles"], 2)                                # counts kept, tiles hidden
        self.assertNotIn("Cycles", panel_html({"session_id": "live_20261026_010203_ab12", "status": "running",
                                               "elapsed_s": 1, "fps": None, "delay_p50_ms": None,
                                               "delay_p95_ms": None, "frames_in": 0, "processed": 0,
                                               "dropped": 0, "paused": False, "recording": False, "notices": [],
                                               "log": [], "steps": [], "station": p}))
        self.assertEqual(dict(st.summary(end + 1)["rows"])["Cycles"], "2")
        self.assertIn("cycles", overlay_texts({**p, "cycles_on": True})[1])


class CombinedSessionTest(unittest.TestCase):
    def test_start_defaults_to_combined_and_summary_lists_all_measures(self):
        from test_live_manager import FakePose, frame
        with tempfile.TemporaryDirectory() as d:
            lm = LiveManager(cfg(model_keep_s=0.0), Path(d) / "live", backend_factory=FakePose,
                             gpu_free=lambda i: 8000, lock_dir=Path(d) / "locks", port=7861)
            lm.start()
            for i in range(15):
                lm.push(frame(i))
                time.sleep(0.03)
            p = lm.panel()
            self.assertEqual(p["station"]["kind"], COMBINED)
            lm.session.set_cycles(True, (0.3, 0.7))
            self.assertTrue(lm.panel()["station"]["cycles_on"])
            s = lm.stop()
            self.assertEqual(len(s["stations"]), 1)
            names = [k for k, _ in s["stations"][0]["rows"]]
            for key in ("Time bent (B1 or worse)", "Longest bent stretch", "Idle time", "Hand use ratio",
                        "Cycles", "Mean cycle"):
                self.assertIn(key, names)
            h = summary_html(s)
            self.assertIn("Measures", h)
            self.assertNotIn("Station 1", h)
            self.assertEqual(log_html(lm.panel()), "")                  # no lift, no cycle step yet


if __name__ == "__main__":
    unittest.main()
