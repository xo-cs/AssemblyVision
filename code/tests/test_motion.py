"""MotionAnalyzer on synthetic subjects with known angles, paths and periods (CPU)."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from assemblyvision.analysis.motion import MotionAnalyzer, acf_period, band_of, cycle_baseline, trunk_angle
from assemblyvision.kts import write_kts
from assemblyvision.types import Step
from helpers import fake_root, make_ctx
from motion_helpers import FPS, cfg, default_wrists, pose_series, shot_for, t_axis

ALL = ["duration", "path", "back_bending", "repetition", "hand_travel", "two_hand", "idle"]


def analyse(series_list, steps, c=None, shot_types=None, scales=None):
    an = MotionAnalyzer(c or cfg())
    subs = []
    for i, s in enumerate(series_list):
        st = shot_types[i] if shot_types else "fixed"
        sc = scales[i] if scales else None
        subs.append(an.subject(s, shot_for(s, st, sc)))
    results, baseline = an.analyse("run", subs, steps, FPS)
    per_step = [{r.metric_name: r for r in recs} for _, recs in results]
    return per_step, an, baseline


def one_step(T):
    return [Step("Work", 0.0, T / FPS, "engine_a")]


class SharedTest(unittest.TestCase):
    def test_angle_and_bands(self):
        self.assertAlmostEqual(float(trunk_angle([0, 0], [0, 100])), 0.0)          # shoulder straight above the hip
        self.assertAlmostEqual(float(trunk_angle([100, 100], [0, 200])), 45.0)
        self.assertAlmostEqual(float(trunk_angle([-100, 200], [0, 200])), 90.0)    # sign of x does not matter
        self.assertEqual(band_of([0, 19.9, 20, 44.9, 45, 59.9, 60, 120], [20, 45, 60]).tolist(),
                         [0, 0, 1, 1, 2, 2, 3, 3])

    def test_acf_period_sine(self):
        t = t_axis(300)
        p, peak = acf_period(np.sin(2 * np.pi * t / 0.7), FPS, 0.4, 2.0)
        self.assertAlmostEqual(p, 0.7, delta=0.02)
        self.assertGreater(peak, 0.9)
        sig = np.sin(2 * np.pi * t / 1.0)
        sig[50:70] = np.nan                                     # missing frames are left out
        self.assertAlmostEqual(acf_period(sig, FPS, 0.4, 2.0)[0], 1.0, delta=0.02)

    def test_baseline(self):
        steps = [Step("A", 0, 6, "e"), Step("other", 6, 7, "e"), Step("A", 7, 15, "e"), Step("A", 15, 25, "e"),
                 Step("B", 25, 30, "e")]
        rows = {r["label"]: r for r in cycle_baseline(steps)}
        self.assertEqual(set(rows), {"A", "B"})
        self.assertEqual((rows["A"]["n"], rows["A"]["median_s"], rows["A"]["q1_s"], rows["A"]["q3_s"]),
                         (3, 8.0, 7.0, 9.0))
        self.assertEqual(rows["A"]["flags"], "")
        self.assertEqual((rows["B"]["n"], rows["B"]["flags"]), (1, "low_n"))


class DurationTest(unittest.TestCase):
    def test_duration_dev(self):   # TC-MOT-01
        s = pose_series(int(30 * FPS))
        steps = [Step("X", 0, 6, "e"), Step("other", 6, 7, "e"), Step("X", 7, 15, "e"), Step("X", 15, 25, "e"),
                 Step("other", 25, 30, "e")]
        per, _, _ = analyse([s], steps, cfg(enabled=["duration"]))
        last = per[2]
        self.assertEqual(last["duration_s"].value, 10.0)
        self.assertAlmostEqual(last["duration_dev"].value, 0.25)
        self.assertEqual((last["duration_s"].unit, last["duration_dev"].unit), ("s", "ratio"))
        self.assertEqual(len(per), 3)                             # other gets no metrics


class PathTest(unittest.TestCase):
    def test_walk_4m(self):   # TC-MOT-03
        T = int(10 * FPS)
        hip = np.stack([np.linspace(100, 500, T), np.full(T, 300.0)], 1)
        s = pose_series(T, hip=hip)
        per, an, _ = analyse([s], one_step(T), cfg(enabled=["path"]), scales=[0.01])
        r = per[0]["movement_path"]
        self.assertEqual(r.unit, "m")
        self.assertAlmostEqual(r.value, 4.0, delta=0.08)
        self.assertEqual(r.coverage, 1.0)
        self.assertEqual(an.paths["run-A-0001"]["t_s"].shape, (50,))     # 5 Hz over 10 s
        per, _, _ = analyse([s], one_step(T), cfg(enabled=["path"]))
        self.assertAlmostEqual(per[0]["movement_path"].value, 400.0, delta=8)
        self.assertEqual(per[0]["movement_path"].unit, "px")
        per, an, _ = analyse([s], one_step(T), cfg(enabled=["path"]), shot_types=["handheld"])
        r = per[0]["movement_path"]
        self.assertIsNone(r.value)
        self.assertEqual(r.flags, "handheld")
        self.assertEqual(an.notices, [])

    def test_fixed_and_handheld_in_one_step(self):
        T = int(5 * FPS)
        a = pose_series(T, shot_id="run-s001")
        b = pose_series(T, shot_id="run-s002", shot_start=T)
        per, _, _ = analyse([a, b], one_step(2 * T), cfg(enabled=["path", "hand_travel"]),
                            shot_types=["fixed", "handheld"])
        self.assertIn("partial_fixed", per[0]["movement_path"].flags)
        self.assertEqual(per[0]["movement_path"].coverage, 0.5)
        self.assertIn("body_relative", per[0]["hand_travel_l"].flags)
        self.assertIn("px_mixed_shots", per[0]["hand_travel_l"].flags)


class BendingTest(unittest.TestCase):
    def test_bands(self):   # TC-MOT-04
        T = int(6 * FPS)
        angle = np.where(t_axis(T) < 4.0, 30.0, 50.0)
        per, _, _ = analyse([pose_series(T, angle=angle)], one_step(T), cfg(enabled=["back_bending"]))
        r = per[0]
        self.assertAlmostEqual(r["bend_b1_s"].value, 4.0, delta=1 / FPS + 1e-9)
        self.assertAlmostEqual(r["bend_b2_s"].value, 2.0, delta=1 / FPS + 1e-9)
        self.assertEqual((r["bend_b0_s"].value, r["bend_b3_s"].value), (0.0, 0.0))
        self.assertAlmostEqual(r["bend_mean_deg"].value, (30 * 4 + 50 * 2) / 6, delta=0.5)
        self.assertEqual(r["bend_b1_s"].flags, "view_dependent")
        self.assertEqual(r["bend_b1_s"].unit, "s")

    def test_view_limited(self):
        T = int(10 * FPS)
        torso = np.where(t_axis(T) < 6.0, 100.0, 50.0)            # 40 percent of frames foreshortened
        per, _, _ = analyse([pose_series(T, torso=torso)], one_step(T), cfg(enabled=["back_bending"]))
        self.assertEqual(per[0]["bend_b0_s"].flags, "view_dependent,view_limited")

    def test_low_coverage(self):   # TC-MOT-11
        T = int(10 * FPS)
        s = pose_series(T)
        s.present[int(3 * FPS):] = False                           # 30 percent of the step
        s.body_xy[int(3 * FPS):] = np.nan
        per, an, _ = analyse([s], one_step(T), cfg(enabled=["back_bending"]))
        r = per[0]["bend_b0_s"]
        self.assertIsNone(r.value)
        self.assertIn("low_coverage", r.flags)
        self.assertAlmostEqual(r.coverage, 0.3)
        self.assertEqual([e.code for e in an.notices], ["AV-MOT-001"])
        self.assertIn("only 30 percent", an.notices[0].text)


def sine_wrists(T, period, amp=30.0, left=True, right=False, hip=(320.0, 300.0)):
    w = default_wrists(T, hip)
    x = amp * np.sin(2 * np.pi * t_axis(T) / period)
    if left:
        w[:, 0, 0] += x
    if right:
        w[:, 1, 0] += x
    return w


class RepetitionTest(unittest.TestCase):
    def test_period_1s(self):   # TC-MOT-06
        T = int(10 * FPS)
        per, _, _ = analyse([pose_series(T, wrists=sine_wrists(T, 1.0))], one_step(T), cfg(enabled=["repetition"]))
        r = per[0]
        self.assertAlmostEqual(r["repetition_period_s"].value, 1.0, delta=0.1)
        self.assertAlmostEqual(r["repetition_count"].value, 10, delta=1)
        self.assertEqual(r["repetition_share"].value, 1.0)

    def test_period_in_long_step(self):
        T = int(40 * FPS)
        per, _, _ = analyse([pose_series(T, wrists=sine_wrists(T, 0.6, right=True))], one_step(T),
                            cfg(enabled=["repetition"]))
        self.assertAlmostEqual(per[0]["repetition_period_s"].value, 0.6, delta=0.05)
        self.assertAlmostEqual(per[0]["repetition_count"].value, 40 / 0.6, delta=2)

    def test_noise_is_not_periodic(self):
        T = int(20 * FPS)
        w = default_wrists(T)
        w += np.random.default_rng(1).normal(0, 4.0, w.shape)
        per, _, _ = analyse([pose_series(T, wrists=w)], one_step(T), cfg(enabled=["repetition"]))
        r = per[0]
        self.assertEqual(r["repetition_share"].value, 0.0)
        self.assertIsNone(r["repetition_period_s"].value)
        self.assertIn("not_periodic", r["repetition_period_s"].flags)

    def test_too_short(self):
        T = int(3 * FPS)
        per, an, _ = analyse([pose_series(T)], one_step(T), cfg(enabled=["repetition"]))
        self.assertIn("too_short", per[0]["repetition_share"].flags)
        self.assertEqual([e.code for e in an.notices], ["AV-MOT-002"])


class HandTravelTest(unittest.TestCase):
    def test_circle_3m(self):   # TC-MOT-07
        T = int(10 * FPS)
        r_px = 300 / (2 * np.pi)                                   # one turn of 300 px in 10 s
        w = default_wrists(T)
        ang = 2 * np.pi * t_axis(T) / 10.0
        w[:, 0] += np.stack([r_px * np.cos(ang), r_px * np.sin(ang)], 1)
        per, _, _ = analyse([pose_series(T, wrists=w)], one_step(T), cfg(enabled=["hand_travel"]), scales=[0.01])
        r = per[0]
        self.assertEqual(r["hand_travel_l"].unit, "m")
        self.assertAlmostEqual(r["hand_travel_l"].value, 3.0, delta=0.06)
        self.assertAlmostEqual(r["hand_travel_l_torso"].value, 3.0, delta=0.06)   # 300 px / 100 px torso
        self.assertEqual(r["hand_travel_r"].value, 0.0)

    def test_still_hand_with_noise(self):
        T = int(10 * FPS)
        w = default_wrists(T) + np.random.default_rng(2).normal(0, 1.0, (T, 2, 2))
        per, _, _ = analyse([pose_series(T, wrists=w)], one_step(T), cfg(enabled=["hand_travel"]))
        # after smoothing 15 percent of the noise steps still pass the 0.5 px jitter limit (LLD rule), about
        # 3 px/s; this stays under 10 percent of the 300 px path of test_circle_3m in the same time
        self.assertLess(per[0]["hand_travel_l"].value, 30.0)

    def test_handheld_is_body_relative(self):
        T = int(10 * FPS)
        pan = np.linspace(0, 200, T)                               # camera pans, the hands stay at the body
        per, _, _ = analyse([pose_series(T, sway=pan)], one_step(T), cfg(enabled=["hand_travel"]),
                            shot_types=["handheld"])
        self.assertLess(per[0]["hand_travel_l"].value, 1.0)
        self.assertEqual(per[0]["hand_travel_l"].flags, "body_relative")


class TwoHandTest(unittest.TestCase):
    def run_two(self, w):
        T = len(w)
        per, _, _ = analyse([pose_series(T, wrists=w)], one_step(T), cfg(enabled=["two_hand"]))
        return per[0]

    def test_same_motion(self):
        T = int(10 * FPS)
        r = self.run_two(sine_wrists(T, 1.0, left=True, right=True))
        self.assertEqual(r["two_hand_simultaneous"].value, 1.0)
        self.assertEqual(r["two_hand_use_ratio"].value, 1.0)
        self.assertAlmostEqual(r["two_hand_speed_corr"].value, 1.0, places=6)

    def test_one_hand_still(self):
        T = int(10 * FPS)
        r = self.run_two(sine_wrists(T, 1.0))
        self.assertEqual(r["two_hand_simultaneous"].value, 0.0)
        self.assertEqual(r["two_hand_use_ratio"].value, 0.0)

    def test_alternating_hands(self):
        T = int(10 * FPS)
        t = t_axis(T)
        w = default_wrists(T)
        burst = 30 * np.sin(2 * np.pi * t / 1.0)
        on_left = (t.astype(int) % 2) == 0                         # left in even seconds, right in odd
        w[:, 0, 0] += np.where(on_left, burst, 0)
        w[:, 1, 0] += np.where(~on_left, burst, 0)
        r = self.run_two(w)
        self.assertLess(r["two_hand_speed_corr"].value, -0.3)
        self.assertLess(r["two_hand_simultaneous"].value, 0.2)
        self.assertAlmostEqual(r["two_hand_use_ratio"].value, 1.0, delta=0.1)


class IdleTest(unittest.TestCase):
    def test_three_seconds(self):   # TC-MOT-10
        T = int(10 * FPS)
        t = t_axis(T)
        clock = np.where(t < 4, t, np.where(t < 7, 4.0, t - 3))   # frozen from 4 to 7 s
        sway = 20 * np.sin(2 * np.pi * clock)
        per, _, _ = analyse([pose_series(T, sway=sway)], one_step(T), cfg(enabled=["idle"]))
        # the Savitzky Golay window (11 frames) blurs each edge of the still stretch by up to 3 frames
        self.assertAlmostEqual(per[0]["idle_time_s"].value, 3.0, delta=0.2)
        self.assertAlmostEqual(per[0]["idle_share"].value, 0.3, delta=0.02)


class GazeTest(unittest.TestCase):
    def test_off_by_default(self):
        T = int(5 * FPS)
        per, _, _ = analyse([pose_series(T)], one_step(T))
        self.assertFalse([k for k in per[0] if k.startswith("gaze")])
        self.assertTrue({"duration_s", "bend_b0_s", "repetition_share", "hand_travel_l", "two_hand_use_ratio",
                         "idle_time_s", "movement_path"} <= set(per[0]))

    def test_classes(self):
        T = int(8 * FPS)
        s = pose_series(T)
        q = T // 4
        s.body_xy[q:2 * q, 0, 0] -= 6                             # nose left of the eye midpoint (eyes 16 px apart)
        s.body_xy[2 * q:3 * q, 0, 0] += 6
        s.body_conf[3 * q:, 0:3] = 0.1                             # no nose, no eyes, ears seen: away
        s.body_xy[:q, 0, 1] = s.body_xy[:q, 3, 1] + 20             # nose 20 px below the ears: down
        per, _, _ = analyse([s], one_step(T), cfg(enabled=ALL + ["gaze"]))
        r = per[0]
        for g in ("front", "left", "right", "away"):
            self.assertAlmostEqual(r[f"gaze_{g}_share"].value, 0.25, delta=0.01, msg=g)
            self.assertIn("rough", r[f"gaze_{g}_share"].flags)
        self.assertAlmostEqual(r["gaze_down_share"].value, 1 / 3, delta=0.01)   # of the frames with nose and ears


class RobustnessTest(unittest.TestCase):
    def test_one_metric_fails(self):   # TC-MOT-15
        T = int(10 * FPS)
        an = MotionAnalyzer(cfg())
        sub = an.subject(pose_series(T), shot_for(pose_series(T)))

        def boom(sv):
            raise RuntimeError("forced")
        an._metrics["back_bending"] = boom
        results, _ = an.analyse("run", [sub], one_step(T), FPS)
        names = {r.metric_name for r in results[0][1]}
        self.assertFalse([n for n in names if n.startswith("bend")])
        self.assertTrue({"duration_s", "idle_time_s", "hand_travel_l", "two_hand_use_ratio"} <= names)
        self.assertEqual([e.code for e in an.notices], ["AV-MOT-004"])
        self.assertIn("forced", an.notices[0].detail)


class StageTest(unittest.TestCase):
    """The metrics stage and a CPU rerun from stored files."""

    def build_run(self, root):
        _, ctx = make_ctx(root)
        rid = ctx.run_id
        T = int(20 * FPS)
        items, tracks = [], []
        for k, typ in ((1, "fixed"), (2, "handheld")):
            sid = f"{rid}-s{k:03d}"
            a = (k - 1) * T
            items.append({"shot_id": sid, "start_frame": a, "end_frame": a + T, "start_s": a / FPS,
                          "end_s": (a + T) / FPS, "shot_type": typ, "scale_m_per_px": None})
            s = pose_series(T, wrists=sine_wrists(T, 1.0), shot_start=a, run_id=rid, shot_id=sid)
            path = write_kts(s, ctx.paths, {"is_subject": True, "subject_source": "auto"})
            tracks.append({"track_id": s.track_id, "shot_id": sid, "kts_path": ctx.rel(path), "is_subject": True})
        ctx.manifest.update(run_id=rid, source="public",
                            ingest={"cfr": {"fps": FPS, "frame_count": 2 * T, "duration_s": 40.0}},
                            shots={"items": items}, tracking={"tracks": tracks, "per_shot": []}, notices=[],
                            stages={}, config={})
        ctx.paths.steps_csv.write_text("step,label,start_s,end_s,source,score\n"
                                       "1,Removal,0.000,15.000,engine_a,1.0\n0,other,15.000,16.000,engine_a,\n"
                                       "2,Disassembly,16.000,40.000,engine_a,1.0\n")
        ctx.write_manifest()
        return ctx

    def test_rerun_metrics_cpu(self):
        from assemblyvision.tools import run_pose
        tmp = fake_root()
        torch_before = "torch" in sys.modules
        try:
            ctx = self.build_run(Path(tmp.name))
            with mock.patch.object(run_pose, "run_source", return_value=("public", ctx.paths.root)), \
                    mock.patch.dict(os.environ), mock.patch("sys.stdout"):
                code = run_pose.main(["--run", str(ctx.paths.root), "--stages", "metrics"])
            self.assertEqual(code, 0)
            m = json.loads(ctx.paths.manifest.read_text())
            self.assertEqual(m["stages"]["metrics"]["status"], "done")
            sec = m["metrics"]
            self.assertEqual([s["segment_id"] for s in sec["steps"]], [f"{ctx.run_id}-A-0001", f"{ctx.run_id}-A-0003"])
            self.assertEqual([p["frames"] for p in sec["steps"][1]["pieces"]], [120, 600])
            lines = ctx.paths.metrics_csv.read_text().splitlines()
            self.assertEqual(lines[0], "segment_id,step,label,start_s,end_s,metric,value,unit,coverage,flags")
            self.assertEqual(len(lines) - 1, sec["counts"]["records"])
            self.assertTrue(ctx.paths.path_npz(f"{ctx.run_id}-A-0001").is_file())
            heat = {h["shot_id"]: h for h in sec["heatmaps"]}
            self.assertEqual(heat[f"{ctx.run_id}-s001"]["status"], "done")
            self.assertTrue((ctx.paths.root / heat[f"{ctx.run_id}-s001"]["png"]).is_file())
            self.assertEqual(heat[f"{ctx.run_id}-s002"]["status"], "skipped")
            self.assertFalse(ctx.paths.heatmap(f"{ctx.run_id}-s002").exists())
            rep = {r["metric"]: r for r in sec["steps"][0]["records"]}
            self.assertAlmostEqual(rep["repetition_period_s"]["value"], 1.0, delta=0.1)
            if not torch_before:
                self.assertNotIn("torch", sys.modules)
        finally:
            tmp.cleanup()

    def test_heatmap_min_samples(self):
        an = MotionAnalyzer(cfg())
        s = pose_series(int(3 * FPS))                              # 15 samples at 5 Hz
        hm = an.heatmap(an.subject(s, shot_for(s)))
        self.assertEqual(hm["status"], "skipped")
        s = pose_series(int(10 * FPS))
        hm = an.heatmap(an.subject(s, shot_for(s)))
        self.assertEqual((hm["status"], int(hm["grid"].sum()), int((hm["grid"] > 0).sum())), ("done", 50, 1))


if __name__ == "__main__":
    unittest.main()
