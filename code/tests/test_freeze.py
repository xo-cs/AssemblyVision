"""TC-FRZ: the validation freeze, the phase file and the decoders without order (Week 12). CPU."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.analysis import clip_classes as C
from assemblyvision.analysis import hand_events as HE
from assemblyvision.analysis import zero_shot as Z
from assemblyvision.storage.files import REPO_ROOT
from assemblyvision.tools import check_outputs as CO
from assemblyvision.tools import freeze as F
from assemblyvision import validation_labels as VL
from assemblyvision.evaluation import metrics as M
from assemblyvision.output import extras as X
from assemblyvision.tools import robustness as R
from assemblyvision.tools import score_measurements as SM
from assemblyvision.tools import train_domain as T
from assemblyvision.tools import validate_seg as V
from assemblyvision.types import Step


def cumulative(labels, K, good=0.0, bad=-3.0):
    x = np.full((len(labels), K), bad)
    x[np.arange(len(labels)), labels] = good
    return np.vstack([np.zeros((1, K)), np.cumsum(x, 0)]).T


class FreezeTest(unittest.TestCase):
    def test_a_phase_file_gives_its_phases(self):
        d = REPO_ROOT / "data/public/tmp"
        d.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".md", dir=d) as f:
            f.write("## Lift\nDistinguishes: a\n> s1\n- d1\n- d2\n## Fit\n> s2\n- d3\n")
            f.flush()
            ph, proc, _ = Z.phase_set("file:" + Path(f.name).relative_to(REPO_ROOT).as_posix())
        self.assertEqual([p["name"] for p in ph], ["Lift", "Fit"])
        self.assertEqual(ph[0]["descriptions"], ["d1", "d2"])
        self.assertIn("railway", proc)
        self.assertIsNone(Z.other_class("file:x.md"))

    def test_a_phase_file_must_be_in_the_allowed_folders(self):
        for name in ("file:code/x.md", "file:data/public/../other/x.md"):
            with self.assertRaises(ValueError):
                Z.phase_set(name)

    def test_the_existing_sets_are_unchanged(self):
        self.assertEqual(len(Z.phase_set("qisens")[0]), 5)
        self.assertIsNotNone(Z.other_class("qisens"))

    def test_ui_is_not_part_of_the_method(self):
        self.assertTrue(all(not m.startswith(F.NOT_METHOD) for m in F.METHOD))


class DecoderTest(unittest.TestCase):
    def test_best_class(self):
        k, lead = C.best_class(cumulative([2] * 30, 4))
        self.assertEqual(k, 2)
        self.assertGreater(lead, 0)

    def test_two_segments_in_either_order(self):
        self.assertEqual(C.two_segments(cumulative([3] * 40 + [1] * 60, 5), 10.0), (3, 1, 40))
        self.assertEqual(C.two_segments(cumulative([1] * 60 + [3] * 40, 5), 10.0), (1, 3, 60))

    def test_unordered_finds_repeats_without_an_order(self):
        labels = [2] * 80 + [0] * 70 + [2] * 90 + [1] * 60
        self.assertEqual(C.unordered(cumulative(labels, 3), 10.0, 5.0), [(2, 0, 80), (0, 80, 150), (2, 150, 240), (1, 240, 300)])

    def test_unordered_keeps_the_minimum_length(self):
        labels = [0] * 100 + [1] * 10 + [0] * 100              # 1 s of noise, 5 s minimum
        self.assertEqual(C.unordered(cumulative(labels, 2), 10.0, 5.0), [(0, 0, 210)])

    def test_unordered_on_a_short_video(self):
        self.assertEqual(C.unordered(cumulative([1] * 20, 3), 10.0, 5.0), [(1, 0, 20)])


class TrainDomainTest(unittest.TestCase):
    def folder(self, d):
        f = Path(d)
        for n in ("a.mp4", "b.mp4"):
            (f / n).write_bytes(b"x")
        (f / "classes.csv").write_text("class,definition\nLift,raise it\nFit,put it on\n")
        (f / "labels.csv").write_text("file,class,second_class,change_s\na.mp4,Lift,,\nb.mp4,Fit,Lift,3.5\n")
        (f / "descriptions.md").write_text("## Lift\n> s\n- d\n## Fit\n> s\n- d\n")
        return f

    def test_files_are_checked(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.folder(d)
            classes, labels = T.read_files(f)
            self.assertEqual(classes, ["Lift", "Fit"])
            self.assertEqual(labels["b.mp4"], {"first": "Fit", "second": "Lift", "change_s": 3.5})
            (f / "labels.csv").write_text("file,class,second_class,change_s\na.mp4,Lift,,\n")
            with self.assertRaisesRegex(ValueError, "one row per clip"):
                T.read_files(f)
            (f / "labels.csv").write_text("file,class,second_class,change_s\na.mp4,Lift,,\nb.mp4,Nope,,\n")
            with self.assertRaisesRegex(ValueError, "not in classes"):
                T.read_files(f)
            (f / "labels.csv").write_text("file,class,second_class,change_s\na.mp4,Lift,,\nb.mp4,Fit,Lift,\n")
            with self.assertRaisesRegex(ValueError, "change time"):
                T.read_files(f)

    def test_descriptions_must_match_the_class_table(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.folder(d)
            (f / "descriptions.md").write_text("## Fit\n> s\n- d\n## Lift\n> s\n- d\n")
            with self.assertRaisesRegex(ValueError, "same order"):
                T.read_files(f)

    def test_init_never_overwrites_a_real_file(self):
        with tempfile.TemporaryDirectory() as d:
            f = self.folder(d)
            before = (f / "labels.csv").read_text()
            T.init(f)
            self.assertEqual((f / "labels.csv").read_text(), before)
            self.assertIn("a.mp4,,,", (f / "labels.csv.template").read_text())

    def test_order_is_seeded(self):
        files = [f"{i}.mp4" for i in range(14)]
        self.assertEqual(T.join_order(files), T.join_order(list(reversed(files))))
        self.assertNotEqual(T.join_order(files), sorted(files))

    def test_joined_truth_merges_neighbours_of_one_class(self):
        labels = {"a": {"first": "X", "second": None, "change_s": None}, "b": {"first": "X", "second": "Y", "change_s": 2.0},
                  "c": {"first": "Z", "second": None, "change_s": None}}
        self.assertEqual(T.joined_truth(["a", "b", "c"], [10.0, 8.0, 5.0], labels),
                         [("X", 0.0, 12.0), ("Y", 12.0, 18.0), ("Z", 18.0, 23.0)])

    def test_scores(self):
        r = T.score_items([("A", "A"), ("A", "B"), ("B", "B"), ("B", "B")], ["A", "B"])
        self.assertEqual((r["accuracy"], r["confusion"]), (75.0, [[1, 1], [0, 2]]))
        self.assertAlmostEqual(r["per_class"]["B"]["precision"], 2 / 3)
        self.assertEqual(r["per_class"]["A"]["recall"], 0.5)

    def test_a_clip_with_two_labels(self):
        ll = cumulative([1] * 40 + [0] * 60, 2)
        lab = {"first": "B", "second": "A", "change_s": 4.0}
        items, info = T.clip_items(ll, 10.0, lab, ["A", "B"])
        self.assertEqual(items, [("B", "B"), ("A", "A")])
        self.assertEqual(info["change_s"], 4.0)


class ValidateSegTest(unittest.TestCase):
    def test_tables(self):
        gt = [Step("A", 2.0, 6.0, "gt"), Step("B", 6.0, 10.0, "gt")]
        pred = [Step("A", 2.0, 7.0, "x"), Step("B", 7.0, 10.0, "x")]
        p, g = M.frames_from_steps(pred, 10.0, 10.0), M.frames_from_steps(gt, 10.0, 10.0)
        cm = V.confusion(p, g, ["other", "A", "B"])
        self.assertEqual(cm.tolist(), [[20, 0, 0], [0, 40, 0], [0, 10, 30]])
        per = V.per_class(cm, ["other", "A", "B"])
        self.assertEqual((per["A"]["precision"], per["B"]["recall"]), (80.0, 75.0))
        rows = V.boundary_errors(pred, gt, 10.0)
        self.assertEqual([(r["between"], r["error_s"]) for r in rows],
                         [("start of A", 0.0), ("end of A = start of B", 1.0)])

    def test_agreement_pairs_each_phase(self):
        a = [Step("A", 1.0, 5.0, "x"), Step("B", 6.0, 10.0, "x")]
        b = [Step("A", 2.0, 5.0, "x"), Step("B", 5.0, 10.0, "x")]             # another number of distinct boundaries
        r = V.agreement(a, b, 10.0, 10.0)
        self.assertEqual((r["boundaries"], r["max_boundary_difference_s"], r["mean_boundary_difference_s"]), (2, 1.0, 1.0))

    def test_an_annotator_file_uses_the_start_times(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "steps_raw_asel.csv"
            p.write_text("step,label,start,end\n1,A,0:23,1:04\n2,B,1:05,2:00\n")
            got = V.read_annotator(p)
            self.assertEqual([(x.label, x.start_s, x.end_s) for x in got], [("A", 23.0, 65.0), ("B", 65.0, 121.0)])
            self.assertEqual(V.annotator_name(p), "asel")


class RobustnessTest(unittest.TestCase):
    def test_the_variants_of_the_rule(self):
        seg = {"vlm_weight": 1.0, "siglip_weight": 1.0, "siglip_crop_weight": 1.0, "siglip_temp": 0.01}
        v = R.weight_variants(seg)
        self.assertEqual(len(v), 8)
        self.assertEqual(dict(v)["vlm_weight x0.5"], {"vlm_weight": 0.5})
        self.assertEqual(dict(v)["siglip_temp x2"], {"siglip_temp": 0.02})

    def test_subsets_are_seeded_halves(self):
        a, b, c = R.subsets([22, 22], 0), R.subsets([22, 22], 0), R.subsets([22, 22], 1)
        self.assertTrue(all((x == y).all() for x, y in zip(a, b)))
        self.assertFalse(all((x == y).all() for x, y in zip(a, c)))
        self.assertEqual([len(x) for x in a], [11, 11])
        self.assertEqual(len(set(a[0].tolist())), 11)

    def test_verdicts(self):
        self.assertEqual(R.summarise([], 93.2)["verdict"], "not tested")
        self.assertEqual(R.summarise([93.0] * 8 + [80.0] * 2, 93.2)["verdict"], "robust")      # exactly 80 %
        self.assertEqual(R.summarise([93.0] * 7 + [80.0] * 3, 93.2)["verdict"], "sensitive")
        self.assertEqual(R.summarise([80.0], 93.2)["verdict"], "sensitive")
        self.assertEqual(R.summarise([88.2], 93.2)["verdict"], "robust")
        g = lambda v: {"summary": {"verdict": v}}
        self.assertEqual(R.overall({"a": g("robust"), "b": g("robust")}), "robust overall")
        self.assertIn("not tested", R.overall({"a": g("robust"), "c": g("not tested")}))
        self.assertEqual(R.overall({"a": g("sensitive"), "c": g("not tested")}), "sensitive to: a")

    def test_bank_is_the_mean_of_the_chosen(self):
        e = [np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])]
        np.testing.assert_allclose(R.bank_from(e, [np.array([0, 1])])[0], [2 ** -0.5, 2 ** -0.5], atol=1e-6)

    def test_the_llm_author_files_and_an_incomplete_one(self):
        from assemblyvision.analysis import phase_text as pt
        wanted = [p["name"] for p in pt.phases("qisens")]
        self.assertEqual(R.AUTHORS, ("llm_llama32", "llm_gemma3", "llm_mistral7b"))
        for a in R.AUTHORS:
            self.assertEqual([p["name"] for p in R.author_phases(a, wanted)], wanted)
        self.assertEqual(set(R.REPLACED), {"llm_qwen3", "llm_phi4mini"})
        self.assertEqual(R.author_phases("llm_qwen3", wanted), [])           # Qwen3 wrote only the first phase
        self.assertIsNone(R.author_phases("nobody", wanted))


class ValidationLabelsTest(unittest.TestCase):
    def test_every_annotator_gets_the_same_seeded_items(self):
        frames = np.arange(100, 5000)
        a = VL.draw_items("trunk_band", "run_x", frames, 30.0, 6000)
        self.assertEqual(a, VL.draw_items("trunk_band", "run_x", frames[::-1], 30.0, 6000))
        self.assertEqual(len(a), 100)
        self.assertNotEqual(a, VL.draw_items("gaze", "run_x", frames, 30.0, 6000))
        self.assertTrue(all(100 <= x["frame"] < 5000 for x in a))

    def test_clips_and_event_windows_stay_inside_the_video(self):
        for x in VL.draw_items("repetition", "r", np.arange(0, 600), 30.0, 600):
            self.assertLessEqual(x["end_s"], 20.0)
            self.assertAlmostEqual(x["end_s"] - x["time_s"], 10.0, places=2)
        w = VL.draw_items("tool_pickup", "r", np.arange(0, 9000), 30.0, 9000)
        self.assertEqual(len(w), 1)
        self.assertAlmostEqual(w[0]["end_s"] - w[0]["time_s"], 120.0, places=2)

    def test_labels_are_per_user_and_replace_the_old_answer(self):
        with tempfile.TemporaryDirectory() as d:
            p = VL.label_path(Path(d), "run_x", "gaze", "asel")
            q = VL.label_path(Path(d), "run_x", "gaze", "kazi")
            it = {"item": 3, "frame": 90, "time_s": 3.0}
            VL.save_item(p, it, "left", "asel")
            VL.save_item(p, it, "front", "asel")
            VL.save_item(q, it, "right", "kazi")
            self.assertEqual([r["value"] for r in VL.read(p)], ["front"])
            self.assertEqual([r["value"] for r in VL.read(q)], ["right"])

    def test_a_label_needs_a_real_account(self):
        for bad in (None, "interface", "../x", "A B"):
            with self.assertRaises(ValueError):
                VL.check_user(bad)

    def test_marks(self):
        with tempfile.TemporaryDirectory() as d:
            p = VL.label_path(Path(d), "r", "tool_pickup", "asel")
            win = {"item": 0, "frame": 300, "time_s": 10.0, "end_s": 130.0}
            VL.add_mark(p, 50.0, win, 30.0, "asel")
            VL.add_mark(p, 20.0, win, 30.0, "asel")
            self.assertEqual([float(r["time_s"]) for r in VL.read(p)], [20.0, 50.0])
            with self.assertRaises(ValueError):
                VL.add_mark(p, 5.0, win, 30.0, "asel")
            VL.remove_last_mark(p)
            self.assertEqual([float(r["time_s"]) for r in VL.read(p)], [50.0])

    def test_clip_class_rules(self):
        with tempfile.TemporaryDirectory() as d:
            p = VL.clip_path(Path(d), "kazi")
            VL.save_clip(p, "a.mp4", ["X", "Y"], "X", None, None, 10.0, "kazi")
            VL.save_clip(p, "a.mp4", ["X", "Y"], "X", "Y", 4.0, 10.0, "kazi")
            self.assertEqual([(r["class"], r["second_class"], r["change_s"]) for r in VL.read(p)], [("X", "Y", "4.0")])
            for bad in (("Z", None, None), ("X", "X", 4.0), ("X", "Y", None), ("X", "Y", 11.0)):
                with self.assertRaises(ValueError):
                    VL.save_clip(p, "a.mp4", ["X", "Y"], bad[0], bad[1], bad[2], 10.0, "kazi")


class FloorPathTest(unittest.TestCase):
    def test_path_is_per_phase_and_never_jumps_between_parts(self):
        n = 60
        j = np.zeros((n, 17, 3))
        j[:, [11, 12], 0] = (np.arange(n) / 30.0)[:, None]                  # 1 m/s along x, for 2 s
        far = j + np.array([100.0, 0, 0])                                   # another part, another world frame
        pieces = [{"frames": np.arange(n), "joints": j}, {"frames": np.arange(n, 2 * n), "joints": far}]
        steps = [{"label": "other", "start_s": 0.0, "end_s": 0.5}, {"label": "A", "start_s": 0.0, "end_s": 4.0}]
        r = CO.floor_path(pieces, steps, 30.0)
        self.assertEqual(len(r), 1)
        self.assertAlmostEqual(r[0]["path_m"], 2 * 59 / 30.0, places=2)
        self.assertEqual((r[0]["parts"], r[0]["coverage"]), (2, 1.0))

    def test_a_phase_without_body3d(self):
        self.assertIsNone(CO.floor_path([], [{"label": "A", "start_s": 0.0, "end_s": 1.0}], 30.0)[0]["path_m"])


class ExtrasTest(unittest.TestCase):
    steps = [{"label": "other", "start_s": 0.0, "end_s": 5.0}, {"label": "A", "start_s": 5.0, "end_s": 15.0},
             {"label": "B", "start_s": 15.0, "end_s": 25.0}, {"label": "A", "start_s": 25.0, "end_s": 40.0}, {"label": "C", "start_s": 40.0, "end_s": 50.0}]

    def test_cycles_count_a_repeated_phase(self):
        c = X.cycles(self.steps)
        self.assertEqual([(r["phase"], r["cycle"], r["duration_s"]) for r in c], [("A", 1, 10.0), ("B", 1, 10.0), ("A", 2, 15.0), ("C", 1, 10.0)])
        self.assertEqual(X.cycle_png(c).shape[2], 3)
        self.assertIsNone(X.cycle_png([]))

    def test_time_ranges(self):
        f = np.arange(0, 300)
        m = (f >= 30) & (f < 120) | (f >= 200) & (f < 210)                  # 3 s, and 0.33 s (too short)
        self.assertEqual(X.stretches(f, m, 30.0, 0.0, 10.0), [[1.0, 4.0]])

    def rows(self, label, start, **values):
        return [{"label": label, "start_s": str(start), "metric": k, "value": str(v), "coverage": "0.9"} for k, v in values.items()]

    def test_rules_carry_their_evidence(self):
        c = X.cycles(self.steps)
        m = self.rows("A", 5.0, bend_b0_s=2, bend_b1_s=2, bend_b2_s=1, bend_b3_s=5, idle_share=0.4, two_hand_simultaneous=0.1)
        m += self.rows("B", 15.0, bend_b0_s=9, bend_b1_s=1, bend_b2_s=0, bend_b3_s=0, idle_share=0.1, two_hand_simultaneous=0.8)
        got = X.rule_suggestions(c, m, [], {(5.0, "bend"): [[6.0, 11.0]]})
        by = {s["rule"]: s for s in got}
        self.assertEqual(set(by), {"deep bending", "long idle", "low two hand use", "long phase"})    # A cycle 2: 15 s, median 10 s
        self.assertEqual((by["deep bending"]["value"], by["deep bending"]["time_ranges"]), (0.5, [[6.0, 11.0]]))
        self.assertEqual(by["long idle"]["time_ranges"], [[5.0, 15.0]])
        self.assertIn("0.25", by["low two hand use"]["threshold"])

    def test_a_badly_covered_or_missing_metric_gives_no_suggestion(self):
        c = X.cycles(self.steps)
        m = self.rows("A", 5.0, idle_share=0.9)
        m[0]["coverage"] = "0.2"
        m += self.rows("B", 15.0, idle_share="")
        self.assertEqual([x["rule"] for x in X.rule_suggestions(c, m, [], {})], ["long phase"])

    def test_the_rows_are_marked_experimental(self):
        e = {"cycles": X.cycles(self.steps), "floor_path": [{"path_m": 1.5, "coverage": 1.0}] * 4, "suggestions": [],
             "event_rows": [{**X.cycles(self.steps)[0], "metric": "tool_pickup_count", "value": 2, "unit": "count"}]}
        r = X.metric_rows(e, "run")
        self.assertIn("not_validated", next(x for x in r if x["metric"] == "movement_path_floor")["flags"])
        self.assertIn("experimental", next(x for x in r if x["metric"] == "tool_pickup_count")["flags"])
        self.assertIn("experimental", next(x for x in r if x["metric"] == "improvable_motions")["flags"])


class HandEventTest(unittest.TestCase):
    def scene(self, object_follows):
        """A wrist reaches an object at x = 300, stops, then leaves to x = 500, with or without the object."""
        T, fps = 120, 30.0
        x = np.r_[np.linspace(100, 300, 30), np.full(30, 300.0), np.linspace(300, 500, 30), np.full(30, 500.0)]
        wrist = np.stack([x, np.full(T, 200.0)], 1)
        speed = np.r_[0.0, np.abs(np.diff(x)) * fps / 100.0]
        frames, boxes, offset = [], [], [0]
        for f in range(0, T, 15):
            cx = x[f] if object_follows and f >= 60 else 300.0
            frames.append(f)
            boxes.append([cx - 10, 190, cx + 10, 210])
            offset.append(len(boxes))
        det = {"frames": np.array(frames), "offset": np.array(offset), "boxes": np.array(boxes, float),
               "cls": np.zeros(len(boxes), int), "conf": np.full(len(boxes), 0.9)}
        return HE.wrist_events(np.arange(T), wrist, speed, 100.0, fps, det, ["bottle"], 0.3, "right")

    def test_the_object_goes_with_the_hand_is_a_pickup(self):
        e = self.scene(True)
        self.assertEqual([x["kind"] for x in e], ["pickup"])
        self.assertAlmostEqual(e[0]["t_s"], 1.0, delta=0.1)
        self.assertGreater(e[0]["reach_s"], 0.5)

    def test_the_object_stays_is_a_placement(self):
        self.assertEqual([x["kind"] for x in self.scene(False)], ["placement"])

    def test_no_object_no_event(self):
        det = {"frames": np.array([0, 60]), "offset": np.array([0, 0, 0]), "boxes": np.zeros((0, 4)), "cls": np.zeros(0, int), "conf": np.zeros(0)}
        x = np.r_[np.linspace(0, 200, 30), np.full(60, 200.0)]
        self.assertEqual(HE.wrist_events(np.arange(90), np.stack([x, x], 1), np.r_[0.0, np.abs(np.diff(x)) * 0.3], 100.0, 30.0, det, ["bottle"], 0.3, "left"), [])


class ScoreMeasurementsTest(unittest.TestCase):
    def test_events_match_one_to_one_within_a_second(self):
        r = SM.match_events([10.0, 20.0, 30.0], [10.4, 10.6, 21.5, 30.9])
        self.assertEqual((r["matched"], r["recall"], r["precision"]), (2, 0.667, 0.5))
        self.assertEqual(SM.match_events([], [1.0])["recall"], None)

    def test_agreement_leaves_out_cannot_tell_and_counts_coverage(self):
        r = SM.agreement([("front", "front"), ("left", "right"), (SM.CANNOT, "front"), ("away", None)])
        self.assertEqual((r["told"], r["scored"], r["agreement"], r["coverage"]), (3, 2, 0.5, 0.667))

    def test_verdicts_use_the_frozen_targets(self):
        self.assertEqual(SM.verdict(9, {"agreement": 0.75}), "meets the target")
        self.assertEqual(SM.verdict(10, {"agreement": 0.69}), "misses the target")
        self.assertEqual(SM.verdict(3, {"agreement": 0.85, "within_one": 0.94}), "misses the target")
        self.assertEqual(SM.verdict(7, {"recall": 0.0, "precision": None}), "misses the target")
        self.assertEqual(SM.verdict(7, {"recall": 0.8, "precision": None}), "-")
        self.assertEqual(SM.verdict(1, {"boundary_mae_s": 8.09, "phases": 5, "phases_within_10_pct": 5}), "meets the target")

    def test_kappa_between_annotators_leaves_out_cannot_see(self):
        a = {"0": "yes", "1": "no", "2": "yes", "3": "cannot see"}
        b = {"0": "yes", "1": "no", "2": "no", "3": "yes"}
        k = SM.kappa_of(a, b, "cannot see")
        self.assertEqual((k["items"], k["agreement"]), (3, 0.667))
        self.assertIsNone(SM.kappa_of({"0": "yes"}, {"0": "yes"}, "cannot see"))

    def test_the_gaze_answers_map_onto_the_label_set(self):
        self.assertEqual(set(v for v in SM.GAZE_MAP.values() if v), set(VL.GAZE2) - {"up", "cannot see"})

    def test_round2_verdicts(self):
        self.assertEqual(SM.verdict(5, {"useful": 0.75}), "meets the target")
        self.assertIn("PCK part missed", SM.verdict(6, {"pck": 0.5}))

    def test_the_pdf_title_without_a_cjk_font(self):
        from unittest import mock
        from assemblyvision.output import report as RP
        rep = {"video": "メンテナンス.mp4", "duration_s": 596.4, "source": "qisens"}
        with mock.patch.object(RP, "cjk_font", return_value=None):
            self.assertEqual(RP.video_title(rep), "QiSens video (9.9 min)")
            self.assertEqual(RP.video_title({"video": "a.mp4"}), "a.mp4")
        with mock.patch.object(RP, "cjk_font", return_value="Noto Sans CJK JP"):
            self.assertEqual(RP.video_title(rep), "メンテナンス.mp4")

    def test_the_clicked_trunk_angle_is_the_systems_definition(self):
        from assemblyvision.analysis.motion import trunk_angle
        self.assertEqual(VL.trunk_angle_band((100, 200), (100, 100)), (0.0, "B0"))        # upright: shoulder above the hip
        a, band = VL.trunk_angle_band((100, 200), (200, 100))
        self.assertAlmostEqual(a, 45.0)
        self.assertEqual(band, "B2")                                                      # 45 is the first value of B2
        self.assertEqual(VL.trunk_angle_band((100, 200), (0, 200))[1], "B3")               # lying flat, either side
        hip, sho = np.array([310.0, 420.0]), np.array([352.0, 300.0])
        self.assertAlmostEqual(VL.trunk_angle_band(hip, sho)[0], float(trunk_angle(sho, hip)))

    def test_old_band_answers_are_dropped_and_hands_and_gaze_kept(self):
        old = [{"item": "0", "frame": "5", "trunk": "B1 (20 to 45 degrees)", "hands": "yes", "gaze": "up"},
               {"item": "1", "frame": "9", "trunk": "cannot see", "hands": "no", "gaze": "away"}]
        rows, n = VL.drop_old_bands(old)
        self.assertEqual(n, 1)
        self.assertEqual((rows[0]["trunk"], rows[0]["hands"], rows[0]["gaze"]), ("", "yes", "up"))
        self.assertEqual(rows[1]["trunk"], "cannot see")

    def test_a_saved_trunk_keeps_points_angle_and_band(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "asel.csv"
            it = {"item": 2, "frame": 30, "time_s": 1.0, "phase": "A"}
            VL.save_frame(p, it, ((100.0, 200.0), (130.0, 100.0)), "yes", VL.GAZE2[0], "asel")
            r = VL.read(p)[0]
            self.assertEqual((r["trunk"], r["band"], r["hip_x"], r["shoulder_y"]), ("clicks", "B0", "100.0", "100.0"))
            self.assertAlmostEqual(float(r["angle_deg"]), 16.7, places=1)
            VL.save_frame(p, it, None, "no", VL.GAZE2[1], "asel")             # trunk not asked: the saved trunk stays
            r = VL.read(p)[0]
            self.assertEqual((r["trunk"], r["band"], r["hands"]), ("clicks", "B0", "no"))
            with self.assertRaises(ValueError):
                VL.save_frame(p, it, None, "", VL.GAZE2[0], "asel")

    def test_the_30_frames_are_a_subset_of_the_100_with_their_item_numbers(self):
        f = np.arange(0, 10000)
        spans = [("A", 0, 60), ("B", 60, 120), ("C", 120, 200), ("D", 200, 260), ("E", 260, 333)]
        full = VL.stratified_frames("frames", "run_x", f, 30.0, spans, 100)
        sub = VL.stratified_frames("frames", "run_x", f, 30.0, spans, 100, 30)
        self.assertEqual(len(sub), 30)
        self.assertTrue(all(x in full for x in sub))
        self.assertEqual([sum(x["phase"] == p for x in sub) for p in "ABCDE"], [6] * 5)

    def test_a_measurement_without_labels_says_so(self):
        self.assertIn("needs a walk of known distance", SM.fmt(2, None)[0][1])
        self.assertIn("no labels yet", SM.fmt(9, None)[0][1])
        self.assertIn("coco_pose", SM.fmt(3, None)[0][1])
        self.assertTrue(all(SM.TARGETS[n] for n in range(1, 11)))


if __name__ == "__main__":
    unittest.main()
