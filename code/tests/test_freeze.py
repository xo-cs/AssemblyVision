"""TC-FRZ: the validation freeze, the phase file and the decoders without order (Week 12). CPU."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.analysis import clip_classes as C
from assemblyvision.analysis import zero_shot as Z
from assemblyvision.storage.files import REPO_ROOT
from assemblyvision.tools import freeze as F
from assemblyvision import validation_labels as VL
from assemblyvision.evaluation import metrics as M
from assemblyvision.tools import robustness as R
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
        self.assertEqual((r["boundaries"], r["max_boundary_difference_s"], r["mean_boundary_difference_s"]), (3, 1.0, 0.67))


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

    def test_the_author_template_has_the_phases_in_order(self):
        from assemblyvision.analysis import phase_text as pt
        got = pt.parse(R.AUTHOR_DIR / "TEMPLATE.md")[None]
        self.assertEqual([p["name"] for p in got], [p["name"] for p in pt.phases("qisens")])


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


if __name__ == "__main__":
    unittest.main()
