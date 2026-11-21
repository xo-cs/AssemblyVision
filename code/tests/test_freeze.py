"""TC-FRZ: the validation freeze, the phase file and the decoders without order (Week 12). CPU."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.analysis import clip_classes as C
from assemblyvision.analysis import zero_shot as Z
from assemblyvision.storage.files import REPO_ROOT
from assemblyvision.tools import freeze as F
from assemblyvision.tools import train_domain as T


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


if __name__ == "__main__":
    unittest.main()
