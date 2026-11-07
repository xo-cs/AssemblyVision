"""Subject and location splits: deterministic, subject disjoint, stratified, stored lists checked."""
import unittest
from pathlib import Path

from assemblyvision.datasets import splits
from assemblyvision.datasets.base import VideoRef
from assemblyvision.tools.dataset_splits import PILOT, rgb_subset
from helpers import fake_root


def refs() -> list[VideoRef]:
    out = []
    for s in range(1, 23):
        out += [VideoRef(f"IDU{s:03d}V{v:03d}", f"S{s:03d}", "lab_a", 30.0) for v in range(1, 4)]
    for s in range(23, 42):
        out += [VideoRef(f"IDU{s:03d}V{v:03d}", f"S{s:03d}", "lab_b", 30.0) for v in range(1, 7)]
    return out


class SplitTest(unittest.TestCase):
    def test_cross_subject_is_seeded_disjoint_and_stratified(self):
        a = splits.cross_subject(refs(), 0.3, 42)
        self.assertEqual(a, splits.cross_subject(refs(), 0.3, 42))
        self.assertNotEqual(a["test_subjects"], splits.cross_subject(refs(), 0.3, 7)["test_subjects"])
        splits.check_disjoint(a)
        test = set(a["test_subjects"])
        self.assertEqual(sum(1 for s in test if int(s[1:]) <= 22), round(0.3 * 22))      # 7 of lab a
        self.assertEqual(sum(1 for s in test if int(s[1:]) > 22), round(0.3 * 19))       # 6 of lab b
        vids = {v.video_id: v.subject for v in refs()}
        self.assertTrue(all(vids[v] in test for v in a["test_videos"]))
        self.assertEqual(len(a["train_videos"]) + len(a["test_videos"]), len(refs()))

    def test_cross_location(self):
        folds = splits.cross_location(refs())
        self.assertEqual([f["name"] for f in folds], ["cross_location_test_lab_a", "cross_location_test_lab_b"])
        self.assertEqual(len(folds[0]["test_videos"]), 66)
        self.assertEqual(len(folds[1]["test_videos"]), 114)
        for f in folds:
            splits.check_disjoint(f)

    def test_stored_lists_are_checked(self):
        tmp = fake_root()
        path = Path(tmp.name) / "experiments/datasets/ha4m_splits.json"
        s = splits.make_splits("ha4m", refs(), 0.3, 42)
        splits.write_splits(path, s)
        splits.check_stored(path, s)
        with self.assertRaises(ValueError):
            splits.check_stored(path, splits.make_splits("ha4m", refs()[:-3], 0.3, 42))
        tmp.cleanup()

    def test_rgb_subset(self):
        s = splits.make_splits("ha4m", refs(), 0.3, 42)
        sub = rgb_subset(s, refs(), 6, 2, 42)
        self.assertEqual(sub["n_videos"], 24)
        self.assertTrue(set(PILOT) <= set(sub["videos"]))
        labs = [int(x[1:]) <= 22 for x in sub["subjects"]]
        self.assertEqual((labs.count(True), labs.count(False)), (6, 6))
        self.assertEqual(len(sub["test_subjects_in_subset"]), 4)            # 2 of 6 per lab, as in the split
        self.assertEqual(sub, {**rgb_subset(s, refs(), 6, 2, 42), "written_utc": sub["written_utc"]})

    def test_overlap_is_refused(self):
        with self.assertRaises(ValueError):
            splits.check_disjoint({"name": "x", "train_subjects": ["S1"], "test_subjects": ["S1"],
                                   "train_videos": ["a"], "test_videos": ["b"]})


if __name__ == "__main__":
    unittest.main()
