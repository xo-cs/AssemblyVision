"""Segmentation metrics and kappa against examples worked out by hand (CPU)."""
import unittest

from assemblyvision.evaluation import metrics as M
from assemblyvision.types import Step


def seq(*parts):
    out = []
    for label, n in parts:
        out += [label] * n
    return out


class F1Test(unittest.TestCase):
    def test_hand_worked_f1(self):   # TC-EVL-01
        # GT A[0,10) B[10,20) C[20,30) D[30,39] (MS TCN ends the last segment at its last index)
        gt = seq(("A", 10), ("B", 10), ("C", 10), ("D", 10))
        # pred A[0,10) IoU 1.0, B[16,20) IoU 4/10, C[20,22) IoU 2/10, D[22,39] IoU 9/17
        pred = seq(("A", 10), ("other", 6), ("B", 4), ("C", 2), ("D", 18))
        self.assertEqual(M.f_score(pred, gt, 0.10), (4.0, 0.0, 0.0))
        self.assertAlmostEqual(M.f1_overlap(pred, gt, 0.10), 100.0)
        self.assertEqual(M.f_score(pred, gt, 0.25), (3.0, 1.0, 1.0))
        self.assertAlmostEqual(M.f1_overlap(pred, gt, 0.25), 75.0)
        self.assertEqual(M.f_score(pred, gt, 0.50), (2.0, 2.0, 2.0))
        self.assertAlmostEqual(M.f1_overlap(pred, gt, 0.50), 50.0)

    def test_gt_segment_matched_once(self):
        gt = seq(("A", 20))
        pred = seq(("A", 10), ("other", 1), ("A", 9))
        self.assertEqual(M.f_score(pred, gt, 0.1), (1.0, 1.0, 0.0))

    def test_empty_cases(self):
        self.assertEqual(M.f1_overlap(seq(("other", 5)), seq(("A", 5)), 0.1), 0.0)
        self.assertEqual(M.f1_overlap(seq(("A", 5)), seq(("other", 5)), 0.1), 0.0)


class EditTest(unittest.TestCase):
    def test_edit(self):   # TC-EVL-02
        gt = seq(("A", 5), ("B", 5), ("C", 5), ("D", 5))
        self.assertEqual(M.edit_score(gt, gt), 100.0)
        self.assertEqual(M.edit_score(seq(("A", 5), ("B", 5), ("X", 5), ("D", 5)), gt), 75.0)

    def test_other_removed_and_lengths_ignored(self):
        gt = seq(("A", 5), ("B", 5))
        pred = seq(("other", 3), ("A", 1), ("other", 2), ("B", 9))
        self.assertEqual(M.edit_score(pred, gt), 100.0)
        self.assertEqual(M.levenstein([], [], norm=True), 100.0)


class FrameAndBoundaryTest(unittest.TestCase):
    def test_frame_accuracy_counts_other(self):
        gt = seq(("other", 2), ("A", 8))
        pred = seq(("A", 10))
        self.assertAlmostEqual(M.frame_accuracy(pred, gt), 80.0)

    def test_frames_from_steps(self):
        steps = [Step("A", 0.0, 1.0, "gt"), Step("B", 1.5, 2.0, "gt")]
        self.assertEqual(M.frames_from_steps(steps, 2.0, 2), ["A", "A", "other", "B"])

    def test_boundary_mae_by_hand(self):
        gt = [Step("A", 0, 10, "gt"), Step("B", 10, 20, "gt"), Step("other", 20, 30, "gt")]
        pred = [Step("A", 0, 12, "p"), Step("B", 12, 25, "p"), Step("other", 25, 30, "p")]
        mae, rows = M.boundary_mae(pred, gt, 30.0)
        # GT boundaries 10 and 20, predicted 12 and 25: errors 2 and 5
        self.assertEqual([r["error_s"] for r in rows], [2.0, 5.0])
        self.assertEqual(mae, 3.5)


class KappaTest(unittest.TestCase):
    def test_identical(self):   # TC-EVL-05
        a = seq(("A", 5), ("B", 7))
        self.assertEqual(M.cohen_kappa(a, a)[0], 1.0)

    def test_two_by_two(self):
        # yes/yes 20, yes/no 5, no/yes 10, no/no 15: po 0.7, pe 0.5*0.6 + 0.5*0.4 = 0.5, kappa 0.4
        a = seq(("y", 20), ("y", 5), ("n", 10), ("n", 15))
        b = seq(("y", 20), ("n", 5), ("y", 10), ("n", 15))
        k, po, pe = M.cohen_kappa(a, b)
        self.assertAlmostEqual(po, 0.7)
        self.assertAlmostEqual(pe, 0.5)
        self.assertAlmostEqual(k, 0.4)


if __name__ == "__main__":
    unittest.main()
