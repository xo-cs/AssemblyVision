"""TC-HMP: HA4M macro phases and their GT rules (Week 10)."""
import unittest

import numpy as np

from assemblyvision.datasets import ha4m_phases as hp


class MacroPhaseTest(unittest.TestCase):
    def test_grouping_covers_the_12_actions_once_in_order(self):
        acts = [a for _, g in hp.MACRO_PHASES for a in g]
        self.assertEqual(acts, list(range(1, 13)))
        self.assertEqual(len(hp.NAMES), 5)

    def test_dont_care_rules(self):
        # intro 0, Block 1 (1, 0, 2), gap to Block 2 (0 0), Block 2 (5), outro 0
        a = np.array([0, 0, 1, 0, 2, 0, 0, 0, 5, 5, 0])
        idx = hp.macro_index(a)
        self.assertEqual(idx.tolist(), [-1, -1, 0, 0, 0, 0, 0, 1, 1, 1, -1])    # gap split at the midpoint
        steps = hp.steps_from_index(idx, 1.0)
        self.assertEqual([(s.label, s.start_s, s.end_s) for s in steps],
                         [("other", 0.0, 2.0), ("Block 1", 2.0, 7.0), ("Block 2", 7.0, 10.0), ("other", 10.0, 11.0)])
        self.assertFalse(hp.in_task_order(steps))                  # three phases are missing

    def test_out_of_order_video_keeps_its_order(self):
        a = np.array([5, 1, 9, 10, 12, 1])
        steps = hp.steps_from_index(hp.macro_index(a), 1.0)
        self.assertEqual([s.label for s in steps], ["Block 2", "Block 1", "Join blocks", "Cover and screws", "Fasten",
                                                    "Block 1"])
        self.assertFalse(hp.in_task_order(steps))


if __name__ == "__main__":
    unittest.main()
