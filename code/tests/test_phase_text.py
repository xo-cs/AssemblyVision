"""TC-PHT: the phase description files are complete and match the phase names."""
import unittest

from assemblyvision.analysis import phase_text as pt
from assemblyvision.config import Config
from assemblyvision.datasets import ikea_phases as ip


class PhaseTextTest(unittest.TestCase):
    def check(self, phases):
        for p in phases:
            self.assertGreaterEqual(len(p["descriptions"]), 20, p["name"])
            self.assertEqual(len(p["short"]), 3, p["name"])
            self.assertTrue(p["distinguishes"], p["name"])

    def test_qisens_matches_steps_toml(self):
        ph = pt.phases("qisens")
        self.assertEqual([p["name"] for p in ph], [s["name"] for s in Config.load()["steps"]["step"]])
        self.assertEqual(len(ph), 5)
        self.check(ph)

    def test_ikea_matches_macro_phases(self):
        for furn in ip.MACRO_PHASES:
            ph = pt.phases("ikea_asm", furn)
            self.assertEqual([p["name"] for p in ph], ip.names(furn))
            self.check(ph)


if __name__ == "__main__":
    unittest.main()
