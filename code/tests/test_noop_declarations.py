"""Every setting of configs/default.toml declares whether its default is a no-op
(experiments/noop.py), so a new key can never make stored experiments unusable by accident."""
import unittest

from assemblyvision.config import Config
from assemblyvision.experiments import noop


class DeclarationTest(unittest.TestCase):
    def test_every_key_is_declared(self):
        missing = noop.undeclared(Config.load().snapshot())
        self.assertEqual(missing, [], "declare these keys in assemblyvision/experiments/noop.py: " + ", ".join(missing))

    def test_no_stale_declarations(self):
        self.assertEqual(noop.stale(Config.load().snapshot()), [])

    def test_off_and_dependent_rules_hold_for_the_defaults(self):
        d = Config.load().snapshot()
        for s, vals in noop.DECLARATIONS.items():
            for k, rule in vals.items():
                if rule == noop.OFF:                    # OFF must really be off by default
                    self.assertTrue(noop.is_no_op(f"{s}.{k}", d[s][k], d), f"{s}.{k}")
        on = {**d, "seg_a": {**d["seg_a"], "engine_b_weight": 0.4, "start_detect": True}}
        self.assertFalse(noop.is_no_op("engine_b.pool_s", d["engine_b"]["pool_s"], on))
        self.assertFalse(noop.is_no_op("seg_a.start_sustain_s", 10.0, on))
        self.assertTrue(noop.is_no_op("engine_b.action_weight", 1.0, on))     # training only
        self.assertFalse(noop.is_no_op("seg_a.engine_b_weight", 0.4, d))
        self.assertFalse(noop.is_no_op("pose.batch", 8, d))


if __name__ == "__main__":
    unittest.main()
