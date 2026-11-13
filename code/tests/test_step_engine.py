"""TC-ENG: the step engine choice seg_a.engine ("B zero shot" default from 2026-11-13, "B v1", "A", "custom";
"A+B v1" is an alias) (Week 10, 11)."""
import unittest

from assemblyvision.config import ENGINE_PRESETS, Config, engine_label
from assemblyvision.errors import AvError
from assemblyvision.tools.run_pose import stored_config


class StepEngineTest(unittest.TestCase):
    def test_default_is_b_zero_shot(self):
        c = Config.load()
        self.assertEqual(c["seg_a"]["engine"], "B zero shot")
        self.assertEqual(engine_label(c), "B zero shot")
        # the Z2 row of w11-qisens-zero-shot (93.20): equal weights, B v1 boundary evidence, dense candidates
        self.assertEqual([c["seg_a"][k] for k in ("siglip_weight", "siglip_crop_weight", "vlm_weight", "siglip_temp",
                                                  "shot_smooth", "engine_a_scale", "visual_weight", "intro_detect")],
                         [1.0, 1.0, 1.0, 0.01, False, 1.0, 0.0, False])
        self.assertEqual(c["engine_b"]["candidates"], "dense")
        self.assertTrue(c["zero_shot"]["in_pipeline"])
        self.assertEqual((c["zero_shot"]["vlm_window_s"], c["zero_shot"]["vlm_frames"]), (2.0, 4))

    def test_b_v1_preset(self):
        c = Config.load(overrides={"seg_a": {"engine": "B v1"}})
        self.assertEqual(c["seg_a"]["engine"], "B v1")
        self.assertEqual((c["seg_a"]["embed_weight"], c["seg_a"]["embed_window_s"], c["engine_b"]["model"]),
                         (0.2, 30.0, "engine_b/w10-engine-b-v1.pt"))
        self.assertEqual(c["seg_a"]["engine_b_weight"], 0.0)
        self.assertEqual(engine_label(c), "B v1")
        self.assertTrue(c["shots"]["parallel"])

    def test_old_name_is_an_alias(self):
        self.assertEqual(Config.load(overrides={"seg_a": {"engine": "A+B v1"}})["seg_a"]["engine"], "B v1")

    def test_a_turns_engine_b_off(self):
        c = Config.load(overrides={"seg_a": {"engine": "A"}})
        self.assertEqual((c["seg_a"]["embed_weight"], c["seg_a"]["engine_b_weight"]), (0.0, 0.0))
        self.assertEqual(engine_label(c), "A")

    def test_preset_values_given_again_are_fine_other_values_conflict(self):
        Config.load(overrides={"seg_a": {"embed_weight": 0.2, "embed_window_s": 30.0},
                               "engine_b": {"model": "engine_b/w10-engine-b-v1.pt"}})
        with self.assertRaises(AvError) as cm:
            Config.load(overrides={"seg_a": {"embed_weight": 0.4}})
        self.assertIn("custom", str(cm.exception))
        with self.assertRaises(AvError):
            Config.load(overrides={"seg_a": {"engine": "A", "engine_b_weight": 0.4}})
        with self.assertRaises(AvError):
            Config.load(overrides={"seg_a": {"engine": "B"}})

    def test_custom_keeps_the_keys(self):
        c = Config.load(overrides={"seg_a": {"engine": "custom", "embed_weight": 0.4}})
        self.assertEqual(c["seg_a"]["embed_weight"], 0.4)
        self.assertEqual(engine_label(c), "B custom")
        c = Config.load(overrides={"seg_a": {"engine": "custom"}})
        self.assertEqual(engine_label(c), "A")                    # shipped weights are 0

    def test_resolved_snapshot_loads_again(self):
        for e in ENGINE_PRESETS:
            snap = Config.load(overrides={"seg_a": {"engine": e}}).snapshot()
            self.assertEqual(Config.load(overrides=snap).snapshot(), snap)

    def test_a_stored_run_from_before_the_engine_runs_as_stored(self):
        self.assertEqual(stored_config({"config": {"seg_a": {"embed_weight": 0.0}}})["seg_a"]["engine"], "custom")
        self.assertEqual(stored_config({"config": {"seg_a": {"engine": "A"}}})["seg_a"]["engine"], "A")

    def test_a_preset_set_on_a_stored_run_replaces_its_stored_values(self):
        from assemblyvision.errors import AvError
        from assemblyvision.tools.run_pose import stored_with
        stored = Config.load(overrides={"seg_a": {"engine": "custom", "embed_window_s": 60.0}}).snapshot()
        manifest = {"config": stored}
        with self.assertRaises(AvError):                     # the old merge: the stored 60 s conflicts
            Config.load(overrides={**stored, "seg_a": {**stored["seg_a"], "engine": "B v1"}})
        c = Config.load(overrides=stored_with(manifest, {"seg_a": {"engine": "B v1"}}))
        self.assertEqual((c["seg_a"]["embed_window_s"], c["seg_a"]["embed_weight"]), (30.0, 0.2))
        self.assertEqual(c["seg_a"]["pause_energy"], stored["seg_a"]["pause_energy"])   # other stored keys stay
        with self.assertRaises(AvError):                     # a value given explicitly still conflicts
            Config.load(overrides=stored_with(manifest, {"seg_a": {"engine": "B v1", "embed_window_s": 60.0}}))
        c = Config.load(overrides=stored_with(manifest, {"seg_a": {"embed_weight": 0.4}}))
        self.assertEqual(c["seg_a"]["embed_window_s"], 60.0)  # no preset given: the stored values stay

    def test_reference_profile_turns_parallel_shots_off(self):
        from assemblyvision.config import DEFAULT_LAYERS
        from assemblyvision.storage.files import REPO_ROOT
        c = Config.load(layers=[*DEFAULT_LAYERS, REPO_ROOT / "code/configs/reference.toml"])
        self.assertFalse(c["shots"]["parallel"])


if __name__ == "__main__":
    unittest.main()
