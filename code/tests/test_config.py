import tomllib
import unittest

from assemblyvision.config import Config
from assemblyvision.errors import AvError
from helpers import fake_root


class ConfigTest(unittest.TestCase):
    def test_defaults_load(self):
        cfg = Config.load()
        self.assertEqual(cfg.get("pose.imgsz"), 640)
        self.assertEqual(cfg["pose"]["det_conf_min"], 0.10)
        self.assertIs(cfg["pose"]["half"], False)

    def test_unknown_key_refused(self):
        with self.assertRaises(AvError) as cm:
            Config.load(overrides={"pose": {"imgsize": 640}})
        self.assertEqual(cm.exception.code, "AV-CFG-001")
        with self.assertRaises(AvError) as cm:
            Config.load(overrides={"posee": {"imgsz": 640}})
        self.assertEqual(cm.exception.code, "AV-CFG-001")

    def test_bad_value_refused(self):
        for over in ({"pose": {"variant": "x"}}, {"pose": {"imgsz": "640"}},
                     {"pose": {"half": True}}, {"pose": {"det_conf_min": 1.5}}):
            with self.assertRaises(AvError) as cm:
                Config.load(overrides=over)
            self.assertEqual(cm.exception.code, "AV-CFG-002", over)

    def test_int_accepted_for_float(self):
        self.assertEqual(Config.load(overrides={"ingest": {"disk_factor": 2}}).get("ingest.disk_factor"), 2)

    def test_layer_file(self):
        with fake_root() as d:
            from pathlib import Path
            from assemblyvision.config import DEFAULT_TOML
            layer = Path(d) / "run_config.toml"
            layer.write_text('[pose]\nvariant = "l"\n')
            self.assertEqual(Config.load([DEFAULT_TOML], run_config=layer).get("pose.variant"), "l")
            with self.assertRaises(AvError) as cm:
                Config.load([DEFAULT_TOML, Path(d) / "missing.toml"])
            self.assertEqual(cm.exception.code, "AV-CFG-003")

    def test_snapshot_round_trip(self):
        cfg = Config.load(overrides={"pose": {"variant": "n"}})
        self.assertEqual(tomllib.loads(cfg.to_toml()), cfg.snapshot())


if __name__ == "__main__":
    unittest.main()
