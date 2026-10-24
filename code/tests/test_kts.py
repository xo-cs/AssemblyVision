import unittest
from pathlib import Path

import numpy as np

from assemblyvision.errors import AvError
from assemblyvision.kts import empty_series, read_kts, write_kts
from helpers import fake_root, make_ctx


class KtsTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.cfg, self.ctx = make_ctx(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        s = empty_series("r", "s", "s-t01", 30.0, 854, 480, 10, 50)
        s.present[5:20] = True
        s.body_xy[5:20] = 1.5
        s.body_conf[5:20] = 0.8
        path = write_kts(s, self.ctx.paths, {"pose_variant": "l", "is_subject": True, "subject_source": "auto"})
        got, meta = read_kts(path)
        self.assertEqual(got.frame_idx[0], 10)
        self.assertTrue(np.array_equal(got.present, s.present))
        self.assertTrue(np.isnan(got.body_xy[0]).all())
        self.assertEqual((meta["frame_start"], meta["frame_count"], meta["subject_source"]), (10, 50, "auto"))
        self.assertFalse(meta["hands_done"])

    def test_damaged_file(self):
        s = empty_series("r", "s", "s-t02", 30.0, 854, 480, 0, 5)
        path = write_kts(s, self.ctx.paths, {})
        path.write_bytes(b"broken")
        with self.assertRaises(AvError) as cm:
            read_kts(path)
        self.assertEqual(cm.exception.code, "AV-STO-003")


if __name__ == "__main__":
    unittest.main()
