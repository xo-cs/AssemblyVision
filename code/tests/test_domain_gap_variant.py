"""TC-DGV: the domain gap check for another YOLO26 variant on the same frames (Week 10)."""
import json
import tempfile
import unittest
from pathlib import Path

from assemblyvision.tools import ha4m_domain_gap as dg


class DomainGapVariantTest(unittest.TestCase):
    def test_yolo_files_per_variant(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self.assertEqual(dg.yolo_stem(d, "x"), d / "yolo_x")              # nothing stored yet
            (d / "yolo.json").write_text(json.dumps({"variant": "x"}))
            self.assertEqual(dg.yolo_stem(d, "x"), d / "yolo")                # the first run keeps its name
            self.assertEqual(dg.yolo_stem(d, "l"), d / "yolo_l")
        self.assertEqual(dg.exp_id("x"), "w10-ha4m-domain-gap")
        self.assertEqual(dg.exp_id("l"), "w10-ha4m-domain-gap-l")

    def test_against_lists_both_and_the_difference(self):
        def rec(p05, p1, px):
            return {"summary": {"found_share": 1.0, "mean_px": px, "median_px": px, "pck_0.05": p05, "pck_0.1": p1,
                                "pck_0.05_visible": p05, "pck_0.1_visible": p1},
                    "joints": [{"joint": "nose", "pck_0.05": p05, "pck_0.1": p1, "mean_px": px}]}
        a = dg.against(rec(0.6, 0.9, 30.0), rec(0.5, 0.85, 40.0), "w10-ha4m-domain-gap", "x")
        self.assertEqual(a["summary"]["pck_0.1"], [0.9, 0.85, 0.05])
        self.assertEqual(a["summary"]["mean_px"], [30.0, 40.0, -10.0])
        self.assertEqual(a["joints"][0]["pck_0.05"], [0.6, 0.5, 0.1])
        res = {"id": "w10-ha4m-domain-gap-l", "title": "t", "variant": "l", "videos": {"V": {
            **rec(0.6, 0.9, 30.0), "location": "lab_a", "camera": "c", "against": a}}}
        res["videos"]["V"]["summary"].update(frames=1, frames_with_kinect=1, frames_found=1)
        res["videos"]["V"]["joints"] = [{**j, "n": 1, "median_px": 30.0, "pck_0.1_visible": 0.9}
                                        for j in res["videos"]["V"]["joints"]]
        md = dg.table_md(res)
        self.assertIn("against YOLO26 x", md)
        self.assertIn("| nose | 0.6 | 0.5 | 0.1 |", md)


if __name__ == "__main__":
    unittest.main()
