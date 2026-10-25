"""Label file formats, the data area guard of eval_steps and the rerun stage list (CPU)."""
import unittest
from pathlib import Path

from assemblyvision.config import Config
from assemblyvision.errors import AvError
from assemblyvision.evaluation.steps_io import parse_clock, read_raw_csv, read_steps_csv
from assemblyvision.storage.files import data_area, run_source
from assemblyvision.tools.eval_steps import kappa_files
from assemblyvision.tools.run_pose import parse_stages
from helpers import fake_root, make_ctx


class FormatTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, text):
        p = self.dir / name
        p.write_text(text)
        return p

    def test_clock(self):
        self.assertEqual(parse_clock("1:05"), 65.0)
        self.assertEqual(parse_clock("1:02:03"), 3723.0)
        self.assertEqual(parse_clock("12.5"), 12.5)
        for bad in ("1:75", "", "a:00", "1::2"):
            with self.assertRaises(ValueError):
                parse_clock(bad)

    def test_raw_inclusive_end(self):
        p = self.write("a.csv", "step,label,start,end\n1,Removal,0:00,1:04\n2,Disassembly,1:05,2:10\n")
        steps = read_raw_csv(p)
        self.assertEqual([(s.start_s, s.end_s) for s in steps], [(0.0, 65.0), (65.0, 131.0)])

    def test_gt_format_errors(self):
        for text in ("label,start_s\nA,0\n", "step,label,start_s,end_s,source\n1,A,5,2,gt\n",
                     "step,label,start_s,end_s,source\n1,A,0,10,gt\n2,B,9,20,gt\n"):
            with self.assertRaises(AvError) as cm:
                read_steps_csv(self.write("gt.csv", text))
            self.assertEqual(cm.exception.code, "AV-EVL-001")
        ok = read_steps_csv(self.write("gt.csv", "step,label,start_s,end_s,source\n2,B,10,20,gt\n1,A,0,10,gt\n"))
        self.assertEqual([s.label for s in ok], ["A", "B"])

    def test_kappa_files(self):
        a = self.write("a.csv", "step,label,start,end\n1,A,0:00,0:09\n2,B,0:10,0:19\n")
        b = self.write("b.csv", "step,label,start,end\n1,A,0:00,0:09\n2,B,0:10,0:19\n")
        self.assertEqual(kappa_files(a, b, Config.load())["kappa"], 1.0)
        c = self.write("c.csv", "step,label,start,end\n1,A,0:00,0:14\n2,B,0:15,0:19\n")
        r = kappa_files(a, c, Config.load())
        # 20 s at 5 fps: 25 of 100 frames differ; po 0.75, pe 0.5*0.75 + 0.5*0.25 = 0.5, kappa 0.5
        self.assertEqual((r["frames"], r["po"], r["kappa"]), (100, 0.75, 0.5))


class GuardTest(unittest.TestCase):
    def test_areas_and_run_folder(self):
        tmp = fake_root()
        try:
            root = Path(tmp.name)
            cfg, ctx = make_ctx(root, source="qisens")
            ctx.write_manifest()
            p = cfg["paths"]
            self.assertEqual(data_area(ctx.paths.root / "x.csv", p, root), "qisens")
            self.assertEqual(data_area(root / p["public_root"] / "gt.csv", p, root), "public")
            self.assertIsNone(data_area(root / "elsewhere.csv", p, root))
            self.assertEqual(run_source(ctx.paths.root, p, root), ("qisens", ctx.paths.root.resolve()))
            with self.assertRaises(AvError) as cm:
                run_source(root / p["public_root"], p, root)
            self.assertEqual(cm.exception.code, "AV-STO-005")
        finally:
            tmp.cleanup()

    def test_stage_list(self):
        self.assertEqual(parse_stages("eval,engine,overlay"), ["segmentation", "overlay", "eval"])
        with self.assertRaises(ValueError):
            parse_stages("pose_body")


if __name__ == "__main__":
    unittest.main()
