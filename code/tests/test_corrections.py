"""TC-COR: boundary correction of a run as a new revision (Week 11; output/corrections.py). CPU."""
import json
import tempfile
import unittest
from pathlib import Path

from test_report_export import make_run


class CorrectionTest(unittest.TestCase):
    def test_move_refuse_and_new_revision(self):
        from unittest import mock
        from assemblyvision.output import corrections as C
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_run(root)
            steps, rev = C.latest_steps(root)
            self.assertIsNone(rev)
            self.assertEqual([b["time_s"] for b in C.boundaries(steps)], [10.0])
            new = C.apply(steps, {1: 12.5})
            self.assertEqual((new[0]["end_s"], new[1]["start_s"], new[1]["source"]), (12.5, 12.5, "corrected"))
            self.assertEqual(steps[0]["end_s"], 10.0)                              # the input is not changed
            for bad in ({1: 0.5}, {1: 19.5}, {2: 15.0}):
                with self.assertRaises(ValueError):
                    C.apply(steps, bad)
            original = (root / "steps.csv").read_bytes()
            fake = b"segment_id,step,label,start_s,end_s,metric,value,unit,coverage,flags\n"
            with mock.patch.object(C, "metrics_csv", return_value=(fake, [])):
                out = C.approve(root, new, "khurshid", "moved")
            self.assertEqual(out.name, "rev1")
            self.assertTrue((root / "export" / "rev0" / "report.pdf").is_file())   # the original is rev0
            self.assertEqual((root / "steps.csv").read_bytes(), original)          # the run files stay
            rep = json.loads((out / "report.json").read_text())
            self.assertEqual((rep["revision"]["based_on"], rep["revision"]["by"]), (0, "khurshid"))
            self.assertIsNone(rep["eval"])                                         # the eval belongs to rev0
            self.assertEqual(C.latest_steps(root)[1], 1)
            self.assertEqual(json.loads((root / "export" / "revisions.json").read_text())[0]["revision"], 1)



if __name__ == "__main__":
    unittest.main()
