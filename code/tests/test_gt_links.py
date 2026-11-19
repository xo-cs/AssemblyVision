"""TC-GTL: hand labels linked to a video by its SHA256 (Week 11; evaluation/gt_links.py). CPU."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from assemblyvision.config import Config
from assemblyvision.evaluation import gt_links as L
from test_report_export import make_run


class GtLinksTest(unittest.TestCase):
    def test_link_and_the_evaluation_card(self):
        from assemblyvision.ui.app import GradioApp
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            area = root / "public"
            cfg = Config.load(overrides={"paths": {"public_root": str(area), "qisens_root": str(root / "q")}})
            runs = []
            for rid, sha in (("run_20260101_000000_aaaa", "ab" * 32), ("run_20260102_000000_bbbb", "ab" * 32),
                             ("run_20260103_000000_cccc", "cd" * 32)):
                run = area / "runs" / rid
                run.mkdir(parents=True)
                make_run(run)
                m = json.loads((run / "manifest.json").read_text())
                m.update(run_id=rid, shots={"items": []})
                m["ingest"]["input_sha256"] = sha
                (run / "manifest.json").write_text(json.dumps(m))
                runs.append(run)
            gt = area / "annotations" / "gt.csv"
            gt.parent.mkdir(parents=True)
            with open(gt, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["step", "label", "start_s", "end_s", "source"])
                w.writerow([1, "A", 0.0, 9.0, "gt"])
                w.writerow([2, "B", 9.0, 20.0, "gt"])
            with self.assertRaises(ValueError):                       # labels outside the area are refused
                L.add(area, "ab" * 32, root / "elsewhere.csv")
            L.add(area, "ab" * 32, gt, "clip.mp4")
            L.add(area, "ab" * 32, gt, "clip.mp4")                    # again: replaced, not doubled
            self.assertEqual(len(L.load(area)), 1)
            self.assertEqual(L.gt_for_run(area, json.loads((runs[1] / "manifest.json").read_text())), gt)
            app = GradioApp(cfg, ["public"], repo_root=root)
            html = app.eval_html("run_20260102_000000_bbbb")          # no eval.json, found by the video's SHA256
            self.assertIn("linked to this video", html)
            self.assertIn("Boundary", html)
            self.assertIn("1.0 s", html)                              # the boundary found at 10 s, labelled at 9 s
            self.assertEqual(app.eval_html("run_20260103_000000_cccc"), "")   # no hand labels: no card


if __name__ == "__main__":
    unittest.main()
