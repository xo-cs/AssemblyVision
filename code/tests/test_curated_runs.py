"""TC-CUR: the curated Results list (Week 11; ui/app.py curated_runs). CPU, synthetic runs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from assemblyvision.config import Config
from test_report_export import make_run


class CuratedRunsTest(unittest.TestCase):
    def test_selection(self):
        from assemblyvision.ui.app import GradioApp
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = Config.load(overrides={"paths": {"public_root": str(root / "public"), "qisens_root": str(root / "qisens")}})

            def run(source, rid, video, overlay=True, milestone=None, body3d=False):
                r = root / source / "runs" / rid
                r.mkdir(parents=True)
                make_run(r)
                m = json.loads((r / "manifest.json").read_text())
                m.update(run_id=rid, source=source, video=f"/x/{video}", shots={"items": []}, status="done")
                if milestone:
                    m["milestone"] = {"label": milestone}
                (r / "manifest.json").write_text(json.dumps(m))
                if overlay:
                    (r / "overlay").mkdir()
                    (r / "overlay" / "tracks_overlay.mp4").write_bytes(b"x")
                if body3d:
                    (r / "body3d").mkdir()
                    np.savez(r / "body3d" / f"{rid}-s001_p1.npz", frames=np.arange(3))
            run("qisens", "run_20260101_000000_aaaa", "q.mp4")                              # pinned
            run("qisens", "run_20260102_000000_bbbb", "q.mp4")                              # not pinned: hidden
            run("qisens", "run_20260103_000000_cccc", "q.mp4", milestone="Engine B v1")     # a milestone copy
            run("qisens", "run_20260104_000000_dddd", "q.mp4", overlay=False, milestone="Engine B v2")   # no overlay
            run("public", "run_20260105_000000_eeee", "a.mp4", body3d=True)                 # a.mp4 with 3D output
            run("public", "run_20260106_000000_ffff", "a.mp4")                              # newer, but no 3D
            run("public", "run_20260107_000000_abab", "b.mp4")
            run("public", "run_20260108_000000_cdcd", "b.mp4", body3d=True, milestone="4D test")   # a hidden test copy
            m = json.loads((root / "public/runs/run_20260108_000000_cdcd/manifest.json").read_text())
            m["milestone"]["hidden"] = True
            (root / "public/runs/run_20260108_000000_cdcd/manifest.json").write_text(json.dumps(m))
            fe = {"qisens": [{"id": "run_20260101_000000_aaaa", "label": "Flagship", "score": 93.2}], "public": {"max_videos": 10}}
            app = GradioApp(cfg, ["qisens", "public"], repo_root=root)
            with mock.patch.object(GradioApp, "featured", return_value=fe):
                ids = [x["run_id"] for x in app.curated_runs()]
                labels = {x["run_id"]: x["label"] for x in app.curated_runs()}
                self.assertEqual(set(ids), {"run_20260101_000000_aaaa", "run_20260103_000000_cccc",
                                            "run_20260105_000000_eeee", "run_20260107_000000_abab"})
                self.assertTrue(labels["run_20260101_000000_aaaa"].startswith("Flagship · 93.2 %"))
                self.assertEqual(ids[0], "run_20260101_000000_aaaa")
                self.assertIn("[3D]", labels["run_20260105_000000_eeee"])
                hidden = "run_20260106_000000_ffff"
                self.assertIn(hidden, [v for _, v in app.run_choices(hidden)])      # the shown run is always a choice
                self.assertNotIn(hidden, [v for _, v in app.run_choices()])
                fe["public"]["max_videos"] = 1
                self.assertEqual(sum(x["run_id"].endswith(("eeee", "abab")) for x in app.curated_runs()), 1)


    def test_a_cached_view_is_made_again_when_the_run_gets_new_output(self):
        import os
        import time
        from assemblyvision.ui.app import GradioApp
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = Config.load(overrides={"paths": {"public_root": str(root / "public"), "qisens_root": str(root / "qisens")}})
            r = root / "public" / "runs" / "run_20260101_000000_aaaa"
            r.mkdir(parents=True)
            make_run(r)
            m = json.loads((r / "manifest.json").read_text())
            m.update(run_id=r.name, source="public", video="/x/a.mp4", shots={"items": []}, status="done")
            (r / "manifest.json").write_text(json.dumps(m))
            app = GradioApp(cfg, ["public"], repo_root=root)
            calls = []
            make = lambda: calls.append(1) or len(calls)
            self.assertEqual((app.memo("x", r.name, make), app.memo("x", r.name, make)), (1, 1))      # cached
            (r / "scene4d").mkdir()                                                                  # new output
            self.assertEqual(app.memo("x", r.name, make), 2)
            later = time.time() + 5
            os.utime(r / "steps.csv", (later, later))                                                 # new steps
            self.assertEqual(app.memo("x", r.name, make), 3)


if __name__ == "__main__":
    unittest.main()
