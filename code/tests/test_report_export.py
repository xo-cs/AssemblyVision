"""TC-REP: report export of a run (Week 11; output/report.py). CPU, synthetic run folder."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from assemblyvision.config import Config
from assemblyvision.output import report as R


def make_run(root: Path, zero_shot: bool = True) -> None:
    cfg = Config.load(overrides={"steps": {"use": True, "step": [{"name": "A"}, {"name": "B"}]}}).snapshot()
    if not zero_shot:
        cfg["seg_a"].update(siglip_weight=0.0, siglip_crop_weight=0.0, vlm_weight=0.0)
    manifest = {"run_id": "run_20260101_000000_abcd", "source": "public", "video": "/x/clip.mp4", "variant": "x",
                "gpu": 0, "status": "done", "created_utc": "2026-01-01T00:00:00Z", "finished_utc": "2026-01-01T00:05:00Z",
                "config": cfg, "versions": {"torch": "t"}, "provenance": {"git": {"commit": "abc"}, "gpu_name": "G"},
                "ingest": {"cfr": {"fps": 10.0, "frame_count": 200, "duration_s": 20.0}},
                "segmentation": {"vocabulary": ["A", "B"]},
                "notices": [{"stage": "segmentation", "code": "AV-SEG-006", "text": "fallback", "detail": "d"}]}
    (root / "manifest.json").write_text(json.dumps(manifest))
    with open(root / "steps.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "label", "start_s", "end_s", "source", "score"])
        w.writerow([1, "A", 0.0, 10.0, "engine_a", 0.5])
        w.writerow([2, "B", 10.0, 20.0, "engine_a", 0.5])
    with open(root / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(R.METRIC_COLS)
        w.writerow(["s-1", 1, "A", 0.0, 10.0, "duration_s", 10, "s", 1.0, ""])
    t = np.arange(40, dtype=np.float32) / 2
    cos = np.zeros((40, 2), np.float32)
    cos[:20, 0], cos[20:, 1] = 0.1, 0.1                          # A for 10 s, then B
    emb = np.zeros((40, 4), np.float16)
    np.savez(root / "siglip.npz", time_s=t, full_cos=cos, crop_cos=cos, full_emb=emb, crop_emb=emb,
             crop_ok=np.ones(40, bool), meta=json.dumps({"fps": 2.0}))
    probs = np.zeros((10, 2), np.float32)
    probs[:5, 0], probs[5:, 1] = 0.9, 0.9
    probs[:5, 1], probs[5:, 0] = 0.1, 0.1
    np.savez(root / "vlm.npz", time_s=np.arange(10, dtype=np.float32) * 2, probs=probs, meta=json.dumps({"window_s": 2.0}))


class ReportExportTest(unittest.TestCase):
    def test_export_and_revisions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_run(root)
            out = R.export_run(root, 0)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["metrics.csv", "report.json", "report.pdf", "steps.csv"])
            rep = json.loads((out / "report.json").read_text())
            self.assertEqual([s["label"] for s in rep["steps"]], ["A", "B"])
            self.assertGreater(rep["steps"][0]["phase_confidence"], 0.95)       # the scores agree with the steps
            self.assertEqual(rep["timestamps"]["run_finished_utc"], "2026-01-01T00:05:00Z")
            self.assertIn("exported_utc", rep["timestamps"])
            self.assertEqual(rep["notices"][0]["code"], "AV-SEG-006")
            self.assertEqual(rep["provenance"]["run_commit"], "abc")
            self.assertEqual(rep["provenance"]["gpu"]["name"], "G")
            self.assertEqual({m["role"] for m in rep["provenance"]["models"]}, {"pose", "engine_b", "siglip", "vlm",
                                                                                "vlm_mmproj"})
            self.assertEqual(len(rep["provenance"]["config_sha256"]), 64)
            self.assertTrue((out / "report.pdf").read_bytes().startswith(b"%PDF"))
            with self.assertRaises(FileExistsError):                             # never overwritten
                R.export_run(root, 0)
            self.assertEqual(R.next_revision(root), 1)

    def test_no_zero_shot_scores_gives_no_confidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_run(root, zero_shot=False)
            rep = R.build(root)
            self.assertEqual([s["phase_confidence"] for s in rep["steps"]], [None, None])
            self.assertEqual({m["role"] for m in rep["provenance"]["models"]}, {"pose", "engine_b"})


if __name__ == "__main__":
    unittest.main()
