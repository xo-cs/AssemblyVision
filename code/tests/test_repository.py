"""TC-REPO: the SQLite Repository of a data area (Week 11; storage/repository.py). CPU."""
import json
import tempfile
import unittest
from pathlib import Path

from assemblyvision.config import Config
from assemblyvision.storage.repository import Repository
from assemblyvision.storage.runstore import RunStore
from test_report_export import make_run

RID = "run_20260101_000000_abcd"


class RepositoryTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.cfg = Config.load(overrides={"paths": {"public_root": str(self.root / "public"),
                                                    "qisens_root": str(self.root / "qisens")}})
        self.run = self.root / "public" / "runs" / RID
        self.run.mkdir(parents=True)
        make_run(self.run)
        (self.run / "eval.json").write_text(json.dumps({"gt_file": "gt.csv", "metrics": {"frame_accuracy": 90.0}}))

    def tearDown(self):
        self._tmp.cleanup()

    def test_migrate_index_and_sync(self):
        repo = Repository("public", self.cfg["paths"])
        self.assertEqual(repo.db_path, self.root / "public" / "av_dev.db")      # inside its own data area
        r = repo.migrate()
        self.assertEqual((r["total"], r["indexed"], r["failed"]), (1, 1, []))
        self.assertEqual(repo.migrate()["indexed"], 0)                          # unchanged runs are not indexed again
        self.assertEqual([s["label"] for s in repo.get_steps(RID)], ["A", "B"])
        self.assertEqual(len(repo.get_metrics(RID)), 1)
        self.assertEqual(repo.get_eval(RID)["metrics"]["frame_accuracy"], 90.0)
        self.assertEqual(repo.get_notices(RID)[0]["code"], "AV-SEG-006")
        with open(self.run / "steps.csv", "a") as f:                            # a rerun changes the files
            f.write("3,B,20.0,25.0,engine_a,0.5\n")
        self.assertEqual(repo.sync(RID)["indexed"], 1)
        self.assertEqual(len(repo.get_steps(RID)), 3)
        import shutil
        shutil.rmtree(self.run)
        self.assertEqual(repo.sync()["removed"], 1)
        self.assertEqual(repo.list_runs(), [])

    def test_runstore_reads_through_the_repository(self):
        st = RunStore(["public"], self.cfg)
        self.assertEqual([r["run_id"] for r in st.list_runs()], [RID])
        self.assertTrue((self.root / "public" / "av_dev.db").is_file())
        self.assertFalse((self.root / "qisens" / "av.db").exists())             # a source not listed is never opened
        self.assertEqual([s["label"] for s in st.get_segments(RID)], ["Phase 1", "Phase 2"])   # public display names
        self.assertEqual(st.get_metrics(RID)[0]["value"], 10.0)
        self.assertEqual(st.get_eval(RID)["metrics"]["frame_accuracy"], 90.0)

    def test_falls_back_to_files_when_the_database_fails(self):
        st = RunStore(["public"], self.cfg)
        (self.root / "public" / "av_dev.db").mkdir()                            # not a database file
        self.assertEqual([r["run_id"] for r in st.list_runs()], [RID])
        self.assertEqual(len(st.get_segments(RID)), 2)


if __name__ == "__main__":
    unittest.main()
