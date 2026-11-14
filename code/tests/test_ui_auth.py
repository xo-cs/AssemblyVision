"""TC-SEC: interface login, safe media paths and the audit log (Week 11; ui/auth.py). CPU."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from assemblyvision.config import Config
from assemblyvision.ui import auth as A
from test_report_export import make_run

RID = "run_20260101_000000_abcd"


class AuthTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.d = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_hash_and_users(self):
        h = A.hash_password("correct horse")
        self.assertTrue(h.startswith("scrypt$"))
        self.assertNotIn("correct horse", h)
        self.assertTrue(A.verify_password("correct horse", h))
        self.assertFalse(A.verify_password("wrong horse", h))
        self.assertNotEqual(h, A.hash_password("correct horse"))               # salted
        with mock.patch.object(A, "LOGIN_LOG", self.d / "login.jsonl"):
            u = A.Users(self.d / "users.json")
            u.add("asel", "long enough pw")
            self.assertEqual(oct(os.stat(u.path).st_mode & 0o777), "0o600")
            self.assertNotIn("long enough pw", u.path.read_text())
            self.assertTrue(u.check("asel", "long enough pw"))
            self.assertFalse(u.check("asel", "nope"))
            self.assertFalse(u.check("nobody", "long enough pw"))
            for bad in (("Bad Name", "long enough pw"), ("kazi", "short")):
                with self.assertRaises(ValueError):
                    u.add(*bad)
            log = [json.loads(x) for x in (self.d / "login.jsonl").read_text().splitlines()]
            self.assertEqual([x["ok"] for x in log], [True, False, False])
            self.assertTrue(all("pw" not in json.dumps(x) for x in log))       # no password in the log
            self.assertTrue(u.remove("asel"))
            self.assertEqual(u.names(), [])

    def test_safe_media(self):
        area = self.d / "public"
        (area / "runs").mkdir(parents=True)
        inside = area / "runs" / "a.pdf"
        inside.write_text("x")
        outside = self.d / "secret.pdf"
        outside.write_text("x")
        link = area / "runs" / "link.pdf"
        link.symlink_to(outside)
        roots = {"public": area}
        self.assertEqual(A.safe_media(inside, roots), str(inside))
        self.assertIsNone(A.safe_media(outside, roots))
        self.assertIsNone(A.safe_media(link, roots))                            # a link out of the area
        self.assertIsNone(A.safe_media(area / "runs" / ".." / ".." / "secret.pdf", roots))
        self.assertIsNone(A.safe_media(None, roots))

    def test_audit_goes_to_the_run_area(self):
        from assemblyvision.storage.runstore import RunStore
        cfg = Config.load(overrides={"paths": {"public_root": str(self.d / "public"),
                                               "qisens_root": str(self.d / "qisens")}})
        run = self.d / "public" / "runs" / RID
        run.mkdir(parents=True)
        make_run(run)
        log = A.AuditLog(RunStore(["public"], cfg))
        log("asel", "export_report", RID, "rev0")
        rec = json.loads((self.d / "public" / "audit.jsonl").read_text())
        self.assertEqual((rec["user"], rec["action"], rec["run_id"]), ("asel", "export_report", RID))
        self.assertFalse((self.d / "qisens").exists())

    def test_the_page_refuses_to_start_without_users(self):
        from assemblyvision.ui import app
        with mock.patch.object(A, "USERS_FILE", self.d / "none.json"), \
                mock.patch.object(app.GradioApp, "launch") as launch, \
                mock.patch.object(A.Users, "__init__", lambda s, path=self.d / "none.json": setattr(s, "path", path)):
            code = app.main(["--sources", "public", "--port", "7869"])
        self.assertEqual(code, 1)
        launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
