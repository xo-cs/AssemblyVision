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


def _init(self, path, clock=None):
    """Users.__init__ with another default file (for main() in the tests)."""
    import threading
    import time
    self.path, self._clock, self._fails, self._lock = path, time.monotonic, {}, threading.Lock()


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
            self.assertTrue(all("long enough" not in json.dumps(x) for x in log))   # no password in the log
            self.assertTrue(u.remove("asel"))
            self.assertEqual(u.names(), [])

    def test_login_names_reasons_and_reset(self):
        """TC-SEC: the login of 2026-11-15: a user name with a capital or a space, a wrong password,
        and a new password that works at the next login without a restart."""
        with mock.patch.object(A, "LOGIN_LOG", self.d / "login.jsonl"):
            u = A.Users(self.d / "users.json")
            u.add("khurshid", "first password 1")
            self.assertTrue(u.check("Khurshid ", "first password 1"))       # trimmed, lower case
            self.assertFalse(u.check("khurshid", "first password 2"))
            self.assertFalse(u.check("kazi", "first password 1"))
            u.add("Khurshid", "second password 2")                          # passwd: a new hash, same user
            self.assertEqual(u.names(), ["khurshid"])
            self.assertTrue(u.check("khurshid", "second password 2"))       # the file is read at every login
            self.assertFalse(u.check("khurshid", "first password 1"))
            log = [json.loads(x) for x in (self.d / "login.jsonl").read_text().splitlines()]
            self.assertEqual([(x["user"], x["reason"]) for x in log],
                             [("khurshid", "ok"), ("khurshid", "wrong password"), ("kazi", "unknown user"),
                              ("khurshid", "ok"), ("khurshid", "wrong password")])
            self.assertTrue(all(x["users_file"] == str(u.path) for x in log))
            self.assertNotIn("password 1", (self.d / "login.jsonl").read_text())

    def test_lockout_after_five_wrong_passwords(self):
        now = [1000.0]
        with mock.patch.object(A, "LOGIN_LOG", self.d / "login.jsonl"):
            u = A.Users(self.d / "users.json", clock=lambda: now[0])
            u.add("khurshid", "right password 1")
            for _ in range(4):
                self.assertFalse(u.check("khurshid", "wrong"))
            self.assertTrue(u.check("khurshid", "right password 1"))       # a login resets the count
            for _ in range(5):
                self.assertFalse(u.check("khurshid", "wrong"))
            self.assertFalse(u.check("khurshid", "right password 1"))      # locked: even the right one
            now[0] += 299.0
            self.assertFalse(u.check("Khurshid", "right password 1"))      # the same user, still locked
            now[0] += 2.0                                                   # after 5 minutes
            self.assertTrue(u.check("khurshid", "right password 1"))
            for _ in range(5):
                u.check("nobody", "x")                                      # unknown names lock the same way
            self.assertEqual(u._fails["nobody"][0], 0)
            log = [json.loads(x) for x in (self.d / "login.jsonl").read_text().splitlines()]
            locks = [x for x in log if x["action"] == "lockout"]
            self.assertEqual([x["user"] for x in locks], ["khurshid", "nobody"])
            self.assertEqual([x["reason"] for x in log if x.get("reason") == "locked"], ["locked", "locked"])
            self.assertEqual((log[3]["fails_in_a_row"], log[4]["reason"]), (4, "ok"))

    def test_host_option(self):
        from assemblyvision.ui import app
        users = self.d / "users.json"
        with mock.patch.object(A.Users, "__init__", lambda s, path=users, clock=None: _init(s, path)), \
                mock.patch.object(app.GradioApp, "launch") as launch:
            self.assertEqual(app.main(["--host", "0.0.0.0", "--port", "7869"]), 1)          # no user: refused
            launch.assert_not_called()
            self.assertEqual(app.main(["--host", "0.0.0.0", "--port", "7869", "--no-login"]), 1)
            launch.assert_not_called()
            A.Users(users).add("mentor", "a long password")
            self.assertEqual(app.main(["--host", "0.0.0.0", "--port", "7869"]), 0)
            self.assertEqual(launch.call_args.kwargs, {"host": "0.0.0.0", "port": 7869})
            self.assertEqual(app.main(["--port", "7868"]), 0)                                # ui.host default
            self.assertEqual(launch.call_args.kwargs, {"host": "127.0.0.1", "port": 7868})
            self.assertEqual(app.main(["--port", "7868", "--no-login"]), 0)                  # localhost: allowed
        from assemblyvision.config import Config
        self.assertEqual((Config.load()["ui"]["host"], Config.load()["ui"]["port"]), ("127.0.0.1", 7860))

    def test_users_file_does_not_depend_on_the_working_directory(self):
        here = os.getcwd()
        try:
            os.chdir(self.d)
            self.assertEqual(A.Users().path, A.REPO_ROOT / ".config" / "ui_users.json")
            self.assertTrue(A.Users().path.is_absolute())
        finally:
            os.chdir(here)

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
