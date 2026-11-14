"""Login, safe media paths and the audit log of the interface (Week 11, TA review security).

- Users: a local file (default .config/ui_users.json, never in git) of user names with
  scrypt password hashes (hashlib, salt per user); tools/ui_users.py adds, lists and
  removes users. The page asks for a login (Gradio auth) whenever the file has a user;
  without one it does not start, unless --no-login is given (local tests only).
- Safe media: every file the page hands to the browser must resolve (symlinks too) to a
  path inside the data area of an open source (data/<source>/).
- Audit log: one JSON line per action (who, what, when, which run) in the data area of the
  run (data/<source>/audit.jsonl), so QiSens entries stay under data/qisens; logins go to
  logs/ui_audit.jsonl (user names and times only).
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path

from ..storage.files import REPO_ROOT, utc_now

USERS_FILE = REPO_ROOT / ".config" / "ui_users.json"
LOGIN_LOG = REPO_ROOT / "logs" / "ui_audit.jsonl"
NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,31}$")
N, R, P = 2 ** 14, 8, 1


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    h = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=N, r=R, p=P, dklen=32)
    return f"scrypt${N}${R}${P}${base64.b64encode(salt).decode()}${base64.b64encode(h).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        kind, n, r, p, salt, h = stored.split("$")
        if kind != "scrypt":
            return False
        got = hashlib.scrypt(password.encode("utf-8"), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p),
                             dklen=32)
        return hmac.compare_digest(got, base64.b64decode(h))
    except (ValueError, TypeError):
        return False


class Users:
    def __init__(self, path: Path = USERS_FILE):
        self.path = path

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, users: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(users, indent=1))
        os.chmod(tmp, 0o600)
        tmp.replace(self.path)

    def add(self, name: str, password: str) -> None:
        if not NAME_RE.fullmatch(name):
            raise ValueError("a user name is 2 to 32 lower case letters, digits, _ . or -, starting with a letter")
        if len(password) < 10:
            raise ValueError("a password needs at least 10 characters")
        users = self.load()
        users[name] = {"hash": hash_password(password), "created_utc": utc_now()}
        self._save(users)

    def remove(self, name: str) -> bool:
        users = self.load()
        if users.pop(name, None) is None:
            return False
        self._save(users)
        return True

    def names(self) -> list[str]:
        return sorted(self.load())

    def check(self, name: str, password: str) -> bool:
        """The Gradio auth function; every attempt goes to the login log."""
        u = self.load().get(name or "")
        ok = bool(u) and verify_password(password or "", u["hash"])
        append_jsonl(LOGIN_LOG, {"utc": utc_now(), "user": name if NAME_RE.fullmatch(name or "") else "?",
                                 "action": "login", "ok": ok})
        return ok


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")


class AuditLog:
    """Callable (user, action, run_id, detail): one line in the run's data area."""

    def __init__(self, store):
        self.store = store

    def path(self, source: str) -> Path:
        return self.store.roots[source] / "audit.jsonl"

    def __call__(self, user: str, action: str, run_id: str | None, detail: str = "", source: str | None = None) -> None:
        if source is None:
            source = self.store._locate(run_id)[0]
        append_jsonl(self.path(source), {"utc": utc_now(), "user": user, "action": action, "run_id": run_id,
                                         "detail": detail})


def safe_media(path: str | Path | None, roots: dict) -> str | None:
    """path if it resolves (symlinks too) inside one of the open data areas, else None."""
    if not path:
        return None
    real = Path(os.path.realpath(path))
    return str(path) if any(real.is_relative_to(Path(os.path.realpath(r))) for r in roots.values()) else None
