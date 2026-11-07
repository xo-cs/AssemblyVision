"""Local image server for the labelling tasks (127.0.0.1 only).

Label Studio's tasks point here instead of holding the images, so Label Studio
never stores or exports a frame. Only PNG files of one folder (and its crops/
subfolder) are served, with headers that forbid caching; no directory listing,
no other file types, no path outside the folder. The log keeps counts only, no
file names.
"""
from __future__ import annotations

import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

NO_STORE = {"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0", "Pragma": "no-cache",
            "Expires": "0", "Content-Disposition": "inline", "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer"}
NAME_RE = re.compile(r"^/(?:crops/)?[A-Za-z0-9_.-]+\.png$")
ALLOWED_HOSTS = ("127.0.0.1", "localhost", "::1")


def make_handler(folder: Path, allow_origin: str, counts: dict):
    root = folder.resolve()

    class Handler(BaseHTTPRequestHandler):
        server_version = "av-frames"
        sys_version = ""

        def _headers(self, status: int, length: int = 0, ctype: str | None = None) -> None:
            self.send_response(status)
            for k, v in NO_STORE.items():
                self.send_header(k, v)
            if allow_origin:
                self.send_header("Access-Control-Allow-Origin", allow_origin)
                self.send_header("Vary", "Origin")
            if ctype:
                self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.end_headers()

        def _refuse(self, status: int) -> None:
            with lock:
                counts["refused"] += 1
            self._headers(status)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if not NAME_RE.fullmatch(path):
                return self._refuse(HTTPStatus.NOT_FOUND)
            target = (root / path.lstrip("/")).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                return self._refuse(HTTPStatus.NOT_FOUND)
            data = target.read_bytes()
            with lock:
                counts["served"] += 1
            self._headers(HTTPStatus.OK, len(data), "image/png")
            self.wfile.write(data)

        def do_HEAD(self):
            self._refuse(HTTPStatus.METHOD_NOT_ALLOWED)

        def do_POST(self):
            self._refuse(HTTPStatus.METHOD_NOT_ALLOWED)

        do_PUT = do_DELETE = do_POST

        def log_message(self, fmt, *args):     # no file names in any log
            pass

    lock = threading.Lock()
    return Handler


def serve(folder: Path, host: str = "127.0.0.1", port: int = 8081, allow_origin: str = "http://127.0.0.1:8080"):
    """(server, counts); call server.serve_forever(). Refuses any host but loopback."""
    if host not in ALLOWED_HOSTS:
        raise ValueError("the frame server binds to 127.0.0.1 only (reach it through the SSH tunnel)")
    if not Path(folder).is_dir():
        raise FileNotFoundError(folder)
    counts = {"served": 0, "refused": 0}
    srv = ThreadingHTTPServer((host, port), make_handler(Path(folder), allow_origin, counts))
    return srv, counts
