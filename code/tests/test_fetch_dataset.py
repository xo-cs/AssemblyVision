"""Selective download from a remote zip (range requests), resume, checks and the 5 GB guard.
A zip built in memory is served through a fake opener; no network is used."""
import io
import json
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from assemblyvision.config import Config
from assemblyvision.datasets import ha4m_fetch as hf
import urllib.error
from email.message import Message

from assemblyvision.datasets import remote
from assemblyvision.datasets.remote import HttpRangeFile, Pacer, member_span, open_remote_zip, parse_retry_after
from assemblyvision.errors import AvError
from assemblyvision.tools import fetch_dataset
from helpers import fake_root
from test_datasets import skeleton_text

CAM = "000124702712"


def setUpModule():
    global _interval
    _interval = remote.DEFAULT_PACER.min_interval_s
    remote.DEFAULT_PACER.min_interval_s = 0.0            # no real pauses in the tests


def tearDownModule():
    remote.DEFAULT_PACER.min_interval_s = _interval
VID = "IDU001V001"


def png(i: int) -> bytes:
    img = np.full((48, 64, 3), (i * 40) % 255, np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def make_zip(frames=(0, 1, 3), labels=8) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr(f"{VID}/", b"")
        for f in frames:
            z.writestr(f"{VID}/Color/{CAM}/FrameID{f:06d}_DeviceTimeStamp{f}00us.png", png(f))
        z.writestr(f"{VID}/Depth/{CAM}/FrameID000000_DeviceTimeStamp0us.png", b"x" * 5000)
        z.writestr(f"{VID}/Labels.txt", "".join(f"{f} {1 if f > 3 else 0} 0\n" for f in range(labels)))
        for f in range(labels):
            z.writestr(f"{VID}/Skeletons/{CAM}/FrameID{f:06d}_DeviceTimeStamp{f}00us.txt", skeleton_text({1: 0.0}))
    return buf.getvalue()


class Server:
    """Fake opener: answers range requests from bytes, counts requests."""

    def __init__(self, data: bytes):
        self.data, self.requests = bytearray(data), 0

    def __call__(self, req, timeout):
        self.requests += 1
        a, b = req.get_header("Range").removeprefix("bytes=").split("-")
        a, b = int(a), min(int(b), len(self.data) - 1)
        body = bytes(self.data[a:b + 1])
        return SimpleNamespace(status=206, headers={"Content-Range": f"bytes {a}-{b}/{len(self.data)}"},
                               read=lambda: body)


class RemoteZipTest(unittest.TestCase):
    def test_range_file_reads_members(self):
        data = make_zip()
        srv = Server(data)
        zf, raw = open_remote_zip("http://x/z", opener=srv, buffer_size=256)
        self.assertEqual(raw.size, len(data))
        self.assertIn(f"{VID}/Labels.txt", zf.namelist())
        self.assertTrue(zf.read(f"{VID}/Labels.txt").startswith(b"0 0 0"))
        infos = [i for i in zf.infolist() if "/Skeletons/" in i.filename]
        lo, hi = member_span(zf, infos)
        self.assertLess(hi - lo, len(data))
        self.assertGreater(hi - lo, sum(i.compress_size for i in infos))

    def test_server_without_ranges_is_refused(self):
        with self.assertRaises(OSError):
            HttpRangeFile("http://x", opener=lambda r, t: SimpleNamespace(status=206, headers={}, read=lambda: b""))

    def test_retries_then_fails(self):
        calls = []

        def flaky(req, timeout):
            calls.append(1)
            raise OSError("reset")
        with self.assertRaises(OSError):
            HttpRangeFile("http://x", opener=flaky, retries=2, pacer=Pacer(min_interval_s=0, sleep=lambda s: None))
        self.assertEqual(len(calls), 3)


class FakeClock:
    def __init__(self):
        self.t, self.sleeps = 0.0, []

    def clock(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(round(s, 3))
        self.t += s


def http_429(retry_after=None):
    h = Message()
    if retry_after is not None:
        h["Retry-After"] = retry_after
    return urllib.error.HTTPError("http://x", 429, "Too Many Requests", h, None)


class RateLimitTest(unittest.TestCase):
    def pacer(self, **kw):
        c = FakeClock()
        return Pacer(sleep=c.sleep, clock=c.clock, **kw), c

    def test_retry_after_then_exponential_backoff(self):
        srv = Server(make_zip())
        errors = [http_429("7"), http_429(), http_429()]

        def opener(req, timeout):
            if errors:
                raise errors.pop(0)
            return srv(req, timeout)
        p, c = self.pacer(min_interval_s=0, backoff_start_s=60, backoff_max_s=1800)
        f = HttpRangeFile("http://x", opener=opener, pacer=p)
        self.assertEqual([w for _, w in p.waits], [7.0, 120.0, 240.0])      # Retry-After, then 60 x 2^1, 2^2
        self.assertEqual(f.stats["limited_waits"], 3)
        self.assertEqual(f.size, len(srv.data))

    def test_backoff_is_capped_and_then_fails(self):
        p, c = self.pacer(min_interval_s=0, backoff_start_s=60, backoff_max_s=1800, limited_retries=7)
        self.assertEqual([p.limited_wait(a, None) for a in range(7)], [60, 120, 240, 480, 960, 1800, 1800])

        def always(req, timeout):
            raise http_429()
        with self.assertRaises(urllib.error.HTTPError):
            HttpRangeFile("http://x", opener=always, pacer=p)
        self.assertEqual(len(p.waits), 7)                                     # 7 waits, then counted as failed

    def test_other_http_errors_are_not_retried(self):
        p, c = self.pacer(min_interval_s=0)

        def not_found(req, timeout):
            raise urllib.error.HTTPError("http://x", 404, "Not Found", Message(), None)
        with self.assertRaises(urllib.error.HTTPError):
            HttpRangeFile("http://x", opener=not_found, pacer=p)
        self.assertEqual(p.waits, [])

    def test_requests_are_paced(self):
        p, c = self.pacer(min_interval_s=3.0)
        f = HttpRangeFile("http://x", opener=Server(make_zip()), pacer=p)
        f.read_range(0, 10)
        f.read_range(10, 10)
        self.assertEqual(c.sleeps, [3.0, 3.0])                                # no wait before the first request

    def test_parse_retry_after(self):
        self.assertEqual(parse_retry_after("120"), 120.0)
        self.assertIsNone(parse_retry_after(None))
        self.assertIsNone(parse_retry_after("soon"))
        self.assertAlmostEqual(parse_retry_after("Sat, 07 Nov 2026 08:00:30 GMT", now=1794038400.0), 30.0)

    def test_one_download_at_a_time(self):
        tmp = fake_root()
        root = Path(tmp.name) / "ha4m"
        first = fetch_dataset.take_lock(root)
        self.assertIsNotNone(first)
        self.assertIsNone(fetch_dataset.take_lock(root))
        first.close()
        again = fetch_dataset.take_lock(root)
        self.assertIsNotNone(again)
        again.close()
        tmp.cleanup()


class FetchTest(unittest.TestCase):
    def setUp(self):
        self._tmp = fake_root()
        self.repo = Path(self._tmp.name)
        self.root = self.repo / "data/public/datasets/ha4m"
        self.data = make_zip()
        self.entry = {"video": VID, "file_id": "f1", "size": len(self.data), "md5": "0" * 32}

    def tearDown(self):
        self._tmp.cleanup()

    def budget(self, gb=5.0, yes=False):
        return hf.Budget(gb, yes)

    def test_annotations_are_stored_checked_and_resumed(self):
        srv = Server(self.data)
        m = hf.fetch_annotations(self.root, VID, self.entry, self.budget(), opener=srv)
        self.assertEqual((m["n_frames"], m["n_skeleton_files"], m["n_color_frames"], m["camera"]), (8, 8, 3, CAM))
        self.assertEqual(len(m["annotations"]["members"]), 9)                    # Labels.txt + 8 skeletons
        for name, digest in m["annotations"]["stored_sha256"].items():
            self.assertEqual(hf.sha256_path(self.root / name), digest)
        with zipfile.ZipFile(self.root / "raw" / f"{VID}_annot.zip") as z:
            self.assertEqual(len(z.namelist()), 9)
            self.assertEqual(z.read(f"{VID}/Labels.txt"), zipfile.ZipFile(io.BytesIO(self.data)).read(f"{VID}/Labels.txt"))
        with np.load(self.root / "skeleton" / f"{VID}.npz") as z:
            self.assertEqual(z["joints_2d_color"].shape, (8, 32, 2))
        before = srv.requests
        hf.fetch_annotations(self.root, VID, self.entry, self.budget(), opener=srv)   # done: no request
        self.assertEqual(srv.requests, before)
        sums = hf.write_sha256sums(self.root).read_text()
        self.assertIn(f"labels/{VID}.csv", sums)

    def test_crc_error_is_reported(self):
        data = bytearray(self.data)
        i = data.index(b"0 0 0\n1 0 0")
        data[i] = ord("9")                                                      # corrupt Labels.txt
        with self.assertRaises(AvError) as cm:
            hf.fetch_annotations(self.root, VID, {**self.entry, "size": len(data)}, self.budget(),
                                 opener=Server(bytes(data)))
        self.assertEqual(cm.exception.code, "AV-DAT-001")
        self.assertFalse((self.root / "meta" / f"{VID}.json").exists())

    def test_size_mismatch_is_reported(self):
        with self.assertRaises(AvError):
            hf.fetch_annotations(self.root, VID, {**self.entry, "size": 1}, self.budget(), opener=Server(self.data))

    def test_budget_needs_approval(self):
        b = self.budget(gb=1e-9)
        with self.assertRaises(AvError) as cm:
            b.check(10)
        self.assertEqual(cm.exception.code, "AV-DAT-002")
        self.budget(gb=1e-9, yes=True).check(10 ** 12)

    def test_rgb_mp4_fills_gaps_and_resumes(self):
        srv = Server(self.data)
        hf.fetch_annotations(self.root, VID, self.entry, self.budget(), opener=srv)
        m = hf.fetch_rgb(self.root, VID, self.entry, self.budget(), opener=srv)
        info = json.loads((self.root / "rgb" / f"{VID}.frames.json").read_text())
        self.assertEqual((info["n_frames"], info["filled"], info["first_frame_id"]), (4, [2], 0))
        cap = cv2.VideoCapture(str(self.root / "rgb" / f"{VID}.mp4"))
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 4)
        cap.release()
        self.assertEqual(m["rgb"]["frames"], 4)
        before = srv.requests
        hf.fetch_rgb(self.root, VID, self.entry, self.budget(), opener=srv)
        self.assertEqual(srv.requests, before)
        self.assertFalse(list((self.root / "rgb").glob("*.tmp*")))

    def test_dataset_root_must_be_public(self):
        cfg = Config.load(overrides={"datasets": {"root": "data/qisens/datasets"}})
        with self.assertRaises(AvError) as cm:
            fetch_dataset.dataset_root(cfg, "ha4m", self.repo)
        self.assertEqual(cm.exception.code, "AV-STO-005")
        self.assertEqual(fetch_dataset.dataset_root(Config.load(), "ha4m", self.repo), self.root)

    def test_listing_parse(self):
        body = json.dumps({"code": 20000, "data": [
            {"fileName": "IDU002V001.zip", "id": "b", "size": 5, "md5": "AB", "path": "/V2/IDU002V001.zip", "dir": False},
            {"fileName": "IDU001V001.zip", "id": "a", "size": 7, "md5": "CD", "path": "/V2/IDU001V001.zip", "dir": False}]})
        files = hf.fetch_listing(post=lambda url, b: body.encode())
        self.assertEqual([f["video"] for f in files], ["IDU001V001", "IDU002V001"])
        self.assertEqual(files[0]["md5"], "cd")
        src = json.loads(hf.write_source(self.root, files).read_text())
        self.assertEqual((src["version"], src["total_bytes"], src["files"]["IDU002V001"]["file_id"]), ("V2", 12, "b"))
        with self.assertRaises(AvError):
            hf.fetch_listing(post=lambda url, b: b'{"code": 1, "messageEn": "no"}')

    def test_inhard_downloads_nothing(self):
        self.assertEqual(fetch_dataset.main(["inhard", "listing"], repo_root=self.repo), 2)
        self.assertFalse((self.repo / "data").exists())


if __name__ == "__main__":
    unittest.main()
