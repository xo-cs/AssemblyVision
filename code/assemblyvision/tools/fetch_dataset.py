"""Download public datasets for Engine B, selectively (Week 10; datasets/ha4m_fetch.py).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.fetch_dataset ha4m listing
    python -m assemblyvision.tools.fetch_dataset ha4m annotations [--videos all | IDU001V001,...]
    python -m assemblyvision.tools.fetch_dataset ha4m rgb --videos IDU001V001,... [--yes]
    python -m assemblyvision.tools.fetch_dataset ha4m sums

Everything is written under data/public/datasets/ha4m/ (refused anywhere else).
annotations reads Labels.txt and the Skeletons of each video from its remote zip
(about 10 MB per video); rgb streams the Color frames of the chosen videos into
one mp4 each. A command that would transfer more than datasets.max_transfer_gb
(5 GB) stops before the transfer unless --yes is given (only after approval).
Videos that are already complete are skipped, so a failed or stopped command is
resumed by running it again. Progress goes to logs/ha4m_download.log.
One stream only: requests are at least datasets.request_interval_s apart, HTTP
429/503 waits for Retry-After or backs off (datasets.backoff_start_s doubling up
to backoff_max_s, limited_retries times), and a lock file lets only one HA4M
download run at a time (annotations and RGB never together).
InHARD: access pending, nothing is downloaded.
Exit codes: 0 success, 1 a video failed, 2 wrong argument or refused.
"""
from __future__ import annotations

import argparse
import fcntl
import logging
import sys
import time
from pathlib import Path

from ..config import Config
from ..datasets import ha4m, ha4m_fetch as hf, remote
from ..errors import AvError
from ..storage.files import REPO_ROOT, data_area


def dataset_root(cfg: Config, name: str, repo_root: Path = REPO_ROOT) -> Path:
    root = repo_root / cfg["datasets"]["root"] / name
    if data_area(root, cfg["paths"], repo_root) != "public":
        raise AvError("AV-STO-005", path=str(root), detail="datasets are stored under data/public only")
    return root


def setup_log(repo_root: Path) -> logging.Logger:
    lg = logging.getLogger("assemblyvision.fetch_dataset")
    if not lg.handlers:
        (repo_root / "logs").mkdir(exist_ok=True)
        fh = logging.FileHandler(repo_root / "logs" / "ha4m_download.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(message)s"))
        lg.addHandler(fh)
        lg.addHandler(sh)
        lg.setLevel(logging.INFO)
    return lg


def pick_videos(arg: str, src: dict) -> list[str]:
    known = sorted(src["files"])
    if arg == "all":
        return known
    vids = [v.strip() for v in arg.split(",") if v.strip()]
    bad = [v for v in vids if v not in src["files"]]
    if bad:
        raise ValueError(f"unknown HA4M videos: {', '.join(bad)}")
    return vids


def run_many(videos: list[str], fn, lg: logging.Logger) -> list[str]:
    """fn(video) for each video, one after the other; returns the failed videos."""
    failed = []
    for v in videos:
        try:
            fn(v)
        except AvError as e:
            if e.code == "AV-DAT-002":
                raise
            lg.error("%s failed: %s", v, e)
            failed.append(v)
        except Exception as e:                # network errors: logged, the next run resumes
            lg.error("%s failed: %s: %s", v, type(e).__name__, e)
            failed.append(v)
    return failed


def take_lock(root: Path):
    """The dataset's download lock (one HA4M download at a time), or None if it is held."""
    root.mkdir(parents=True, exist_ok=True)
    f = open(root / ".download.lock", "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    return f


def configure_pacer(cfg: Config) -> None:
    d = cfg["datasets"]
    p = remote.DEFAULT_PACER
    p.min_interval_s, p.backoff_start_s = d["request_interval_s"], d["backoff_start_s"]
    p.backoff_max_s, p.limited_retries = d["backoff_max_s"], d["limited_retries"]


def main(argv: list[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.fetch_dataset")
    p.add_argument("dataset", choices=("ha4m", "inhard"))
    p.add_argument("what", choices=("listing", "annotations", "rgb", "sums"))
    p.add_argument("--videos", default="all", help="all or comma separated video ids")
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--yes", action="store_true", help="allow a transfer above datasets.max_transfer_gb (approved)")
    args = p.parse_args(argv)
    cfg = Config.load()
    if args.dataset == "inhard":
        print("InHARD: access is still pending; nothing is downloaded.", file=sys.stderr)
        return 2
    try:
        root = dataset_root(cfg, "ha4m", repo_root)
        lg = setup_log(repo_root)
        lock = take_lock(root)
        if lock is None:
            lg.error("another HA4M download is running (lock %s); start this one after it", root / ".download.lock")
            return 2
        configure_pacer(cfg)
        if args.what == "listing":
            files = hf.fetch_listing()
            path = hf.write_source(root, files)
            lg.info("HA4M %s listing: %d zips, %.2f TB -> %s", ha4m.VERSION, len(files),
                    sum(f["size"] for f in files) / 1e12, path)
            return 0
        if args.what == "sums":
            lg.info("wrote %s", hf.write_sha256sums(root))
            return 0
        src = hf.load_source(root)
        videos = pick_videos(args.videos, src)
        budget = hf.Budget(cfg["datasets"]["max_transfer_gb"], args.yes)
        t0 = time.monotonic()
        if args.what == "annotations":
            todo = [v for v in videos if not hf.read_meta(root, v).get("annotations", {}).get("done")]
            lg.info("annotations: %d videos to fetch (%d already done)", len(todo), len(videos) - len(todo))
            videos = todo

            def fn(v):
                m = hf.fetch_annotations(root, v, {"video": v, **src["files"][v]}, budget)
                a = m["annotations"]
                lg.info("%s annotations: %d frames, %d skeleton files, camera %s; %.1f MB in %d requests; total "
                        "%.2f GB, %.2f MB/s", v, m["n_frames"], m["n_skeleton_files"], m["camera"],
                        a["transfer_bytes"] / 1e6, a["requests"], budget.used / 1e9,
                        budget.used / 1e6 / max(time.monotonic() - t0, 1e-6))
        else:
            todo = [v for v in videos if not hf.read_meta(root, v).get("rgb", {}).get("done")]
            missing = [v for v in todo if not hf.read_meta(root, v).get("color_span_bytes")]
            if missing:
                lg.error("fetch the annotations of %s first (they record the Color size)", ", ".join(missing))
                return 2
            need = sum(hf.read_meta(root, v)["color_span_bytes"] for v in todo)
            lg.info("rgb: %d videos to fetch (%d already done), %.1f GB to transfer", len(todo),
                    len(videos) - len(todo), need / 1e9)
            budget.check(need)
            videos = todo

            def fn(v):
                m = hf.fetch_rgb(root, v, {"video": v, **src["files"][v]}, budget, crf=args.crf,
                                 progress=lg.info)
                lg.info("%s rgb done: %d frames (%d filled), %.1f GB in %.0f s", v, m["rgb"]["frames"],
                        m["rgb"]["filled"], m["rgb"]["transfer_bytes"] / 1e9, m["rgb"]["seconds"])
        failed = run_many(videos, fn, lg)
        hf.write_sha256sums(root)
        waits = remote.DEFAULT_PACER.waits
        lg.info("%s: %d done, %d failed%s; transferred %.2f GB at %.2f MB/s; %d waits after HTTP 429/503 (%.0f s)",
                args.what, len(videos) - len(failed), len(failed), f" ({', '.join(failed)})" if failed else "",
                budget.used / 1e9, budget.used / 1e6 / max(time.monotonic() - t0, 1e-6), len(waits),
                sum(w for _, w in waits))
        return 1 if failed else 0
    except AvError as e:
        print(f"{e.code}: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
