"""Index the existing runs of a data area into its SQLite Repository (Week 11; CPU).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.migrate_runs --source public      # data/public/av_dev.db
    python -m assemblyvision.tools.migrate_runs --source qisens      # data/qisens/av.db (Khurshid runs it)

Reads the run folders of the area (manifest.json, steps.csv, metrics.csv, eval.json,
export/revisions.json) and writes their rows into the database of the same area; the
run folders are not changed. Safe to run again: only runs whose files changed are
indexed again. Prints only counts.
"""
from __future__ import annotations

import argparse
import sys
import time

from ..config import Config
from ..storage.files import SOURCES


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.migrate_runs")
    p.add_argument("--source", choices=SOURCES, required=True)
    args = p.parse_args(argv)
    from ..storage.repository import Repository
    repo = Repository(args.source, Config.load()["paths"])
    t0 = time.monotonic()
    r = repo.migrate()
    counts = {t: repo.con().execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("runs", "steps", "metrics", "evaluations", "notices", "revisions")}
    print(f"migrate_runs {args.source}: {r['total']} run folders, {r['indexed']} indexed, {r['removed']} removed, "
          f"{len(r['failed'])} failed in {time.monotonic() - t0:.1f} s; rows {counts}")
    for f in r["failed"]:
        print(f"  failed: {f}", file=sys.stderr)
    return 1 if r["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
