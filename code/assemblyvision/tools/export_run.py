"""Report export of one run (Week 11; output/report.py). CPU only.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.export_run --run RUN_DIR

Writes RUN_DIR/export/rev0/ (report.pdf, report.json, steps.csv, metrics.csv) from the
stored files of the run; a QiSens run's export stays under data/qisens. Revisions are
never overwritten: rev0 is the run as computed, later revisions come from the boundary
correction in the interface. Prints only the output folder and counts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import Config
from ..storage.files import run_source


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.export_run")
    p.add_argument("--run", type=Path, required=True)
    args = p.parse_args(argv)
    from ..output.report import export_run
    _, root = run_source(args.run, Config.load()["paths"])
    if (root / "export" / "rev0").exists():
        print(f"{root / 'export' / 'rev0'} exists; exports are never overwritten", file=sys.stderr)
        return 1
    out = export_run(root, 0)
    print(f"exported {out}: " + ", ".join(sorted(f.name for f in out.iterdir())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
