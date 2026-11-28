"""Drop the band radio answers of the first frames task (2026-11-28) from saved labels; hands and gaze are kept. CPU.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.migrate_frame_labels --run RUN_DIR

Rewrites <data area>/validation/labels/<run>/frames/<user>.csv in the new format (validation_labels.FRAME_FIELDS) and keeps
the old file next to it as <user>.csv.v1. Prints only counts per annotator, no label value.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .. import validation_labels as L
from ..config import Config
from ..storage.files import run_source


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.migrate_frame_labels")
    p.add_argument("--run", type=Path, required=True)
    a = p.parse_args(argv)
    _, root = run_source(a.run, Config.load()["paths"])
    d = root.parent.parent / "validation" / "labels" / root.name / "frames"
    files = sorted(d.glob("*.csv"))
    if not files:
        print("no frames labels for this run")
        return 0
    for f in files:
        rows, n = L.drop_old_bands(L.read(f))
        if n:
            shutil.copy2(f, f.with_suffix(".csv.v1"))
            L._write(f, rows, L.FRAME_FIELDS)
        print(f"{f.stem}: {len(rows)} frames kept, {n} old band answers dropped (hands and gaze kept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
