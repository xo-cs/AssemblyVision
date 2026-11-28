"""Delete every label file of one account in every data area (Week 12). Prints only counts, never a label value.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.purge_labels NAME

Removes <area>/validation/labels/*/*/NAME.csv, their .v1 backups and <area>/videos/*/annotations/NAME.csv. Remove the
account itself with tools/ui_users.py remove NAME.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.purge_labels")
    p.add_argument("name")
    a = p.parse_args(argv)
    from ..config import Config
    from ..storage.files import SOURCES, source_root
    from ..ui.auth import NAME_RE, norm_name
    name = norm_name(a.name)
    if not NAME_RE.fullmatch(name):
        print("not a valid user name", file=sys.stderr)
        return 1
    paths = Config.load()["paths"]
    for src in SOURCES:
        area = source_root(src, paths)
        files = [f for pat in (f"validation/labels/*/*/{name}.csv", f"validation/labels/*/*/{name}.csv.v1",
                               f"videos/*/annotations/{name}.csv") for f in area.glob(pat)]
        for f in files:
            f.unlink()
        print(f"{src}: {len(files)} label files of {name} deleted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
