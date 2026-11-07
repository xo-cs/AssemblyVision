"""Serve the sampled frames and crops of one run to Label Studio (labelling/server.py).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.label_frames_server --run RUN_DIR [--port 8081]

Binds 127.0.0.1 only; reach it through the SSH tunnel together with Label
Studio. Serves only RUN_DIR/labelling/frames/*.png and crops/*.png, with
no-store headers. Stop it with Ctrl+C after the session; it prints counts only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..config import Config
from ..errors import AvError
from ..labelling.server import serve
from ..storage.files import REPO_ROOT, run_source


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.label_frames_server")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--label-studio", default="http://127.0.0.1:8080", help="origin allowed to load the images")
    args = p.parse_args(argv)
    try:
        _, root = run_source(args.run, Config.load()["paths"], REPO_ROOT)
        srv, counts = serve(root / "labelling" / "frames", port=args.port, allow_origin=args.label_studio)
    except (AvError, FileNotFoundError, ValueError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(f"serving the labelling frames on http://127.0.0.1:{args.port} (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        print(f"stopped: {counts['served']} images served, {counts['refused']} requests refused")
    return 0


if __name__ == "__main__":
    sys.exit(main())
