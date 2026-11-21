"""The validation freeze (Week 12, git tag validation-freeze-1). No GPU.

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.freeze --check [--models]   # is the working tree still the frozen method?
    python -m assemblyvision.tools.freeze --write              # writes the checksum list (only before the tag)

The tag fixes the code, the configs and the phase descriptions; docs_team/freeze/validation-freeze-1.json
holds their SHA256 and the SHA256 of the model weights. After the tag only new files and the user
interface may change: a changed or removed file of the method makes --check fail. Every validation
result records record() (the tag, its commit, the commit it ran on and this check).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

from ..storage.files import REPO_ROOT, utc_now

TAG = "validation-freeze-1"
MANIFEST = REPO_ROOT / "docs_team" / "freeze" / f"{TAG}.json"
METHOD = ("code/assemblyvision", "code/configs", "code/requirements-pinned.txt", "code/requirements-ci.lock",
          "docs_team/phase_descriptions.md", "docs_team/phase_descriptions_other.md",
          "docs_team/phase_descriptions_ikea_asm.md")
NOT_METHOD = ("code/assemblyvision/ui/",)         # the app's pages do not change a result
MODELS = ("yolo26x-pose.pt", "yolo26l-pose.pt", "engine_b/w10-engine-b-v1.pt", "engine_b/w10-engine-b-v2.pt",
          "Qwen3VL-8B-Instruct-Q4_K_M.gguf", "mmproj-Qwen3VL-8B-Instruct-F16.gguf", "siglip2",
          "scene4d/pi3/model.safetensors")


def _git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True, check=True).stdout.strip()


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def model_files() -> list:
    out = []
    for m in MODELS:
        p = REPO_ROOT / "models" / m
        out += sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
    return out


def write() -> dict:
    files = [REPO_ROOT / f for f in _git("ls-files", "code/configs", *METHOD[2:]).splitlines()]
    rec = {"tag": TAG, "written_utc": utc_now(),
           "files": {f.relative_to(REPO_ROOT).as_posix(): sha256(f) for f in files},
           "models": {f.relative_to(REPO_ROOT).as_posix(): sha256(f) for f in model_files()}}
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def changed() -> list[str]:
    """Files of the method that differ from the tag (committed or not); new files are allowed."""
    out = []
    for line in _git("diff", "--name-status", "--no-renames", TAG, "--", *METHOD).splitlines():
        st, name = line.split("\t", 1)
        if st != "A" and not name.startswith(NOT_METHOD):
            out.append(name)
    return out


def record() -> dict:
    """What a validation result stores about the code it ran on."""
    ch = changed()
    return {"tag": TAG, "tag_commit": _git("rev-parse", f"{TAG}^{{commit}}"), "ran_on": _git("rev-parse", "HEAD"),
            "method_unchanged": not ch, "changed": ch}


def check_models() -> list[str]:
    want = json.loads(MANIFEST.read_text())["models"]
    return [k for k, v in want.items() if not (REPO_ROOT / k).is_file() or sha256(REPO_ROOT / k) != v]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.freeze")
    p.add_argument("--check", action="store_true")
    p.add_argument("--models", action="store_true", help="with --check: also hash the weights (a few minutes)")
    p.add_argument("--write", action="store_true")
    a = p.parse_args(argv)
    if a.write:
        r = write()
        print(f"{MANIFEST.relative_to(REPO_ROOT)}: {len(r['files'])} files, {len(r['models'])} model files")
        return 0
    r = record()
    print(json.dumps(r, indent=1))
    bad = check_models() if a.models else []
    if a.models:
        print(f"model files that differ: {bad or 'none'}")
    return 0 if r["method_unchanged"] and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
