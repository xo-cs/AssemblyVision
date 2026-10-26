"""Stand in for tools/run_pose as the worker of RunManager tests: takes the
same arguments, walks the stages of the queued run with progress and
finishes. The data root comes from AV_TEST_ROOT."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

p = argparse.ArgumentParser()
for a in ("--source", "--video", "--variant", "--gpu", "--run-id"):
    p.add_argument(a)
p.add_argument("--fail", action="store_true")
args = p.parse_args()
manifest = Path(os.environ["AV_TEST_ROOT"]) / "data" / args.source / "runs" / args.run_id / "manifest.json"
m = json.loads(manifest.read_text())
assert m["status"] == "queued", m["status"]
m["status"] = "running"
m["argv"] = sys.argv[1:]


def write():
    tmp = manifest.with_name("manifest.json.tmp")
    tmp.write_text(json.dumps(m))
    os.replace(tmp, manifest)


for stage in ("ingest", "pose_body", "segmentation"):
    for frac in (0.0, 0.5):
        m["stages"][stage] = {"status": "running", "progress": frac}
        write()
        time.sleep(float(os.environ.get("AV_TEST_STEP_S", "0.05")))
    m["stages"][stage] = {"status": "done", "progress": 1.0, "seconds": 0.1}
m["status"] = "done"
m["summary"] = f"{args.run_id} done: fake worker"
write()
