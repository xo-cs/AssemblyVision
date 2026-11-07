"""Compare labelled frames with a run's DET v1 output (labelling/evaluate.py).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.eval_pose_labels --run RUN_DIR --boxes BOX_EXPORT.json \
        --keypoints KEYPOINT_EXPORT.json [--det-conf-min 0.25]

Reports detection precision and recall, OKS AP, AP50 and AP75 (pycocotools, and
our own implementation as a cross check), PCK@0.1 per joint, and the agreement
between labellers on the overlap set. Label Studio exports the labeller as a user
id; --labellers gives the names (the ids are on the Label Studio account page). Writes RUN_DIR/labelling/eval_pose.json
and eval_pose.md. Both exports must lie in the run's own data area.
Exit codes: 0 success, 1 error, 2 refused.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import Config
from ..errors import AvError
from ..labelling import evaluate, sampling, studio
from ..storage.files import REPO_ROOT, data_area, run_source


def main(argv: list[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.eval_pose_labels")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--boxes", type=Path, required=True)
    p.add_argument("--keypoints", type=Path, required=True)
    p.add_argument("--det-conf-min", type=float, default=0.25)
    p.add_argument("--labellers", default="", help="names for the Label Studio user ids, e.g. 1=khurshid,2=asel,3=kazi")
    args = p.parse_args(argv)
    cfg = Config.load()
    try:
        source, root = run_source(args.run, cfg["paths"], repo_root)
        for f in (args.boxes, args.keypoints):
            if data_area(f, cfg["paths"], repo_root) != source:
                raise AvError("AV-STO-005", path=str(f), detail=f"the export must lie in the {source} data area")
        manifest = json.loads((root / "manifest.json").read_text())
        names = dict(x.split("=", 1) for x in args.labellers.split(",") if "=" in x)
        boxes = studio.parse_boxes(studio.load_export(args.boxes), names)
        kps = studio.parse_keypoints(studio.load_export(args.keypoints), names)
        gt, agree = evaluate.ground_truth(boxes, kps)
        preds = evaluate.predictions(manifest, root, {g["frame_idx"] for g in gt.values()})
        res = evaluate.evaluate(gt, preds, agree, args.det_conf_min)
        res["run_id"] = manifest["run_id"]
        out = root / "labelling"
        sampling.write_json(out / "eval_pose.json", res)
        (out / "eval_pose.md").write_text(evaluate.report_md(res, manifest["run_id"]))
        d, ap = res["detection"], res["oks_ap"]
        print(f"precision {d['precision']}, recall {d['recall']}; OKS AP {ap['pycocotools']['AP']} (own "
              f"{ap['own']['AP']}), AP50 {ap['pycocotools']['AP50']}, AP75 {ap['pycocotools']['AP75']}; "
              f"PCK@0.1 {res['pck']['mean']}; labeller OKS {res['agreement']['keypoints']['mean_oks']}")
        print(f"wrote {out / 'eval_pose.json'} and eval_pose.md")
        return 0
    except AvError as e:
        print(f"{e.code}: {e}", file=sys.stderr)
        return 2 if e.code == "AV-STO-005" else 1


if __name__ == "__main__":
    sys.exit(main())
