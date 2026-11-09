"""Sensitivity of Engine A to the pose model on public clips (Week 10).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.pose_sensitivity [--ids w10-pose-sens-x,w10-pose-sens-l]

CPU only: reads the results.json of full run experiments that differ only in
pose.variant (stored settings are checked) and ran with a steps GT per clip, and
writes experiments/w10-pose-sensitivity/ (results.json, table.md) and INDEX.md:
per clip and variant the Engine A eval (frame accuracy, edit, F1, boundary MAE),
the phases found and the subject coverage of the first repeat, and the eval
change against the first id.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import Config
from ..experiments import registry as reg
from ..storage.files import REPO_ROOT, utc_now
from .run_diff import flat

EXP_ID = "w10-pose-sensitivity"
IDS = "w10-pose-sens-x,w10-pose-sens-l"
METRICS = ("frame_accuracy", "edit", "f1_10", "f1_25", "f1_50", "boundary_mae_s")


def settings_differ(a: reg.Experiment, b: reg.Experiment) -> list[str]:
    """Settings keys whose values differ between two experiments, besides pose.variant."""
    fa, fb = flat(a.settings), flat(b.settings)
    return sorted(k for k in set(fa) | set(fb) if k != "pose.variant" and fa.get(k) != fb.get(k))


def clip_row(rec: dict, repo_root: Path) -> dict:
    first = rec["repeats"][0]
    m = json.loads((repo_root / first["run_dir"] / "manifest.json").read_text())
    seg, tr = m.get("segmentation", {}), m.get("tracking", {})
    cov = [round(t.get("coverage", 0), 3) for t in tr.get("tracks", []) if t.get("is_subject")]
    ev = rec["summary"].get("eval") or {}
    return {"run": Path(first["run_dir"]).name, "eval": {k: ev.get(k) for k in METRICS},
            "gt_file": ev.get("gt_file"), "phases": f"{seg.get('phases_found')}/{seg.get('phases_expected')}",
            "subject_coverage": cov, "pose_fps": rec["summary"].get("pose_fps"),
            "eval_same_in_repeats": rec["summary"].get("eval_same_in_repeats", True)}


def build(ids: list[str], paths_cfg: dict, repo_root: Path = REPO_ROOT) -> dict:
    exps = [reg.load(i, "public", paths_cfg, repo_root, fill_new=True) for i in ids]
    diff = [k for e in exps[1:] for k in settings_differ(exps[0], e)]
    if diff:
        raise reg.ExperimentError(f"the experiments differ in more than pose.variant: {', '.join(sorted(set(diff)))}")
    if len({tuple(e.meta["videos"]) for e in exps}) != 1:
        raise reg.ExperimentError("the experiments must use the same clips")
    res = {"id": EXP_ID, "kind": "pose_sensitivity", "source": "public", "reference": ids[0],
           "title": "Engine A sensitivity to the pose model (YOLO26 " + " vs ".join(e.meta["variant"] for e in exps)
                    + "), public clips with steps GT", "experiments": ids, "clips": {}}
    for v in exps[0].meta["videos"]:
        rows = {}
        for e in exps:
            r = reg.load_results(e.id, "public", paths_cfg, repo_root)
            row = clip_row(r["videos"][v], repo_root)
            if not row["gt_file"]:
                raise reg.ExperimentError(f"{e.id}: {Path(v).name} has no eval (no GT stored)")
            rows[e.meta["variant"]] = row
        base = rows[exps[0].meta["variant"]]["eval"]
        for row in rows.values():
            row["change"] = {k: None if row["eval"][k] is None or base[k] is None else round(row["eval"][k] - base[k], 2)
                             for k in METRICS}
        res["clips"][Path(v).name] = rows
    res["finished_utc"] = utc_now()
    vs = [e.meta["variant"] for e in exps]
    res["index"] = {"cells": [f"{c}: frame acc " + ", ".join(f"{k} {r[k]['eval']['frame_accuracy']}" for k in vs)
                              for c, r in res["clips"].items()], "verdict": "measurement"}
    return res


def table_md(res: dict) -> str:
    lines = [f"# {res['id']}", "", res["title"], "",
             f"Experiments {', '.join(res['experiments'])}: the same stored settings except pose.variant "
             f"(configs/reference.toml, batch 1, strict FP32). Change: against {res['reference']}.", ""]
    for clip, rows in res["clips"].items():
        lines += [f"## {clip} (GT {next(iter(rows.values()))['gt_file']})", "",
                  "| pose | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s | phases | subject coverage | pose fps | run |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for var, r in rows.items():
            e, c = r["eval"], r["change"]
            cell = lambda k: f"{e[k]}" + (f" ({c[k]:+})" if c[k] else "")
            lines.append(f"| YOLO26 {var} | " + " | ".join(cell(k) for k in METRICS)
                         + f" | {r['phases']} | {r['subject_coverage']} | {r['pose_fps']} | {r['run']} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.pose_sensitivity")
    p.add_argument("--ids", default=IDS)
    args = p.parse_args(argv)
    pc = Config.load()["paths"]
    try:
        res = build([i.strip() for i in args.ids.split(",") if i.strip()], pc)
    except reg.ExperimentError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    d = reg.area_dir("public", pc) / EXP_ID
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n")
    (d / "table.md").write_text(table_md(res))
    reg.write_index(reg.area_dir("public", pc))
    print(table_md(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
