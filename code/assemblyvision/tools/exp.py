"""Experiment registry from the command line (Week 10; experiments/registry.py).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.exp new <id> --video PATH [--video PATH ...] [--from <id>]
        [--set SECTION.KEY=VALUE ...] [--repeats 3] [--reference <id>] [--gpu 0] [--variant x]
        [--gt GT_CSV] [--title TEXT] [--source public|qisens]
    python -m assemblyvision.tools.exp run <id> [--source qisens]
    python -m assemblyvision.tools.exp recheck <id> [--source qisens]
    python -m assemblyvision.tools.exp table <id> [<id> ...] [--source qisens] [--out FILE.md]
    python -m assemblyvision.tools.exp noise-floor [--from w10-ref-notf32]
    python -m assemblyvision.tools.exp gpu-check [--gpu 0]
    python -m assemblyvision.tools.exp index

new stores the full configuration (the --from experiment's, else the defaults,
then --set) in <area>/<id>/config.toml. run executes every repeat as a separate
run_pose process with the profiler on (one GPU job; the GPU must be free of
other processes) and writes results.json, table.md and, for public data,
experiments/INDEX.md. noise-floor stores the floor of the rule "equivalent
within numerical noise" (evaluation/equivalence.py NOISE_RULE) in
experiments/noise_floor.json, from the equivalence records of the strict FP32
experiment against the TF32 reference. recheck compares the stored runs again with the reference
experiment's current runs (CPU only). With --source qisens everything (config, results, temp
files, run folders) stays under data/qisens/; without it nothing there is opened.
Exit codes: 0 success, 1 error, 2 wrong argument, 3 GPU in use by another process.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ..config import Config, _read_toml, deep_merge
from ..errors import AvError
from ..experiments import registry as reg
from ..storage.files import REPO_ROOT
from .run_pose import _fail, parse_sets


def layered_sets(configs: list[Path], sets: list[str]) -> dict:
    """exp new overrides: the --config layers in order, then --set (later wins)."""
    return deep_merge(*[_read_toml(p) for p in configs], parse_sets(sets))


def _print_filled(exp) -> None:
    f = exp.meta.get("filled_at_load") or {}
    for key in f.get("keys", []):
        print(f"filled from the defaults at commit {f['commit'][:12]}: {key} (not in the stored config; a no-op default)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.exp")
    sub = p.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="store the configuration of a new experiment")
    n.add_argument("id")
    n.add_argument("--video", action="append", required=True)
    n.add_argument("--source", choices=("public", "qisens"), default="public")
    n.add_argument("--from", dest="base", default="", help="copy the settings of this experiment")
    n.add_argument("--config", action="append", default=[], type=Path, metavar="FILE",
                   help="a config layer (e.g. configs/reference.toml) applied before --set")
    n.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    n.add_argument("--repeats", type=int, default=3)
    n.add_argument("--reference", default="", help="experiment whose runs this one must be equivalent to")
    n.add_argument("--gpu", type=int, default=0)
    n.add_argument("--variant", default=None)
    n.add_argument("--gt", default="")
    n.add_argument("--title", default="")
    n.add_argument("--overwrite", action="store_true", help="replace an existing config.toml (results stay)")
    r = sub.add_parser("run", help="run an experiment")
    r.add_argument("id")
    r.add_argument("--source", choices=("public", "qisens"), default="public")
    rc = sub.add_parser("recheck", help="compare the stored runs again with the reference's current runs")
    rc.add_argument("id")
    rc.add_argument("--source", choices=("public", "qisens"), default="public")
    t = sub.add_parser("table", help="comparison table of experiments")
    t.add_argument("ids", nargs="+")
    t.add_argument("--source", choices=("public", "qisens"), default="public")
    t.add_argument("--out", type=Path, default=None, help="also write the table to this file (in the area)")
    t.add_argument("--summary", action="append", default=[], metavar="ID",
                   help="append the table.md of an approved summary (public area, e.g. w10-qisens)")
    nf = sub.add_parser("noise-floor", help="store the noise floor from the strict FP32 experiment")
    nf.add_argument("--from", dest="base", default="w10-ref-notf32")
    g = sub.add_parser("gpu-check", help="list processes on a GPU; exit 3 if there are any")
    g.add_argument("--gpu", type=int, default=0)
    sub.add_parser("index", help="rebuild experiments/INDEX.md")
    args = p.parse_args(argv)

    os.environ["XDG_CONFIG_HOME"] = str(REPO_ROOT / ".config")
    try:
        paths_cfg = Config.load()["paths"]
        if args.cmd == "new":
            base = None
            if args.base:
                try:
                    base = reg.load(args.base, args.source, paths_cfg, fill_new=True)
                except reg.ExperimentError as err:
                    if args.source != "qisens" or "no experiment" not in str(err):
                        raise
                    base = reg.load(args.base, "public", paths_cfg, fill_new=True)   # public settings for a QiSens run
            gt = Path(args.gt).absolute().relative_to(REPO_ROOT).as_posix() if args.gt else ""
            exp = reg.create(args.id, source=args.source, videos=[str(Path(v).absolute()) for v in args.video],
                             paths_cfg=paths_cfg, title=args.title, repeats=args.repeats, reference=args.reference,
                             gpu=args.gpu, variant=args.variant, gt=gt, base=base,
                             sets=layered_sets(args.config, args.set),
                             overwrite=args.overwrite)
            for key in exp.meta.get("filled_from_defaults", []):
                print(f"filled from the defaults at commit {exp.meta['filled_at_commit'][:12]}: {key} "
                      f"(added after {base.id} was stored; a no-op default)")
            print(f"wrote {exp.dir / 'config.toml'}")
            return 0
        if args.cmd == "run":
            exp = reg.load(args.id, args.source, paths_cfg, fill_new=True)
            _print_filled(exp)
            res = reg.run(exp, paths_cfg)
            print(reg.format_table(res))
            print(f"wrote {exp.dir / 'results.json'} and table.md")
            return 0
        if args.cmd == "recheck":
            exp = reg.load(args.id, args.source, paths_cfg, fill_new=True)
            _print_filled(exp)
            res = reg.recheck(exp, paths_cfg)
            for v, rec in res["videos"].items():
                print(f"{Path(v).name}: {reg.video_verdict(rec) or 'reference'}")
            return 0
        if args.cmd == "table":
            text = reg.noise_text(reg.area_dir("public", paths_cfg)) + \
                reg.compare_table([reg.load_results(i, args.source, paths_cfg) for i in args.ids]) + \
                "".join("\n" + reg.summary_table(i, paths_cfg) for i in args.summary)
            print(text)
            if args.out is not None:
                area = reg.area_dir(args.source, paths_cfg)
                out = args.out if args.out.is_absolute() else area / args.out
                if not out.resolve().is_relative_to(area.resolve()):
                    print(f"--out must lie in {area}", file=sys.stderr)
                    return 2
                out.write_text(text)
                print(f"wrote {out}")
            return 0
        if args.cmd == "noise-floor":
            print(f"wrote {reg.write_noise_floor(args.base, paths_cfg)}")
            return 0
        if args.cmd == "gpu-check":
            procs = reg.default_gpu_check(args.gpu)
            for pr in procs:
                print(f"GPU {args.gpu}: pid {pr['pid']} {pr['name']} {pr['used_mb']} MB")
            print(f"GPU {args.gpu}: {'in use by ' + str(len(procs)) + ' process(es)' if procs else 'free'}")
            return 3 if procs else 0
        if args.cmd == "index":
            print(f"wrote {reg.write_index(reg.area_dir('public', paths_cfg))}")
            return 0
    except AvError as e:
        return _fail(e)
    except (reg.ExperimentError, OSError) as e:
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
