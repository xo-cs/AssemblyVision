"""Write the split lists and the RGB subset of a dataset into experiments/datasets/ (tracked).

    source ~/AssemblyVision/av.sh && cd ~/AssemblyVision/code
    python -m assemblyvision.tools.dataset_splits ha4m subjects
    python -m assemblyvision.tools.dataset_splits ha4m splits
    python -m assemblyvision.tools.dataset_splits ha4m rgb-subset [--subjects-per-location 6] [--videos-per-subject 2]

subjects: the fixed cross subject split of all 41 subject IDs (from SOURCE.json,
location from the camera serial of each video), written once to
experiments/datasets/ha4m_subject_split.json; it never changes as videos arrive.
splits: cross subject (by that fixed split when it exists, else (seeded, datasets.cross_subject_test held out per location,
no subject in both)) and cross location (train on one lab, test on the other),
from the fetched labels. rgb-subset: subjects chosen seeded per location from
both folds of the cross subject split (in its train and test proportion), the
pilot videos first, then seeded videos per subject.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from ..config import Config
from ..datasets import splits
from ..datasets.ha4m import HA4MLoader
from ..storage.files import REPO_ROOT, utc_now

PILOT = ("IDU001V001", "IDU023V001")


def rgb_subset(split: dict, videos: list, per_location: int, per_subject: int, seed: int) -> dict:
    fold = next(f for f in split["folds"] if f["name"] == "cross_subject")
    test = set(fold["test_subjects"])
    rng = random.Random(seed)
    by_subject: dict[str, list[str]] = {}
    for v in videos:
        by_subject.setdefault(v.subject, []).append(v.video_id)
    pilot_subjects = {v.subject for v in videos if v.video_id in PILOT}
    chosen_subjects, picks = [], {}
    for loc, subs in sorted(split["locations"].items()):
        n_test = round(per_location * len([s for s in subs if s in test]) / len(subs))
        groups = {"test": [s for s in subs if s in test], "train": [s for s in subs if s not in test]}
        sel = []
        for name, k in (("test", n_test), ("train", per_location - n_test)):
            pool = sorted(groups[name])
            rng.shuffle(pool)
            pool.sort(key=lambda s: s not in pilot_subjects)              # pilot subjects stay in
            sel += pool[:k]
        chosen_subjects += sorted(sel)
    for s in chosen_subjects:
        vids = sorted(by_subject[s])
        first = [v for v in vids if v in PILOT]
        rest = [v for v in vids if v not in PILOT]
        rng.shuffle(rest)
        picks[s] = sorted((first + rest)[:per_subject])
    flat = sorted(v for vs in picks.values() for v in vs)
    return {"dataset": split["dataset"], "rule": f"{per_location} subjects per location in the cross subject "
            f"train/test proportion, {per_subject} videos each, seed {seed}; pilot {', '.join(PILOT)} included",
            "pilot": list(PILOT), "subjects": picks, "videos": flat, "n_videos": len(flat),
            "test_subjects_in_subset": sorted(s for s in picks if s in test), "written_utc": utc_now()}


def subject_locations(loader: HA4MLoader) -> dict[str, str]:
    """Every subject of the dataset listing (SOURCE.json files) with its location, the
    camera serial of its fetched videos (all of them must agree)."""
    from ..datasets.ha4m import video_subject
    from ..errors import AvError
    names = sorted(Path(f).stem for f in loader.source()["files"])
    out: dict[str, str] = {}
    for vid in names:
        sub = video_subject(vid)
        try:
            loc = loader.video_location(vid)
        except AvError:
            continue                                     # not fetched yet
        if out.setdefault(sub, loc) != loc:
            raise ValueError(f"subject {sub} has videos of two cameras")
    missing = sorted({video_subject(v) for v in names} - set(out))
    if missing:
        raise ValueError(f"no fetched video gives the location of {', '.join(missing)}")
    return out


def main(argv: list[str] | None = None, repo_root: Path = REPO_ROOT) -> int:
    p = argparse.ArgumentParser(prog="python -m assemblyvision.tools.dataset_splits")
    p.add_argument("dataset", choices=("ha4m",))
    p.add_argument("what", choices=("subjects", "splits", "rgb-subset"))
    p.add_argument("--subjects-per-location", type=int, default=6)
    p.add_argument("--videos-per-subject", type=int, default=2)
    args = p.parse_args(argv)
    cfg = Config.load()
    d = cfg["datasets"]
    out_dir = repo_root / "experiments" / "datasets"
    loader = HA4MLoader(cfg, repo_root=repo_root)
    fixed_path = out_dir / f"{args.dataset}_subject_split.json"
    if args.what == "subjects":
        subj = subject_locations(loader)
        if fixed_path.is_file():
            splits.check_subject_split(fixed_path, subj, d["cross_subject_test"], d["split_seed"])
            print(f"{fixed_path} exists and matches the rule; it is never rewritten")
            return 0
        rec = splits.subject_split(subj, d["cross_subject_test"], d["split_seed"])
        out_dir.mkdir(parents=True, exist_ok=True)
        fixed_path.write_text(json.dumps({"dataset": args.dataset, **rec, "written_utc": utc_now()}, indent=1) + "\n")
        print(f"{len(rec['train_subjects'])} train / {len(rec['test_subjects'])} test subjects -> {fixed_path}")
        return 0
    fixed = splits.check_subject_split(fixed_path, subject_locations(loader), d["cross_subject_test"],
                                       d["split_seed"]) if fixed_path.is_file() else None
    videos = loader.videos()
    if not videos:
        print("no fetched labels yet (fetch_dataset ha4m annotations)", file=sys.stderr)
        return 2
    s = splits.make_splits(args.dataset, videos, d["cross_subject_test"], d["split_seed"], fixed)
    if args.what == "splits":
        path = splits.write_splits(out_dir / f"{args.dataset}_splits.json", s)
        for f in s["folds"]:
            print(f"{f['name']}: train {len(f['train_videos'])} videos / {len(f['train_subjects'])} subjects, "
                  f"test {len(f['test_videos'])} / {len(f['test_subjects'])}")
        print(f"wrote {path}")
        return 0
    sub = rgb_subset(s, videos, args.subjects_per_location, args.videos_per_subject, d["split_seed"])
    path = out_dir / f"{args.dataset}_rgb_subset.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sub, indent=1) + "\n")
    print(f"{sub['n_videos']} videos of {len(sub['subjects'])} subjects ({len(sub['test_subjects_in_subset'])} "
          f"in the cross subject test fold) -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
