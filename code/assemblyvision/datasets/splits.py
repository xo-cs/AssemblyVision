"""Subject and location splits (whole subjects or videos, never random frames).

The lists are written to experiments/datasets/<name>_splits.json (tracked), and
a loader checks a stored file against what the rule gives, so every result
names the exact videos it was trained and tested on.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from ..storage.files import utc_now
from .base import VideoRef


def subject_split(subject_location: dict[str, str], test_share: float, seed: int) -> dict:
    """Seeded split of a fixed list of subjects into train and test, stratified by
    location: in each location round(test_share * subjects) subjects go to test.
    Made from the dataset's subject IDs (not the fetched videos), so it never changes."""
    by_loc: dict[str, list[str]] = {}
    for sub, loc in sorted(subject_location.items()):
        by_loc.setdefault(loc, []).append(sub)
    rng = random.Random(seed)
    test: set[str] = set()
    for loc in sorted(by_loc):
        subs = sorted(by_loc[loc])
        rng.shuffle(subs)
        k = max(1, round(test_share * len(subs))) if len(subs) > 1 else 0
        test.update(subs[:k])
    return {"name": "cross_subject", "rule": f"seeded shuffle of the subjects per location, {test_share:.0%} to test",
            "seed": seed, "test_share": test_share, "subjects": len(subject_location),
            "locations": {loc: subs for loc, subs in sorted(by_loc.items())},
            "train_subjects": sorted(set(subject_location) - test), "test_subjects": sorted(test)}


def cross_subject(videos: list[VideoRef], test_share: float, seed: int, fixed: dict | None = None) -> dict:
    """Cross subject fold of the videos: by the fixed subject split if given (its test
    subjects), else a subject split of the subjects present in videos."""
    if fixed is None:
        fixed = subject_split({v.subject: v.location for v in videos}, test_share, seed)
    test = set(fixed["test_subjects"])
    return _fold("cross_subject", videos, lambda v: v.subject in test,
                 {"rule": fixed["rule"], "seed": fixed["seed"]})


def cross_location(videos: list[VideoRef]) -> list[dict]:
    """One fold per location: train on the others, test on it."""
    return [_fold(f"cross_location_test_{loc}", videos, lambda v, loc=loc: v.location == loc,
                  {"rule": f"train on every other location, test on {loc}"})
            for loc in sorted({v.location for v in videos})]


def _fold(name: str, videos: list[VideoRef], is_test, info: dict) -> dict:
    train = sorted(v.video_id for v in videos if not is_test(v))
    test = sorted(v.video_id for v in videos if is_test(v))
    subj = lambda ids: sorted({v.subject for v in videos if v.video_id in set(ids)})
    return {"name": name, **info, "train_videos": train, "test_videos": test,
            "train_subjects": subj(train), "test_subjects": subj(test)}


def check_disjoint(fold: dict) -> None:
    both = set(fold["train_subjects"]) & set(fold["test_subjects"])
    if both:
        raise ValueError(f"{fold['name']}: subjects in train and test: {sorted(both)}")
    if set(fold["train_videos"]) & set(fold["test_videos"]):
        raise ValueError(f"{fold['name']}: a video is in train and test")


def make_splits(dataset: str, videos: list[VideoRef], test_share: float, seed: int,
                fixed: dict | None = None) -> dict:
    folds = [cross_subject(videos, test_share, seed, fixed)] + cross_location(videos)
    for f in folds:
        check_disjoint(f)
    return {"dataset": dataset, "videos": len(videos), "subjects": len({v.subject for v in videos}),
            "locations": {loc: sorted({v.subject for v in videos if v.location == loc})
                          for loc in sorted({v.location for v in videos})},
            "folds": folds}


def write_splits(path: Path, splits: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**splits, "written_utc": utc_now()}, indent=1) + "\n")
    return path


def check_stored(path: Path, splits: dict) -> None:
    """The stored lists must equal the rule's (same videos, subjects, folds)."""
    stored = json.loads(path.read_text())
    stored.pop("written_utc", None)
    if stored != splits:
        raise ValueError(f"{path} does not match the split rule for the current videos; regenerate it")


def check_subject_split(path: Path, subject_location: dict[str, str], test_share: float, seed: int) -> dict:
    """The stored subject split, checked against the rule; ValueError if it differs."""
    stored = json.loads(path.read_text())
    want = subject_split(subject_location, test_share, seed)
    if {k: stored.get(k) for k in want} != want:
        raise ValueError(f"{path} does not match the subject split rule; it must never change")
    return stored
