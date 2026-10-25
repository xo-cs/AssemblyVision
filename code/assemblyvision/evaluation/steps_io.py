"""Step label files.

GT and prediction format (seconds, end exclusive):
    step,label,start_s,end_s,source[,score]
Raw annotator format (m:ss or h:mm:ss or plain seconds, end second inclusive):
    step,label,start,end        "1:05,2:10" covers [65 s, 131 s)
Errors are AV-EVL-001 with the file name and the line.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..errors import AvError
from ..types import Step

GT_COLUMNS = ["step", "label", "start_s", "end_s", "source"]
RAW_COLUMNS = ["step", "label", "start", "end"]


def _bad(path: Path, detail: str) -> AvError:
    return AvError("AV-EVL-001", file=Path(path).name, detail=detail)


def _rows(path: Path, columns: list[str]) -> list[tuple[int, dict]]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            head = [h.strip() for h in (reader.fieldnames or [])]
            if head[:len(columns)] != columns:
                raise _bad(path, f"header is {','.join(head)}, expected {','.join(columns)}")
            reader.fieldnames = head
            return [(n, {k: (v or "").strip() for k, v in r.items() if k})
                    for n, r in enumerate(reader, start=2) if any((v or "").strip() for v in r.values() if v)]
    except OSError as e:
        raise _bad(path, str(e)) from e


def _check_order(path: Path, steps: list[Step]) -> list[Step]:
    steps = sorted(steps, key=lambda s: s.start_s)
    for a, b in zip(steps, steps[1:]):
        if b.start_s < a.end_s - 1e-6:
            raise _bad(path, f"steps {a.label!r} and {b.label!r} overlap")
    return steps


def read_steps_csv(path: Path) -> list[Step]:
    steps = []
    for n, r in _rows(path, GT_COLUMNS):
        try:
            a, b = float(r["start_s"]), float(r["end_s"])
        except ValueError as e:
            raise _bad(path, f"line {n}: start_s and end_s must be numbers") from e
        if not r["label"]:
            raise _bad(path, f"line {n}: empty label")
        if not b > a >= 0:
            raise _bad(path, f"line {n}: needs 0 <= start_s < end_s")
        score = r.get("score")
        steps.append(Step(r["label"], a, b, r["source"] or "gt", float(score) if score else None))
    return _check_order(path, steps)


def parse_clock(text: str) -> float:
    """'1:05' -> 65, '1:02:03' -> 3723, '12.5' -> 12.5."""
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3 or any(p.strip() == "" for p in parts):
        raise ValueError(text)
    vals = [float(p) for p in parts]
    if any(v < 0 for v in vals) or any(v >= 60 for v in vals[1:]):
        raise ValueError(text)
    t = 0.0
    for v in vals:
        t = t * 60 + v
    return t


def read_raw_csv(path: Path) -> list[Step]:
    steps = []
    for n, r in _rows(path, RAW_COLUMNS):
        try:
            a, b = parse_clock(r["start"]), parse_clock(r["end"]) + 1.0   # inclusive end second
        except ValueError as e:
            raise _bad(path, f"line {n}: times must be m:ss, h:mm:ss or seconds") from e
        if not r["label"]:
            raise _bad(path, f"line {n}: empty label")
        if not b > a:
            raise _bad(path, f"line {n}: end before start")
        steps.append(Step(r["label"], a, b, "annotator"))
    return _check_order(path, steps)
