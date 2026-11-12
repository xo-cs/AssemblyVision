"""IKEA ASM macro phases: the 33 atomic actions grouped into phases per furniture type
(Week 11). The grouping and its rationale are in experiments/datasets/ikea_asm_macro_phases.md.

The GT rules are the HA4M ones (ha4m_phases.macro_index): NA and "other" frames are
don't care; between two frames of the same phase they belong to it, between two
phases they are split at the midpoint, before the first and after the last action they
are "other". Annotation frames are video frames (25 fps).
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from ..types import Step
from . import ha4m_phases as hp

SOURCE = "ikea_macro"
FPS = 25.0
LEGS = ("pick up leg", "align leg screw with table thread", "spin leg", "tighten leg", "lay down leg", "rotate table",
        "flip table top", "pick up table top", "lay down table top", "push table top")
FLIP = ("flip table", "push table")
SHELF = ("pick up shelf", "attach shelf to table", "flip shelf", "lay down shelf")
MACRO_PHASES: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "Lack_TV_Bench": [("Attach legs", LEGS), ("Flip table", FLIP), ("Attach shelf", SHELF)],
    "Lack_Coffee_Table": [("Attach legs", LEGS), ("Flip table", FLIP), ("Attach shelf", SHELF)],
    "Lack_Side_Table": [("Attach legs", LEGS), ("Flip table", FLIP)],
    "Kallax_Shelf_Drawer": [
        ("Side panels", ("pick up front panel", "lay down front panel", "pick up side panel", "lay down side panel",
                         "align side panel holes with front panel dowels", "attach drawer side panel")),
        ("Bottom panel", ("pick up bottom panel", "lay down bottom panel", "slide bottom of drawer")),
        ("Back panel", ("pick up back panel", "lay down back panel", "attach drawer back panel")),
        ("Pins and finish", ("pick up pin", "insert drawer pin", "position the drawer right side up")),
    ],
}
DONT_CARE = ("NA", "other")


def names(furniture: str) -> list[str]:
    return [n for n, _ in MACRO_PHASES[furniture]]


def frame_labels(segments: list[dict], n_frames: int) -> list[str]:
    """Per frame action label from gt_segments.json segments (inclusive frame ranges)."""
    lab = ["NA"] * n_frames
    for s in segments:
        a, b = int(s["segment"][0]), int(s["segment"][1])
        for f in range(max(a, 0), min(b + 1, n_frames)):
            lab[f] = s["label"]
    return lab


def phase_index(furniture: str, labels: list[str]) -> np.ndarray:
    groups = MACRO_PHASES[furniture]
    ids = {lab: k + 1 for k, (_, labs) in enumerate(groups) for lab in labs}      # 0 = don't care
    acts = np.array([ids.get(x, 0) for x in labels], np.int64)
    unknown = {x for x in labels if x not in ids and x not in DONT_CARE}
    if unknown:
        raise ValueError(f"{furniture}: actions without a phase: {sorted(unknown)}")
    return hp.macro_index(acts, {k + 1: k for k in range(len(groups))})


def gt_steps(furniture: str, segments: list[dict], n_frames: int, fps: float = FPS) -> list[Step]:
    return hp.steps_from_index(phase_index(furniture, frame_labels(segments, n_frames)), fps, SOURCE, names(furniture))


def in_task_order(furniture: str, steps: list[Step]) -> bool:
    return hp.in_task_order(steps, names(furniture))


def load_annotations(root: Path) -> dict:
    """{video id "<furniture>/<recording>": {"subset", "annotation"}} from raw/action_annotations.zip."""
    with zipfile.ZipFile(root / "raw/action_annotations.zip") as z:
        return json.loads(z.read("gt_segments.json"))["database"]
