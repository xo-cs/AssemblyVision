"""HA4M macro phases: the 12 actions grouped into 5 phases in task order (Week 10).

The grouping and its rationale are in experiments/datasets/ha4m_macro_phases.md.
HA4M (Cicirelli et al. 2022, Table 3) builds the Epicyclic Gear Train in three parts:
Block 1 (actions 1 to 4), Block 2 (5 to 8) and the final EGT (9 to 12). The final part
lasts about as long as the other two together, so it is split along its own sub tasks.

GT per video: every frame gets the macro phase of its action. Don't care frames
(action 0) between two frames of the same phase belong to it; between two different
phases they are split at the midpoint (an odd middle frame goes to the later phase); before the first and after the last action
they are "other" (intro and outro). A video whose actions leave the task order gives
more than 5 segments (a phase can appear twice); it is kept and counted.
"""
from __future__ import annotations

import numpy as np

from ..types import Step

MACRO_PHASES: list[tuple[str, tuple[int, ...]]] = [
    ("Block 1", (1, 2, 3, 4)),            # carrier, gear bearings x3, planet gears x3, carrier shaft
    ("Block 2", (5, 6, 7, 8)),            # sun shaft, sun gear, sun gear bearing, ring bear
    ("Join blocks", (9,)),                # pick up Block 2 and place it on Block 1
    ("Cover and screws", (10, 11)),       # cover, screws x2
    ("Fasten", (12,)),                    # Allen key, turn both screws, return key and EGT
]
NAMES = [n for n, _ in MACRO_PHASES]
OTHER = "other"
SOURCE = "ha4m_macro"
_OF_ACTION = {a: k for k, (_, acts) in enumerate(MACRO_PHASES) for a in acts}


def macro_index(actions: np.ndarray, of_action: dict | None = None) -> np.ndarray:
    """Per frame action ids (0 = don't care) -> macro phase index per frame, -1 = other.
    of_action: action id -> phase index (default: the HA4M grouping)."""
    a = np.asarray(actions, np.int64)
    m = _OF_ACTION if of_action is None else of_action
    k = np.array([m.get(int(x), -1) for x in a], np.int64)
    known = np.flatnonzero(k >= 0)
    if known.size == 0:
        return k
    out = np.full(len(k), -1, np.int64)
    out[known] = k[known]
    for left, right in zip(known[:-1], known[1:]):           # fill each gap of don't care frames
        if right - left <= 1:
            continue
        if k[left] == k[right]:
            out[left + 1:right] = k[left]
        else:
            mid = (left + right + 1) // 2
            out[left + 1:mid] = k[left]
            out[mid:right] = k[right]
    return out


def steps_from_index(idx: np.ndarray, fps: float, source: str = SOURCE, names: list[str] | None = None) -> list[Step]:
    """Contiguous segments of a per frame phase index as steps (other included)."""
    names = NAMES if names is None else names
    steps = []
    start = 0
    for f in range(1, len(idx) + 1):
        if f == len(idx) or idx[f] != idx[start]:
            label = OTHER if idx[start] < 0 else names[int(idx[start])]
            steps.append(Step(label, round(start / fps, 3), round(f / fps, 3), source))
            start = f
    return steps


def gt_steps(labels: np.ndarray, fps: float) -> list[Step]:
    """GT macro phase steps of one video from its [T,2] (action, repetition) labels."""
    return steps_from_index(macro_index(np.asarray(labels)[:, 0]), fps)


def in_task_order(steps: list[Step], names: list[str] | None = None) -> bool:
    """True if the phases (other left out) appear once each, in MACRO_PHASES order."""
    seq = [s.label for s in steps if s.label != OTHER]
    return seq == (NAMES if names is None else names)
