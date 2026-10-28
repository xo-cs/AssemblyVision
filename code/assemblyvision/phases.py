"""Phase vocabulary per data area.

QiSens runs use the five process phases of configs/steps.toml. Public clips
(motorbike assembly, track work, coupling, welding and so on) are not that
process, so Engine A gives them neutral labels "Phase 1" to "Phase N". Public
runs made before this rule still carry QiSens names in their steps.csv; the
interface shows them through public_label_map (display only, files unchanged).
"""
from __future__ import annotations

import dataclasses

OTHER = "other"

PUBLIC_NOTE = ("Public test clip. Used to test pose estimation and the pipeline. Phase labels are placeholders, "
               "so phase accuracy is not meaningful here.")


def neutral_label(k: int) -> str:
    return f"Phase {k}"


def phase_defs_for(step_defs: list[dict], source: str | None) -> list[dict]:
    """The step definitions Engine A uses for a run of this data area: QiSens
    (and no source, as in unit tests) keep steps.toml, public gets Phase 1..N
    with the same limits."""
    if source != "public":
        return list(step_defs)
    return [{**d, "name": neutral_label(i + 1)} for i, d in enumerate(step_defs)]


def public_label_map(labels: list[str]) -> dict[str, str]:
    """Stored label -> 'Phase k', k in order of first appearance; other stays other."""
    out: dict[str, str] = {}
    for label in labels:
        if label != OTHER and label not in out:
            out[label] = neutral_label(len(out) + 1)
    return out


def display_steps(steps: list | None, source: str | None) -> list | None:
    """Step objects with display labels: public runs get Phase 1..N (public_label_map)."""
    if not steps or source != "public":
        return steps
    names = public_label_map([s.label for s in steps])
    return [dataclasses.replace(s, label=names.get(s.label, s.label)) for s in steps]
