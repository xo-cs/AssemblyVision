"""Phase descriptions for zero shot phase scoring (Week 11): parses
docs_team/phase_descriptions.md (QiSens) and docs_team/phase_descriptions_ikea_asm.md.

Format: optional "# Furniture: <type>" sections; one "## <phase>" heading per phase in
task order; a "Distinguishes: ..." line; three "> " lines (the short version for the VLM
prompt); "- " lines (one visual description each, the SigLIP prompt ensemble).
"""
from __future__ import annotations

from pathlib import Path

from ..storage.files import REPO_ROOT

QISENS = REPO_ROOT / "docs_team/phase_descriptions.md"
IKEA = REPO_ROOT / "docs_team/phase_descriptions_ikea_asm.md"
OTHER = REPO_ROOT / "docs_team/phase_descriptions_other.md"
IKEA_GROUP = {"Lack_TV_Bench": "Lack", "Lack_Coffee_Table": "Lack", "Lack_Side_Table": "Lack",
              "Kallax_Shelf_Drawer": "Kallax_Shelf_Drawer"}


def parse(path: Path) -> dict[str | None, list[dict]]:
    """{furniture or None: [{"name", "distinguishes", "short": [3 lines], "descriptions": [...]}]}."""
    out: dict[str | None, list[dict]] = {}
    group, phase = None, None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith("# Furniture:"):
            group, phase = line.split(":", 1)[1].strip(), None
        elif line.startswith("## "):
            phase = {"name": line[3:].strip(), "distinguishes": "", "short": [], "descriptions": []}
            out.setdefault(group, []).append(phase)
        elif phase is None:
            continue
        elif line.startswith("Distinguishes:"):
            phase["distinguishes"] = line.split(":", 1)[1].strip()
        elif line.startswith("> "):
            phase["short"].append(line[2:].strip())
        elif line.startswith("- "):
            phase["descriptions"].append(line[2:].strip())
    return out


def phases(dataset: str, furniture: str | None = None) -> list[dict]:
    """The phases of QiSens ("qisens") or of an IKEA furniture type ("ikea_asm"), in task order;
    for the Lack side table only the phases it has."""
    if dataset == "qisens":
        return parse(QISENS)[None]
    from ..datasets import ikea_phases as ip
    groups = parse(IKEA)[IKEA_GROUP[furniture]]
    want = ip.names(furniture)
    return [p for p in groups if p["name"] in want]


def other(dataset: str) -> dict:
    """The "other" class (no work in progress) of seg_a.other_class for "qisens" or "ikea_asm"."""
    return parse(OTHER)[{"qisens": "QiSens", "ikea_asm": "IKEA ASM"}[dataset]][0]
