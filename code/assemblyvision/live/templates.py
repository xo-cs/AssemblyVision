"""Process templates of the Live tab (LLD 3.12): configs/live/<name>.toml.

kind workbench: [zones] name = [x0, y0, x1, y1] as fractions of the image and
  [[steps]] name, start_event, end_event, min_s, max_s; cyclic = true. Events:
  zone_enter:<zone>, zone_exit:<zone>, both_in:<zone>.
kind lift: [lift] thresholds of the floor pick station.
kind posture: [posture] settings of the side work station.
Every template has title, camera (the recommended camera view) and description.
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from ..config import CODE_DIR
from ..errors import AvError

TEMPLATE_DIR = CODE_DIR / "configs" / "live"
KINDS = ("workbench", "lift", "posture")
_EVENT = re.compile(r"(zone_enter|zone_exit|both_in):(\w+)")


@dataclass
class ProcessTemplate:
    name: str
    kind: str
    title: str
    camera: str = ""
    description: str = ""
    cyclic: bool = False
    zones: dict = field(default_factory=dict)       # name -> [x0, y0, x1, y1] fractions
    steps: list = field(default_factory=list)       # [{name, start_event, end_event, min_s, max_s}]
    params: dict = field(default_factory=dict)      # [lift] or [posture]

    def with_borders(self, b1: float, b2: float) -> "ProcessTemplate":
        """Copy with the left, center and right zones moved to the image x borders b1 < b2."""
        b1, b2 = sorted((min(max(b1, 0.05), 0.95), min(max(b2, 0.05), 0.95)))
        zones = dict(self.zones)
        for name, (x0, x1) in {"left": (0.0, b1), "center": (b1, b2), "right": (b2, 1.0)}.items():
            if name in zones:
                zones[name] = [x0, zones[name][1], x1, zones[name][3]]
        return ProcessTemplate(self.name, self.kind, self.title, self.camera, self.description, self.cyclic,
                               zones, self.steps, self.params)


def list_templates(folder: Path = TEMPLATE_DIR) -> list[str]:
    return sorted(p.stem for p in folder.glob("*.toml"))


def load_template(name: str, folder: Path = TEMPLATE_DIR) -> ProcessTemplate:
    """Read and check one template; AV-LIV-003 if missing or invalid."""
    def bad(detail: str) -> AvError:
        return AvError("AV-LIV-003", name=name, detail=detail)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name or ""):
        raise bad("not a template name")
    path = folder / f"{name}.toml"
    try:
        d = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise bad(str(e)) from e
    kind = d.get("kind")
    if kind not in KINDS:
        raise bad(f"kind must be one of {', '.join(KINDS)}")
    t = ProcessTemplate(name, kind, str(d.get("title", name)), str(d.get("camera", "")),
                        str(d.get("description", "")), bool(d.get("cyclic", False)), dict(d.get("zones", {})),
                        list(d.get("steps", [])), dict(d.get(kind if kind != "workbench" else "params", {})))
    for z, box in t.zones.items():
        if not (isinstance(box, list) and len(box) == 4 and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise bad(f"zone {z} must be [x0, y0, x1, y1] fractions with x0 < x1 and y0 < y1")
    if kind == "workbench":
        if not t.steps:
            raise bad("a workbench template needs [[steps]]")
        for i, st in enumerate(t.steps):
            for key in ("name", "start_event", "end_event"):
                if not isinstance(st.get(key), str):
                    raise bad(f"steps[{i}] needs {key}")
            for key in ("start_event", "end_event"):
                m = _EVENT.fullmatch(st[key])
                if not m or m.group(2) not in t.zones:
                    raise bad(f"steps[{i}].{key} {st[key]!r} is not an event of a known zone")
            st.setdefault("min_s", 0.0)
            st.setdefault("max_s", 3600.0)
    return t
