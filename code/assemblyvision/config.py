"""Load, layer, validate and snapshot the TOML configuration (LLD 3.0.5).

The first layer (configs/default.toml) defines every key. Later layers may only
override existing keys (AV-CFG-001) with a value of the same type that passes
the range rules below (AV-CFG-002).
"""
from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any, Callable

from .errors import AvError

CODE_DIR = Path(__file__).resolve().parent.parent          # ~/AssemblyVision/code
DEFAULT_TOML = CODE_DIR / "configs" / "default.toml"

VARIANTS = ("n", "s", "m", "l")

# dotted key: (check, allowed range shown in AV-CFG-002)
_RULES: dict[str, tuple[Callable[[Any], bool], str]] = {
    "run.progress_every_s": (lambda v: v > 0, "> 0"),
    "ingest.max_fps": (lambda v: 1 <= v <= 240, "1 to 240"),
    "ingest.default_fps": (lambda v: 1 <= v <= 240, "1 to 240"),
    "ingest.crf": (lambda v: 0 <= v <= 51, "0 to 51"),
    "ingest.min_duration_s": (lambda v: v >= 0, ">= 0"),
    "ingest.disk_factor": (lambda v: v >= 0, ">= 0"),
    "ingest.vfr_tolerance_ms": (lambda v: v >= 0, ">= 0"),
    "split.train_frac": (lambda v: 0 < v < 1, "0 to 1"),
    "split.val_frac": (lambda v: 0 <= v < 1, "0 to 1"),
    "shots.cut_threshold": (lambda v: 0 < v <= 255, "0 to 255"),
    "shots.min_shot_s": (lambda v: v >= 0, ">= 0"),
    "shots.fixed_motion_max": (lambda v: 0 <= v <= 1, "0 to 1"),
    "shots.motion_pairs": (lambda v: v >= 1, ">= 1"),
    "shots.motion_pair_s": (lambda v: v > 0, "> 0"),
    "pose.variant": (lambda v: v in VARIANTS, "n, s, m, l"),
    "pose.fallback_variant": (lambda v: v in VARIANTS + ("",), "n, s, m, l or empty"),
    "pose.imgsz": (lambda v: 32 <= v <= 1920 and v % 32 == 0, "32 to 1920, multiple of 32"),
    "pose.det_conf_min": (lambda v: 0 <= v <= 1, "0 to 1"),
    "pose.ref_conf_min": (lambda v: 0 <= v <= 1, "0 to 1"),
    "pose.ref_search_s": (lambda v: v > 0, "> 0"),
    "pose.half": (lambda v: v is False, "false (Pascal GPU, FP32 only)"),
    "pose.hand_crop_margin": (lambda v: 0 <= v <= 1, "0 to 1"),
    "track.algorithm": (lambda v: v == "bytetrack", "bytetrack"),
    "track.high_thresh": (lambda v: 0 <= v <= 1, "0 to 1"),
    "track.low_thresh": (lambda v: 0 <= v <= 1, "0 to 1"),
    "track.new_thresh": (lambda v: 0 <= v <= 1, "0 to 1"),
    "track.match_thresh": (lambda v: 0 < v <= 1, "0 to 1"),
    "track.buffer_s": (lambda v: v >= 0, ">= 0"),
    "track.min_track_s": (lambda v: v >= 0, ">= 0"),
    "track.auto_subject_min_coverage": (lambda v: 0 <= v <= 1, "0 to 1"),
    "track.dup_iou": (lambda v: 0 < v <= 1, "0 to 1"),
    "track.dup_contain": (lambda v: 0 < v <= 1, "0 to 1"),
    "track.dup_kp_agree": (lambda v: 0 < v <= 1, "0 to 1"),
    "track.dup_kp_conf": (lambda v: 0 <= v <= 1, "0 to 1"),
    "track.dup_kp_dist": (lambda v: v > 0, "> 0"),
    "gpu.pose": (lambda v: v >= 0, ">= 0"),
    "overlay.kp_conf_min": (lambda v: 0 <= v <= 1, "0 to 1"),
    "overlay.max_width": (lambda v: v >= 64, ">= 64"),
    "overlay.crf": (lambda v: 0 <= v <= 51, "0 to 51"),
}


def _read_toml(path: Path) -> dict:
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise AvError("AV-CFG-003", detail=str(e), file=str(path)) from e


def _same_type(default: Any, value: Any) -> bool:
    if isinstance(default, bool) or isinstance(value, bool):
        return type(default) is type(value)
    if isinstance(default, float):
        return isinstance(value, (int, float))
    return type(default) is type(value)


class Config:
    """Merged configuration. Read with cfg["pose"]["imgsz"] or cfg.get("pose.imgsz")."""

    def __init__(self, data: dict, sources: list[str]):
        self._data = data
        self.sources = sources

    @classmethod
    def load(cls, layers: list[Path] | None = None, run_config: Path | None = None,
             overrides: dict | None = None) -> "Config":
        layers = list(layers) if layers else [DEFAULT_TOML]
        base = _read_toml(layers[0])
        sources = [str(layers[0])]
        for path in layers[1:] + ([run_config] if run_config else []):
            cls._merge(base, _read_toml(path), str(path))
            sources.append(str(path))
        if overrides:
            cls._merge(base, overrides, "<command line>")
            sources.append("<command line>")
        for key, (check, allowed) in _RULES.items():
            section, name = key.split(".")
            value = base.get(section, {}).get(name)
            if value is not None and not check(value):
                raise AvError("AV-CFG-002", key=key, value=repr(value), range=allowed)
        cls._check_cross(base)
        return cls(base, sources)

    @staticmethod
    def _check_cross(d: dict) -> None:
        """Rules between keys: the ByteTrack low band must contain detections."""
        t, low = d["track"], d["track"]["low_thresh"]
        if not low <= t["high_thresh"]:
            raise AvError("AV-CFG-002", key="track.low_thresh", value=repr(low), range="<= track.high_thresh")
        if d["pose"]["det_conf_min"] > low:
            raise AvError("AV-CFG-002", key="pose.det_conf_min", value=repr(d["pose"]["det_conf_min"]),
                          range="<= track.low_thresh, or the low score band of ByteTrack stays empty")

    @staticmethod
    def _merge(base: dict, layer: dict, file: str) -> None:
        for section, values in layer.items():
            if section not in base or not isinstance(values, dict):
                raise AvError("AV-CFG-001", key=section, file=file)
            for name, value in values.items():
                key = f"{section}.{name}"
                if name not in base[section]:
                    raise AvError("AV-CFG-001", key=key, file=file)
                if not _same_type(base[section][name], value):
                    allowed = type(base[section][name]).__name__
                    raise AvError("AV-CFG-002", key=key, value=repr(value), range=allowed)
                base[section][name] = value

    def __getitem__(self, section: str) -> dict:
        return self._data[section]

    def get(self, dotted: str) -> Any:
        section, name = dotted.split(".")
        return self._data[section][name]

    def snapshot(self) -> dict:
        """Deep copy for the manifest and run_config.toml (FR-STO-03)."""
        return copy.deepcopy(self._data)

    def to_toml(self) -> str:
        """Serialise the snapshot (flat sections of scalars) as TOML."""
        lines = []
        for section, values in self._data.items():
            lines.append(f"[{section}]")
            for name, value in values.items():
                if isinstance(value, bool):
                    text = "true" if value else "false"
                elif isinstance(value, str):
                    text = '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
                else:
                    text = repr(value)
                lines.append(f"{name} = {text}")
            lines.append("")
        return "\n".join(lines)
