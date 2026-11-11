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
STEPS_TOML = CODE_DIR / "configs" / "steps.toml"
DEFAULT_LAYERS = [DEFAULT_TOML, STEPS_TOML]          # LLD 3.0.5 order (gpu.toml comes later)

VARIANTS = ("n", "s", "m", "l", "x")
# step engine presets (seg_a.engine): the keys each preset sets; "custom" leaves them as configured
ENGINE_PRESETS: dict[str, dict[str, Any]] = {
    "A": {"seg_a.engine_b_weight": 0.0, "seg_a.embed_weight": 0.0},
    "B v1": {"seg_a.engine_b_weight": 0.0, "seg_a.embed_weight": 0.2, "seg_a.embed_window_s": 30.0,
               "engine_b.model": "engine_b/w10-engine-b-v1.pt"},
}
ENGINES = (*ENGINE_PRESETS, "custom")
# Engine B = the learned model with the DP decoder; "A+B v1" was the name of "B v1" from 2026-11-11 to 11-12
ENGINE_ALIASES = {"A+B v1": "B v1"}
MOTION_METRICS = ("duration", "path", "back_bending", "repetition", "hand_travel", "two_hand", "idle", "gaze")

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
    "pose.variant": (lambda v: v in VARIANTS, "n, s, m, l, x"),
    "pose.fallback_variant": (lambda v: v in VARIANTS + ("",), "n, s, m, l, x or empty"),
    "pose.imgsz": (lambda v: 32 <= v <= 1920 and v % 32 == 0, "32 to 1920, multiple of 32"),
    "pose.det_conf_min": (lambda v: 0 <= v <= 1, "0 to 1"),
    "pose.ref_conf_min": (lambda v: 0 <= v <= 1, "0 to 1"),
    "pose.ref_search_s": (lambda v: v > 0, "> 0"),
    "pose.half": (lambda v: isinstance(v, bool), "true or false"),
    "pose.tf32": (lambda v: isinstance(v, bool), "true or false"),
    "pose.hand_crop_margin": (lambda v: 0 <= v <= 1, "0 to 1"),
    "pose.batch": (lambda v: 1 <= v <= 256, "1 to 256"),
    "pose.decode": (lambda v: v in ("opencv", "prefetch", "ffmpeg_cuda"), "opencv, prefetch or ffmpeg_cuda"),
    "pose.prefetch_frames": (lambda v: 1 <= v <= 4096, "1 to 4096"),
    "profile.gpu_sample_s": (lambda v: 0.05 <= v <= 60, "0.05 to 60"),
    "datasets.cross_subject_test": (lambda v: 0.05 <= v <= 0.95, "0.05 to 0.95"),
    "datasets.max_transfer_gb": (lambda v: 0 < v <= 10000, "above 0"),
    "datasets.request_interval_s": (lambda v: 0 <= v <= 600, "0 to 600 s"),
    "datasets.backoff_start_s": (lambda v: 1 <= v <= 3600, "1 to 3600 s"),
    "datasets.backoff_max_s": (lambda v: 1 <= v <= 86400, "1 to 86400 s"),
    "datasets.limited_retries": (lambda v: 0 <= v <= 100, "0 to 100"),
    "boundaries.tol_s": (lambda v: 0 <= v <= 5, "0 to 5 s"),
    "engine_b.jitter_torso": (lambda v: 0 <= v <= 1, "0 to 1"),
    "engine_b.joint_dropout": (lambda v: 0 <= v < 1, "0 to below 1"),
    "engine_b.frame_dropout": (lambda v: 0 <= v < 1, "0 to below 1"),
    "engine_b.conf_visible": (lambda v: 0 <= v <= 1, "0 to 1"),
    "engine_b.conf_predicted": (lambda v: 0 <= v <= 1, "0 to 1"),
    "engine_b.conf_noise": (lambda v: 0 <= v <= 1, "0 to 1"),
    "engine_b.model_fps": (lambda v: 1 <= v <= 60, "1 to 60"),
    "engine_b.stages": (lambda v: 1 <= v <= 8, "1 to 8"),
    "engine_b.layers": (lambda v: 1 <= v <= 16, "1 to 16"),
    "engine_b.channels": (lambda v: 4 <= v <= 1024, "4 to 1024"),
    "engine_b.dropout": (lambda v: 0 <= v < 1, "0 to below 1"),
    "engine_b.epochs": (lambda v: 1 <= v <= 10000, "1 to 10000"),
    "engine_b.lr": (lambda v: 0 < v <= 1, "above 0, up to 1"),
    "engine_b.weight_decay": (lambda v: 0 <= v <= 1, "0 to 1"),
    "engine_b.batch_videos": (lambda v: 1 <= v <= 512, "1 to 512"),
    "engine_b.crop_s": (lambda v: v > 0, "> 0"),
    "engine_b.pos_weight": (lambda v: v >= 0, ">= 0"),
    "engine_b.peak_threshold": (lambda v: 0 <= v <= 1, "0 to 1"),
    "engine_b.peak_min_gap_s": (lambda v: v >= 0, ">= 0"),
    "engine_b.eval_tols_s": (lambda v: len(v) > 0 and all(isinstance(x, (int, float)) and x > 0 for x in v),
                             "a non empty list of positive numbers"),
    "engine_b.pool_s": (lambda v: 0 <= v <= 600, "0 to 600 s"),
    "boundaries.ignore_dont_care": (lambda v: isinstance(v, bool), "true or false"),
    "shots.downscale": (lambda v: 0 <= v <= 16, "0 to 16 (0 = automatic)"),
    "overlay.encoder": (lambda v: v in ("libx264", "h264_nvenc"), "libx264 or h264_nvenc"),
    "overlay.workers": (lambda v: 1 <= v <= 32, "1 to 32"),
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
    "clean.body_conf_min": (lambda v: 0 <= v <= 1, "0 to 1"),
    "clean.gap_max_s": (lambda v: v >= 0, ">= 0"),
    "clean.sg_window_s": (lambda v: v >= 0, ">= 0 (0 = no smoothing)"),
    "clean.sg_order": (lambda v: 0 <= v <= 5, "0 to 5"),
    "seg_a.phase_solver": (lambda v: v in ("dp", "forward"), "dp or forward"),
    "seg_a.min_factor": (lambda v: v >= 0, ">= 0"),
    "seg_a.max_factor": (lambda v: v > 0, "> 0"),
    "seg_a.pause_energy": (lambda v: v >= 0, ">= 0"),
    "seg_a.pause_min_s": (lambda v: v > 0, "> 0"),
    "seg_a.long_pause_s": (lambda v: v > 0, "> 0"),
    "seg_a.cut_pause_window_s": (lambda v: v >= 0, ">= 0"),
    "seg_a.no_person_min_s": (lambda v: v > 0, "> 0"),
    "seg_a.fallback_pause_min_s": (lambda v: v > 0, "> 0"),
    "seg_a.other_min_s": (lambda v: v >= 0, ">= 0"),
    "seg_a.strength_no_person": (lambda v: v >= 0, ">= 0"),
    "seg_a.strength_cut_pause": (lambda v: v >= 0, ">= 0"),
    "seg_a.strength_pause": (lambda v: v >= 0, ">= 0"),
    "seg_a.strength_cut": (lambda v: v >= 0, ">= 0"),
    "seg_a.ac_weight": (lambda v: v >= 0, ">= 0"),
    "seg_a.ac_window_s": (lambda v: v > 0, "> 0"),
    "seg_a.engine_b_weight": (lambda v: v >= 0, ">= 0"),
    "seg_a.engine_b_window_s": (lambda v: v >= 0, ">= 0"),
    "seg_a.engine_b_norm": (lambda v: v in ("max", "none"), "max or none"),
    "seg_a.engine_b_candidates": (lambda v: isinstance(v, bool), "true or false"),
    "shots.parallel": (lambda v: isinstance(v, bool), "true or false"),
    "pose.overlap_stages": (lambda v: isinstance(v, bool), "true or false"),
    "seg_a.embed_candidates": (lambda v: isinstance(v, bool), "true or false"),
    "seg_a.embed_peak_min": (lambda v: 0 <= v <= 1, "0 to 1"),
    "seg_a.embed_peak_gap_s": (lambda v: v > 0, "> 0"),
    "seg_a.refine_window_s": (lambda v: 0 <= v <= 120, "0 to 120 s"),
    "seg_a.other_ends": (lambda v: isinstance(v, bool), "true or false"),
    "seg_a.other_max_s": (lambda v: v > 0, "> 0"),
    "seg_a.other_min_strength": (lambda v: v >= 0, ">= 0"),
    "engine_b.video_norm": (lambda v: isinstance(v, bool), "true or false"),
    "engine_b.contrast_weight": (lambda v: v >= 0, ">= 0"),
    "engine_b.contrast_window_s": (lambda v: 0.5 <= v <= 60, "0.5 to 60 s"),
    "engine_b.contrast_temp": (lambda v: 0 < v <= 10, "above 0, up to 10"),
    "seg_a.engine": (lambda v: v in ENGINES or v in ENGINE_ALIASES, ", ".join(ENGINES)),
    "seg_a.embed_weight": (lambda v: v >= 0, ">= 0"),
    "seg_a.embed_window_s": (lambda v: 1 <= v <= 1200, "1 to 1200 s"),
    "seg_a.start_detect": (lambda v: isinstance(v, bool), "true or false"),
    "seg_a.start_sustain_s": (lambda v: 0 < v <= 600, "above 0, up to 600 s"),
    "seg_a.start_active_share": (lambda v: 0 < v <= 1, "above 0, up to 1"),
    "seg_a.start_max_share": (lambda v: 0 <= v < 1, "0 to below 1"),
    "seg_a.start_min_intro_s": (lambda v: v >= 0, ">= 0"),
    "engine_b.action_weight": (lambda v: v >= 0, ">= 0"),
    "engine_b.n_actions": (lambda v: 1 <= v <= 1000, "1 to 1000"),
    "steps.mode": (lambda v: v == "ordered", "ordered"),
    "eval.seg_fps": (lambda v: 1 <= v <= 240, "1 to 240"),
    "eval.kappa_fps": (lambda v: 1 <= v <= 240, "1 to 240"),
    "eval.overlaps": (lambda v: len(v) > 0 and all(isinstance(x, (int, float)) and 0 < x <= 1 for x in v),
                      "a non empty list of numbers in (0, 1]"),
    "motion.enabled": (lambda v: all(x in MOTION_METRICS for x in v), ", ".join(MOTION_METRICS)),
    "motion.angle_bands": (lambda v: len(v) == 3 and all(isinstance(x, (int, float)) for x in v)
                           and 0 < v[0] < v[1] < v[2] < 180, "three increasing angles in (0, 180) deg"),
    "motion.cov_min": (lambda v: 0 <= v <= 1, "0 to 1"),
    "motion.path_hz": (lambda v: v > 0, "> 0"),
    "motion.jitter_torso": (lambda v: v >= 0, ">= 0"),
    "motion.rep_lag_min_s": (lambda v: v > 0, "> 0"),
    "motion.rep_lag_max_s": (lambda v: v > 0, "> 0"),
    "motion.rep_window_s": (lambda v: v > 0, "> 0"),
    "motion.rep_hop_s": (lambda v: v > 0, "> 0"),
    "motion.rep_acf_min": (lambda v: 0 <= v <= 1, "0 to 1"),
    "motion.idle_energy": (lambda v: v >= 0, ">= 0"),
    "motion.idle_min_s": (lambda v: v > 0, "> 0"),
    "motion.foreshort_ratio": (lambda v: 0 < v <= 1, "0 to 1"),
    "motion.view_limited_share": (lambda v: 0 <= v <= 1, "0 to 1"),
    "motion.hand_active": (lambda v: v >= 0, ">= 0"),
    "motion.gaze_conf_min": (lambda v: 0 <= v <= 1, "0 to 1"),
    "motion.gaze_yaw": (lambda v: v > 0, "> 0"),
    "motion.gaze_down_torso": (lambda v: v >= 0, ">= 0"),
    "motion.stream_rep_every_s": (lambda v: v > 0, "> 0"),
    "motion.stream_torso_s": (lambda v: v > 0, "> 0"),
    "heatmap.cell_m": (lambda v: v > 0, "> 0"),
    "heatmap.cell_px": (lambda v: v >= 1, ">= 1"),
    "heatmap.min_samples": (lambda v: v >= 1, ">= 1"),
    "gpu.pose": (lambda v: v >= 0, ">= 0"),
    "overlay.kp_conf_min": (lambda v: 0 <= v <= 1, "0 to 1"),
    "overlay.max_width": (lambda v: v >= 64, ">= 64"),
    "overlay.crf": (lambda v: 0 <= v <= 51, "0 to 51"),
    "live.gpu": (lambda v: v >= 0, ">= 0"),
    "live.input_px": (lambda v: 160 <= v <= 1920 and v % 32 == 0, "160 to 1920, multiple of 32"),
    "live.det_conf": (lambda v: 0 <= v <= 1, "0 to 1"),
    "live.kp_conf_min": (lambda v: 0 <= v <= 1, "0 to 1"),
    "live.stream_every_s": (lambda v: 0.02 <= v <= 1, "0.02 to 1"),
    "live.zone_hold_s": (lambda v: v >= 0, ">= 0"),
    "live.one_euro_min_cutoff": (lambda v: v > 0, "> 0"),
    "live.one_euro_beta": (lambda v: v >= 0, ">= 0"),
    "live.one_euro_d_cutoff": (lambda v: v > 0, "> 0"),
    "live.lost_s": (lambda v: v > 0, "> 0"),
    "live.idle_stop_s": (lambda v: v > 0, "> 0"),
    "live.max_session_s": (lambda v: v > 0, "> 0"),
    "live.rec_fps": (lambda v: 1 <= v <= 60, "1 to 60"),
    "live.min_free_mb": (lambda v: v >= 0, ">= 0"),
    "live.warn_fps": (lambda v: v >= 0, ">= 0"),
    "live.model_keep_s": (lambda v: v >= 0, ">= 0"),
    "live.lift_knee_conf": (lambda v: 0 <= v <= 1, "0 to 1"),
    "live.rate_steps": (lambda v: len(v) >= 1 and all(1 <= x <= 60 for x in v)
                        and list(v) == sorted(v, reverse=True), "1 to 60 fps, highest first"),
    "live.rate_p95_ms": (lambda v: v > 0, "> 0"),
    "live.rate_drop_max": (lambda v: 0 < v <= 1, "0 to 1"),
    "live.rate_hold_s": (lambda v: v >= 0, ">= 0"),
    "live.rate_up_s": (lambda v: v > 0, "> 0"),
    "live.lift_knee_seen": (lambda v: 0 <= v <= 1, "0 to 1"),
    "ui.poll_s": (lambda v: 0.5 <= v <= 60, "0.5 to 60"),
    "ui.max_upload_mb": (lambda v: v >= 1, ">= 1"),
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


def engine_label(d) -> str:
    """The step engine of resolved settings: the preset name, or for "custom" "A" (no Engine
    B weight) or "B custom"."""
    e = d["seg_a"].get("engine", "custom")
    if e != "custom":
        return e
    return "A" if d["seg_a"]["engine_b_weight"] == 0 and d["seg_a"]["embed_weight"] == 0 else "B custom"


class Config:
    """Merged configuration. Read with cfg["pose"]["imgsz"] or cfg.get("pose.imgsz")."""

    def __init__(self, data: dict, sources: list[str]):
        self._data = data
        self.sources = sources

    @classmethod
    def load(cls, layers: list[Path] | None = None, run_config: Path | None = None,
             overrides: dict | None = None) -> "Config":
        layers = list(layers) if layers else list(DEFAULT_LAYERS)
        base = _read_toml(layers[0])
        shipped = copy.deepcopy(base)
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
        cls._apply_engine(base, shipped)
        cls._check_cross(base)
        cls._check_steps(base.get("steps", {}).get("step", []))
        return cls(base, sources)

    @staticmethod
    def _check_steps(steps: list) -> None:
        """steps.step: [{name, min_s?, max_s?}], names unique, 0 < min_s < max_s."""
        names = set()
        for i, st in enumerate(steps):
            key = f"steps.step[{i}]"
            if not isinstance(st, dict):
                raise AvError("AV-CFG-002", key=key, value=repr(st), range="a table {name, min_s, max_s}")
            for name in st:
                if name not in ("name", "min_s", "max_s"):
                    raise AvError("AV-CFG-001", key=f"{key}.{name}", file="steps")
            name = st.get("name")
            if not isinstance(name, str) or not name.strip() or name in names \
                    or name == "other" or name.startswith("segment_"):
                raise AvError("AV-CFG-002", key=f"{key}.name", value=repr(name),
                              range="a unique non empty name, not other or segment_N")
            names.add(name)
            for lim in ("min_s", "max_s"):
                v = st.get(lim)
                if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0):
                    raise AvError("AV-CFG-002", key=f"{key}.{lim}", value=repr(v), range="> 0")
            if "min_s" in st and "max_s" in st and not st["min_s"] < st["max_s"]:
                raise AvError("AV-CFG-002", key=f"{key}.min_s", value=repr(st["min_s"]), range="< max_s")

    @staticmethod
    def _apply_engine(d: dict, shipped: dict) -> None:
        """seg_a.engine "A" or "B v1" sets its preset keys (ENGINE_PRESETS). A preset key
        given with another value than the preset's and the shipped default's is a conflict
        (AV-CFG-002): change it with seg_a.engine = "custom"."""
        d["seg_a"]["engine"] = ENGINE_ALIASES.get(d["seg_a"]["engine"], d["seg_a"]["engine"])
        preset = ENGINE_PRESETS.get(d["seg_a"]["engine"])
        if preset is None:
            return
        for key, value in preset.items():
            section, name = key.split(".")
            have = d[section][name]
            if have != value and have != shipped[section][name]:
                raise AvError("AV-CFG-002", key=key, value=repr(have),
                              range=f"{value!r} (set by seg_a.engine = {d['seg_a']['engine']!r}); "
                                    f"set seg_a.engine = \"custom\" to change it")
            d[section][name] = value

    @staticmethod
    def _check_cross(d: dict) -> None:
        """Rules between keys: the ByteTrack low band must contain detections."""
        t, low = d["track"], d["track"]["low_thresh"]
        if not low <= t["high_thresh"]:
            raise AvError("AV-CFG-002", key="track.low_thresh", value=repr(low), range="<= track.high_thresh")
        if d["pose"]["det_conf_min"] > low:
            raise AvError("AV-CFG-002", key="pose.det_conf_min", value=repr(d["pose"]["det_conf_min"]),
                          range="<= track.low_thresh, or the low score band of ByteTrack stays empty")
        if t["overlap_pose"] and not d["pose"]["reference_in_pass"]:
            raise AvError("AV-CFG-002", key="track.overlap_pose", value="true",
                          range="false unless pose.reference_in_pass is true (a shot is tracked with its reference)")
        if d["shots"]["parallel"] and not d["pose"]["reference_in_pass"]:
            raise AvError("AV-CFG-002", key="shots.parallel", value="true",
                          range="false unless pose.reference_in_pass is true (the reference frames come from the pose pass)")
        m = d["motion"]
        if not m["rep_lag_min_s"] < m["rep_lag_max_s"]:
            raise AvError("AV-CFG-002", key="motion.rep_lag_min_s", value=repr(m["rep_lag_min_s"]),
                          range="< motion.rep_lag_max_s")
        if m["rep_window_s"] < 2 * m["rep_lag_max_s"]:
            raise AvError("AV-CFG-002", key="motion.rep_window_s", value=repr(m["rep_window_s"]),
                          range=">= 2 * motion.rep_lag_max_s")

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
        """Serialise the snapshot (sections of scalars, lists and inline tables) as TOML."""
        lines = []
        for section, values in self._data.items():
            lines.append(f"[{section}]")
            for name, value in values.items():
                lines.append(f"{name} = {_toml_value(value)}")
            lines.append("")
        return "\n".join(lines)


def deep_merge(*layers: dict) -> dict:
    """Section wise merge of override dicts, later layers win (values are copied)."""
    out: dict = {}
    for layer in layers:
        for section, values in layer.items():
            if isinstance(values, dict):
                out.setdefault(section, {}).update(copy.deepcopy(values))
            else:
                out[section] = copy.deepcopy(values)
    return out


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(v)}" for k, v in value.items()) + " }"
    return repr(value)
