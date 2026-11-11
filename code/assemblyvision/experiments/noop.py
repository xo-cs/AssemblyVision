"""No-op declarations of every setting in configs/default.toml (Week 10).

An experiment stores every setting, so reusing or rerunning one that was stored
before a new setting existed needs the new key filled in. A missing key may be
filled from the current default only when that default changes nothing in a
batch run of the stored experiment. Every key of default.toml is declared here,
once (tests/test_noop_declarations.py fails on an undeclared or stale key):

  NO        not a no-op: a stored experiment without it is refused
  RUN_FREE  never read by a batch pipeline run (training, dataset tools, the UI,
            live mode): any default is a no-op for exp new, run and recheck
  OFF       the default itself switches its feature off (false, 0 or "")
  a function (default, full settings) -> bool, for keys that only act while
            another setting is on (e.g. the Engine B keys while its weights are 0)
  Legacy(v) a feature switched on by default later: a stored experiment without the
            key ran with v, so it is filled with v (not the current default)

When you add a key to default.toml, declare it here: OFF or a dependent rule for
a feature that is off by default, RUN_FREE for a training or tool setting, NO
otherwise (then old experiments are refused and must be recreated).
"""
from __future__ import annotations

NO = "no"
RUN_FREE = "run_free"
OFF = "off"


class Legacy:
    """Filled with the value stored experiments ran with (the behaviour before the key)."""

    def __init__(self, value):
        self.value = value

    def __repr__(self) -> str:
        return f"Legacy({self.value!r})"


def _seg(st: dict, key: str, default):
    return st.get("seg_a", {}).get(key, default)


def engine_b_off(v, st) -> bool:
    """Engine B is read by a run only while seg_a.engine_b_weight or seg_a.embed_weight > 0."""
    return _seg(st, "engine_b_weight", 0) == 0 and _seg(st, "embed_weight", 0) == 0


def boundary_term_off(v, st) -> bool:
    return _seg(st, "engine_b_weight", 0) == 0


def embed_off(v, st) -> bool:
    return _seg(st, "embed_weight", 0) == 0


def start_detect_off(v, st) -> bool:
    return _seg(st, "start_detect", False) is False


def candidates_off(v, st) -> bool:
    return v is False or boundary_term_off(v, st)


def _all(rule: str, keys: str) -> dict:
    return {k: rule for k in keys.split()}


DECLARATIONS: dict[str, dict] = {
    "run": _all(NO, "seed progress_every_s"),
    "profile": _all(NO, "enabled gpu_sample_s"),
    "paths": _all(NO, "qisens_root public_root models_dir"),
    "ingest": _all(NO, "max_fps default_fps crf min_duration_s disk_factor vfr_tolerance_ms"),
    "split": _all(NO, "train_frac val_frac"),
    "shots": {**_all(NO, "cut_threshold min_shot_s fixed_motion_max motion_pairs motion_pair_s downscale motion_seek"),
              "parallel": Legacy(False)},         # default true from 2026-11-11
    "pose": {**_all(NO, "variant fallback_variant imgsz det_conf_min ref_conf_min ref_search_s half tf32 "
                     "hand_crop_margin batch decode prefetch_frames reference_in_pass"),
             "overlap_stages": Legacy(False)},   # default true from 2026-11-11
    "track": _all(NO, "algorithm high_thresh low_thresh new_thresh match_thresh buffer_s min_track_s "
                      "auto_subject_min_coverage dup_suppress dup_iou dup_contain dup_kp_agree dup_kp_conf "
                      "dup_kp_dist overlap_pose"),
    "clean": _all(NO, "body_conf_min gap_max_s sg_window_s sg_order"),
    "seg_a": {
        **_all(NO, "phase_solver min_factor max_factor pause_energy pause_min_s long_pause_s cut_pause_window_s "
                   "no_person_min_s fallback_pause_min_s other_min_s strength_no_person strength_cut_pause "
                   "strength_pause strength_cut activity_change ac_weight ac_window_s"),
        "engine": Legacy("custom"),          # added 2026-11-11 (default "A+B v1"); custom = the stored keys
        "engine_b_weight": OFF,
        "engine_b_window_s": engine_b_off,
        "engine_b_norm": engine_b_off,
        "engine_b_candidates": candidates_off,
        "embed_weight": OFF,
        "embed_window_s": embed_off,
        "start_detect": OFF,
        "start_sustain_s": start_detect_off,
        "start_active_share": start_detect_off,
        "start_max_share": start_detect_off,
        "start_min_intro_s": start_detect_off,
    },
    "steps": _all(NO, "use mode step"),
    "eval": _all(NO, "seg_fps kappa_fps overlaps label"),
    "datasets": _all(RUN_FREE, "root split_seed cross_subject_test max_transfer_gb request_interval_s "
                               "backoff_start_s backoff_max_s limited_retries"),
    "boundaries": _all(RUN_FREE, "tol_s ignore_dont_care dont_care_label"),
    "engine_b": {
        **_all(RUN_FREE, "augment jitter_torso joint_dropout frame_dropout conf_visible conf_predicted conf_noise "
                         "aug_seed model_fps stages layers channels dropout epochs lr weight_decay batch_videos "
                         "crop_s pos_weight train_seed eval_tols_s action_weight n_actions hip_calibration "
                         "calibration_from"),
        "model": engine_b_off,                  # a run reads the model file only with a weight > 0
        "pool_s": boundary_term_off,
        "peak_threshold": boundary_term_off,    # the firing rate and engine_b candidates
        "peak_min_gap_s": boundary_term_off,
    },
    "motion": _all(NO, "enabled angle_bands cov_min path_hz jitter_torso rep_lag_min_s rep_lag_max_s rep_window_s "
                       "rep_hop_s rep_acf_min idle_energy idle_min_s foreshort_ratio view_limited_share hand_active "
                       "gaze_conf_min gaze_yaw gaze_down_torso stream_rep_every_s stream_torso_s"),
    "heatmap": _all(NO, "cell_m cell_px min_samples"),
    "gpu": _all(NO, "pose"),
    "overlay": _all(NO, "kp_conf_min max_width crf enabled encoder workers"),
    "ui": _all(RUN_FREE, "poll_s max_upload_mb"),
    "live": _all(RUN_FREE, "gpu half pose_model input_px det_conf kp_conf_min stream_every_s rate_steps rate_p95_ms "
                           "rate_drop_max rate_hold_s rate_up_s zone_hold_s one_euro_min_cutoff one_euro_beta "
                           "one_euro_d_cutoff lost_s idle_stop_s max_session_s rec_fps min_free_mb warn_fps "
                           "default_template model_keep_s lift_knee_conf lift_knee_seen"),
}


def declaration(key: str):
    """The rule of "section.key", or None if the key is not declared."""
    section, _, name = key.partition(".")
    return DECLARATIONS.get(section, {}).get(name)


def fill_value(key: str, default):
    """The value a missing key is filled with: the Legacy value, else the default."""
    rule = declaration(key)
    return rule.value if isinstance(rule, Legacy) else default


def is_no_op(key: str, default, settings: dict) -> bool:
    """True if filling key with default changes nothing in a run of settings (the
    stored values, completed with the defaults). For a Legacy key: default is the
    old value."""
    rule = declaration(key)
    if rule is None or rule == NO:
        return False
    if isinstance(rule, Legacy):
        return default == rule.value
    if rule == RUN_FREE:
        return True
    if rule == OFF:
        return default in (False, 0, 0.0, "") and not isinstance(default, (list, dict))
    return bool(rule(default, settings))


def undeclared(default: dict) -> list[str]:
    """Keys of the default settings without a declaration."""
    return [f"{s}.{k}" for s, vals in default.items() for k in vals if declaration(f"{s}.{k}") is None]


def stale(default: dict) -> list[str]:
    """Declared keys that default.toml no longer has."""
    return [f"{s}.{k}" for s, vals in DECLARATIONS.items() for k in vals if k not in default.get(s, {})]
