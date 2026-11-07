"""Joint mappings of the public datasets' own skeletons to COCO 17 (our pose format).

The datasets' skeletons are kept only for a comparison experiment; Engine B is
trained on keypoints from our own pose model.
"""
from __future__ import annotations

from pathlib import Path

COCO17 = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
          "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
          "right_knee", "left_ankle", "right_ankle")

# Azure Kinect Body Tracking SDK joint order (HA4M Skeletons, JointID 0 to 31).
KINECT32 = ("pelvis", "spine_navel", "spine_chest", "neck", "clavicle_left", "shoulder_left", "elbow_left",
            "wrist_left", "hand_left", "handtip_left", "thumb_left", "clavicle_right", "shoulder_right",
            "elbow_right", "wrist_right", "hand_right", "handtip_right", "thumb_right", "hip_left", "knee_left",
            "ankle_left", "foot_left", "hip_right", "knee_right", "ankle_right", "foot_right", "head", "nose",
            "eye_left", "ear_left", "eye_right", "ear_right")

# COCO index -> Kinect index. Left and right are the person's own side in both.
KINECT32_TO_COCO17 = (27, 28, 30, 29, 31, 5, 12, 6, 13, 7, 14, 18, 22, 19, 23, 20, 24)

KINECT32_NOTES = {
    "left_hip": "Kinect hip is the estimated joint centre; COCO marks the hip on the visible body. "
                "The offset is measured in the comparison experiment.",
    "right_hip": "Kinect hip is the estimated joint centre; COCO marks the hip on the visible body. "
                 "The offset is measured in the comparison experiment.",
    "nose": "Kinect face points are model estimates from the head pose, less precise than body joints.",
    "left_eye": "Kinect face point (estimated).", "right_eye": "Kinect face point (estimated).",
    "left_ear": "Kinect face point (estimated).", "right_ear": "Kinect face point (estimated).",
}

# InHARD mocap skeleton (BVH, 17 joints at 120 Hz). The joint names are read from the
# BVH header, so the mapping is by name. PROVISIONAL until the first file is seen
# (access pending): the usual Perception Neuron names are assumed. No face points,
# so the COCO nose, eyes and ears have no equivalent.
# COCO index -> InHARD joint name (None = no equivalent). Arm = upper arm joint (shoulder
# position), ForeArm = elbow, Hand = wrist, UpLeg = hip, Leg = knee, Foot = ankle.
INHARD17_TO_COCO17 = (None, None, None, None, None, "LeftArm", "RightArm", "LeftForeArm", "RightForeArm",
                      "LeftHand", "RightHand", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "LeftFoot",
                      "RightFoot")


def unmapped(source: tuple[str, ...], mapping) -> list[str]:
    """Source joints that no COCO joint uses."""
    used = {source[m] if isinstance(m, int) else m for m in mapping if m is not None}
    return [j for j in source if j not in used]


def mapping_report() -> str:
    """Markdown report of both mappings (experiments/datasets/joint_mapping.md)."""
    lines = ["# Joint mappings to COCO 17", "",
             "Engine B uses keypoints from our pose model (YOLO26 x, COCO 17). The datasets' own skeletons",
             "are mapped to COCO 17 only for the comparison experiment. Left and right are the person's own side.",
             "", "## HA4M: Azure Kinect (32 joints, 2D colour projections) to COCO 17", "",
             "| COCO 17 | Kinect joint (id) | note |", "| --- | --- | --- |"]
    for c, k in zip(COCO17, KINECT32_TO_COCO17):
        lines.append(f"| {c} | {KINECT32[k]} ({k}) | {KINECT32_NOTES.get(c, '')} |")
    lines += ["", "Every COCO joint has an equivalent. Kinect joints with no COCO equivalent: "
              + ", ".join(unmapped(KINECT32, KINECT32_TO_COCO17)) + ".", "",
              "## InHARD: mocap skeleton (BVH, 120 Hz) to COCO 17", "",
              "| COCO 17 | InHARD joint | note |", "| --- | --- | --- |"]
    for c, k in zip(COCO17, INHARD17_TO_COCO17):
        lines.append(f"| {c} | {k or 'none'} | {'no face points in the mocap skeleton' if k is None else ''} |")
    lines += ["", "COCO joints with no equivalent: "
              + ", ".join(c for c, k in zip(COCO17, INHARD17_TO_COCO17) if k is None) + ". The InHARD joint names",
              "are provisional (access pending) and are checked against the first BVH file.", "",
              "InHARD is 3D only until its camera calibration is known; the 2D comparison needs a projection.", ""]
    return "\n".join(lines)


def write_mapping_report(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mapping_report())
    return path
