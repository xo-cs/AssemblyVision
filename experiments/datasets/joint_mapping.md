# Joint mappings to COCO 17

Engine B uses keypoints from our pose model (YOLO26 x, COCO 17). The datasets' own skeletons
are mapped to COCO 17 only for the comparison experiment. Left and right are the person's own side.

## HA4M: Azure Kinect (32 joints, 2D colour projections) to COCO 17

| COCO 17 | Kinect joint (id) | note |
| --- | --- | --- |
| nose | nose (27) | Kinect face points are model estimates from the head pose, less precise than body joints. |
| left_eye | eye_left (28) | Kinect face point (estimated). |
| right_eye | eye_right (30) | Kinect face point (estimated). |
| left_ear | ear_left (29) | Kinect face point (estimated). |
| right_ear | ear_right (31) | Kinect face point (estimated). |
| left_shoulder | shoulder_left (5) |  |
| right_shoulder | shoulder_right (12) |  |
| left_elbow | elbow_left (6) |  |
| right_elbow | elbow_right (13) |  |
| left_wrist | wrist_left (7) |  |
| right_wrist | wrist_right (14) |  |
| left_hip | hip_left (18) | Kinect hip is the estimated joint centre; COCO marks the hip on the visible body. The offset is measured in the comparison experiment. |
| right_hip | hip_right (22) | Kinect hip is the estimated joint centre; COCO marks the hip on the visible body. The offset is measured in the comparison experiment. |
| left_knee | knee_left (19) |  |
| right_knee | knee_right (23) |  |
| left_ankle | ankle_left (20) |  |
| right_ankle | ankle_right (24) |  |

Every COCO joint has an equivalent. Kinect joints with no COCO equivalent: pelvis, spine_navel, spine_chest, neck, clavicle_left, hand_left, handtip_left, thumb_left, clavicle_right, hand_right, handtip_right, thumb_right, foot_left, foot_right, head.

## InHARD: mocap skeleton (BVH, 120 Hz) to COCO 17

| COCO 17 | InHARD joint | note |
| --- | --- | --- |
| nose | none | no face points in the mocap skeleton |
| left_eye | none | no face points in the mocap skeleton |
| right_eye | none | no face points in the mocap skeleton |
| left_ear | none | no face points in the mocap skeleton |
| right_ear | none | no face points in the mocap skeleton |
| left_shoulder | LeftArm |  |
| right_shoulder | RightArm |  |
| left_elbow | LeftForeArm |  |
| right_elbow | RightForeArm |  |
| left_wrist | LeftHand |  |
| right_wrist | RightHand |  |
| left_hip | LeftUpLeg |  |
| right_hip | RightUpLeg |  |
| left_knee | LeftLeg |  |
| right_knee | RightLeg |  |
| left_ankle | LeftFoot |  |
| right_ankle | RightFoot |  |

COCO joints with no equivalent: nose, left_eye, right_eye, left_ear, right_ear. The InHARD joint names
are provisional (access pending) and are checked against the first BVH file.

InHARD is 3D only until its camera calibration is known; the 2D comparison needs a projection.
