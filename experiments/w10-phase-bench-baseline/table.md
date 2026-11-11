# w10-phase-bench-baseline

Public phase benchmark baseline: Engine A and Engine B v1 on the HA4M test split

HA4M test split of the cross subject fold (65 videos, 63 in task order), 5 macro phases (experiments/datasets/ha4m_macro_phases.md), Kinect COCO 17 skeletons with YOLO like confidences. Mean over videos; at 85: videos with frame accuracy >= 85.

| row | engine | frame acc | median | at 85 | edit | F1@10 | F1@25 | F1@50 | MAE s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Engine A | A | 24.01 | 23.83 | 0/65 | 43.59 | 29.53 | 28.09 | 9.45 | 23.75 |
| Engine B v1 | B v1 | 24.01 | 23.83 | 0/65 | 43.59 | 29.53 | 28.09 | 9.45 | 23.75 |

Public clips (stored YOLO26 x runs of w10-pose-sens-x):

| row | clip | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Engine A | process_480p30.mp4 | 92.33 | 100.0 | 100.0 | 100.0 | 100.0 | 2.5 |
| Engine A | shots_check_480p30.mp4 | 30.04 | 100.0 | 40.0 | 40.0 | 40.0 | 1.65 |
| Engine B v1 | process_480p30.mp4 | 92.33 | 100.0 | 100.0 | 100.0 | 100.0 | 2.5 |
| Engine B v1 | shots_check_480p30.mp4 | 34.78 | 100.0 | 60.0 | 40.0 | 40.0 | 2.25 |

Settings per row: results.json (rows[].sets).
