# w10-phase-bench-v2-test

Engine B v2 rule: the single test split evaluation (without cuts and with the synthetic cuts of each p)

HA4M test split of the cross subject fold (65 videos, 63 in task order), 5 macro phases (experiments/datasets/ha4m_macro_phases.md), Kinect COCO 17 skeletons with YOLO like confidences. Mean over videos; at 85: videos with frame accuracy >= 85.

| row | engine | frame acc | median | at 85 | edit | F1@10 | F1@25 | F1@50 | MAE s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Engine A | A | 24.01 | 23.83 | 0/65 | 43.59 | 29.53 | 28.09 | 9.45 | 23.75 |
| Engine B v1 as shipped | B v1 | 54.97 | 53.8 | 0/65 | 98.51 | 75.81 | 64.04 | 41.22 | 10.12 |
| v1 selected | B custom | 63.24 | 63.06 | 5/65 | 99.13 | 81.45 | 76.53 | 60.22 | 10.22 |
| v2 selected | B custom | 88.53 | 93.0 | 54/65 | 99.13 | 96.17 | 94.94 | 92.17 | 6.16 |
| v1 selected, refinement 10 s (ablation) | B custom | 59.8 | 60.22 | 1/65 | 92.97 | 80.31 | 72.3 | 47.67 | 9.94 |
| v1 selected, video_norm off (ablation) | B custom | 59.38 | 58.33 | 3/65 | 99.13 | 80.17 | 72.78 | 46.99 | 9.2 |
| v2 selected, refinement 10 s (ablation) | B custom | 68.81 | 70.3 | 9/65 | 96.36 | 91.09 | 88.25 | 58.59 | 8.17 |
| v2 selected, video_norm on (ablation) | B custom | 87.37 | 93.0 | 54/65 | 99.13 | 95.86 | 94.63 | 90.94 | 6.26 |

With synthetic cuts (experiments/datasets/ha4m_synthetic_cuts.json; 3.62 cuts per minute, the QiSens cut rate; a share p of the true phase boundaries gets a cut):

| row | p | frame acc | median | at 85 | edit | F1@10 | F1@25 | F1@50 | MAE s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Engine A | 0.5 | 56.94 | 59.31 | 4/65 | 88.97 | 74.55 | 68.48 | 47.91 | 9.47 |
| Engine A | 0.75 | 64.24 | 69.25 | 15/65 | 92.67 | 77.08 | 73.69 | 58.35 | 8.18 |
| Engine A | 1 | 88.11 | 95.91 | 51/65 | 99.13 | 96.17 | 94.94 | 89.09 | 6.59 |
| Engine B v1 as shipped | 0.5 | 61.28 | 63.72 | 5/65 | 98.51 | 80.4 | 74.55 | 50.78 | 8.62 |
| Engine B v1 as shipped | 0.75 | 68.19 | 68.69 | 17/65 | 99.13 | 82.69 | 78.99 | 59.64 | 7.7 |
| Engine B v1 as shipped | 1 | 84.81 | 95.64 | 46/65 | 99.13 | 92.48 | 91.24 | 84.17 | 6.23 |
| v1 selected | 0.5 | 70.73 | 72.71 | 18/65 | 99.13 | 85.09 | 81.09 | 68.5 | 8.61 |
| v1 selected | 0.75 | 76.09 | 79.42 | 27/65 | 99.13 | 86.38 | 84.53 | 72.56 | 7.78 |
| v1 selected | 1 | 83.13 | 94.93 | 42/65 | 99.13 | 90.01 | 88.48 | 82.01 | 7.36 |
| v2 selected | 0.5 | 87.93 | 93.37 | 54/65 | 99.13 | 95.86 | 94.32 | 91.55 | 6.1 |
| v2 selected | 0.75 | 89.34 | 94.25 | 57/65 | 99.13 | 96.17 | 94.94 | 92.48 | 5.95 |
| v2 selected | 1 | 89.79 | 94.52 | 57/65 | 99.13 | 96.17 | 95.24 | 92.48 | 5.9 |
| v1 selected, refinement 10 s (ablation) | 0.5 | 59.6 | 60.22 | 1/65 | 93.28 | 80.51 | 72.2 | 46.95 | 9.95 |
| v1 selected, refinement 10 s (ablation) | 0.75 | 58.6 | 59.62 | 0/65 | 93.28 | 79.59 | 71.89 | 46.34 | 9.96 |
| v1 selected, refinement 10 s (ablation) | 1 | 59.16 | 59.82 | 1/65 | 93.28 | 79.89 | 72.2 | 46.95 | 9.93 |
| v1 selected, video_norm off (ablation) | 0.5 | 64.78 | 68.73 | 10/65 | 99.13 | 81.09 | 75.24 | 58.38 | 8.66 |
| v1 selected, video_norm off (ablation) | 0.75 | 68.93 | 69.66 | 20/65 | 99.13 | 82.99 | 78.38 | 61.79 | 8.33 |
| v1 selected, video_norm off (ablation) | 1 | 79.4 | 85.56 | 35/65 | 99.13 | 87.86 | 87.24 | 77.4 | 7.5 |
| v2 selected, refinement 10 s (ablation) | 0.5 | 68.28 | 70.3 | 9/65 | 96.67 | 89.72 | 86.89 | 58.56 | 8.19 |
| v2 selected, refinement 10 s (ablation) | 0.75 | 68.28 | 70.3 | 9/65 | 96.67 | 89.72 | 86.89 | 58.56 | 8.19 |
| v2 selected, refinement 10 s (ablation) | 1 | 68.28 | 70.3 | 9/65 | 96.67 | 89.72 | 86.89 | 58.56 | 8.19 |
| v2 selected, video_norm on (ablation) | 0.5 | 87.57 | 93.37 | 53/65 | 99.13 | 95.24 | 94.01 | 90.32 | 6.25 |
| v2 selected, video_norm on (ablation) | 0.75 | 87.74 | 94.27 | 55/65 | 99.13 | 94.94 | 93.71 | 90.32 | 6.13 |
| v2 selected, video_norm on (ablation) | 1 | 87.23 | 94.75 | 53/65 | 99.13 | 94.32 | 92.78 | 89.4 | 6.17 |

Public clips (stored YOLO26 x runs of w10-pose-sens-x):

| row | clip | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Engine A | process_480p30.mp4 | 92.33 | 100.0 | 100.0 | 100.0 | 100.0 | 2.5 |
| Engine A | shots_check_480p30.mp4 | 30.04 | 100.0 | 40.0 | 40.0 | 40.0 | 1.65 |
| Engine B v1 as shipped | process_480p30.mp4 | 92.33 | 100.0 | 100.0 | 100.0 | 100.0 | 2.5 |
| Engine B v1 as shipped | shots_check_480p30.mp4 | 34.78 | 100.0 | 60.0 | 40.0 | 40.0 | 2.25 |
| v1 selected | process_480p30.mp4 | 88.81 | 100.0 | 100.0 | 100.0 | 100.0 | 7.062 |
| v1 selected | shots_check_480p30.mp4 | 56.52 | 100.0 | 80.0 | 80.0 | 60.0 | 3.0 |
| v2 selected | process_480p30.mp4 | 62.41 | 100.0 | 100.0 | 100.0 | 40.0 | 24.375 |
| v2 selected | shots_check_480p30.mp4 | 38.74 | 100.0 | 80.0 | 60.0 | 20.0 | 5.0 |
| v1 selected, refinement 10 s (ablation) | process_480p30.mp4 | 71.08 | 100.0 | 100.0 | 100.0 | 40.0 | 20.013 |
| v1 selected, refinement 10 s (ablation) | shots_check_480p30.mp4 | 26.88 | 80.0 | 44.44 | 44.44 | 22.22 | 6.592 |
| v1 selected, video_norm off (ablation) | process_480p30.mp4 | 74.54 | 100.0 | 100.0 | 100.0 | 100.0 | 17.938 |
| v1 selected, video_norm off (ablation) | shots_check_480p30.mp4 | 56.52 | 100.0 | 80.0 | 80.0 | 60.0 | 3.0 |
| v2 selected, refinement 10 s (ablation) | process_480p30.mp4 | 37.16 | 100.0 | 80.0 | 40.0 | 0.0 | 33.892 |
| v2 selected, refinement 10 s (ablation) | shots_check_480p30.mp4 | 53.62 | 100.0 | 80.0 | 40.0 | 20.0 | 3.775 |
| v2 selected, video_norm on (ablation) | process_480p30.mp4 | 51.37 | 100.0 | 100.0 | 80.0 | 20.0 | 29.938 |
| v2 selected, video_norm on (ablation) | shots_check_480p30.mp4 | 42.69 | 100.0 | 80.0 | 60.0 | 20.0 | 4.75 |

Settings per row: results.json (rows[].sets).
