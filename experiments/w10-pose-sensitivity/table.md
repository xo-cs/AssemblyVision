# w10-pose-sensitivity

Engine A sensitivity to the pose model (YOLO26 x vs l), public clips with steps GT

Experiments w10-pose-sens-x, w10-pose-sens-l: the same stored settings except pose.variant (configs/reference.toml, batch 1, strict FP32). Change: against w10-pose-sens-x.

## process_480p30.mp4 (GT process_480p30_gt.csv)

| pose | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s | phases | subject coverage | pose fps | run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26 x | 92.33 | 100.0 | 100.0 | 100.0 | 100.0 | 2.5 | 5/5 | [0.997, 1.0, 1.0, 1.0, 0.941, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.897, 1.0, 1.0, 1.0, 1.0, 1.0, 0.633, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | 51.67 | run_20261109_115508_02bf |
| YOLO26 l | 92.33 | 100.0 | 100.0 | 100.0 | 100.0 | 2.5 | 5/5 | [1.0, 1.0, 1.0, 1.0, 0.941, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.922, 1.0, 1.0, 1.0, 1.0, 1.0, 0.653, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | 71.1 | run_20261109_120150_fb97 |

## shots_check_480p30.mp4 (GT shots_check_480p30_gt.csv)

| pose | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s | phases | subject coverage | pose fps | run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26 x | 30.04 | 100.0 | 40.0 | 40.0 | 40.0 | 1.65 | 5/5 | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.952] | 51.38 | run_20261109_115904_6e30 |
| YOLO26 l | 30.04 | 100.0 | 40.0 | 40.0 | 40.0 | 1.65 | 5/5 | [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.887] | 70.89 | run_20261109_120451_888f |
