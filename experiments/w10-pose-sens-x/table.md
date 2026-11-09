# w10-pose-sens-x: pose sensitivity: Engine A with YOLO26 x poses, reference profile, steps GT

Source public, reference none, commit 140110d3101402c6594812ed985696dc96a22f89, GPU NVIDIA RTX A5000, date 2026-11-09T11:59:47Z.

## process_480p30.mp4 (1 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 16.19 |
| reference | 19.75 |
| pose_body | 189.28 |
| tracking | 7.02 |
| segmentation | 0.32 |
| metrics | 0.25 |
| eval | 0.00 |
| full run (process wall) | 236.74 (min 236.74, max 236.74) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 51.7 |
| inference fps (model only) | 61.0 |
| mean GPU util, run / pose stage (%) | 73.4 / 84.9 |
| peak GPU memory, nvidia-smi / torch (MB) | 913.0 / 582.7 |

| pose breakdown | median s |
| --- | --- |
| decode | 3.66 |
| infer_call | 181.99 |
| preprocess | 8.83 |
| inference | 160.35 |
| postprocess | 10.18 |
| extract | 1.48 |
| writing | 0.73 |

Repeat consistency against repeat 1: n/a.

Engine A eval against process_480p30_gt.csv: frame accuracy 92.33, edit 100.0, F1@10 100.0, F1@25 100.0, F1@50 100.0, boundary MAE 2.5 s.

## shots_check_480p30.mp4 (1 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.18 |
| shots | 3.34 |
| reference | 4.92 |
| pose_body | 29.56 |
| tracking | 1.15 |
| segmentation | 0.08 |
| metrics | 0.06 |
| eval | 0.00 |
| full run (process wall) | 42.94 (min 42.94, max 42.94) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 51.4 |
| inference fps (model only) | 61.1 |
| mean GPU util, run / pose stage (%) | 68.7 / 84.6 |
| peak GPU memory, nvidia-smi / torch (MB) | 913.0 / 582.7 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.62 |
| infer_call | 28.31 |
| preprocess | 1.38 |
| inference | 24.86 |
| postprocess | 1.66 |
| extract | 0.24 |
| writing | 0.14 |

Repeat consistency against repeat 1: n/a.

Engine A eval against shots_check_480p30_gt.csv: frame accuracy 30.04, edit 100.0, F1@10 40.0, F1@25 40.0, F1@50 40.0, boundary MAE 1.65 s.
