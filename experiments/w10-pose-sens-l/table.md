# w10-pose-sens-l: pose sensitivity: Engine A with YOLO26 l poses, reference profile, steps GT

Source public, reference none, commit 140110d3101402c6594812ed985696dc96a22f89, GPU NVIDIA RTX A5000, date 2026-11-09T12:05:24Z.

## process_480p30.mp4 (1 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 15.50 |
| reference | 15.49 |
| pose_body | 137.56 |
| tracking | 7.54 |
| segmentation | 0.30 |
| metrics | 0.24 |
| eval | 0.00 |
| full run (process wall) | 179.99 (min 179.99, max 179.99) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 71.1 |
| inference fps (model only) | 89.4 |
| mean GPU util, run / pose stage (%) | 55.0 / 66.5 |
| peak GPU memory, nvidia-smi / torch (MB) | 617.0 / 302.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 3.69 |
| infer_call | 130.69 |
| preprocess | 8.80 |
| inference | 109.34 |
| postprocess | 10.02 |
| extract | 1.47 |
| writing | 0.64 |

Repeat consistency against repeat 1: n/a.

Engine A eval against process_480p30_gt.csv: frame accuracy 92.33, edit 100.0, F1@10 100.0, F1@25 100.0, F1@50 100.0, boundary MAE 2.5 s.

## shots_check_480p30.mp4 (1 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.17 |
| shots | 3.41 |
| reference | 4.16 |
| pose_body | 21.43 |
| tracking | 1.27 |
| segmentation | 0.08 |
| metrics | 0.06 |
| eval | 0.00 |
| full run (process wall) | 33.78 (min 33.78, max 33.78) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 70.9 |
| inference fps (model only) | 91.0 |
| mean GPU util, run / pose stage (%) | 49.5 / 64.7 |
| peak GPU memory, nvidia-smi / torch (MB) | 617.0 / 302.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.59 |
| infer_call | 19.93 |
| preprocess | 1.36 |
| inference | 16.68 |
| postprocess | 1.51 |
| extract | 0.22 |
| writing | 0.23 |

Repeat consistency against repeat 1: n/a.

Engine A eval against shots_check_480p30_gt.csv: frame accuracy 30.04, edit 100.0, F1@10 40.0, F1@25 40.0, F1@50 40.0, boundary MAE 1.65 s.
