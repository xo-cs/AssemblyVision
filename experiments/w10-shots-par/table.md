# w10-shots-par: shot detection in parallel with the pose pass (shots.parallel), default profile

Source public, reference w10-best-b8, commit 140110d3101402c6594812ed985696dc96a22f89, GPU NVIDIA RTX A5000, date 2026-11-09T11:50:28Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 18.88 |
| reference | 1.02 |
| pose_body | 93.28 |
| tracking | 0.00 |
| segmentation | 0.31 |
| metrics | 0.28 |
| overlay | 24.40 |
| full run (process wall) | 123.32 (min 122.93, max 123.51) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 105.1 |
| inference fps (model only) | 142.5 |
| mean GPU util, run / pose stage (%) | 60.3 / 76.6 |
| peak GPU memory, nvidia-smi / torch (MB) | 2793.0 / 1153.5 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.05 |
| infer_call | 89.93 |
| preprocess | 10.77 |
| inference | 68.65 |
| postprocess | 8.67 |
| extract | 1.31 |
| writing | 0.65 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-best-b8: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.34 |
| shots | 13.18 |
| reference | 1.20 |
| pose_body | 21.33 |
| tracking | 2.65 |
| segmentation | 0.12 |
| metrics | 0.03 |
| overlay | 14.81 |
| full run (process wall) | 44.18 (min 43.29, max 44.85) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 84.7 |
| inference fps (model only) | 141.5 |
| mean GPU util, run / pose stage (%) | 33.5 / 60.0 |
| peak GPU memory, nvidia-smi / torch (MB) | 2932.0 / 1153.5 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.15 |
| infer_call | 20.42 |
| preprocess | 4.01 |
| inference | 12.60 |
| postprocess | 2.05 |
| extract | 0.31 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-best-b8: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).
