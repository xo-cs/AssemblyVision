# w10-refpass: reference frames from the pose pass

Source public, reference w10-ref, commit 3ae5b9806ca1c8ca8196860ae2c9d3505d36ac4e, GPU NVIDIA RTX A5000, date 2026-11-06T09:10:43Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 15.53 |
| reference | 0.53 |
| pose_body | 147.59 |
| tracking | 7.16 |
| segmentation | 0.31 |
| metrics | 0.25 |
| overlay | 42.37 |
| full run (process wall) | 217.91 (min 214.35, max 219.37) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 66.3 |
| inference fps (model only) | 82.6 |
| mean GPU util, run / pose stage (%) | 40.6 / 58.8 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 3.56 |
| infer_call | 140.25 |
| preprocess | 8.69 |
| inference | 118.33 |
| postprocess | 9.77 |
| extract | 2.13 |
| writing | 0.69 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.34 |
| shots | 9.91 |
| reference | 0.51 |
| pose_body | 30.72 |
| tracking | 1.61 |
| segmentation | 0.09 |
| metrics | 0.03 |
| overlay | 24.81 |
| full run (process wall) | 71.66 (min 70.69, max 72.21) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 58.3 |
| inference fps (model only) | 80.2 |
| mean GPU util, run / pose stage (%) | 23.1 / 51.1 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 2.50 |
| infer_call | 27.47 |
| preprocess | 2.12 |
| inference | 22.21 |
| postprocess | 1.92 |
| extract | 0.43 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).
