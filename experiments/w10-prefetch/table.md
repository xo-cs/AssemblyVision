# w10-prefetch: threaded prefetch reader (batch 1)

Source public, reference w10-ref, commit 3ae5b9806ca1c8ca8196860ae2c9d3505d36ac4e, GPU NVIDIA RTX A5000, date 2026-11-06T08:00:16Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 15.31 |
| reference | 16.22 |
| pose_body | 142.13 |
| tracking | 7.15 |
| segmentation | 0.31 |
| metrics | 0.25 |
| overlay | 42.41 |
| full run (process wall) | 227.88 (min 226.70, max 227.89) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 68.8 |
| inference fps (model only) | 84.1 |
| mean GPU util, run / pose stage (%) | 41.5 / 61.6 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.11 |
| infer_call | 138.97 |
| preprocess | 10.63 |
| inference | 116.23 |
| postprocess | 9.68 |
| extract | 1.62 |
| writing | 0.68 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.35 |
| shots | 9.59 |
| reference | 1.80 |
| pose_body | 28.77 |
| tracking | 1.68 |
| segmentation | 0.10 |
| metrics | 0.03 |
| overlay | 24.64 |
| full run (process wall) | 70.47 (min 70.28, max 70.73) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 62.2 |
| inference fps (model only) | 79.9 |
| mean GPU util, run / pose stage (%) | 23.2 / 53.4 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.07 |
| infer_call | 28.13 |
| preprocess | 3.46 |
| inference | 22.30 |
| postprocess | 1.91 |
| extract | 0.32 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).
