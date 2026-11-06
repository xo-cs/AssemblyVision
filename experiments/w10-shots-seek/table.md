# w10-shots-seek: shots motion pairs read by seeking

Source public, reference w10-ref, commit 50a7f2b95fbcef36439bcdbf144eaab0148ffa71, GPU NVIDIA RTX A5000, date 2026-11-06T10:15:23Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 15.34 |
| reference | 16.28 |
| pose_body | 148.82 |
| tracking | 7.10 |
| segmentation | 0.31 |
| metrics | 0.25 |
| overlay | 42.95 |
| full run (process wall) | 234.36 (min 232.31, max 237.56) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 65.7 |
| inference fps (model only) | 81.2 |
| mean GPU util, run / pose stage (%) | 40.6 / 58.8 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 3.57 |
| infer_call | 141.95 |
| preprocess | 8.79 |
| inference | 120.42 |
| postprocess | 10.12 |
| extract | 1.66 |
| writing | 0.59 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.35 |
| shots | 12.43 |
| reference | 1.79 |
| pose_body | 30.05 |
| tracking | 1.65 |
| segmentation | 0.10 |
| metrics | 0.03 |
| overlay | 25.04 |
| full run (process wall) | 75.20 (min 74.40, max 75.22) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 59.5 |
| inference fps (model only) | 79.4 |
| mean GPU util, run / pose stage (%) | 21.8 / 51.2 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 2.44 |
| infer_call | 27.04 |
| preprocess | 2.13 |
| inference | 22.45 |
| postprocess | 1.95 |
| extract | 0.33 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).
