# w10-trackoverlap: tracking in a worker during pose (with reference from the pass)

Source public, reference w10-ref, commit 3ae5b9806ca1c8ca8196860ae2c9d3505d36ac4e, GPU NVIDIA RTX A5000, date 2026-11-06T09:24:58Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.40 |
| shots | 15.32 |
| reference | 0.51 |
| pose_body | 147.44 |
| tracking | 0.14 |
| segmentation | 0.32 |
| metrics | 0.25 |
| overlay | 41.84 |
| full run (process wall) | 209.82 (min 209.13, max 212.71) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 66.3 |
| inference fps (model only) | 82.4 |
| mean GPU util, run / pose stage (%) | 42.2 / 59.1 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 3.60 |
| infer_call | 140.28 |
| preprocess | 8.75 |
| inference | 118.66 |
| postprocess | 9.85 |
| extract | 2.17 |
| writing | 0.58 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.34 |
| shots | 9.60 |
| reference | 0.71 |
| pose_body | 30.12 |
| tracking | 2.76 |
| segmentation | 0.10 |
| metrics | 0.03 |
| overlay | 25.04 |
| full run (process wall) | 72.77 (min 71.40, max 73.05) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 59.4 |
| inference fps (model only) | 81.2 |
| mean GPU util, run / pose stage (%) | 22.6 / 51.8 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 2.44 |
| infer_call | 26.98 |
| preprocess | 2.11 |
| inference | 21.96 |
| postprocess | 1.90 |
| extract | 0.43 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).
