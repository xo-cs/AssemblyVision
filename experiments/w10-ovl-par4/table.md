# w10-ovl-par4: overlay in 4 parallel chunks (libx264)

Source public, reference w10-ref, commit 3ae5b9806ca1c8ca8196860ae2c9d3505d36ac4e, GPU NVIDIA RTX A5000, date 2026-11-06T08:56:12Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 15.42 |
| reference | 15.96 |
| pose_body | 144.52 |
| tracking | 7.03 |
| segmentation | 0.31 |
| metrics | 0.24 |
| overlay | 31.73 |
| full run (process wall) | 218.71 (min 218.29, max 220.20) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 67.7 |
| inference fps (model only) | 83.9 |
| mean GPU util, run / pose stage (%) | 43.2 / 60.2 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 3.59 |
| infer_call | 137.45 |
| preprocess | 8.65 |
| inference | 116.56 |
| postprocess | 9.74 |
| extract | 1.61 |
| writing | 0.59 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.34 |
| shots | 9.61 |
| reference | 1.84 |
| pose_body | 29.58 |
| tracking | 1.65 |
| segmentation | 0.09 |
| metrics | 0.03 |
| overlay | 23.83 |
| full run (process wall) | 69.95 (min 69.75, max 71.79) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 60.4 |
| inference fps (model only) | 80.8 |
| mean GPU util, run / pose stage (%) | 23.1 / 52.3 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 2.47 |
| infer_call | 26.57 |
| preprocess | 2.13 |
| inference | 22.06 |
| postprocess | 1.90 |
| extract | 0.32 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).
