# w10-ovl-off: overlay off

Source public, reference w10-ref, commit 3ae5b9806ca1c8ca8196860ae2c9d3505d36ac4e, GPU NVIDIA RTX A5000, date 2026-11-06T08:27:15Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 15.35 |
| reference | 15.80 |
| pose_body | 143.54 |
| tracking | 7.24 |
| segmentation | 0.33 |
| metrics | 0.24 |
| full run (process wall) | 186.05 (min 185.88, max 190.16) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 68.1 |
| inference fps (model only) | 84.3 |
| mean GPU util, run / pose stage (%) | 50.9 / 60.6 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 3.58 |
| infer_call | 136.78 |
| preprocess | 8.64 |
| inference | 115.97 |
| postprocess | 9.68 |
| extract | 1.60 |
| writing | 0.63 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.34 |
| shots | 9.66 |
| reference | 1.77 |
| pose_body | 29.19 |
| tracking | 1.69 |
| segmentation | 0.09 |
| metrics | 0.03 |
| full run (process wall) | 46.22 (min 45.83, max 46.35) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 61.2 |
| inference fps (model only) | 82.3 |
| mean GPU util, run / pose stage (%) | 36.5 / 53.4 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 2.48 |
| infer_call | 26.08 |
| preprocess | 2.10 |
| inference | 21.65 |
| postprocess | 1.86 |
| extract | 0.32 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).
