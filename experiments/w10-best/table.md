# w10-best: equivalent winners combined (batch 1, FP32)

Source public, reference w10-ref, commit 50a7f2b95fbcef36439bcdbf144eaab0148ffa71, GPU NVIDIA RTX A5000, date 2026-11-06T10:39:23Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 15.49 |
| reference | 0.50 |
| pose_body | 147.44 |
| tracking | 0.13 |
| segmentation | 0.31 |
| metrics | 0.25 |
| overlay | 24.59 |
| full run (process wall) | 192.72 (min 191.10, max 194.91) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 66.4 |
| inference fps (model only) | 81.7 |
| mean GPU util, run / pose stage (%) | 46.6 / 58.9 |
| peak GPU memory, nvidia-smi / torch (MB) | 1881.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.12 |
| infer_call | 143.62 |
| preprocess | 10.76 |
| inference | 119.76 |
| postprocess | 10.01 |
| extract | 2.19 |
| writing | 0.72 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.36 |
| shots | 12.54 |
| reference | 0.56 |
| pose_body | 29.67 |
| tracking | 2.80 |
| segmentation | 0.14 |
| metrics | 0.03 |
| overlay | 13.70 |
| full run (process wall) | 63.09 (min 63.02, max 63.63) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 60.3 |
| inference fps (model only) | 80.1 |
| mean GPU util, run / pose stage (%) | 27.0 / 52.5 |
| peak GPU memory, nvidia-smi / torch (MB) | 2020.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.08 |
| infer_call | 28.91 |
| preprocess | 3.50 |
| inference | 22.25 |
| postprocess | 1.91 |
| extract | 0.43 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).
