# w10-shots-ds: cut detection downscale 8 (automatic: 3 at 480p, 7 at 1080p)

Source public, reference w10-ref, commit 7cb09bd3771988a3de49c02d1216db79dae624cf, GPU NVIDIA RTX A5000, date 2026-11-07T06:52:32Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 14.93 |
| reference | 16.38 |
| pose_body | 147.80 |
| tracking | 7.21 |
| segmentation | 0.32 |
| metrics | 0.27 |
| overlay | 42.55 |
| full run (process wall) | 233.31 (min 232.41, max 233.46) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 66.2 |
| inference fps (model only) | 81.6 |
| mean GPU util, run / pose stage (%) | 40.3 / 58.9 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 3.57 |
| infer_call | 141.04 |
| preprocess | 8.67 |
| inference | 119.88 |
| postprocess | 9.96 |
| extract | 1.65 |
| writing | 0.58 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.34 |
| shots | 9.59 |
| reference | 1.83 |
| pose_body | 29.58 |
| tracking | 1.69 |
| segmentation | 0.09 |
| metrics | 0.03 |
| overlay | 24.78 |
| full run (process wall) | 70.97 (min 70.75, max 71.33) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 60.5 |
| inference fps (model only) | 81.7 |
| mean GPU util, run / pose stage (%) | 22.8 / 53.1 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 2.48 |
| infer_call | 26.37 |
| preprocess | 2.16 |
| inference | 21.81 |
| postprocess | 1.88 |
| extract | 0.33 |
| writing | 0.10 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).
