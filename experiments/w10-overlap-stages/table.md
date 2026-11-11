# w10-overlap-stages: pre and postprocessing in threads around inference (pose.overlap_stages), default pose profile

Source public, reference w10-shots-par, commit 6debf98415dd64e2ea1822e6514158d4e8d261a6, GPU NVIDIA RTX A5000, date 2026-11-11T02:16:59Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.39 |
| shots | 22.38 |
| reference | 0.86 |
| pose_body | 83.26 |
| tracking | 0.00 |
| segmentation | 0.33 |
| metrics | 0.25 |
| overlay | 25.97 |
| full run (process wall) | 114.56 (min 114.46, max 115.52) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 117.8 |
| inference fps (model only) | 135.3 |
| mean GPU util, run / pose stage (%) | 65.9 / 86.8 |
| peak GPU memory, nvidia-smi / torch (MB) | 3345.0 / 1652.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.09 |
| infer_call | 73.70 |
| preprocess | 38.22 |
| inference | 72.27 |
| postprocess | 75.20 |
| extract | 2.07 |
| writing | 0.85 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-shots-par: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.35 |
| shots | 16.45 |
| reference | 1.13 |
| pose_body | 19.40 |
| tracking | 2.60 |
| segmentation | 0.10 |
| metrics | 0.03 |
| overlay | 14.91 |
| full run (process wall) | 42.21 (min 41.63, max 43.27) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 93.2 |
| inference fps (model only) | 115.4 |
| mean GPU util, run / pose stage (%) | 34.8 / 67.9 |
| peak GPU memory, nvidia-smi / torch (MB) | 3484.0 / 1653.6 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.30 |
| infer_call | 17.23 |
| preprocess | 8.73 |
| inference | 15.44 |
| postprocess | 14.93 |
| extract | 1.12 |
| writing | 0.10 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-shots-par: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).
