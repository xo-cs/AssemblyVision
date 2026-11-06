# w10-ovl-nvenc: overlay encoder h264_nvenc

Source public, reference w10-ref, commit 3ae5b9806ca1c8ca8196860ae2c9d3505d36ac4e, GPU NVIDIA RTX A5000, date 2026-11-06T08:41:37Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 15.34 |
| reference | 16.20 |
| pose_body | 146.31 |
| tracking | 7.21 |
| segmentation | 0.31 |
| metrics | 0.24 |
| overlay | 32.47 |
| full run (process wall) | 223.04 (min 217.15, max 224.01) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 66.8 |
| inference fps (model only) | 82.4 |
| mean GPU util, run / pose stage (%) | 42.7 / 59.6 |
| peak GPU memory, nvidia-smi / torch (MB) | 1206.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 3.58 |
| infer_call | 139.72 |
| preprocess | 8.66 |
| inference | 118.67 |
| postprocess | 9.89 |
| extract | 1.64 |
| writing | 0.59 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.34 |
| shots | 10.02 |
| reference | 1.65 |
| pose_body | 28.77 |
| tracking | 1.63 |
| segmentation | 0.09 |
| metrics | 0.03 |
| overlay | 18.36 |
| full run (process wall) | 64.23 (min 63.47, max 64.47) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 62.1 |
| inference fps (model only) | 83.4 |
| mean GPU util, run / pose stage (%) | 26.5 / 54.3 |
| peak GPU memory, nvidia-smi / torch (MB) | 1241.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 2.43 |
| infer_call | 25.80 |
| preprocess | 2.08 |
| inference | 21.38 |
| postprocess | 1.84 |
| extract | 0.32 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **equivalent** (frames within 1 px 1.0000, identical 1.0000, box max 0.0 px, keypoint max 0.0 px, keypoint mean 0.0 px, conf max 0.0 / 0.0).
