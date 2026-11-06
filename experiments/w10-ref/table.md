# w10-ref: reference: YOLO26 x, batch 1, FP32, current defaults

Source public, reference none, commit 50a7f2b95fbcef36439bcdbf144eaab0148ffa71, GPU NVIDIA RTX A5000, date 2026-11-06T09:47:29Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 15.39 |
| reference | 16.54 |
| pose_body | 149.81 |
| tracking | 7.32 |
| segmentation | 0.32 |
| metrics | 0.25 |
| overlay | 42.90 |
| full run (process wall) | 235.72 (min 232.57, max 237.03) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 65.3 |
| inference fps (model only) | 80.6 |
| mean GPU util, run / pose stage (%) | 39.9 / 58.1 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 3.55 |
| infer_call | 142.89 |
| preprocess | 8.76 |
| inference | 121.27 |
| postprocess | 10.13 |
| extract | 1.67 |
| writing | 0.58 |

Repeat consistency against repeat 1: equivalent, equivalent.

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.35 |
| shots | 9.67 |
| reference | 1.88 |
| pose_body | 29.64 |
| tracking | 1.65 |
| segmentation | 0.10 |
| metrics | 0.03 |
| overlay | 25.14 |
| full run (process wall) | 71.69 (min 71.24, max 71.83) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 60.3 |
| inference fps (model only) | 80.8 |
| mean GPU util, run / pose stage (%) | 22.6 / 52.3 |
| peak GPU memory, nvidia-smi / torch (MB) | 981.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 2.48 |
| infer_call | 26.62 |
| preprocess | 2.14 |
| inference | 22.07 |
| postprocess | 1.91 |
| extract | 0.33 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.
