# w10-b8-notf32: pose batch 8, strict FP32 (TF32 off)

Source public, reference w10-ref-notf32, commit 7cb09bd3771988a3de49c02d1216db79dae624cf, GPU NVIDIA RTX A5000, date 2026-11-07T06:27:25Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.39 |
| shots | 15.42 |
| reference | 19.09 |
| pose_body | 195.52 |
| tracking | 7.31 |
| segmentation | 0.31 |
| metrics | 0.26 |
| overlay | 42.31 |
| full run (process wall) | 284.22 (min 282.54, max 284.34) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 50.0 |
| inference fps (model only) | 56.9 |
| mean GPU util, run / pose stage (%) | 66.0 / 89.1 |
| peak GPU memory, nvidia-smi / torch (MB) | 6685.0 / 5822.9 |

| pose breakdown | median s |
| --- | --- |
| decode | 4.17 |
| infer_call | 188.47 |
| preprocess | 8.28 |
| inference | 171.89 |
| postprocess | 7.71 |
| extract | 1.28 |
| writing | 0.61 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref-notf32: **different** (frames within 1 px 0.9999, identical 0.0000, box max 1.3067 px, keypoint max 0.9381 px, keypoint mean 2e-05 px, conf max 0.0 / 0.0567).
- pose: frame 4902 has 6 detections in the reference, 5 in the candidate
- tracking: 1 detections of shot s018 have another track assignment
- pose: 1 frames not within 1 px (box max 1.3067 px, keypoint max 0.9381 px)
- pose: confidence differs by up to 0.0567

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.34 |
| shots | 9.63 |
| reference | 1.94 |
| pose_body | 39.53 |
| tracking | 1.66 |
| segmentation | 0.10 |
| metrics | 0.04 |
| overlay | 24.90 |
| full run (process wall) | 81.76 (min 81.00, max 82.90) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 45.2 |
| inference fps (model only) | 58.2 |
| mean GPU util, run / pose stage (%) | 39.7 / 78.5 |
| peak GPU memory, nvidia-smi / torch (MB) | 6685.0 / 5822.9 |

| pose breakdown | median s |
| --- | --- |
| decode | 4.31 |
| infer_call | 34.54 |
| preprocess | 2.34 |
| inference | 30.60 |
| postprocess | 1.49 |
| extract | 0.25 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref-notf32: **different** (frames within 1 px 0.9994, identical 0.0000, box max 0.7955 px, keypoint max 2.1403 px, keypoint mean 0.00011 px, conf max 0.0 / 0.0069).
- pose: 1 frames not within 1 px (box max 0.7955 px, keypoint max 2.1403 px)
