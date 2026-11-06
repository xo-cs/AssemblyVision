# w10-b8: pose batch 8, FP32

Source public, reference w10-ref, commit 50a7f2b95fbcef36439bcdbf144eaab0148ffa71, GPU NVIDIA RTX A5000, date 2026-11-06T09:59:37Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.39 |
| shots | 16.25 |
| reference | 16.14 |
| pose_body | 91.23 |
| tracking | 7.31 |
| segmentation | 0.31 |
| metrics | 0.25 |
| overlay | 41.79 |
| full run (process wall) | 176.70 (min 176.34, max 176.97) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 107.2 |
| inference fps (model only) | 142.9 |
| mean GPU util, run / pose stage (%) | 44.3 / 77.6 |
| peak GPU memory, nvidia-smi / torch (MB) | 1799.0 / 1153.5 |

| pose breakdown | median s |
| --- | --- |
| decode | 4.03 |
| infer_call | 84.91 |
| preprocess | 8.28 |
| inference | 68.43 |
| postprocess | 7.68 |
| extract | 1.28 |
| writing | 0.57 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **different** (frames within 1 px 0.9761, identical 0.0000, box max 69.1854 px, keypoint max 51.7557 px, keypoint mean 0.01859 px, conf max 0.2661 / 0.4054).
- pose: frame 333 has 5 detections in the reference, 4 in the candidate
- pose: frame 650 has 5 detections in the reference, 4 in the candidate
- tracking: 2 detections of shot s002 have another track assignment
- tracking: shot s002 duplicates_suppressed 242 vs 241
- pose: frame 763 has 9 detections in the reference, 8 in the candidate
- pose: frame 852 has 7 detections in the reference, 6 in the candidate
- pose: frame 966 has 7 detections in the reference, 8 in the candidate
- pose: frame 1074 has 7 detections in the reference, 6 in the candidate
- pose: frame 1162 has 7 detections in the reference, 8 in the candidate
- tracking: 3 detections of shot s003 have another track assignment
- pose: frame 2371 has 5 detections in the reference, 6 in the candidate
- pose: frame 2399 has 6 detections in the reference, 7 in the candidate
- pose: frame 2453 has 6 detections in the reference, 5 in the candidate
- pose: frame 2495 has 5 detections in the reference, 6 in the candidate
- pose: frame 2556 has 9 detections in the reference, 10 in the candidate
- pose: frame 2596 has 7 detections in the reference, 8 in the candidate
- tracking: 12 detections of shot s006 have another track assignment
- tracking: shot s006 duplicates_suppressed 403 vs 405
- pose: frame 2974 has 3 detections in the reference, 2 in the candidate
- tracking: 1 detections of shot s010 have another track assignment
- tracking: shot s010 duplicates_suppressed 216 vs 215
- tracking: 1 detections of shot s011 have another track assignment
- tracking: 4 detections of shot s012 have another track assignment
- tracking: shot s012 duplicates_suppressed 125 vs 126
- tracking: 7 detections of shot s018 have another track assignment
- tracking: shot s018 short_tracks 8 vs 7
- tracking: shot s018 duplicates_suppressed 337 vs 338
- tracking: 1 detections of shot s019 have another track assignment
- tracking: 8 detections of shot s021 have another track assignment
- tracking: shot s021 duplicates_suppressed 774 vs 772
- tracking: 1 detections of shot s023 have another track assignment
- tracking: 4 detections of shot s024 have another track assignment
- tracking: 7 detections of shot s026 have another track assignment
- tracking: shot s026 duplicates_suppressed 256 vs 255
- tracking: 1 detections of shot s027 have another track assignment
- pose: 218 frames not within 1 px (box max 69.1854 px, keypoint max 51.7557 px)
- pose: confidence differs by up to 0.4054
- metrics: A-0006 hand_travel_l 889.044 vs 898.859
- metrics: A-0006 hand_travel_l_torso 6.83463 vs 6.99917
- metrics: A-0010 idle_share 0.219586 vs 0.237226
- metrics: A-0010 idle_time_s 12.0333 vs 13

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.35 |
| shots | 10.14 |
| reference | 1.63 |
| pose_body | 21.02 |
| tracking | 1.67 |
| segmentation | 0.11 |
| metrics | 0.03 |
| overlay | 25.48 |
| full run (process wall) | 63.76 (min 63.66, max 64.12) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 85.2 |
| inference fps (model only) | 145.2 |
| mean GPU util, run / pose stage (%) | 21.4 / 60.5 |
| peak GPU memory, nvidia-smi / torch (MB) | 1799.0 / 1153.5 |

| pose breakdown | median s |
| --- | --- |
| decode | 4.22 |
| infer_call | 16.20 |
| preprocess | 2.34 |
| inference | 12.27 |
| postprocess | 1.43 |
| extract | 0.24 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **different** (frames within 1 px 0.9248, identical 0.0000, box max 67.9321 px, keypoint max 48.8661 px, keypoint mean 0.04826 px, conf max 0.0964 / 0.2572).
- pose: frame 202 has 5 detections in the reference, 6 in the candidate
- pose: frame 624 has 7 detections in the reference, 6 in the candidate
- pose: frame 1016 has 9 detections in the reference, 10 in the candidate
- pose: frame 1157 has 7 detections in the reference, 6 in the candidate
- pose: frame 1192 has 6 detections in the reference, 5 in the candidate
- tracking: 7 detections of shot s001 have another track assignment
- tracking: shot s001 duplicates_suppressed 1054 vs 1055
- pose: 134 frames not within 1 px (box max 67.9321 px, keypoint max 48.8661 px)
- pose: confidence differs by up to 0.2572
