# w10-b32: pose batch 32, FP32

Source public, reference w10-ref, commit 3ae5b9806ca1c8ca8196860ae2c9d3505d36ac4e, GPU NVIDIA RTX A5000, date 2026-11-06T07:45:16Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 15.35 |
| reference | 16.27 |
| pose_body | 91.69 |
| tracking | 7.11 |
| segmentation | 0.31 |
| metrics | 0.24 |
| overlay | 42.25 |
| full run (process wall) | 177.04 (min 176.77, max 177.29) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 106.7 |
| inference fps (model only) | 146.4 |
| mean GPU util, run / pose stage (%) | 43.8 / 76.3 |
| peak GPU memory, nvidia-smi / torch (MB) | 4441.0 / 3111.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 5.21 |
| infer_call | 84.27 |
| preprocess | 10.33 |
| inference | 66.80 |
| postprocess | 7.00 |
| extract | 1.17 |
| writing | 0.56 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **different** (frames within 1 px 0.9743, identical 0.0000, box max 69.0962 px, keypoint max 51.7246 px, keypoint mean 0.01981 px, conf max 0.2661 / 0.6381).
- pose: frame 333 has 5 detections in the reference, 4 in the candidate
- pose: frame 650 has 5 detections in the reference, 4 in the candidate
- tracking: 2 detections of shot s002 have another track assignment
- tracking: shot s002 duplicates_suppressed 242 vs 241
- pose: frame 763 has 9 detections in the reference, 8 in the candidate
- pose: frame 852 has 7 detections in the reference, 6 in the candidate
- pose: frame 1074 has 7 detections in the reference, 6 in the candidate
- pose: frame 1162 has 7 detections in the reference, 8 in the candidate
- tracking: 3 detections of shot s003 have another track assignment
- tracking: shot s003 duplicates_suppressed 768 vs 767
- pose: frame 2101 has 7 detections in the reference, 6 in the candidate
- pose: frame 2371 has 5 detections in the reference, 6 in the candidate
- pose: frame 2399 has 6 detections in the reference, 7 in the candidate
- pose: frame 2453 has 6 detections in the reference, 5 in the candidate
- pose: frame 2495 has 5 detections in the reference, 6 in the candidate
- pose: frame 2556 has 9 detections in the reference, 10 in the candidate
- tracking: 13 detections of shot s006 have another track assignment
- tracking: shot s006 duplicates_suppressed 403 vs 404
- pose: frame 2841 has 4 detections in the reference, 3 in the candidate
- pose: frame 2974 has 3 detections in the reference, 2 in the candidate
- tracking: 2 detections of shot s010 have another track assignment
- tracking: shot s010 duplicates_suppressed 216 vs 214
- tracking: 1 detections of shot s011 have another track assignment
- tracking: 4 detections of shot s012 have another track assignment
- tracking: shot s012 duplicates_suppressed 125 vs 126
- tracking: 7 detections of shot s018 have another track assignment
- tracking: shot s018 short_tracks 8 vs 7
- tracking: shot s018 duplicates_suppressed 337 vs 338
- tracking: 1 detections of shot s019 have another track assignment
- tracking: 13 detections of shot s021 have another track assignment
- tracking: shot s021 duplicates_suppressed 774 vs 772
- tracking: 1 detections of shot s023 have another track assignment
- tracking: 4 detections of shot s024 have another track assignment
- tracking: 7 detections of shot s026 have another track assignment
- tracking: shot s026 duplicates_suppressed 256 vs 255
- tracking: 1 detections of shot s027 have another track assignment
- pose: 234 frames not within 1 px (box max 69.0962 px, keypoint max 51.7246 px)
- pose: confidence differs by up to 0.6381
- metrics: A-0006 hand_travel_l 889.044 vs 898.33
- metrics: A-0006 hand_travel_l_torso 6.83463 vs 6.98759
- metrics: A-0010 idle_share 0.219586 vs 0.236618
- metrics: A-0010 idle_time_s 12.0333 vs 12.9667

Within numerical noise: **no** (measures not_within_1px_share 0.025652, track_changes_per_frame 0.007455, metric_max_rel_delta 0.072; floor not_within_1px_share 0.016835, track_changes_per_frame 0.003367, metric_max_rel_delta 0.0301).
- not_within_1px_share 0.025652 above the floor 0.016835
- track_changes_per_frame 0.007455 above the floor 0.003367
- metric_max_rel_delta 0.072 above the floor 0.0301
- above the floor: A-0010 idle_share 0.219586 to 0.236618 (+0.017032, 7.2 %)
- above the floor: A-0010 idle_time_s 12.0333 to 12.9667 (+0.9334, 7.2 %)

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.34 |
| shots | 9.60 |
| reference | 1.90 |
| pose_body | 21.12 |
| tracking | 1.61 |
| segmentation | 0.09 |
| metrics | 0.03 |
| overlay | 24.90 |
| full run (process wall) | 63.01 (min 62.80, max 63.17) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 84.7 |
| inference fps (model only) | 148.4 |
| mean GPU util, run / pose stage (%) | 22.9 / 63.2 |
| peak GPU memory, nvidia-smi / torch (MB) | 4441.0 / 3111.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 5.07 |
| infer_call | 15.54 |
| preprocess | 2.09 |
| inference | 12.01 |
| postprocess | 1.39 |
| extract | 0.24 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **different** (frames within 1 px 0.9248, identical 0.0000, box max 67.9622 px, keypoint max 36.7505 px, keypoint mean 0.04792 px, conf max 0.1325 / 0.2571).
- pose: frame 197 has 6 detections in the reference, 7 in the candidate
- pose: frame 202 has 5 detections in the reference, 6 in the candidate
- pose: frame 1016 has 9 detections in the reference, 10 in the candidate
- pose: frame 1157 has 7 detections in the reference, 6 in the candidate
- pose: frame 1192 has 6 detections in the reference, 5 in the candidate
- tracking: 6 detections of shot s001 have another track assignment
- tracking: shot s001 duplicates_suppressed 1054 vs 1055
- pose: 134 frames not within 1 px (box max 67.9622 px, keypoint max 36.7505 px)
- pose: confidence differs by up to 0.2571

Within numerical noise: **no** (measures not_within_1px_share 0.075196, track_changes_per_frame 0.003928, metric_max_rel_delta 0.0001; floor not_within_1px_share 0.016835, track_changes_per_frame 0.003367, metric_max_rel_delta 0.0301).
- not_within_1px_share 0.075196 above the floor 0.016835
- track_changes_per_frame 0.003928 above the floor 0.003367
