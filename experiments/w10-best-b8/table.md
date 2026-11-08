# w10-best-b8: proposal (speed_proposed.toml) at pose batch 8, FP32

Source public, reference w10-ref, commit 7cb09bd3771988a3de49c02d1216db79dae624cf, GPU NVIDIA RTX A5000, date 2026-11-07T06:36:47Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 15.42 |
| reference | 0.50 |
| pose_body | 90.26 |
| tracking | 0.12 |
| segmentation | 0.31 |
| metrics | 0.25 |
| overlay | 24.23 |
| full run (process wall) | 134.70 (min 134.56, max 136.47) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 108.4 |
| inference fps (model only) | 142.9 |
| mean GPU util, run / pose stage (%) | 54.9 / 78.7 |
| peak GPU memory, nvidia-smi / torch (MB) | 2793.0 / 1153.5 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.04 |
| infer_call | 87.22 |
| preprocess | 10.16 |
| inference | 68.42 |
| postprocess | 7.48 |
| extract | 1.79 |
| writing | 0.74 |

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

Within numerical noise: **no** (measures not_within_1px_share 0.023898, track_changes_per_frame 0.006578, metric_max_rel_delta 0.0744; floor not_within_1px_share 0.016835, track_changes_per_frame 0.003367, metric_max_rel_delta 0.0301).
- not_within_1px_share 0.023898 above the floor 0.016835
- track_changes_per_frame 0.006578 above the floor 0.003367
- metric_max_rel_delta 0.0744 above the floor 0.0301
- above the floor: A-0010 idle_share 0.219586 to 0.237226 (+0.01764, 7.4 %)
- above the floor: A-0010 idle_time_s 12.0333 to 13 (+0.9667, 7.4 %)

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.34 |
| shots | 9.66 |
| reference | 0.51 |
| pose_body | 19.02 |
| tracking | 2.76 |
| segmentation | 0.10 |
| metrics | 0.03 |
| overlay | 14.23 |
| full run (process wall) | 50.24 (min 49.99, max 50.58) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 94.3 |
| inference fps (model only) | 145.3 |
| mean GPU util, run / pose stage (%) | 28.8 / 67.2 |
| peak GPU memory, nvidia-smi / torch (MB) | 2932.0 / 1153.5 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.07 |
| infer_call | 18.37 |
| preprocess | 3.74 |
| inference | 12.27 |
| postprocess | 1.45 |
| extract | 0.35 |
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

Within numerical noise: **no** (measures not_within_1px_share 0.075196, track_changes_per_frame 0.004489, metric_max_rel_delta 0.0009; floor not_within_1px_share 0.016835, track_changes_per_frame 0.003367, metric_max_rel_delta 0.0301).
- not_within_1px_share 0.075196 above the floor 0.016835
- track_changes_per_frame 0.004489 above the floor 0.003367
