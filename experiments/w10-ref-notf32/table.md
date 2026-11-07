# w10-ref-notf32: reference in strict FP32 (TF32 off), batch 1

Source public, reference w10-ref, commit 7cb09bd3771988a3de49c02d1216db79dae624cf, GPU NVIDIA RTX A5000, date 2026-11-07T06:09:03Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.40 |
| shots | 15.38 |
| reference | 18.83 |
| pose_body | 188.63 |
| tracking | 7.07 |
| segmentation | 0.31 |
| metrics | 0.25 |
| overlay | 42.34 |
| full run (process wall) | 276.87 (min 274.42, max 278.93) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 51.9 |
| inference fps (model only) | 61.0 |
| mean GPU util, run / pose stage (%) | 62.6 / 85.0 |
| peak GPU memory, nvidia-smi / torch (MB) | 913.0 / 582.7 |

| pose breakdown | median s |
| --- | --- |
| decode | 3.59 |
| infer_call | 181.81 |
| preprocess | 8.71 |
| inference | 160.25 |
| postprocess | 10.16 |
| extract | 1.65 |
| writing | 0.65 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **different** (frames within 1 px 0.9887, identical 0.0000, box max 43.3263 px, keypoint max 26.6921 px, keypoint mean 0.01081 px, conf max 0.1805 / 0.6739).
- tracking: 1 detections of shot s002 have another track assignment
- pose: frame 883 has 9 detections in the reference, 10 in the candidate
- tracking: shot s003 duplicates_suppressed 768 vs 769
- pose: frame 2382 has 5 detections in the reference, 4 in the candidate
- tracking: 8 detections of shot s006 have another track assignment
- pose: frame 4277 has 4 detections in the reference, 5 in the candidate
- pose: frame 4306 has 5 detections in the reference, 6 in the candidate
- tracking: shot s015 duplicates_suppressed 9 vs 10
- pose: frame 4671 has 6 detections in the reference, 5 in the candidate
- tracking: 1 detections of shot s018 have another track assignment
- tracking: shot s018 duplicates_suppressed 337 vs 336
- pose: frame 6210 has 7 detections in the reference, 8 in the candidate
- pose: frame 6499 has 8 detections in the reference, 7 in the candidate
- tracking: 3 detections of shot s021 have another track assignment
- pose: frame 8198 has 3 detections in the reference, 2 in the candidate
- pose: frame 8210 has 3 detections in the reference, 2 in the candidate
- tracking: 2 detections of shot s026 have another track assignment
- tracking: shot s026 duplicates_suppressed 256 vs 255
- pose: frame 9301 has 5 detections in the reference, 6 in the candidate
- tracking: shot s029 duplicates_suppressed 9 vs 10
- pose: 103 frames not within 1 px (box max 43.3263 px, keypoint max 26.6921 px)
- pose: confidence differs by up to 0.6739
- metrics: A-0006 repetition_count 14.6258 vs 15.08
- metrics: A-0006 repetition_period_s 0.683725 vs 0.663132
- metrics: A-0010 hand_travel_l 376.009 vs 371.662
- metrics: A-0010 hand_travel_l_torso 2.99787 vs 2.94857

Within numerical noise: **yes** (measures not_within_1px_share 0.011291, track_changes_per_frame 0.002193, metric_max_rel_delta 0.0301; floor not_within_1px_share 0.016835, track_changes_per_frame 0.003367, metric_max_rel_delta 0.0301).

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.35 |
| shots | 9.75 |
| reference | 1.79 |
| pose_body | 36.07 |
| tracking | 1.65 |
| segmentation | 0.10 |
| metrics | 0.03 |
| overlay | 25.59 |
| full run (process wall) | 78.54 (min 78.26, max 79.57) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 49.5 |
| inference fps (model only) | 62.9 |
| mean GPU util, run / pose stage (%) | 37.1 / 77.8 |
| peak GPU memory, nvidia-smi / torch (MB) | 913.0 / 582.7 |

| pose breakdown | median s |
| --- | --- |
| decode | 2.49 |
| infer_call | 33.03 |
| preprocess | 2.18 |
| inference | 28.34 |
| postprocess | 2.00 |
| extract | 0.34 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **different** (frames within 1 px 0.9832, identical 0.0000, box max 17.5217 px, keypoint max 106.7459 px, keypoint mean 0.02494 px, conf max 0.0086 / 0.1226).
- pose: frame 1541 has 4 detections in the reference, 3 in the candidate
- pose: frame 1707 has 4 detections in the reference, 5 in the candidate
- tracking: 5 detections of shot s001 have another track assignment
- tracking: shot s001 duplicates_suppressed 1054 vs 1053
- pose: 30 frames not within 1 px (box max 17.5217 px, keypoint max 106.7459 px)
- pose: confidence differs by up to 0.1226

Within numerical noise: **yes** (measures not_within_1px_share 0.016835, track_changes_per_frame 0.003367, metric_max_rel_delta 0.0012; floor not_within_1px_share 0.016835, track_changes_per_frame 0.003367, metric_max_rel_delta 0.0301).
