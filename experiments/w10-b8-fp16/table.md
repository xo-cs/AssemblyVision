# w10-b8-fp16: pose batch 8, FP16

Source public, reference w10-ref, commit 50a7f2b95fbcef36439bcdbf144eaab0148ffa71, GPU NVIDIA RTX A5000, date 2026-11-06T10:26:28Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 15.39 |
| reference | 17.90 |
| pose_body | 68.62 |
| tracking | 7.16 |
| segmentation | 0.31 |
| metrics | 0.27 |
| overlay | 42.53 |
| full run (process wall) | 156.38 (min 156.21, max 156.42) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 142.6 |
| inference fps (model only) | 213.5 |
| mean GPU util, run / pose stage (%) | 32.1 / 66.0 |
| peak GPU memory, nvidia-smi / torch (MB) | 1455.0 / 716.9 |

| pose breakdown | median s |
| --- | --- |
| decode | 3.97 |
| infer_call | 62.22 |
| preprocess | 8.19 |
| inference | 45.81 |
| postprocess | 7.64 |
| extract | 1.25 |
| writing | 0.88 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **different** (frames within 1 px 0.8854, identical 0.0000, box max 68.5849 px, keypoint max 91.6597 px, keypoint mean 0.08366 px, conf max 0.2673 / 0.5872).
- pose: frame 222 has 3 detections in the reference, 4 in the candidate
- pose: frame 333 has 5 detections in the reference, 4 in the candidate
- pose: frame 499 has 4 detections in the reference, 3 in the candidate
- pose: frame 658 has 4 detections in the reference, 5 in the candidate
- tracking: 3 detections of shot s002 have another track assignment
- tracking: shot s002 duplicates_suppressed 242 vs 243
- pose: frame 763 has 9 detections in the reference, 8 in the candidate
- pose: frame 776 has 8 detections in the reference, 7 in the candidate
- pose: frame 786 has 8 detections in the reference, 7 in the candidate
- pose: frame 826 has 8 detections in the reference, 7 in the candidate
- pose: frame 827 has 8 detections in the reference, 7 in the candidate
- pose: frame 844 has 10 detections in the reference, 9 in the candidate
- pose: frame 931 has 8 detections in the reference, 7 in the candidate
- pose: frame 966 has 7 detections in the reference, 8 in the candidate
- pose: frame 1024 has 9 detections in the reference, 10 in the candidate
- pose: frame 1059 has 8 detections in the reference, 7 in the candidate
- pose: frame 1074 has 7 detections in the reference, 6 in the candidate
- pose: frame 1162 has 7 detections in the reference, 8 in the candidate
- tracking: 9 detections of shot s003 have another track assignment
- tracking: shot s003 duplicates_suppressed 768 vs 763
- tracking: 1 detections of shot s004 have another track assignment
- tracking: shot s004 duplicates_suppressed 2 vs 1
- tracking: 16 detections of shot s006 have another track assignment
- tracking: 3 detections of shot s010 have another track assignment
- tracking: shot s010 duplicates_suppressed 216 vs 215
- tracking: 1 detections of shot s011 have another track assignment
- tracking: 4 detections of shot s012 have another track assignment
- tracking: shot s015 duplicates_suppressed 9 vs 10
- tracking: 10 detections of shot s018 have another track assignment
- tracking: shot s018 short_tracks 8 vs 7
- tracking: 3 detections of shot s019 have another track assignment
- tracking: shot s019 duplicates_suppressed 492 vs 494
- tracking: 13 detections of shot s021 have another track assignment
- tracking: shot s021 duplicates_suppressed 774 vs 775
- tracking: 2 detections of shot s023 have another track assignment
- tracking: 10 detections of shot s026 have another track assignment
- tracking: shot s028 duplicates_suppressed 233 vs 237
- tracking: shot s029 duplicates_suppressed 9 vs 10
- pose: 1045 frames not within 1 px (box max 68.5849 px, keypoint max 91.6597 px)
- pose: confidence differs by up to 0.5872
- reference: shot s023 boxes differ by more than 1 px
- metrics: A-0002 idle_share 0.204313 vs 0.187062
- metrics: A-0002 idle_time_s 12.6333 vs 11.5667
- metrics: A-0006 hand_travel_l 889.044 vs 912.143
- metrics: A-0006 hand_travel_l_torso 6.83463 vs 7.17358
- metrics: A-0006 idle_share 0.322567 vs 0.298768
- metrics: A-0006 idle_time_s 25.3 vs 23.4333
- metrics: A-0006 repetition_count 14.6258 vs 14.9428
- metrics: A-0006 repetition_period_s 0.683725 vs 0.669216
- metrics: A-0010 idle_share 0.219586 vs 0.237226
- metrics: A-0010 idle_time_s 12.0333 vs 13

Within numerical noise: **no** (measures not_within_1px_share 0.114558, track_changes_per_frame 0.009318, metric_max_rel_delta 0.0844; floor not_within_1px_share 0.016835, track_changes_per_frame 0.003367, metric_max_rel_delta 0.0301).
- not_within_1px_share 0.114558 above the floor 0.016835
- track_changes_per_frame 0.009318 above the floor 0.003367
- metric_max_rel_delta 0.0844 above the floor 0.0301
- above the floor: A-0002 idle_share 0.204313 to 0.187062 (-0.017251, 8.4 %)
- above the floor: A-0002 idle_time_s 12.6333 to 11.5667 (-1.0666, 8.4 %)
- above the floor: A-0010 idle_share 0.219586 to 0.237226 (+0.01764, 7.4 %)
- above the floor: A-0010 idle_time_s 12.0333 to 13 (+0.9667, 7.4 %)
- above the floor: A-0006 idle_share 0.322567 to 0.298768 (-0.023799, 7.4 %)
- above the floor: A-0006 idle_time_s 25.3 to 23.4333 (-1.8667, 7.4 %)
- above the floor: A-0006 hand_travel_l_torso 6.83463 to 7.17358 (+0.33895, 4.7 %)

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.35 |
| shots | 10.01 |
| reference | 3.50 |
| pose_body | 19.19 |
| tracking | 1.72 |
| segmentation | 0.11 |
| metrics | 0.03 |
| overlay | 25.27 |
| full run (process wall) | 63.53 (min 63.09, max 63.84) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 94.0 |
| inference fps (model only) | 164.9 |
| mean GPU util, run / pose stage (%) | 13.2 / 41.1 |
| peak GPU memory, nvidia-smi / torch (MB) | 1455.0 / 716.9 |

| pose breakdown | median s |
| --- | --- |
| decode | 4.08 |
| infer_call | 14.50 |
| preprocess | 2.15 |
| inference | 10.81 |
| postprocess | 1.44 |
| extract | 0.24 |
| writing | 0.22 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **different** (frames within 1 px 0.7374, identical 0.0000, box max 67.9955 px, keypoint max 67.826 px, keypoint mean 0.15918 px, conf max 0.0966 / 0.2981).
- pose: frame 197 has 6 detections in the reference, 7 in the candidate
- pose: frame 414 has 3 detections in the reference, 4 in the candidate
- pose: frame 600 has 8 detections in the reference, 7 in the candidate
- pose: frame 652 has 7 detections in the reference, 6 in the candidate
- pose: frame 1353 has 8 detections in the reference, 7 in the candidate
- pose: frame 1369 has 6 detections in the reference, 7 in the candidate
- pose: frame 1392 has 5 detections in the reference, 6 in the candidate
- pose: frame 1693 has 4 detections in the reference, 5 in the candidate
- tracking: 2025 detections of shot s001 have another track assignment
- tracking: shot s001 tracks_kept 37 vs 38
- tracking: shot s001 short_tracks 31 vs 32
- pose: 468 frames not within 1 px (box max 67.9955 px, keypoint max 67.826 px)
- pose: confidence differs by up to 0.2981

Within numerical noise: **no** (measures not_within_1px_share 0.262626, track_changes_per_frame 1.137486, metric_max_rel_delta 0.0033; floor not_within_1px_share 0.016835, track_changes_per_frame 0.003367, metric_max_rel_delta 0.0301).
- not_within_1px_share 0.262626 above the floor 0.016835
- track_changes_per_frame 1.137486 above the floor 0.003367
