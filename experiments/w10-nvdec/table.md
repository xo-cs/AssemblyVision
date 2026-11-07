# w10-nvdec: ffmpeg NVDEC reader (batch 1)

Source public, reference w10-ref, commit 3ae5b9806ca1c8ca8196860ae2c9d3505d36ac4e, GPU NVIDIA RTX A5000, date 2026-11-06T08:15:28Z.

## process_480p30.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.38 |
| shots | 15.50 |
| reference | 15.97 |
| pose_body | 144.63 |
| tracking | 7.26 |
| segmentation | 0.34 |
| metrics | 0.25 |
| overlay | 42.31 |
| full run (process wall) | 230.03 (min 228.73, max 230.27) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 67.6 |
| inference fps (model only) | 83.3 |
| mean GPU util, run / pose stage (%) | 42.1 / 61.8 |
| peak GPU memory, nvidia-smi / torch (MB) | 1214.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.34 |
| infer_call | 140.93 |
| preprocess | 11.25 |
| inference | 117.47 |
| postprocess | 9.73 |
| extract | 1.63 |
| writing | 0.60 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **different** (frames within 1 px 0.1273, identical 0.0000, box max 646.7772 px, keypoint max 662.3453 px, keypoint mean 0.54891 px, conf max 0.382 / 0.9891).
- pose: frame 98 has 7 detections in the reference, 6 in the candidate
- pose: frame 99 has 7 detections in the reference, 6 in the candidate
- pose: frame 100 has 5 detections in the reference, 4 in the candidate
- pose: frame 109 has 4 detections in the reference, 3 in the candidate
- pose: frame 120 has 3 detections in the reference, 4 in the candidate
- pose: frame 126 has 5 detections in the reference, 4 in the candidate
- pose: frame 127 has 3 detections in the reference, 4 in the candidate
- pose: frame 128 has 3 detections in the reference, 4 in the candidate
- pose: frame 129 has 3 detections in the reference, 4 in the candidate
- pose: frame 139 has 6 detections in the reference, 5 in the candidate
- pose: frame 184 has 3 detections in the reference, 4 in the candidate
- pose: frame 185 has 3 detections in the reference, 4 in the candidate
- pose: frame 186 has 3 detections in the reference, 4 in the candidate
- pose: frame 220 has 3 detections in the reference, 4 in the candidate
- pose: frame 229 has 3 detections in the reference, 4 in the candidate
- pose: frame 254 has 5 detections in the reference, 4 in the candidate
- pose: frame 255 has 5 detections in the reference, 4 in the candidate
- pose: frame 274 has 5 detections in the reference, 4 in the candidate
- pose: frame 275 has 5 detections in the reference, 4 in the candidate
- pose: frame 281 has 5 detections in the reference, 4 in the candidate
- tracking: 77 detections of shot s002 have another track assignment
- tracking: shot s002 short_tracks 6 vs 4
- tracking: shot s002 duplicates_suppressed 242 vs 235
- tracking: 995 detections of shot s003 have another track assignment
- tracking: shot s003 tracks_kept 13 vs 16
- tracking: shot s003 duplicates_suppressed 768 vs 777
- tracking: 3 detections of shot s004 have another track assignment
- tracking: shot s004 duplicates_suppressed 2 vs 3
- tracking: 520 detections of shot s006 have another track assignment
- tracking: shot s006 tracks_kept 12 vs 13
- tracking: shot s006 short_tracks 5 vs 7
- tracking: shot s006 duplicates_suppressed 403 vs 406
- tracking: 17 detections of shot s010 have another track assignment
- tracking: shot s010 duplicates_suppressed 216 vs 223
- tracking: 12 detections of shot s011 have another track assignment
- tracking: 41 detections of shot s012 have another track assignment
- tracking: shot s012 short_tracks 8 vs 7
- tracking: shot s012 duplicates_suppressed 125 vs 130
- tracking: 15 detections of shot s014 have another track assignment
- tracking: shot s014 duplicates_suppressed 35 vs 36
- tracking: 3 detections of shot s015 have another track assignment
- tracking: shot s015 duplicates_suppressed 9 vs 10
- tracking: 3 detections of shot s017 have another track assignment
- tracking: shot s017 duplicates_suppressed 2 vs 1
- tracking: 760 detections of shot s018 have another track assignment
- tracking: shot s018 tracks_kept 11 vs 13
- tracking: shot s018 duplicates_suppressed 337 vs 327
- tracking: 104 detections of shot s019 have another track assignment
- tracking: shot s019 short_tracks 8 vs 9
- tracking: shot s019 duplicates_suppressed 492 vs 491
- tracking: 762 detections of shot s021 have another track assignment
- tracking: shot s021 tracks_kept 13 vs 14
- tracking: shot s021 short_tracks 13 vs 12
- tracking: shot s021 duplicates_suppressed 774 vs 793
- tracking: 24 detections of shot s023 have another track assignment
- tracking: shot s023 duplicates_suppressed 102 vs 104
- tracking: 13 detections of shot s024 have another track assignment
- tracking: shot s024 duplicates_suppressed 2 vs 3
- tracking: 380 detections of shot s026 have another track assignment
- tracking: shot s026 tracks_kept 14 vs 15
- tracking: shot s026 short_tracks 13 vs 9
- tracking: shot s026 duplicates_suppressed 256 vs 252
- tracking: 5 detections of shot s027 have another track assignment
- tracking: shot s027 duplicates_suppressed 23 vs 25
- tracking: 39 detections of shot s028 have another track assignment
- tracking: shot s028 duplicates_suppressed 233 vs 235
- tracking: 3 detections of shot s029 have another track assignment
- tracking: shot s029 duplicates_suppressed 9 vs 10
- tracking: 3 detections of shot s031 have another track assignment
- tracking: shot s031 duplicates_suppressed 2 vs 1
- tracking: 14 detections of shot s032 have another track assignment
- tracking: shot s032 duplicates_suppressed 47 vs 42
- pose: 7967 frames not within 1 px (box max 646.7772 px, keypoint max 662.3453 px)
- pose: confidence differs by up to 0.9891
- metrics: A-0002 hand_travel_r 3546.83 vs 3467.32
- metrics: A-0002 hand_travel_r_torso 21.4699 vs 21.0462
- metrics: A-0002 two_hand_speed_corr 0.337827 vs 0.350171
- metrics: A-0002 two_hand_use_ratio 0.948636 vs 0.935897
- metrics: A-0004 idle_share 0.34137 vs 0.288219
- metrics: A-0004 idle_time_s 20.7667 vs 17.5333
- metrics: A-0004 two_hand_use_ratio 0.757637 vs 0.742455
- metrics: A-0006 hand_travel_l 889.044 vs 994.542
- metrics: A-0006 hand_travel_l_torso 6.83463 vs 7.30532
- metrics: A-0006 hand_travel_r 1646.95 vs 1628.6
- metrics: A-0006 hand_travel_r_torso 11.1236 vs 10.6925
- metrics: A-0006 idle_time_s 25.3 vs 24.7667
- metrics: A-0006 repetition_count 14.6258 vs 0
- metrics: A-0006 repetition_count flags - vs not_periodic
- metrics: A-0006 repetition_period_s reported in one run, NULL in the other
- metrics: A-0006 repetition_period_s flags - vs not_periodic
- metrics: A-0006 repetition_share 0.142857 vs 0
- metrics: A-0006 repetition_share flags - vs not_periodic
- metrics: A-0008 hand_travel_r 1083.34 vs 1104.69
- metrics: A-0010 bend_b0_s 40.5 vs 40.0667
- metrics: 7 more

Within numerical noise: **no** (measures not_within_1px_share 0.872713, track_changes_per_frame 0.418885, metric_max_rel_delta 1.0; floor not_within_1px_share 0.016835, track_changes_per_frame 0.003367, metric_max_rel_delta 0.0301).
- not_within_1px_share 0.872713 above the floor 0.016835
- track_changes_per_frame 0.418885 above the floor 0.003367
- metric_max_rel_delta 1.0 above the floor 0.0301
- metrics: 1 NULL decisions and 3 flags changed

## 12981262_1920_1080_25fps.mp4 (3 repeats, medians)

| stage | median s |
| --- | --- |
| ingest | 0.34 |
| shots | 9.61 |
| reference | 1.79 |
| pose_body | 30.74 |
| tracking | 1.62 |
| segmentation | 0.10 |
| metrics | 0.03 |
| overlay | 24.71 |
| full run (process wall) | 72.39 (min 71.22, max 73.37) |

| cost | value |
| --- | --- |
| pose only fps (pose stage) | 58.2 |
| inference fps (model only) | 72.9 |
| mean GPU util, run / pose stage (%) | 23.1 / 51.8 |
| peak GPU memory, nvidia-smi / torch (MB) | 1305.0 / 583.1 |

| pose breakdown | median s |
| --- | --- |
| decode | 0.51 |
| infer_call | 29.65 |
| preprocess | 2.74 |
| inference | 24.44 |
| postprocess | 1.92 |
| extract | 0.33 |
| writing | 0.09 |

Repeat consistency against repeat 1: equivalent, equivalent.

Equivalence with w10-ref: **different** (frames within 1 px 0.0000, identical 0.0000, box max 1820.3431 px, keypoint max 1870.9377 px, keypoint mean 1.55549 px, conf max 0.311 / 0.989).
- pose: frame 0 has 7 detections in the reference, 6 in the candidate
- pose: frame 4 has 7 detections in the reference, 6 in the candidate
- pose: frame 5 has 5 detections in the reference, 7 in the candidate
- pose: frame 7 has 7 detections in the reference, 8 in the candidate
- pose: frame 8 has 5 detections in the reference, 6 in the candidate
- pose: frame 14 has 4 detections in the reference, 5 in the candidate
- pose: frame 19 has 4 detections in the reference, 5 in the candidate
- pose: frame 21 has 5 detections in the reference, 4 in the candidate
- pose: frame 30 has 6 detections in the reference, 5 in the candidate
- pose: frame 34 has 6 detections in the reference, 5 in the candidate
- pose: frame 36 has 5 detections in the reference, 6 in the candidate
- pose: frame 60 has 4 detections in the reference, 5 in the candidate
- pose: frame 82 has 4 detections in the reference, 3 in the candidate
- pose: frame 107 has 5 detections in the reference, 6 in the candidate
- pose: frame 152 has 4 detections in the reference, 5 in the candidate
- pose: frame 153 has 5 detections in the reference, 4 in the candidate
- pose: frame 161 has 6 detections in the reference, 5 in the candidate
- pose: frame 174 has 5 detections in the reference, 6 in the candidate
- pose: frame 175 has 6 detections in the reference, 7 in the candidate
- pose: frame 177 has 7 detections in the reference, 8 in the candidate
- tracking: 3129 detections of shot s001 have another track assignment
- tracking: shot s001 tracks_kept 37 vs 40
- tracking: shot s001 short_tracks 31 vs 34
- tracking: shot s001 duplicates_suppressed 1054 vs 1052
- pose: 1782 frames not within 1 px (box max 1820.3431 px, keypoint max 1870.9377 px)
- pose: confidence differs by up to 0.989
- metrics: A-0001 two_hand_simultaneous 0.419608 vs 0.442886
- metrics: A-0001 two_hand_use_ratio 0.668203 vs 0.698113
- metrics: A-0002 hand_travel_r coverage 0.0808 vs 0.0932
- metrics: A-0002 hand_travel_r_torso coverage 0.0808 vs 0.0932
- metrics: A-0002 two_hand_simultaneous coverage 0.0808 vs 0.0932
- metrics: A-0002 two_hand_speed_corr coverage 0.0808 vs 0.0932
- metrics: A-0002 two_hand_use_ratio coverage 0.0808 vs 0.0932

Within numerical noise: **no** (measures not_within_1px_share 1.0, track_changes_per_frame 1.757576, metric_max_rel_delta 0.0526; floor not_within_1px_share 0.016835, track_changes_per_frame 0.003367, metric_max_rel_delta 0.0301).
- not_within_1px_share 1.0 above the floor 0.016835
- track_changes_per_frame 1.757576 above the floor 0.003367
- metric_max_rel_delta 0.0526 above the floor 0.0301
