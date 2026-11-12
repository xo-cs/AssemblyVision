Verdicts: **equivalent** = every strict check of evaluation/equivalence.py holds; **equivalent within numerical noise** (within noise) = against the reference, (1) the share of frames not within 1 px, (2) the track changes per frame and (3) the largest relative delta of a reported motion metric (deltas under 0.01 absolute ignored) are each no larger than the noise floor, the same measures of strict FP32 against the default TF32 at batch 1 (w10-ref-notf32 against w10-ref, the larger of the two public clips); (4) the Engine A steps are identical (labels, boundaries within one frame); (5) every NULL decision and every flag of the motion metrics is identical; **different** otherwise.

Noise floor (w10-ref-notf32 against w10-ref, measured 2026-11-07):

| measure | floor | process_480p30.mp4 / 12981262_1920_1080_25fps.mp4 |
| --- | --- | --- |
| not_within_1px_share | 0.016835 | 0.011291 / 0.016835 |
| track_changes_per_frame | 0.003367 | 0.002193 / 0.003367 |
| metric_max_rel_delta | 0.0301 | 0.0301 / 0.0012 |

Default profile (decision of 2026-11-08): the w10-best-b8 settings (pose batch 8, TF32), recorded as **above the noise floor; Engine A steps, NULL decisions and flags identical** (the metric changes above the floor are listed with their absolute values in w10_summary.md). Accuracy checks use configs/reference.toml (batch 1, strict FP32, the w10-ref-notf32 settings). From 2026-11-11 the default also has shots.parallel = true (w10-shots-par, equivalent), pose.overlap_stages = true (w10-overlap-stages, equivalent) and the step engine seg_a.engine = "B v1" (Engine B = the learned model with the DP decoder: the Engine B v1 embedding, weight 0.2, window 30 s; "A", the frozen Engine A rules, stays selectable as the baseline). Headline QiSens result: Engine B v1 on YOLO26 x, 67.26 frame accuracy (w10-qisens-engine-b-v1); Engine A alone on x gives 50.00.

### process_480p30.mp4

| experiment | ingest | shots | ref | pose | tracking | seg | metrics | overlay | full run s | pose fps | infer fps | GPU util % | pose util % | peak MB | torch MB | within 1 px | track changes / frame | metric max rel delta | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w10-ref | 0.4 | 15.4 | 16.5 | 149.8 | 7.3 | 0.3 | 0.2 | 42.9 | 235.7 | 65.3 | 80.6 | 39.9 | 58.1 | 981 | 583 | ref | ref | ref | reference |
| w10-ref-notf32 | 0.4 | 15.4 | 18.8 | 188.6 | 7.1 | 0.3 | 0.2 | 42.3 | 276.9 | 51.9 | 61.0 | 62.6 | 85.0 | 913 | 583 | 0.9887 | 0.0022 | 0.0301 | within noise |
| w10-b8 | 0.4 | 16.2 | 16.1 | 91.2 | 7.3 | 0.3 | 0.2 | 41.8 | 176.7 | 107.2 | 142.9 | 44.3 | 77.6 | 1799 | 1154 | 0.9761 | 0.0066 | 0.0744 | different |
| w10-b8-notf32 | 0.4 | 15.4 | 19.1 | 195.5 | 7.3 | 0.3 | 0.3 | 42.3 | 284.2 | 50.0 | 56.9 | 66.0 | 89.1 | 6685 | 5823 | 0.9999 | 0.0001 | 0.0000 | within noise |
| w10-best | 0.4 | 15.5 | 0.5 | 147.4 | 0.1 | 0.3 | 0.2 | 24.6 | 192.7 | 66.4 | 81.7 | 46.6 | 58.9 | 1881 | 583 | 1.0000 | 0.0000 | 0.0000 | equivalent |
| w10-best-b8 | 0.4 | 15.4 | 0.5 | 90.3 | 0.1 | 0.3 | 0.2 | 24.2 | 134.7 | 108.4 | 142.9 | 54.9 | 78.7 | 2793 | 1154 | 0.9761 | 0.0066 | 0.0744 | different |
| w10-shots-ds | 0.4 | 14.9 | 16.4 | 147.8 | 7.2 | 0.3 | 0.3 | 42.5 | 233.3 | 66.2 | 81.6 | 40.3 | 58.9 | 981 | 583 | 1.0000 | 0.0000 | 0.0000 | equivalent |
| w10-shots-par | 0.4 | 18.9 | 1.0 | 93.3 | 0.0 | 0.3 | 0.3 | 24.4 | 123.3 | 105.1 | 142.5 | 60.3 | 76.6 | 2793 | 1154 | 1.0000 | 0.0000 | 0.0000 | equivalent |
| w10-overlap-stages | 0.4 | 22.4 | 0.9 | 83.3 | 0.0 | 0.3 | 0.2 | 26.0 | 114.6 | 117.8 | 135.3 | 65.9 | 86.8 | 3345 | 1652 | 1.0000 | 0.0000 | 0.0000 | equivalent |

Metrics above the noise floor (reference to candidate, absolute change, relative change):
- w10-b8: A-0010 idle_share 0.219586 to 0.237226 (+0.01764, 7.4 %)
- w10-b8: A-0010 idle_time_s 12.0333 to 13 (+0.9667, 7.4 %)
- w10-best-b8: A-0010 idle_share 0.219586 to 0.237226 (+0.01764, 7.4 %)
- w10-best-b8: A-0010 idle_time_s 12.0333 to 13 (+0.9667, 7.4 %)

### 12981262_1920_1080_25fps.mp4

| experiment | ingest | shots | ref | pose | tracking | seg | metrics | overlay | full run s | pose fps | infer fps | GPU util % | pose util % | peak MB | torch MB | within 1 px | track changes / frame | metric max rel delta | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w10-ref | 0.3 | 9.7 | 1.9 | 29.6 | 1.6 | 0.1 | 0.0 | 25.1 | 71.7 | 60.3 | 80.8 | 22.6 | 52.3 | 981 | 583 | ref | ref | ref | reference |
| w10-ref-notf32 | 0.3 | 9.8 | 1.8 | 36.1 | 1.6 | 0.1 | 0.0 | 25.6 | 78.5 | 49.5 | 62.9 | 37.1 | 77.8 | 913 | 583 | 0.9832 | 0.0034 | 0.0012 | within noise |
| w10-b8 | 0.3 | 10.1 | 1.6 | 21.0 | 1.7 | 0.1 | 0.0 | 25.5 | 63.8 | 85.2 | 145.2 | 21.4 | 60.5 | 1799 | 1154 | 0.9248 | 0.0045 | 0.0009 | different |
| w10-b8-notf32 | 0.3 | 9.6 | 1.9 | 39.5 | 1.7 | 0.1 | 0.0 | 24.9 | 81.8 | 45.2 | 58.2 | 39.7 | 78.5 | 6685 | 5823 | 0.9994 | 0.0000 | 0.0000 | within noise |
| w10-best | 0.4 | 12.5 | 0.6 | 29.7 | 2.8 | 0.1 | 0.0 | 13.7 | 63.1 | 60.3 | 80.1 | 27.0 | 52.5 | 2020 | 583 | 1.0000 | 0.0000 | 0.0000 | equivalent |
| w10-best-b8 | 0.3 | 9.7 | 0.5 | 19.0 | 2.8 | 0.1 | 0.0 | 14.2 | 50.2 | 94.3 | 145.3 | 28.8 | 67.2 | 2932 | 1154 | 0.9248 | 0.0045 | 0.0009 | different |
| w10-shots-ds | 0.3 | 9.6 | 1.8 | 29.6 | 1.7 | 0.1 | 0.0 | 24.8 | 71.0 | 60.5 | 81.7 | 22.8 | 53.1 | 981 | 583 | 1.0000 | 0.0000 | 0.0000 | equivalent |
| w10-shots-par | 0.3 | 13.2 | 1.2 | 21.3 | 2.6 | 0.1 | 0.0 | 14.8 | 44.2 | 84.7 | 141.5 | 33.5 | 60.0 | 2932 | 1154 | 1.0000 | 0.0000 | 0.0000 | equivalent |
| w10-overlap-stages | 0.3 | 16.5 | 1.1 | 19.4 | 2.6 | 0.1 | 0.0 | 14.9 | 42.2 | 93.2 | 115.4 | 34.8 | 67.9 | 3484 | 1654 | 1.0000 | 0.0000 | 0.0000 | equivalent |

### w10-qisens

QiSens video (9.9 min, YOLO26 x, batch 1, FP32 with PyTorch default TF32), GPU0 A5000, median of 3 repeats each. Approved summary numbers only (2026-11-07); everything else stays under the QiSens data area.

w10-qisens-best = configs/speed_proposed.toml (w10-best without shots.motion_seek).

| experiment | ingest | shots | ref | pose | tracking | seg | metrics | overlay | full run s | pose fps | infer fps | GPU util % | pose util % | peak MB | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w10-qisens-ref | 0.9 | 33.0 | 15.8 | 265.6 | 6.7 | 0.3 | 0.4 | 66.9 | 392.2 | 67.4 | 83.2 | 41.1 | 59.8 | 981 | reference |
| w10-qisens-best | 0.9 | 32.6 | 0.5 | 262.6 | 0.4 | 0.3 | 0.4 | 39.9 | 340.7 | 68.1 | 83.5 | 48.0 | 60.8 | 1881 | equivalent (within 1 px 1.0000, keypoint max 0.0 px) |

Gates (TA, full run): 300 s initial, 240 s milestone, under 180 s stretch. Neither configuration meets 300 s.

### w10-qisens-engine-b-v0

QiSens video, Engine B v0 (engine_b/w10-engine-b-v0.pt) as evidence in Engine A, blind test: the setting was chosen on public runs before any QiSens result (w10-engine-b-pool). Approved summary numbers only (2026-11-08); everything else stays under the QiSens data area.

| setting | role | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s | B peaks/min raw / pooled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weight 0 (Engine A alone; YOLO26 l, 25 Oct code) | baseline | 75.02 | 100 | 100 | 100 | 80 | 29.35 | - |
| **pool_s 30, weight 0.4** | pre chosen (w10-engine-b-pool, public runs only) | 55.51 | 100 | 60 | 60 | 20 | 27.13 | 4.95 / 0.51 |
| pool_s 0, weight 0.4 | post hoc sensitivity, never used for choosing | 79.96 | 100 | 100 | 100 | 60 | 23.45 | - |
| pool_s 10, weight 0.4 | post hoc sensitivity, never used for choosing | 65.52 | 100 | 80 | 80 | 60 | 27.13 | - |
| pool_s 60, weight 0.4 | post hoc sensitivity, never used for choosing | 49.72 | 100 | 60 | 40 | 20 | 37.24 | - |

Headline: the pre chosen setting lowers frame accuracy from 75.02 (YOLO26 l, 25 Oct code) to 55.51 and F1@50 from 80 to 20; MAE drops a little (29.35 to 27.13 s). The pooled rate (0.51 peaks/min) was close to the planned phase rate, so the problem is where the peaks fall, not how often.

The weight 0 row reproduces the 2026-10-25 result (YOLO26 l, 25 Oct code), so this compare ran on that run's stored keypoints; the current pipeline gives 50.00 on QiSens (w10-qisens-steps-drop).

Conclusion: A fine action boundary scorer does not transfer to minute long phases.

### w10-qisens-engine-b-v1

QiSens video, Engine B v1 (engine_b/w10-engine-b-v1.pt) embedding change score as evidence in Engine A, blind test: seg_a.embed_weight 0.2 and seg_a.embed_window_s 30 were chosen on public clips by the rule in w10-engine-b-embed before any QiSens result. This is the step engine "B v1" (named "Engine B v1" on 2026-11-11), the default from 2026-11-11. Approved summary numbers only (2026-11-11); everything else stays under the QiSens data area.

| run | pose | setting | role | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | weight 0 (Engine A alone) | baseline | 50.00 | 100 | 80 | 60 | 40 | 20.44 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | **weight 0.2 (Engine B v1)** | pre chosen headline | 67.26 | 100 | 100 | 100 | 40 | 39.06 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | weight 0.4, 0.8 | post hoc, identical to 0.2 | 67.26 | 100 | 100 | 100 | 40 | 39.06 |
| 25 Oct run (run_20261025_053609_fd62) | YOLO26 l | weight 0 (Engine A alone) | baseline | 75.02 | 100 | 100 | 100 | 80 | 29.35 |
| 25 Oct run (run_20261025_053609_fd62) | YOLO26 l | **weight 0.2 (Engine B v1)** | pre chosen | 80.11 | 100 | 100 | 100 | 100 | 23.27 |
| 25 Oct run (run_20261025_053609_fd62) | YOLO26 l | weight 0.4, 0.8 | post hoc, identical to 0.2 | 80.11 | 100 | 100 | 100 | 100 | 23.27 |

Findings:

- Headline (Engine B v1 on YOLO26 x, the default pipeline): frame accuracy 50.00 to 67.26, F1@10 and F1@25 80 and 60 to 100, F1@50 stays 40, edit 100; the boundary MAE rises from 20.44 to 39.06 s.
- On the YOLO26 l keypoints: 75.02 to 80.11, F1 100,100,100, MAE 29.35 to 23.27 s.
- Weights 0.4 and 0.8 give the same result as 0.2 on both runs, so the result does not hang on the exact weight.
- Engine B v1's phase level embedding transfers to QiSens where v0's fine boundary scorer did not (w10-qisens-engine-b-v0).

### w10-qisens-engine-b-v2

QiSens video, Engine B v2 (engine_b/w10-engine-b-v2.pt) and the v1 selected decoder, blind test: both were chosen on the HA4M macro phase benchmark by the rule in experiments/w10-engine-b-v2/rule.md (CV on the train split, the test split once) before any QiSens result. Approved summary numbers only (2026-11-12); everything else stays under the QiSens data area.

| run | pose | setting | role | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | B v1 (engine_a candidates; w10-qisens-engine-b-v1) | reference, the current default | 67.26 | 100 | 100 | 100 | 40 | 39.06 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | B v1 with dense candidates ("B v1 dense") | comparison (corrected A1) | 67.26 | 100 | 100 | 100 | 40 | 39.06 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | **v2 selected: refinement 0, video_norm off (C1)** | pre chosen headline (rule w10-engine-b-v2) | 52.50 | 100 | 80 | 60 | 20 | 35.51 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | v2 selected, refinement 10 s (C1) | ablation, item 6 | 59.77 | 100 | 60 | 60 | 40 | 26.17 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | v2 selected, video_norm on (C2) | ablation, item 7 | 57.44 | 100 | 80 | 80 | 60 | 27.23 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | v2 selected, video_norm on, refinement 10 s (C2) | ablation, items 6 and 7 | 59.77 | 100 | 60 | 60 | 40 | 26.17 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | v1 selected: refinement 0, video_norm on (B1) | selected v1 decoder (CV) | 50.28 | 100 | 80 | 60 | 20 | 32.47 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | v1 selected, refinement 10 s (B1) | ablation, item 6 | 57.23 | 100 | 80 | 80 | 20 | 36.05 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | v1 selected, video_norm off (B2) | ablation, item 7 | 65.59 | 100 | 100 | 100 | 40 | 41.06 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | v1 selected, video_norm off, refinement 10 s (B2) | ablation, items 6 and 7; post hoc, never used for choosing | 67.94 | 100 | 100 | 100 | 60 | 38.25 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | v2 selected, engine_a_scale 0, 0.25, 0.5, 1.0 (D) | post hoc sensitivity, all four identical | 52.50 | 100 | 80 | 60 | 20 | 35.51 |

Findings:

- Headline (pre chosen by the rule, v2 selected on YOLO26 x): 52.50 / 100 / 80,60,20 / MAE 35.51 s, below the current default B v1 (67.26).
- v2 reaches 88.53 on the HA4M macro phase benchmark (w10-phase-bench-v2-test) but 52.50 on QiSens blind, so it does not transfer from HA4M macro phases to the QiSens task.
- v1 selected (50.28) is also below B v1 (67.26); its best ablation (video_norm off, refinement 10 s: 67.94) is post hoc and within one step of B v1, so nothing is chosen from it.
- B v1 with dense candidates gives 67.26, identical to engine_a candidates (corrected A1, 2026-11-12): at weight 0.2 the dense candidates change nothing on QiSens. The first A1 compare was invalid (the stored run has Engine B off, so both rows were Engine A, 50.00); run_pose --compare now warns in that case, and a preset set on a stored run replaces the run's stored values of its keys.
- The decoder settings chosen on the public benchmark (dense candidates, short window, Engine A scale) did not help on QiSens. Refinement and video_norm help v2 there (52.50 to 57.44 to 59.77), the opposite of the benchmark.
- engine_a_scale changes nothing on QiSens (D). Checked (2026-11-12): the value reaches the decoder in the compare path (public process_480p30 run: 62.41 at scale 0, 60.57 at 0.25 to 1.0; test TC in test_engine_b_model). The reason is that Engine A's candidates also get the Engine B term (embed_weight x the highest change score within engine_b_window_s), so at embed_weight 1.0 the base strengths (cut 0.2) are small next to it, and no_person stretches get no Engine B term at all.
- The YOLO26 l run (run_20261025_053609_fd62) has not been reported for v2 yet.

Conclusion: v2 reaches 88.53 on the HA4M benchmark but 52.50 on QiSens blind, so it does not transfer from HA4M macro phases to the QiSens task.

### w10-qisens-steps-drop

Engine A on the QiSens video, approved summary numbers only (2026-11-08, updated 2026-11-11); everything else stays under the QiSens data area.

| run | pose | GPU | precision | code | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s | how |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run of 2026-10-25 05:36 UTC | YOLO26 l | GTX 1080 Ti | batch 1, FP32 | 25 Oct (uncommitted, became cb949b6) | 75.02 | 100 | 100 | 100 | 80 | 29.35 | its stored eval |
| run_20261025_053609_fd62 (the 25 Oct run) rescored: weight 0 row of the Engine B v0 compare | YOLO26 l (its stored keypoints) | GTX 1080 Ti (keypoints) | batch 1, FP32 (keypoints) | Engine A, eval and GT current on 2026-11-08 | 75.02 | 100 | 100 | 100 | 80 | 29.35 | run_pose --compare on the stored run (confirmed by Khurshid 2026-11-09) |
| w10-qisens-l-replica | YOLO26 l | RTX A5000 | batch 1, strict FP32 (configs/reference.toml) | 140110d (current code) | 75.02 | 100 | 100 | 100 | 80 | 29.35 | its eval; identical to the 25 Oct run |
| w10-qisens-ref, repeat 1 | YOLO26 x | RTX A5000 | batch 1, TF32 default | run 2026-11-07 (07a6977), scored with 06883b7 | 50.00 | 100 | 80 | 60 | 40 | 20.44 | run_pose --compare on the stored run (no GT was stored with it) |
| w10-qisens-b8 | YOLO26 x | RTX A5000 | batch 8, TF32 | 06883b7 | 50.00 | 100 | 80 | 60 | 40 | 20.44 | its eval |
| seg_a.start_detect = true (blind, defaults fixed on public clips) | YOLO26 x | RTX A5000 | as the run | 06883b7 | 50.00 | 100 | 80 | 60 | 40 | 20.44 | run_pose --compare; identical to start_detect = false (no effect) |

Findings:

- w10-qisens-l-replica (YOLO26 l, current code, RTX A5000, batch 1 strict FP32) gives 75.02 / 100 / 100,100,80 / MAE 29.35, identical to the 25 Oct run. The pose model (l against x) explains the whole drop; the GPU numerics (GTX 1080 Ti against RTX A5000) do not.
- Engine A on QiSens is sensitive to the pose model (l 75.02, x 50.00), while on the public clips with a GT l and x give identical steps (w10-pose-sensitivity). The HA4M domain gap differs by 0.02 to 0.03 PCK@0.1, mostly in the lower body (w10-ha4m-domain-gap-l).
- Today's Engine A and GT reproduce 75.02 on the 25 Oct keypoints (run_20261025_053609_fd62), so the GT and Engine A were ruled out first.
- Batching does not change the steps on QiSens: batch 1 and batch 8 give the same 50.00 / 100 / 80,60,40 / MAE 20.44.
- seg_a.start_detect = true has no effect on QiSens (blind result).
- Pose model decision by the rule in experiments/INDEX.md (public criteria only): YOLO26 x stays the default (better HA4M PCK and COCO mAP; the speed gate is met with x). The QiSens l result is not used to choose.

Next: Headline from 2026-11-11: step engine Engine B v1 on YOLO26 x (w10-qisens-engine-b-v1); v2 does not transfer (w10-qisens-engine-b-v2), so B v1 stays the default. Timing run with the new defaults: w10-qisens-defaults.
