# Experiments

One line per experiment (public data only; QiSens rows only as approved summaries).
Regenerate any table with `python -m assemblyvision.tools.exp table <id> ...`.

Verdicts: **equivalent** = every strict check of evaluation/equivalence.py holds; **equivalent within numerical noise** (within noise) = against the reference, (1) the share of frames not within 1 px, (2) the track changes per frame and (3) the largest relative delta of a reported motion metric (deltas under 0.01 absolute ignored) are each no larger than the noise floor, the same measures of strict FP32 against the default TF32 at batch 1 (w10-ref-notf32 against w10-ref, the larger of the two public clips); (4) the Engine A steps are identical (labels, boundaries within one frame); (5) every NULL decision and every flag of the motion metrics is identical; **different** otherwise.

Noise floor (w10-ref-notf32 against w10-ref, measured 2026-11-07):

| measure | floor | process_480p30.mp4 / 12981262_1920_1080_25fps.mp4 |
| --- | --- | --- |
| not_within_1px_share | 0.016835 | 0.011291 / 0.016835 |
| track_changes_per_frame | 0.003367 | 0.002193 / 0.003367 |
| metric_max_rel_delta | 0.0301 | 0.0301 / 0.0012 |

Default profile (decision of 2026-11-08): the w10-best-b8 settings (pose batch 8, TF32), recorded as **above the noise floor; Engine A steps, NULL decisions and flags identical** (the metric changes above the floor are listed with their absolute values in w10_summary.md). Accuracy checks use configs/reference.toml (batch 1, strict FP32, the w10-ref-notf32 settings). The QiSens run w10-qisens-b8 decides whether this default stays.

Pose model choice rule (written 2026-11-09): the pose model (YOLO26 variant) is chosen only by public criteria, namely the HA4M domain gap PCK on the 540 frames (w10-ha4m-domain-gap and its per variant runs), the published COCO mAP of the variant, and speed (pose fps on the public clips). QiSens segmentation results are never used to choose it. Disclosure: when this rule was written, the QiSens Engine A results for YOLO26 l (75.02, 25 Oct code) and YOLO26 x (50.00) were already known (w10-qisens-steps-drop).

| id | date | title | per video: full run median s, pose fps, mean GPU util %, peak GPU MB | verdict |
| --- | --- | --- | --- | --- |
| [w10-b16](w10-b16/table.md) | 2026-11-06 | pose batch 16, FP32 | process_480p30.mp4: 175.8 s, 108.2 fps, 44.4 %, 2717 MB<br>12981262_1920_1080_25fps.mp4: 62.5 s, 84.3 fps, 21.6 %, 2717 MB | different |
| [w10-b32](w10-b32/table.md) | 2026-11-06 | pose batch 32, FP32 | process_480p30.mp4: 177.0 s, 106.7 fps, 43.8 %, 4441 MB<br>12981262_1920_1080_25fps.mp4: 63.0 s, 84.7 fps, 22.9 %, 4441 MB | different |
| [w10-b8](w10-b8/table.md) | 2026-11-06 | pose batch 8, FP32 | process_480p30.mp4: 176.7 s, 107.2 fps, 44.3 %, 1799 MB<br>12981262_1920_1080_25fps.mp4: 63.8 s, 85.2 fps, 21.4 %, 1799 MB | different |
| [w10-b8-fp16](w10-b8-fp16/table.md) | 2026-11-06 | pose batch 8, FP16 | process_480p30.mp4: 156.4 s, 142.6 fps, 32.1 %, 1455 MB<br>12981262_1920_1080_25fps.mp4: 63.5 s, 94.0 fps, 13.2 %, 1455 MB | different |
| [w10-b8-notf32](w10-b8-notf32/table.md) | 2026-11-07 | pose batch 8, strict FP32 (TF32 off) | process_480p30.mp4: 284.2 s, 50.0 fps, 66.0 %, 6685 MB<br>12981262_1920_1080_25fps.mp4: 81.8 s, 45.2 fps, 39.7 %, 6685 MB | within noise |
| [w10-best](w10-best/table.md) | 2026-11-06 | equivalent winners combined (batch 1, FP32) | process_480p30.mp4: 192.7 s, 66.4 fps, 46.6 %, 1881 MB<br>12981262_1920_1080_25fps.mp4: 63.1 s, 60.3 fps, 27.0 %, 2020 MB | equivalent |
| [w10-best-b8](w10-best-b8/table.md) | 2026-11-07 | proposal (speed_proposed.toml) at pose batch 8, FP32 | process_480p30.mp4: 134.7 s, 108.4 fps, 54.9 %, 2793 MB<br>12981262_1920_1080_25fps.mp4: 50.2 s, 94.3 fps, 28.8 %, 2932 MB | different |
| [w10-engine-b-embed](w10-engine-b-embed/table.md) | 2026-11-08 | Engine B seg_a.embed_weight and seg_a.embed_window_s | chosen seg_a.embed_window_s = 30, seg_a.embed_weight = 0.2 (process 92.33 unchanged, shots_check frame acc 30.04 to 78.26) | measurement |
| [w10-engine-b-firing](w10-engine-b-firing/table.md) | 2026-11-08 | Engine B firing rate on the HA4M test subjects (w10-engine-b-v0) | true 10.87 changes/min; peaks/min pool 0 s: 11.3, pool 2 s: 8.93, pool 5 s: 3.86, pool 10 s: 2.13, pool 30 s: 0.88, pool 60 s: 0.69 | measurement |
| [w10-engine-b-pool](w10-engine-b-pool/table.md) | 2026-11-08 | Engine B weight and pool window | chosen engine_b.pool_s = 30, seg_a.engine_b_weight = 0.4 (process clip unchanged, frame acc 92.33; shots_check frame acc 30.04 to 46.11) | measurement |
| [w10-engine-b-v0](w10-engine-b-v0/table.md) | 2026-11-08 | Engine B v0: MS TCN boundary scorer, HA4M | cross subject: boundary F1@0.2 s 0.7471 (P 0.7329, R 0.7618)<br>cross subject: boundary F1@0.5 s 0.8916 (P 0.8747, R 0.9092)<br>cross location, test lab_a: F1@0.2 s 0.6294, F1@0.5 s 0.7876<br>cross location, test lab_b: F1@0.2 s 0.5454, F1@0.5 s 0.7549 | measurement |
| [w10-engine-b-v0-hipcal](w10-engine-b-v0-hipcal/table.md) | 2026-11-08 | Engine B v0 with hip calibration | cross subject: boundary F1@0.2 s 0.7447 (P 0.7291, R 0.7609)<br>cross subject: boundary F1@0.5 s 0.8872 (P 0.8686, R 0.9066)<br>cross location, test lab_a: F1@0.2 s 0.6312, F1@0.5 s 0.7833<br>cross location, test lab_b: F1@0.2 s 0.5348, F1@0.5 s 0.7506 | measurement |
| [w10-engine-b-v1](w10-engine-b-v1/table.md) | 2026-11-08 | Engine B v1: multitask MS TCN (boundary plus 12 actions), HA4M | cross subject: boundary F1@0.2 s 0.7303 (P 0.7164, R 0.7446)<br>cross subject: boundary F1@0.5 s 0.8958 (P 0.8788, R 0.9135)<br>cross location, test lab_a: F1@0.2 s 0.6065, F1@0.5 s 0.7653<br>cross location, test lab_b: F1@0.2 s 0.5055, F1@0.5 s 0.7513 | measurement |
| [w10-ha4m-domain-gap](w10-ha4m-domain-gap/table.md) | 2026-11-07 | Domain gap: YOLO26 x against the HA4M Kinect 2D joints, one video per lab | IDU001V001 (lab_a): found 1.0, PCK@0.1 0.8682, 45.7 px<br>IDU023V001 (lab_b): found 1.0, PCK@0.1 0.8316, 30.89 px | measurement |
| [w10-nvdec](w10-nvdec/table.md) | 2026-11-06 | ffmpeg NVDEC reader (batch 1) | process_480p30.mp4: 230.0 s, 67.6 fps, 42.1 %, 1214 MB<br>12981262_1920_1080_25fps.mp4: 72.4 s, 58.2 fps, 23.1 %, 1305 MB | different |
| [w10-ovl-nvenc](w10-ovl-nvenc/table.md) | 2026-11-06 | overlay encoder h264_nvenc | process_480p30.mp4: 223.0 s, 66.8 fps, 42.7 %, 1206 MB<br>12981262_1920_1080_25fps.mp4: 64.2 s, 62.1 fps, 26.5 %, 1241 MB | equivalent |
| [w10-ovl-off](w10-ovl-off/table.md) | 2026-11-06 | overlay off | process_480p30.mp4: 186.0 s, 68.1 fps, 50.9 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 46.2 s, 61.2 fps, 36.5 %, 981 MB | equivalent |
| [w10-ovl-par4](w10-ovl-par4/table.md) | 2026-11-06 | overlay in 4 parallel chunks (libx264) | process_480p30.mp4: 218.7 s, 67.7 fps, 43.2 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 70.0 s, 60.4 fps, 23.1 %, 981 MB | equivalent |
| [w10-prefetch](w10-prefetch/table.md) | 2026-11-06 | threaded prefetch reader (batch 1) | process_480p30.mp4: 227.9 s, 68.8 fps, 41.5 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 70.5 s, 62.2 fps, 23.2 %, 981 MB | equivalent |
| [w10-qisens](w10-qisens/table.md) | 2026-11-07 | QiSens video (9.9 min), 3 repeats each: reference vs speed_proposed.toml (approved summary) | w10-qisens-ref: 392.2 s, 67.4 fps, 41.1 %, 981 MB<br>w10-qisens-best: 340.7 s, 68.1 fps, 48.0 %, 1881 MB | equivalent |
| [w10-qisens-engine-b-v0](w10-qisens-engine-b-v0/table.md) | 2026-11-08 | QiSens blind test of Engine B v0 in Engine A (approved summary) | headline (pre chosen pool 30, weight 0.4): frame acc 55.51 vs 75.02 Engine A alone (YOLO26 l, 25 Oct code), F1@50 20 vs 80, MAE 27.13 vs 29.35 s<br>A fine action boundary scorer does not transfer to minute long phases. | measurement |
| [w10-qisens-steps-drop](w10-qisens-steps-drop/table.md) | 2026-11-08 | QiSens Engine A result: 75.02 (25 Oct, YOLO26 l) to 50.00 (Nov, YOLO26 x); GT and Engine A ruled out; start_detect blind result (approved summary) | Engine A on QiSens: 75.02 (YOLO26 l, 25 Oct code) to 50.00 / MAE 20.44 (YOLO26 x, Nov), same at batch 1 and 8<br>today's Engine A and GT on the 25 Oct keypoints: 75.02 (GT and Engine A ruled out)<br>start_detect = true: no effect (blind) | measurement |
| [w10-ref](w10-ref/table.md) | 2026-11-06 | reference: YOLO26 x, batch 1, FP32, current defaults | process_480p30.mp4: 235.7 s, 65.3 fps, 39.9 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 71.7 s, 60.3 fps, 22.6 %, 981 MB | reference |
| [w10-ref-notf32](w10-ref-notf32/table.md) | 2026-11-07 | reference in strict FP32 (TF32 off), batch 1 | process_480p30.mp4: 276.9 s, 51.9 fps, 62.6 %, 913 MB<br>12981262_1920_1080_25fps.mp4: 78.5 s, 49.5 fps, 37.1 %, 913 MB | within noise |
| [w10-refpass](w10-refpass/table.md) | 2026-11-06 | reference frames from the pose pass | process_480p30.mp4: 217.9 s, 66.3 fps, 40.6 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 71.7 s, 58.3 fps, 23.1 %, 981 MB | equivalent |
| [w10-shots-ds](w10-shots-ds/table.md) | 2026-11-07 | cut detection downscale 8 (automatic: 3 at 480p, 7 at 1080p) | process_480p30.mp4: 233.3 s, 66.2 fps, 40.3 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 71.0 s, 60.5 fps, 22.8 %, 981 MB | equivalent |
| [w10-shots-seek](w10-shots-seek/table.md) | 2026-11-06 | shots motion pairs read by seeking | process_480p30.mp4: 234.4 s, 65.7 fps, 40.6 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 75.2 s, 59.5 fps, 21.8 %, 981 MB | equivalent |
| [w10-start-of-work](w10-start-of-work/table.md) | 2026-11-08 | Start of work detection on the public clips (no harm check) | public GT files have no intro; with start_min_intro_s 10 s both clips unchanged (first version cut false intros of 6.3 s and 1.3 s) | measurement |
| [w10-trackoverlap](w10-trackoverlap/table.md) | 2026-11-06 | tracking in a worker during pose (with reference from the pass) | process_480p30.mp4: 209.8 s, 66.3 fps, 42.2 %, 981 MB<br>12981262_1920_1080_25fps.mp4: 72.8 s, 59.4 fps, 22.6 %, 981 MB | equivalent |
