# w10-qisens-steps-drop

Engine A on the QiSens video, approved summary numbers only (2026-11-08, updated 2026-11-09); everything else stays under the QiSens data area.

| run | pose | GPU | precision | code | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s | how |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run of 2026-10-25 05:36 UTC | YOLO26 l | GTX 1080 Ti | batch 1, FP32 | 25 Oct (uncommitted, became cb949b6) | 75.02 | 100 | 100 | 100 | 80 | 29.35 | its stored eval |
| run_20261025_053609_fd62 (the 25 Oct run) rescored: weight 0 row of the Engine B v0 compare | YOLO26 l (its stored keypoints) | GTX 1080 Ti (keypoints) | batch 1, FP32 (keypoints) | Engine A, eval and GT current on 2026-11-08 | 75.02 | 100 | 100 | 100 | 80 | 29.35 | run_pose --compare on the stored run (confirmed by Khurshid 2026-11-09) |
| w10-qisens-ref, repeat 1 | YOLO26 x | RTX A5000 | batch 1, TF32 default | run 2026-11-07 (07a6977), scored with 06883b7 | 50.00 | 100 | 80 | 60 | 40 | 20.44 | run_pose --compare on the stored run (no GT was stored with it) |
| w10-qisens-b8 | YOLO26 x | RTX A5000 | batch 8, TF32 | 06883b7 | 50.00 | 100 | 80 | 60 | 40 | 20.44 | its eval |
| seg_a.start_detect = true (blind, defaults fixed on public clips) | YOLO26 x | RTX A5000 | as the run | 06883b7 | 50.00 | 100 | 80 | 60 | 40 | 20.44 | run_pose --compare; identical to start_detect = false (no effect) |

Findings:

- Confirmed (2026-11-09): the Engine B v0 compare ran on run_20261025_053609_fd62, and its weight 0 row was 75.02. Today's Engine A and today's GT reproduce 75.02 on the 25 Oct keypoints, so the GT and Engine A are ruled out.
- The drop therefore lies in the keypoints: the pose model (YOLO26 l to x) or the GPU numerics (GTX 1080 Ti to RTX A5000). w10-qisens-l-replica (YOLO26 l on the A5000, current code, batch 1 strict FP32) separates the two.
- Batching does not change the steps on QiSens: w10-qisens-ref (batch 1) and w10-qisens-b8 (batch 8) give the same 50.00 / 100 / 80,60,40 / MAE 20.44.
- The drop happened between 2026-10-25 and 2026-11-07.
- Engine A code from cb949b6 (25 Oct) and from now give identical boundaries on three public runs.
- seg_a.start_detect = true has no effect on QiSens (blind result).
- The pose model is not chosen from these results: see the pose model choice rule in experiments/INDEX.md.

Next: w10-qisens-l-replica (running 2026-11-09): about 75 means the model explains the drop, about 50 the GPU numerics. Public side: w10-ha4m-domain-gap-l and w10-pose-sensitivity.
