# w10-qisens-steps-drop

Engine A on the QiSens video, approved summary numbers only (2026-11-08); everything else stays under the QiSens data area.

| run | pose | GPU | precision | code | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s | how |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run of 2026-10-25 05:36 UTC | YOLO26 l | GTX 1080 Ti | batch 1, FP32 | 25 Oct (uncommitted, became cb949b6) | 75.02 | 100 | 100 | 100 | 80 | 29.35 | its stored eval |
| w10-qisens-ref, repeat 1 | YOLO26 x | RTX A5000 | batch 1, TF32 default | run 2026-11-07 (07a6977), scored with 06883b7 | 50.00 | 100 | 80 | 60 | 40 | 20.44 | run_pose --compare on the stored run (no GT was stored with it) |
| w10-qisens-b8 | YOLO26 x | RTX A5000 | batch 8, TF32 | 06883b7 | 50.00 | 100 | 80 | 60 | 40 | 20.44 | its eval |
| seg_a.start_detect = true (blind, defaults fixed on public clips) | YOLO26 x | RTX A5000 | as the run | 06883b7 | 50.00 | 100 | 80 | 60 | 40 | 20.44 | run_pose --compare; identical to start_detect = false (no effect) |

Findings:

- Batching does not change the steps on QiSens: w10-qisens-ref (batch 1) and w10-qisens-b8 (batch 8) give the same 50.00 / 100 / 80,60,40 / MAE 20.44.
- The drop happened between 2026-10-25 and 2026-11-07.
- Engine A code from cb949b6 (25 Oct) and from now give identical boundaries on three public runs, so Engine A itself does not explain the drop.
- seg_a.start_detect = true has no effect on QiSens (blind result).

Next: w10-qisens-l-replica: current code, configs/reference.toml, pose.variant = l, 1 repeat, with the GT.
