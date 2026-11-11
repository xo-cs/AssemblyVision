# w10-qisens-steps-drop

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

Next: Headline from 2026-11-11: step engine Engine B v1 on YOLO26 x (w10-qisens-engine-b-v1). Timing run with the new defaults: w10-qisens-defaults.
