# w10-qisens-engine-b-v1

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
