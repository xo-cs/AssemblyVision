# w10-qisens-engine-b-v2

QiSens video, Engine B v2 (engine_b/w10-engine-b-v2.pt) and the v1 selected decoder, blind test: both were chosen on the HA4M macro phase benchmark by the rule in experiments/w10-engine-b-v2/rule.md (CV on the train split, the test split once) before any QiSens result. Approved summary numbers only (2026-11-12); everything else stays under the QiSens data area.

| run | pose | setting | role | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | B v1 (engine_a candidates; w10-qisens-engine-b-v1) | reference, the current default | 67.26 | 100 | 100 | 100 | 40 | 39.06 |
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
- The decoder settings chosen on the public benchmark (dense candidates, short window, Engine A scale) did not help on QiSens. Refinement and video_norm help v2 there (52.50 to 57.44 to 59.77), the opposite of the benchmark.
- engine_a_scale changes nothing on QiSens (D). Checked (2026-11-12): the value reaches the decoder in the compare path (public process_480p30 run: 62.41 at scale 0, 60.57 at 0.25 to 1.0; test TC in test_engine_b_model). The reason is that Engine A's candidates also get the Engine B term (embed_weight x the highest change score within engine_b_window_s), so at embed_weight 1.0 the base strengths (cut 0.2) are small next to it, and no_person stretches get no Engine B term at all.
- The first B v1 dense against engine_a compare (A1) was invalid: the stored run has Engine B off, so both rows were Engine A (50.00). A corrected command is in the report of 2026-11-12; compare now warns in that case.
- The YOLO26 l run (run_20261025_053609_fd62) has not been reported for v2 yet.

Conclusion: v2 reaches 88.53 on the HA4M benchmark but 52.50 on QiSens blind, so it does not transfer from HA4M macro phases to the QiSens task.
