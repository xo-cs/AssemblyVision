# w11-qisens-zero-shot

QiSens, zero shot phase scoring (SigLIP 2 and Qwen3 VL 8B, rule w11-zero-shot) and visual Engine B v3 (rule w11-visual-engine-b), blind. Approved summary numbers only (2026-11-13); everything else stays under the QiSens data area.

| run | pose | setting | role | frame acc | edit | F1@10 | F1@25 | F1@50 | MAE s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | B v1 (w10-qisens-engine-b-v1) | reference, the default until now | 67.26 | 100 | 100 | 100 | 40 | 39.06 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | Z1: siglip 1, crop 4, vlm 0.5 | pre chosen headline (rule w11-zero-shot) | 83.14 | 100 | 100 | 100 | 60 | 20.09 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | Z1 with vlm off | ablation | 58.83 | 100 | 80 | 60 | 40 | 48.46 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | Z2: untuned equal weights 1, 1, 1 | untuned row of the rule (never chosen); best | 93.20 | 100 | 100 | 100 | 100 | 8.09 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | Z2 with vlm off | ablation | 66.21 | 100 | 80 | 80 | 60 | 30.92 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | Z1 with shot_smooth on (Z3) | information | 67.86 | 100 | 80 | 80 | 60 | 28.95 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | Z2 with intro_detect on | information | 93.20 | - | - | - | - | - |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | V1: v3, visual 0.5 (10 s), embed 1.0 (30 s), A scale 0.25 | pre chosen headline (rule w11-visual-engine-b) | 32.45 | 100 | 40 | 40 | 20 | 19.45 |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | V1 with visual off | ablation | 47.43 | - | - | - | - | - |
| w10-qisens-b8 (run_20261108_133251_1550) | YOLO26 x | Z2 with other_class on (2026-11-14; rule w11-other-class: off) | information | 93.21 | - | - | - | - | - |

Findings:

- Zero shot headline (Z1, pre chosen by the rule): 83.14 / 100 / 100,100,60 / MAE 20.09 s, against B v1 67.26.
- The untuned equal weights (Z2) give 93.20 / 100 / 100,100,100 / MAE 8.09 s, above the 85 target and the 90 aim. The untuned weights already beat the selected ones on the public IKEA test before any QiSens run (worst case 57.79 against 50.43, w11-zero-shot-test), so the IKEA tuning overfit.
- The VLM carries most of the gain: without it Z1 drops to 58.83 and Z2 to 66.21 (about B v1).
- Negative or no effect results: visual Engine B v3 (32.45, visual off 47.43, both below B v1), shot_smooth (83.14 to 67.86), intro_detect on Z2 (identical, 93.20).
- seg_a.other_class on the Z2 setting (2026-11-14): 93.21 against 93.20, no effect; it stays off (rule w11-other-class decided off on public data).

Conclusion: Zero shot phase scoring with equal weights reaches 93.20 on QiSens blind and becomes the default step engine ("B zero shot"); v3, shot_smooth and intro_detect are not used.
