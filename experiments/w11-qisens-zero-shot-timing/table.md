# w11-qisens-zero-shot-timing

QiSens, the full default pipeline with B zero shot (YOLO26 x on GPU0, Qwen3 VL 8B on GPU1, VLM 3 s windows, 2 frames, 4 slots), 3 repeats with the GT. Approved summary numbers only (2026-11-14); everything else stays under the QiSens data area.

| measure | value |
| --- | --- |
| full run s, median of 3 (min, max) | 270.62 (270.28, 270.96) |
| eval: frame acc / edit / F1@10,25,50 / MAE s | 92.97 / 100 / 100,100,100 / 8.37 (differs between repeats) |
| gates | 300 s met; 240 s milestone not met; 180 s stretch not met; 85 % accuracy met |

Runs: run_20261114_045653_5eff, run_20261114_050124_9d97, run_20261114_050555_2576.

Findings:

- Headline: the full default pipeline runs in 270.62 s median (270.28 to 270.96) on the 9.9 min QiSens video, under the 300 s gate, with 92.97 / 100 / 100,100,100 / MAE 8.37 s frame accuracy (target 85).
- The eval differs between the repeats (VLM outputs are not fully deterministic with parallel slots); min, median and max are to be added.
- The VLM setting chosen on public data (3 s, 2 frames, 4 slots) keeps the accuracy of the 2 s, 4 frame run (93.20, w11-qisens-zero-shot).

Conclusion: B zero shot meets both critical gates on QiSens: 92.97 frame accuracy and 270.62 s for the full run.
