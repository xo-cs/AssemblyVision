# w10-qisens

QiSens video (9.9 min, YOLO26 x, batch 1, FP32 with PyTorch default TF32), GPU0 A5000, median of 3 repeats each. Approved summary numbers only (2026-11-07); everything else stays under the QiSens data area.

w10-qisens-best = configs/speed_proposed.toml (w10-best without shots.motion_seek).

| experiment | ingest | shots | ref | pose | tracking | seg | metrics | overlay | full run s | pose fps | infer fps | GPU util % | pose util % | peak MB | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w10-qisens-ref | 0.9 | 33.0 | 15.8 | 265.6 | 6.7 | 0.3 | 0.4 | 66.9 | 392.2 | 67.4 | 83.2 | 41.1 | 59.8 | 981 | reference |
| w10-qisens-best | 0.9 | 32.6 | 0.5 | 262.6 | 0.4 | 0.3 | 0.4 | 39.9 | 340.7 | 68.1 | 83.5 | 48.0 | 60.8 | 1881 | equivalent (within 1 px 1.0000, keypoint max 0.0 px) |

Gates (TA, full run): 300 s initial, 240 s milestone, under 180 s stretch. Neither configuration meets 300 s.
