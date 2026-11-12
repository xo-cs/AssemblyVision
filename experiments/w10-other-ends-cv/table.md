# w10-other-ends-cv

Item 8: optional intro and outro on synthetic intros (train split; rule w10-engine-b-v2, amendment 3)

Decoder: w10-engine-b-v2-w4-c10 (CV) with {'embed_weight': 1.0, 'engine_a_scale': 0.0, 'embed_window_s': 10.0, 'refine_window_s': 0.0, 'video_norm': False}, plus seg_a.other_ends. Every video plain and with a still intro of 5 to 30 s (seeded), no cuts.

| other_min_strength | all | plain | intro | intro F1@50 | intro MAE s |
| --- | --- | --- | --- | --- | --- |
| off | 81.77 | 89.35 | 74.18 | 85.51 | 6.24 |
| 0.05 | 39.41 | 33.03 | 45.79 | 32.63 | 4.53 |
| 0.1 | 39.41 | 33.03 | 45.79 | 32.63 | 4.53 |
| 0.2 | 39.41 | 33.03 | 45.79 | 32.63 | 4.53 |
| 0.3 | 38.83 | 34.11 | 43.55 | 31.45 | 4.84 |
| 0.5 | 45.32 | 42.93 | 47.71 | 41.32 | 5.71 |
| 0.75 | 73.19 | 78.91 | 67.47 | 76.2 | 5.86 |
| 1 | 81.76 | 89.35 | 74.17 | 85.51 | 6.24 |

Chosen bias (rule): 1.
