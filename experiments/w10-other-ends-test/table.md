# w10-other-ends-test

Item 8: optional intro and outro on synthetic intros (test split; rule w10-engine-b-v2, amendment 3)

Decoder: w10-engine-b-v2 with {'embed_weight': 1.0, 'engine_a_scale': 0.0, 'embed_window_s': 10.0, 'refine_window_s': 0.0, 'video_norm': False}, plus seg_a.other_ends. Every video plain and with a still intro of 5 to 30 s (seeded), no cuts.

| other_min_strength | all | plain | intro | intro F1@50 | intro MAE s |
| --- | --- | --- | --- | --- | --- |
| off | 80.65 | 88.53 | 72.77 | 82.66 | 6.23 |
| 1 | 80.65 | 88.53 | 72.76 | 82.66 | 6.23 |
