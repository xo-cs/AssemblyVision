# w11-visual-engine-b-test

Visual Engine B: the single test evaluation (rule w11-visual-engine-b)

Selected by w11-visual-engine-b-cv: v3 {'visual_weight': 0.5, 'visual_window_s': 10.0, 'embed_weight': 1.0, 'embed_window_s': 30.0, 'engine_a_scale': 0.25, 'intro_detect': False}.

| row | HA4M p0.5 | HA4M plain | Kallax | Lack Coffee | F1@50 Kallax | MAE s Kallax |
| --- | --- | --- | --- | --- | --- | --- |
| Engine A | 56.94 | 24.01 | 33.01 | 60.61 | 10.36 | 23.48 |
| B v1 as shipped | 54.9 | 24.01 | 33.01 | 60.46 | 10.36 | 23.48 |
| selected: v3 | 81.17 | 76.6 | 59.09 | 81.53 | 43.75 | 12.35 |
| selected, intro on | 79.38 | 76.61 | 59.12 | 80.47 | 43.75 | 12.35 |
