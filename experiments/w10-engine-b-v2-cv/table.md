# w10-engine-b-v2-cv

Engine B v2 selection: decoder grid by 3 fold CV on the 152 train videos, dense candidates, worst case over the synthetic cut conditions (rule: w10-engine-b-v2, amendment 2)

Rule: experiments/w10-engine-b-v2/rule.md. 152 train videos, each decoded by the model of the fold that held its subject out, dense candidates. Frame accuracy (mean over videos) with the synthetic cuts of each p (experiments/datasets/ha4m_synthetic_cuts.json) and without cuts. Ranked by the worst case over p, then F1@50 and MAE in that condition, then simplicity.

| rank | model | embed weight | Engine A scale | window s | refine s | video_norm | worst | p 0.5 | p 0.75 | p 1 | no cuts | F1@50 (worst) | MAE s (worst) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | w10-engine-b-v2-w4-c10 | 1.0 | 0.0 | 10.0 | 0.0 | False | 90.47 | 90.47 | 90.49 | 91.36 | 89.35 | 94.26 | 6.06 |
| 2 | w10-engine-b-v2-w4-c10 | 2.0 | 0.25 | 10.0 | 0.0 | False | 90.46 | 90.53 | 90.46 | 91.71 | 89.18 | 94.26 | 5.97 |
| 3 | w10-engine-b-v2-w4-c10 | 0.2 | 0.0 | 10.0 | 0.0 | False | 90.46 | 90.46 | 90.47 | 91.33 | 89.35 | 94.26 | 6.07 |
| 4 | w10-engine-b-v2-w4-c10 | 0.5 | 0.0 | 10.0 | 0.0 | False | 90.46 | 90.46 | 90.48 | 91.35 | 89.35 | 94.26 | 6.07 |
| 5 | w10-engine-b-v2-w4-c10 | 2.0 | 0.0 | 10.0 | 0.0 | False | 90.46 | 90.46 | 90.48 | 91.35 | 89.35 | 94.26 | 6.07 |
| 6 | w10-engine-b-v2-w4-c10 | 2.0 | 1.0 | 10.0 | 0.0 | False | 90.27 | 90.27 | 91.17 | 92.23 | 89.14 | 94.26 | 6.06 |
| 7 | w10-engine-b-v2-w4-c10 | 0.5 | 0.25 | 10.0 | 0.0 | False | 90.27 | 90.27 | 91.17 | 92.23 | 89.14 | 94.26 | 6.06 |
| 8 | w10-engine-b-v2-w4-c10 | 1.0 | 0.5 | 10.0 | 0.0 | False | 90.26 | 90.26 | 91.17 | 92.22 | 89.14 | 94.26 | 6.06 |
| 9 | w10-engine-b-v2-w4-c10 | 2.0 | 0.5 | 10.0 | 0.0 | False | 90.22 | 90.22 | 90.93 | 91.54 | 89.18 | 94.26 | 6.07 |
| 10 | w10-engine-b-v2-w4-c10 | 1.0 | 0.25 | 10.0 | 0.0 | False | 90.21 | 90.21 | 90.93 | 91.53 | 89.18 | 94.26 | 6.07 |
| 11 | w10-engine-b-v2-w4-c05 | 0.5 | 0.0 | 10.0 | 0.0 | False | 89.72 | 90.12 | 89.72 | 91.04 | 88.86 | 93.74 | 6.04 |
| 12 | w10-engine-b-v2-w4-c05 | 1.0 | 0.0 | 10.0 | 0.0 | False | 89.72 | 90.12 | 89.72 | 91.02 | 88.86 | 93.74 | 6.04 |
| 13 | w10-engine-b-v2-w4-c05 | 0.2 | 0.0 | 10.0 | 0.0 | False | 89.71 | 90.12 | 89.71 | 91.01 | 88.86 | 93.74 | 6.04 |
| 14 | w10-engine-b-v2-w4-c05 | 2.0 | 0.0 | 10.0 | 0.0 | False | 89.71 | 90.12 | 89.71 | 91.04 | 88.86 | 93.74 | 6.04 |
| 15 | w10-engine-b-v2-w4-c05 | 2.0 | 1.0 | 10.0 | 0.0 | False | 89.58 | 89.95 | 89.58 | 91.13 | 88.59 | 93.08 | 5.89 |
| 16 | w10-engine-b-v2-w4-c05 | 1.0 | 0.5 | 10.0 | 0.0 | False | 89.58 | 89.94 | 89.58 | 91.12 | 88.59 | 93.08 | 5.89 |
| 17 | w10-engine-b-v2-w4-c05 | 0.5 | 0.25 | 10.0 | 0.0 | False | 89.58 | 89.95 | 89.58 | 91.13 | 88.59 | 93.08 | 5.89 |
| 18 | w10-engine-b-v2-w4-c05 | 2.0 | 0.25 | 10.0 | 0.0 | False | 89.35 | 89.7 | 89.35 | 90.9 | 88.6 | 93.1 | 5.95 |
| 19 | w10-engine-b-v2-w4-c05 | 2.0 | 0.5 | 10.0 | 0.0 | False | 89.27 | 89.75 | 89.27 | 90.71 | 88.6 | 92.71 | 5.94 |
| 20 | w10-engine-b-v2-w4-c05 | 1.0 | 0.25 | 10.0 | 0.0 | False | 89.27 | 89.75 | 89.27 | 90.69 | 88.6 | 92.71 | 5.94 |
| 21 | w10-engine-b-v2-w8-c10 | 2.0 | 0.5 | 10.0 | 0.0 | False | 89.08 | 89.08 | 89.41 | 90.55 | 87.14 | 92.3 | 6.09 |
| 22 | w10-engine-b-v2-w8-c10 | 1.0 | 0.25 | 10.0 | 0.0 | False | 89.08 | 89.08 | 89.41 | 90.53 | 87.14 | 92.3 | 6.09 |
| 23 | w10-engine-b-v2-w8-c10 | 2.0 | 1.0 | 10.0 | 0.0 | False | 88.9 | 88.9 | 89.4 | 91.52 | 86.97 | 92.43 | 6.1 |
| 24 | w10-engine-b-v2-w8-c10 | 1.0 | 0.5 | 10.0 | 0.0 | False | 88.89 | 88.89 | 89.4 | 91.52 | 86.97 | 92.43 | 6.1 |
| 25 | w10-engine-b-v2-w8-c10 | 0.5 | 0.25 | 10.0 | 0.0 | False | 88.89 | 88.89 | 89.4 | 91.51 | 86.97 | 92.43 | 6.1 |

Best per model:

| model | embed weight | Engine A scale | window s | refine s | video_norm | worst | p 0.5 | p 0.75 | p 1 | no cuts | F1@50 (worst) | MAE s (worst) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w10-engine-b-v1-cv | 0.5 | 0.5 | 10.0 | 0.0 | True | 69.91 | 69.91 | 76.33 | 83.69 | 65.79 | 66.04 | 8.46 |
| w10-engine-b-v2-w4-c05 | 0.5 | 0.0 | 10.0 | 0.0 | False | 89.72 | 90.12 | 89.72 | 91.04 | 88.86 | 93.74 | 6.04 |
| w10-engine-b-v2-w4-c10 | 1.0 | 0.0 | 10.0 | 0.0 | False | 90.47 | 90.47 | 90.49 | 91.36 | 89.35 | 94.26 | 6.06 |
| w10-engine-b-v2-w8-c05 | 2.0 | 1.0 | 10.0 | 0.0 | False | 88.25 | 88.76 | 88.25 | 91.34 | 85.78 | 91.23 | 5.98 |
| w10-engine-b-v2-w8-c10 | 2.0 | 0.5 | 10.0 | 0.0 | False | 89.08 | 89.08 | 89.41 | 90.55 | 87.14 | 92.3 | 6.09 |

Best per Engine A scale (all models):

| model | embed weight | Engine A scale | window s | refine s | video_norm | worst | p 0.5 | p 0.75 | p 1 | no cuts | F1@50 (worst) | MAE s (worst) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| w10-engine-b-v2-w4-c10 | 2.0 | 1.0 | 10.0 | 0.0 | False | 90.27 | 90.27 | 91.17 | 92.23 | 89.14 | 94.26 | 6.06 |
| w10-engine-b-v2-w4-c10 | 1.0 | 0.5 | 10.0 | 0.0 | False | 90.26 | 90.26 | 91.17 | 92.22 | 89.14 | 94.26 | 6.06 |
| w10-engine-b-v2-w4-c10 | 2.0 | 0.25 | 10.0 | 0.0 | False | 90.46 | 90.53 | 90.46 | 91.71 | 89.18 | 94.26 | 5.97 |
| w10-engine-b-v2-w4-c10 | 1.0 | 0.0 | 10.0 | 0.0 | False | 90.47 | 90.47 | 90.49 | 91.36 | 89.35 | 94.26 | 6.06 |

v1 selected: {'embed_weight': 0.5, 'engine_a_scale': 0.5, 'embed_window_s': 10.0, 'refine_window_s': 0.0, 'video_norm': True}.
v2 selected: w10-engine-b-v2-w4-c10 {'embed_weight': 1.0, 'engine_a_scale': 0.0, 'embed_window_s': 10.0, 'refine_window_s': 0.0, 'video_norm': False}.
