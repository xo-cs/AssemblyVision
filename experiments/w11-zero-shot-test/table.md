# w11-zero-shot-test

Zero shot phase scoring: the single test evaluation (rule w11-zero-shot)

Selected on w11-zero-shot-train: {'siglip_weight': 1.0, 'siglip_crop_weight': 4.0, 'vlm_weight': 0.5, 'siglip_temp': 0.01, 'shot_smooth': False}.

With synthetic cuts p 0.5:

| row | siglip | crop | vlm | temp | shot smooth | worst | Kallax | Lack Coffee | F1@50 Kallax | MAE s Kallax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Engine A | - | - | - | - | - | 48.23 | 53.3 | 48.23 | 45.71 | 9.79 |
| B v1 as shipped | - | - | - | - | - | 42.47 | 47.47 | 42.47 | 38.21 | 10.41 |
| untuned equal weights | 1.0 | 1.0 | 1.0 | 0.01 | False | 57.79 | 57.79 | 68.54 | 44.67 | 13.98 |
| selected (rule) | 1.0 | 4.0 | 0.5 | 0.01 | False | 50.43 | 62.66 | 50.43 | 51.6 | 8.89 |

Without cuts:

| row | siglip | crop | vlm | temp | shot smooth | worst | Kallax | Lack Coffee | F1@50 Kallax | MAE s Kallax |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Engine A | - | - | - | - | - | 33.01 | 33.01 | 60.61 | 10.36 | 23.48 |
| B v1 as shipped | - | - | - | - | - | 33.01 | 33.01 | 60.46 | 10.36 | 23.48 |
| untuned equal weights | 1.0 | 1.0 | 1.0 | 0.01 | False | 56.79 | 56.79 | 68.62 | 39.67 | 14.97 |
| selected (rule) | 1.0 | 4.0 | 0.5 | 0.01 | False | 52.5 | 62.11 | 52.5 | 51.7 | 9.61 |
