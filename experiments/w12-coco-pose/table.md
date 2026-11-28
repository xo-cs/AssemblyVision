# w12-coco-pose

COCO 2017 val, YOLO26 x pose with the flagship's settings (rule.md). Freeze validation-freeze-1, method unchanged: True. 2878 persons with all four shoulders and hips labelled visible; 2828 matched to a prediction (98.3 %).

**COCO is not factory footage:** these numbers say how well the keypoints are found in everyday photos, not on QiSens.

| joint | PCK@0.1, all persons | PCK@0.1, matched persons | labelled visible |
| --- | --- | --- | --- |
| left shoulder | 0.964 | 0.981 | 2878 |
| right shoulder | 0.967 | 0.984 | 2878 |
| left hip | 0.938 | 0.954 | 2878 |
| right hip | 0.939 | 0.955 | 2878 |
| left wrist | 0.934 | 0.947 | 2170 |
| right wrist | 0.936 | 0.949 | 2195 |

| trunk angle | persons | with a system angle | mean abs difference deg | median deg | exact band | within one band |
| --- | --- | --- | --- | --- | --- | --- |
| all | 2878 | 2800 | 2.58 | 1.61 | 0.955 | 0.999 |
| bent (human angle over 45 deg) | 155 | 150 | 7.0 | 4.52 | 0.747 | 1.0 |

Human bands B0 to B3: [2407, 316, 77, 78]. Targets (frozen 2026-11-26): #3 exact band at least 0.80 and within one band at least 0.95; #6 wrist PCK@0.1 at least 0.85 (here 0.935, mean of both wrists, all persons).
