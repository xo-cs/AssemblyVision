# w11-other-class

seg_a.other_class on IKEA ASM: natural ends and synthetic intros and outros (rule w11-other-class).

40 IKEA ASM training videos (w11-vlm-speed/videos.json), untuned B zero shot weights, synthetic cuts p 0.5, VLM file vlm_w3f2s4.npz (on: its _other version). Frame accuracy, mean over videos.

| condition | Kallax | Lack Coffee | worst | F1@50 Kallax | MAE s Kallax |
| --- | --- | --- | --- | --- | --- |
| natural, off | 55.13 | 74.14 | 55.13 | 38.73 | 10.92 |
| natural, on | 52.77 | 81.11 | 52.77 | 32.87 | 10.03 |
| synthetic intro and outro, off | 20.3 | 35.13 | 20.3 | 6.68 | 13.69 |
| synthetic intro and outro, on | 48.75 | 39.01 | 39.01 | 10.72 | 9.36 |

Change of the worst case with other_class on: natural -2.36, synthetic +18.71.
Decision (rule: natural at least -1.0 and synthetic at least +2.0): **off**.
