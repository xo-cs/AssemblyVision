# w11-vlm-speed

VLM speed: window and frames per request on IKEA ASM training videos (rule w11-vlm-speed).

40 videos (68.8 min), untuned equal weights, synthetic cuts p 0.5; frame accuracy, mean over videos. Speed: VLM scoring seconds per minute of video on GPU1 (server start not included).

Step 1, slots (3 videos, window 2 s, 4 frames):

| slots | windows | s per window |
| --- | --- | --- |
| 1 | 127 | 1.644 |
| 2 | 127 | 1.226 |
| 4 | 127 | 1.097 |

Step 2:

| file | window s | frames | slots | Kallax | Lack Coffee | worst | s per video min | s per window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vlm.npz (reference) | 2 | 4 | 2 | 54.92 | 75.63 | 54.92 | - | - |
| vlm_w2f4s4.npz | 2 | 4 | 4 | 56.1 | 73.99 | 56.1 | 31.67 | 1.048 |
| vlm_w2f2s4.npz | 2 | 2 | 4 | 50.33 | 73.1 | 50.33 | 24.34 | 0.807 |
| vlm_w3f4s4.npz | 3 | 4 | 4 | 53.93 | 74.14 | 53.93 | 21.36 | 1.055 |
| vlm_w3f2s4.npz | 3 | 2 | 4 | 55.13 | 74.14 | 55.13 | 16.41 | 0.815 |
| vlm_w4f4s4.npz | 4 | 4 | 4 | 51.74 | 80.4 | 51.74 | 16.11 | 1.058 |
| vlm_w4f2s4.npz | 4 | 2 | 4 | 50.47 | 74.14 | 50.47 | 12.36 | 0.815 |

Selected (rule): {'window_s': 3.0, 'frames': 2, 'slots': 4}.
Speed up against the current setting at the same slots: 1.93x.

Notes:

- Batching changes the VLM outputs: the current setting with 4 slots instead of 2 moves the worst case from 54.92 to 56.10 (Kallax +1.18, Lack Coffee -1.64). The 1 point margin of the rule is at that noise level, so the selected setting is as accurate as the current one within noise, not shown to be better.
- Speed: 3 s windows with 2 frames need 16.41 s of VLM per minute of video, against 31.67 s (current setting, 4 slots) and about 36.8 s (current setting, 2 slots, 1.226 s per window): 1.9x and 2.2x faster.
- Expected QiSens full run (9.9 min of video, estimate, not measured): VLM about 162 s plus 6 s server start, running on GPU1 during the pose pass (ingest to tracking about 265 s in w10-qisens-best), so it adds no time; SigLIP on GPU0 after tracking about 70 s (1190 samples, from 79 samples in 8.5 s on IKEA); the rest as in w10-qisens-best (340.7 s). Total about 410 s (6.8 min), against about 340 s for B v1. The pose pass ran about 10 % slower beside the VLM on one IKEA video (70.9 against 79.0 fps).
