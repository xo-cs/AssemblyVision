# w11-siglip-speed

SigLIP 2 at 1 fps against 2 fps (rule w11-siglip-speed): 40 IKEA ASM training videos, untuned B zero shot weights, VLM vlm_w3f2s4.npz, synthetic cuts p 0.5. Frame accuracy, mean over videos.

| SigLIP fps | Kallax | Lack Coffee | worst |
| --- | --- | --- | --- |
| 2 (siglip.npz) | 55.13 | 74.14 | 55.13 |
| 1 (siglip_f1.npz) | 55.12 | 73.79 | 55.12 |

Change of the worst case at 1 fps: -0.01. Decision (at most -1.0): **1 fps**.
