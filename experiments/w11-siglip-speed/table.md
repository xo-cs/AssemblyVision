# w11-siglip-speed

SigLIP 2 at 1 fps against 2 fps (rule w11-siglip-speed): 40 IKEA ASM training videos, untuned B zero shot weights, VLM vlm_w3f2s4.npz, synthetic cuts p 0.5. Frame accuracy, mean over videos.

| SigLIP fps | Kallax | Lack Coffee | worst |
| --- | --- | --- | --- |
| 2 (siglip.npz) | 55.13 | 74.14 | 55.13 |
| 1 (siglip_f1.npz) | 55.12 | 73.79 | 55.12 |

Change of the worst case at 1 fps: -0.01. Decision (at most -1.0): **1 fps**.
Notes:

- Timing (GPU0, one public 148 s 1920x1080 video, model already loaded, two repeats each): 14.3 s at 2 fps, 9.4 s at 1 fps; the video decode alone is about 5 s of either. Scaled to the 9.9 min QiSens video (if it is 1080p, which was not checked): about 57 s at 2 fps and 38 s at 1 fps, so about 20 s less. Expected QiSens full run: about 251 s instead of 270.62 s, still above the 240 s milestone (estimate, not measured).
