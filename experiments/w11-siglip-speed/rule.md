# SigLIP speed rule (written 2026-11-14, before any result at 1 fps)

**Question.** Can the siglip_scores stage sample 1 frame per second instead of 2 (half the SigLIP time
of a full run), accuracy first? QiSens is never used to decide.

**Known before this rule.** QiSens full default run 270.62 s (w11-qisens-zero-shot-timing; the SigLIP
stage time on QiSens was not reported). On one IKEA video SigLIP took 8.5 s for 79 samples, about 4 s of
it loading the model.

**Method.** The stored siglip.npz files (2 fps) of the 40 IKEA ASM training videos of w11-vlm-speed hold
the 1 fps samples too: every other sample (0, 1, 2 s, ...). A 1 fps file is made from them on the CPU
(siglip_f1.npz, each sample covering 1 s), so no GPU job is needed.

**Decoder.** "B zero shot" untuned equal weights, the VLM setting chosen by w11-vlm-speed
(vlm_w3f2s4.npz), synthetic cuts p 0.5.

**Criterion.** 1 fps becomes the default if the worst case over the two sets (Kallax Shelf Drawer,
Lack Coffee Table) of the mean frame accuracy drops by at most 1 point against 2 fps. Otherwise 2 fps stays.

**Reported.** Both rows per set, and the expected QiSens full run time against the 240 s milestone.
