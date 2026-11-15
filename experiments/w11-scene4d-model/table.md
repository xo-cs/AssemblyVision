# w11-scene4d-model

Scene 4D model choice (rule w11-scene4d-model): public clips of process_480p30, fixed s021 (120 frames) and handheld s024 (138 frames) at 6 fps, RTX A4000 (16 GB).

| model | licence | gates | fixed: camera rotation deg | fixed: camera shift (share of depth) | fixed: background spread | handheld: camera jitter | s per video min (fixed / handheld) | peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pi3 (yyfz233/Pi3, 3.83 GB) | BSD-2-Clause | pass | 1.17 | 0.0212 | 0.0193 | 0.0007 | 135.0 / 157.0 | 13204 |
| CUT3R (cut3r_512_dpt_4_64, 3.17 GB, HF mirrors of the Drive file, 3 agree on SHA256) | CC BY-NC-SA 4.0 (repository LICENSE) | pass | 1.75 | 0.0324 | 0.0457 | 0.0009 | 44.3 / 44.8 | 8474 |
| MonST3R (Junyi42, 2.28 GB; + SEA-RAFT, SAM 2) | CC BY-NC-SA 4.0 | fails gate 2: 314 s for the 20 s fixed clip before finishing, about 15.7 min per video minute on the A4000 | - | - | - | - | - | - |
| MegaSaM (+ Depth Anything ViT-L 1.34 GB, UniDepth v2 1.45 GB, RAFT) | Apache 2.0 code; UniDepth CC BY-NC | fails gate 2: 521 s for the 20 s fixed clip, about 26 min per video minute on the A4000 | - | - | - | - | - | - |
| VGGT-1B | CC BY-NC 4.0 (a commercial variant exists) | not run: 5.03 GB download, needs approval | - | - | - | - | - | - |
| Pi3X | CC BY-NC 4.0 | not run: 5.44 GB download, needs approval | - | - | - | - | - | - |
| Open-d4rt (48 frame checkpoint) | Apache 2.0 code; checkpoint licence not stated | not run: 13.95 GB download, needs approval | - | - | - | - | - | - |

Ranking of the candidates that pass (rule: the two fixed clip measures, ranks summed): Pi3 first (1.17 deg, 0.019), CUT3R second (1.75 deg, 0.046). Pi3 needs 13.2 GB for 138 frames at once, so longer shots go in chunks; CUT3R works frame by frame (constant 8 GB) and is 3x faster.

Screenshots: logs/body3d_shots/scene4d_{pi3,cut3r}_{fixed,handheld}.png. Not complete: VGGT, Pi3X and Open-d4rt wait for the download approval.
