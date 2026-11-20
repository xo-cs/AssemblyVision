# w11-scene4d-quality

Scene 4D quality (rule w11-scene4d-quality): public process_480p30, fixed s021_p1 (120 samples) and handheld s024_p1 (138 samples) at 6 fps, through the stage on the RTX A4000 (16 GB), chunks of 80 frames with 12 overlap unless noted. "before" is the stored output of the stage of 15 Nov (measured from its files; its raw points were not kept, so no background spread).

| model | licence | gates | chunk | fixed: camera rotation deg | fixed: camera shift (share of depth) | fixed: background spread | fixed: cloud noise (share of depth) | handheld: camera jitter | background points (fixed / handheld) | s per video min (fixed / handheld) | peak MB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pi3 before (stage of 15 Nov) | BSD-2-Clause | pass | 80 | 8.9 | 0.131 | - | 0.00072 | 0.0008 | 106780 / 67778 | - / - | - |
| Pi3 improved | BSD-2-Clause | pass | 80 | 1.13 | 0.0187 | 0.0308 | 0.00055 | 0.0008 | 120291 / 132227 | 288.3 / 341.0 | 10068 |
| VGGT-1B (facebook/VGGT-1B, 5.03 GB) | CC BY-NC 4.0 | pass | 80 | 0.93 | 0.0112 | 0.0272 | 0.00059 | 0.0002 | 137450 / 144255 | 396.0 / 292.4 | 13956 |
| Pi3X (yyfz233/Pi3X, 5.44 GB) | CC BY-NC 4.0 | fails gate 2 on the fixed clip (748 s per video minute on the A4000, 15.7 GB; 40 frame chunks, 80 run out of memory) | 40 | 1.11 | 0.0103 | 0.0346 | 0.00057 | 0.0004 | 120963 / 114134 | 747.6 / 429.4 | 15753 |

The improved stage (the same code for every model): chunks are stitched by the cameras of their shared frames (the fit on the points alone let the rotation jump by 8.9 degrees at the chunk border of the fixed clip, which laid two copies of the scene over each other); per frame confidence (the least sure 35 % out); depth edges out; voxels of depth / 700 seen in 2 or more frames, with the median colour; statistical outlier removal; round points sized by the neighbour gap in the player. The times include writing the raw points for the measures (tests only).

Ranking of the models that pass (rule: fixed rotation, background spread, cloud noise; ranks summed): VGGT-1B 1 + 1 + 3 = 5, Pi3 improved 3 + 2 + 1 = 6 (Pi3X 2 + 3 + 2 = 7, and it fails the time gate). The differences are small (0.93 against 1.13 degrees, 0.027 against 0.031, 0.00059 against 0.00055). In the screenshots VGGT and Pi3X give denser surfaces but leave black holes behind the person, and Pi3X's cloud is cut to a round patch; Pi3 improved is the most complete. Not switched: Khurshid decides (scene4d.model = "pi3", "vggt" or "pi3x").

Screenshots (Results page player in the logged in app, the same zoom): logs/ui_shots/scene4d_compare_{fixed,handheld}.png (2 x 2: Pi3 before, Pi3 improved, VGGT-1B, Pi3X) and scene4d_<model>_<clip>.png. The hidden test copies (Runs page, show all runs): before run_20261120_081955_aa95, Pi3 run_20261120_080210_2e74, VGGT run_20261120_081437_3ccf, Pi3X run_20261120_080709_d203; they are removed after the decision.

Weights: models/scene4d/vggt/model.safetensors SHA256 f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e; models/scene4d/pi3x/model.safetensors SHA256 69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a.
