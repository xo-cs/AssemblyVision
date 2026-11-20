# Scene 4D quality rule (written 2026-11-20, before the candidates are run)

**Goal.** The 4D point clouds look scattered and low quality. First improve the Pi3 output, then test
VGGT-1B and Pi3X (downloads approved by Khurshid on 2026-11-20) the same way, and show screenshots side
by side before the stage switches model.

## Clips (public only; the same as rule w11-scene4d-model)

From the public run run_20261025_063706_7fc4 (process_480p30, 854x480, 30 fps): shot part s021_p1
(fixed camera, 20 s) and s024_p1 (handheld, 23 s), at 6 fps, through the real stage
(scene4d.run with only these two parts, chunks of 80 frames with 12 overlap, as in a run). The test copies
of the run are hard linked (milestone_run.make_copy), hidden from the Results list, and removed after the
screenshots. The models' raw points are kept under tmp/ only for the measures.

## Improvements of the stage (the same for every model)

Confidence: per frame the least confident 35 % of the pixels are left out (and conf < 0.3 always).
Depth edges: a point whose 4 neighbour differs by more than 3 % in distance to the camera is left out.
Background: voxels of median depth / 700 (was / 300) that are seen in at least 2 sampled frames, the median
colour per voxel (was the mean), statistical outlier removal (8 neighbours, 3 robust deviations). Player:
round points of their own world size (1.6 x the mean gap to the 8 neighbours), so they get smaller with
the distance to the viewer and close the gaps of sparse areas.

## Measures

The four camera and background measures of rule w11-scene4d-model (fixed: rotation, shift, background
spread; handheld: camera jitter), on the raw points, plus two of the final cloud:
- **cloud noise:** on the fixed clip, the median distance of a background point to the plane through its
  16 nearest neighbours, as a share of the median depth (lower is crisper);
- **background points** kept in the player, and time per video minute, peak GPU memory.

## Choice

Gates as in w11-scene4d-model (runs on our GPUs in chunks, at most 10 GPU minutes per video minute,
licence recorded: Pi3 BSD-2-Clause; VGGT-1B and Pi3X CC BY-NC 4.0, non commercial research only).
Rank the passing models by the fixed rotation, the background spread and the cloud noise (ranks summed),
ties broken by the screenshots. Screenshots in logs/ui_shots/scene4d_<model>_<clip>.png, taken in the
Results page player of the logged in app. Khurshid decides before scene4d.model changes.
