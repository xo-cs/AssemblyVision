# Scene 4D fusion rule (written 2026-11-21, before any fusion run)

**Goal.** Test whether a fusion of Pi3 improved and VGGT-1B gives a better 4D scene than either model
alone. Asked by Khurshid on 2026-11-21.

## Clips (public only; the same as w11-scene4d-quality)

Public run run_20261025_063706_7fc4 (process_480p30, 854x480, 30 fps): shot part s021_p1 (fixed camera,
120 samples) and s024_p1 (handheld, 138 samples), at 6 fps, chunks of 80 frames with 12 overlap, on the
RTX A4000. All three candidates run in this session with the same code and the same measure script, so
their numbers are compared with each other and not with the table of w11-scene4d-quality.

## Candidates

- **Pi3 improved** (scene4d.model = "pi3").
- **VGGT-1B** (scene4d.model = "vggt").
- **Fusion** (scene4d.model = "fusion"), fixed here:
  1. Pi3 is the base: the cameras of the part are Pi3's, unchanged.
  2. VGGT runs on the same chunks. Each VGGT chunk is aligned to the Pi3 part by a similarity transform
     on the cameras of its frames (camera_sim3).
  3. Both models' background points (after each model's confidence and depth edge filters, no person
     boxes) go into one voxel grid of median Pi3 depth / 700. Per voxel and model: mean distance to the
     Pi3 camera of the frame, mean confidence, number of frames.
  4. **Agree:** both models have the voxel or a voxel next to it, and their depths differ by at most
     2 % of the depth. Kept; both contribute points.
  5. **Only one model** (the other has no point within the neighbourhood): kept only if that model's
     mean confidence in the voxel is at least its own 70th percentile over all its voxels, and the voxel
     is seen in at least 2 frames.
  6. **Both, but they disagree:** the voxel of the model with the higher confidence percentile is kept.
  7. Colour: median per voxel. Then the outlier removal and the point sizes of the stage.
  The settings 2 %, 70th percentile and the grid are not tuned on these clips.

## Measures

The first four as in w11-scene4d-quality; for each measure lower is better unless noted.

1. **Fixed camera rotation** (degrees), on the fixed clip.
2. **Background spread**, on the fixed clip: the spread over time of the 3D point of each background
   pixel as a share of its depth. The fusion has no per pixel track of its own, so it is measured on
   the kept points: for each kept voxel the spread of the raw points of its source frames that fall in it
   or in a neighbour voxel, as a share of the depth (median). The same computation is applied to Pi3 and
   VGGT alone, so all three are measured the same way.
3. **Cloud noise:** median distance of a background point to the plane through its 16 nearest
   neighbours, as a share of the median depth (fixed clip).
4. **Coverage (higher is better):** the share of the image pixels of the sampled frames, outside the
   person boxes, that the final background cloud covers when projected into the Pi3 camera of the frame
   (a pixel counts when a point lands within 1 pixel at 1/4 image resolution). Mean of both clips.
5. **Holes behind the person (lower is better):** the same projection, on the pixels inside the union of
   the person boxes over the part that are outside the person box in at least 20 % of the frames (so
   the background there was visible at some time): the share not covered. Mean of both clips.
6. **Time per video minute** (s), the larger of the two clips, on the A4000. Gate: at most 600 s, as in
   w11-scene4d-model. Peak GPU memory is recorded.

## Choice

Gates: runs on our GPUs in chunks; at most 600 s per video minute.

The candidates that pass are ranked 1, 2, 3 on each of the five quality measures (1 to 5), a tie shares
the better rank, and the ranks are summed; the lowest sum is first.

**The fusion wins only if** it has the strictly lowest sum, is not worse than Pi3 improved by more than
10 % on fixed camera rotation or cloud noise, and the screenshots show no visible double surfaces. Then
scene4d.model = "fusion", and the 4D of the flagship (command for Khurshid) and the 6 public demo runs is
made with it. **Otherwise** Pi3 improved stays: those 4D outputs are made with Pi3 improved, and the VGGT
and Pi3X test files (weights, their hidden test run copies, raw test points) are removed.

Licence: the fusion contains VGGT-1B (CC BY-NC 4.0), so it is non commercial research only; recorded.

Screenshots: logs/ui_shots/scene4d_fusion_{fixed,handheld}.png (Pi3 improved, VGGT, fusion, side by
side, Results page player, the same zoom).
