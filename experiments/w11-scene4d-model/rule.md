# Scene 4D model choice rule (written 2026-11-15, before any candidate is run)

**Goal.** A "Scene 4D" view like the D4RT demo: a coloured point cloud of the scene from the video, the
moving person inside it with motion trails, and the camera as a moving wireframe. The model makes per
frame depth (or points) and cameras; the stage scene4d turns them into points per shot part.

## Candidates

Open-d4rt (released checkpoint), VGGT, Pi3, CUT3R, MonST3R, MegaSaM. Each gets at most about one hour
of setup and test. A download above 5 GB is asked for first (VGGT-1B 5.03 GB, Open-d4rt 13.95 GB).

## Clips (public only)

From the public run run_20261025_063706_7fc4 (process_480p30, 854x480, 30 fps):
- **fixed:** shot part s021_p1 (600 frames, 20 s, the camera does not move);
- **handheld:** shot part s024_p1 (690 frames, 23 s, the worker walks about 1.3 m, the camera follows).

Sampled at 6 frames per second (every 5th frame: 120 and 138 frames), as the stage would (5 to 10 fps).
For this test only, the sampled public frames may be written to a temporary folder when a candidate's
code reads image files; the stage itself writes no frames.

## Gates (a candidate that fails one is out)

1. Runs on the RTX A5000 (24 GB) or the RTX A4000 (16 GB) for a whole clip, in chunks if needed.
2. At most 10 minutes of GPU time per minute of video at 6 fps (the clip time scaled to one minute).
3. Its licence allows our use (non commercial research is accepted); recorded.

## Measures

- **Fixed camera stays fixed:** on the fixed clip, the camera path's total rotation (degrees) and
  translation (as a share of the median scene depth). Lower is better; a static camera should be near 0.
- **Static background is stable:** on the fixed clip, for pixels outside the subject's box (our
  tracking), the spread over time of the 3D point of each pixel in the world frame, as a share of its
  depth (median over pixels). Lower is better.
- **Handheld:** the camera path is smooth (median frame to frame change of the camera rotation and
  position against its 5 frame moving average); and a visual check of the screenshots: the background
  stays in place while the camera moves, the person is separate from it and in proportion.
- Time per minute of video, peak GPU memory.

## Choice

Among the candidates that pass the gates: rank by the two fixed clip measures (each candidate's rank
summed) and break ties by the handheld visual check. The best two are shown to Khurshid with
screenshots before the stage is built; he decides.
